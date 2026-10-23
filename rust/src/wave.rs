//! Wavefront engines: all rows of a block walk ONE tree together, level by level. The rows that are still inside the tree sit in a
//! compact list of (row, node) items; every step takes the next items of the list (the lanes are refilled from it, never idle),
//! advances them one node, writes the finished ones' leaf values and appends the others to the next list. There is no branch
//! that depends on the data, and the tree (a few KB) stays in L1 while the block's rows stream through it. The leaves are added
//! to the row's margin one tree at a time, in tree order, so the float32 sums are bit-identical to xgboost's and to the walker's.
//!   engine 1: scalar, any CPU (x86-64, aarch64, ...)      engine 2: AVX2 gathers + left-pack by permute table (x86-64 only), chosen at run time;
//!   on any other CPU engine 2 runs engine 1.

use crate::finish::{predict_blocks, Meta, Out};
use crate::{Model, FLAG_LATE};

pub const ENGINE_WAVE: u32 = 1;
pub const ENGINE_AVX2: u32 = 2;

/// Downfall (gather data sampling): its microcode fix makes every AVX2 gather cost ~26 cycles (measured on an i7-8550U), which makes the
/// gather engine slower than the scalar one. Linux reports it in this file; "Not affected" CPUs keep the gather engine.
/// TREEFLAT_FORCE_GATHER=1 (tests, benchmarks) keeps the gather engine on a mitigated CPU.
#[cfg(target_arch = "x86_64")]
fn gathers_slow() -> bool {
    static SLOW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SLOW.get_or_init(|| {
        std::env::var_os("TREEFLAT_FORCE_GATHER").is_none()
            && std::fs::read_to_string(
                "/sys/devices/system/cpu/vulnerabilities/gather_data_sampling",
            )
            .is_ok_and(|s| {
                let s = s.trim_start();
                s.starts_with("Mitigation") || s.starts_with("Vulnerable")
            })
    })
}

pub fn avx2_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("popcnt")
            && !gathers_slow()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Rows per block: the block's features (rows * n_features * 4 bytes) plus a tree should fit the L1 data cache.
fn block_for(n_features: usize) -> usize {
    (24 * 1024 / ((n_features + 1) * 4)).clamp(32, 256) / 8 * 8
}

// the gather engine's lists are only read by `tree_avx2`, which exists on x86-64 only
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub struct Scratch {
    ent: [Vec<u64>; 2],
    rows: [Vec<i32>; 2],
    cur: [Vec<i32>; 2],
    done_row: Vec<i32>,
    done_val: Vec<u32>,
    /// the block's rows copied with one spare column per row (`stride = n_features + 1`): the spare column receives the leaf value
    xs: Vec<f32>,
}

impl Scratch {
    fn new(block: usize, stride: usize) -> Scratch {
        let n = block + 16;
        Scratch {
            ent: [vec![0; n], vec![0; n]],
            rows: [vec![0; n], vec![0; n]],
            cur: [vec![0; n], vec![0; n]],
            done_row: vec![0; n],
            done_val: vec![0; n],
            xs: vec![0.0; block * stride + 16],
        }
    }
}

/// Copies the block's rows into `xs` with a spare column per row; says whether any value is NaN.
fn copy_block(xb: &[f32], rows: usize, nf: usize, xs: &mut [f32]) -> bool {
    let mut nan = false;
    for (src, dst) in xb
        .chunks_exact(nf)
        .zip(xs.chunks_exact_mut(nf + 1))
        .take(rows)
    {
        dst[..nf].copy_from_slice(src);
        nan |= src.iter().fold(false, |a, v| a | v.is_nan());
    }
    nan
}

/// One tree, scalar, over the block copy `xs` (row stride `nf + 1`). With `NAN` false no value is NaN, so the comparison is the bare
/// `v < threshold`. SAFETY: see `Model::walk` (every reachable node index is inside the tree, features < n_features, a leaf has
/// distance 0); `xs` holds `rows` rows of `nf + 1` floats, `acc` has room for `rows` floats, the scratch for `rows + 16` items.
#[inline(never)]
unsafe fn tree_scalar<const NAN: bool>(
    nodes: *const u64,
    root: usize,
    nf: usize,
    rows: usize,
    s: &mut Scratch,
    acc: *mut f32,
) {
    let stride = nf + 1;
    let xs = s.xs.as_mut_ptr();
    let (mut a, mut b) = (s.ent[0].as_mut_ptr(), s.ent[1].as_mut_ptr());
    // The first level is the same node for every row: read it once and make the first list directly.
    let n0 = nodes.add(root);
    let (thr0, dist0) = (*n0.cast::<f32>(), u64::from(*n0.cast::<u16>().add(2)));
    if dist0 == 0 {
        for r in 0..rows {
            *acc.add(r) += thr0; // a tree that is a single leaf
        }
        return;
    }
    let w0 = *n0;
    let feat0 = ((w0 >> 48) & 0x7fff) as usize;
    for r in 0..rows {
        let v = *xs.add(r * stride + feat0);
        let left = if NAN {
            (v < thr0) | (v.is_nan() & (w0 >> 63 != 0))
        } else {
            v < thr0
        };
        *a.add(r) = ((r * stride) as u64) << 32
            | (root as u64 + std::hint::select_unpredictable(left, 1u64, dist0));
    }
    let mut n = rows;
    while n > 0 {
        let mut m = 0usize;
        for i in 0..n {
            let e = *a.add(i);
            let node = nodes.add(e as u32 as usize);
            let at = (e >> 32) as usize;
            // the node's fields read one by one: threshold, distance to the right child (0 at a leaf), feature (and the default bit)
            let thr = *node.cast::<f32>();
            let dist = u64::from(*node.cast::<u16>().add(2));
            let ft = *node.cast::<u16>().add(3);
            let v = *xs.add(at + usize::from(ft & 0x7fff));
            // every visit writes the node's threshold field to the row's spare column: at a leaf (the row's last visit) it is the
            // leaf value, and the last write wins
            *xs.add(at + nf) = thr;
            // `left` is `v < thr`, or for a missing value the node's default direction (bit 15 of `ft`)
            let left = if NAN {
                (v < thr) | (v.is_nan() & (ft >> 15 != 0))
            } else {
                v < thr
            };
            // a conditional move: LLVM turns a plain `if` here into a branch on the coin toss
            *b.add(m) = e + std::hint::select_unpredictable(left, 1u64, dist);
            m += usize::from(dist != 0);
        }
        std::mem::swap(&mut a, &mut b);
        n = m;
    }
    for r in 0..rows {
        *acc.add(r) += *xs.add(r * stride + nf);
    }
}

#[cfg(target_arch = "x86_64")]
const fn build_lut() -> [[i32; 8]; 256] {
    let mut t = [[0i32; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut k, mut j) = (0, 0);
        while j < 8 {
            if (m >> j) & 1 == 1 {
                t[m][k] = j;
                k += 1;
            }
            j += 1;
        }
        m += 1;
    }
    t
}
/// For each 8-bit mask, the permutation that moves the lanes whose bit is set to the front, in order.
#[cfg(target_arch = "x86_64")]
static LUT: [[i32; 8]; 256] = build_lut();

/// One tree, AVX2: eight (row, node) items per step. SAFETY: as `tree_scalar`, and the CPU has AVX2 and POPCNT.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
#[inline(never)]
unsafe fn tree_avx2(
    nodes: *const u64,
    root: usize,
    xb: *const f32,
    nf: usize,
    rows: usize,
    s: &mut Scratch,
    acc: *mut f32,
) {
    use std::arch::x86_64::*;
    let nfv = _mm256_set1_epi32(nf as i32);
    let m_ffff = _mm256_set1_epi32(0xffff);
    let m_7fff = _mm256_set1_epi32(0x7fff);
    let one = _mm256_castsi256_ps(_mm256_set1_epi32(1));
    let zero = _mm256_setzero_si256();
    let nodes32 = nodes as *const i32;
    let (mut ra, mut rb) = (s.rows[0].as_mut_ptr(), s.rows[1].as_mut_ptr());
    let (mut ca, mut cb) = (s.cur[0].as_mut_ptr(), s.cur[1].as_mut_ptr());
    let (dr, dv) = (s.done_row.as_mut_ptr(), s.done_val.as_mut_ptr());
    for r in 0..rows {
        *ra.add(r) = r as i32;
        *ca.add(r) = root as i32;
    }
    let (mut n, mut nd) = (rows, 0usize);
    while n > 0 {
        let pad = (8 - n % 8) % 8; // the unused lanes of the last vector point at the root of row 0 and are masked out below
        for k in n..n + pad {
            *ra.add(k) = 0;
            *ca.add(k) = root as i32;
        }
        let mut nb = 0usize;
        let mut i = 0usize;
        while i < n {
            let valid: u32 = if n - i >= 8 {
                0xff
            } else {
                (1u32 << (n - i)) - 1
            };
            let rv = _mm256_loadu_si256(ra.add(i).cast());
            let cv = _mm256_loadu_si256(ca.add(i).cast());
            let thr_bits = _mm256_i32gather_epi32::<8>(nodes32, cv);
            let meta = _mm256_i32gather_epi32::<8>(nodes32.add(1), cv);
            let dist = _mm256_and_si256(meta, m_ffff);
            let feat = _mm256_and_si256(_mm256_srli_epi32::<16>(meta), m_7fff);
            let idx = _mm256_add_epi32(_mm256_mullo_epi32(rv, nfv), feat);
            let v = _mm256_i32gather_ps::<4>(xb, idx);
            let thr = _mm256_castsi256_ps(thr_bits);
            let lt = _mm256_cmp_ps::<_CMP_LT_OQ>(v, thr);
            let nge = _mm256_cmp_ps::<_CMP_NGE_UQ>(v, thr);
            let dl = _mm256_castsi256_ps(_mm256_srai_epi32::<31>(meta));
            // left = v < thr, or, for NaN, the default direction: !(v >= thr) is true for NaN
            let left = _mm256_blendv_ps(lt, nge, dl);
            let step = _mm256_blendv_ps(_mm256_castsi256_ps(dist), one, left);
            let ncur = _mm256_add_epi32(cv, _mm256_castps_si256(step));
            let leaf_bits =
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(dist, zero))) as u32;
            let (ms, md) = (!leaf_bits & valid, leaf_bits & valid);
            let p = _mm256_loadu_si256(LUT[ms as usize].as_ptr().cast());
            _mm256_storeu_si256(rb.add(nb).cast(), _mm256_permutevar8x32_epi32(rv, p));
            _mm256_storeu_si256(cb.add(nb).cast(), _mm256_permutevar8x32_epi32(ncur, p));
            nb += ms.count_ones() as usize;
            let q = _mm256_loadu_si256(LUT[md as usize].as_ptr().cast());
            _mm256_storeu_si256(dr.add(nd).cast(), _mm256_permutevar8x32_epi32(rv, q));
            _mm256_storeu_si256(dv.add(nd).cast(), _mm256_permutevar8x32_epi32(thr_bits, q));
            nd += md.count_ones() as usize;
            i += 8;
        }
        std::mem::swap(&mut ra, &mut rb);
        std::mem::swap(&mut ca, &mut cb);
        n = nb;
    }
    for j in 0..nd {
        *acc.add(*dr.add(j) as usize) += f32::from_bits(*dv.add(j));
    }
}

impl Model {
    /// Predicts with a wavefront engine (`engine` is ENGINE_WAVE or ENGINE_AVX2; AVX2 falls back to the scalar one when the CPU
    /// lacks it). Same results, bit for bit, as the walker.
    pub(crate) fn predict_wave<O: Out>(
        &self,
        x: &[f32],
        out: &mut [O],
        bm: Option<&[f64]>,
        threads: usize,
        engine: u32,
    ) {
        let avx2 = engine == ENGINE_AVX2 && avx2_ok();
        let block = block_for(self.n_features);
        let nf = self.n_features;
        let nodes = self.nodes();
        let meta = Meta {
            n_features: nf,
            n_groups: self.n_groups,
            n_folds: self.n_folds,
            objective: self.objective,
            average: self.average,
            base: self.base.clone(),
            late: self.flags & FLAG_LATE != 0,
        };
        let fold_block =
            |s: &mut Scratch, f: usize, xb: &[f32], rows: usize, margin: &mut [f32]| {
                let (first, count) = self.folds[f];
                let nan = !avx2 && copy_block(xb, rows, nf, &mut s.xs);
                for &(root, group) in &self.trees[first..first + count] {
                    // SAFETY: `from_storage` validated the tree; `xb` has `rows * nf` floats; `margin[group * block..]` has `block >= rows`
                    // floats; the scratch was sized for `block`. The AVX2 path is taken only when the CPU has it.
                    unsafe {
                        let acc = margin.as_mut_ptr().add(group * block);
                        #[cfg(target_arch = "x86_64")]
                        if avx2 {
                            tree_avx2(nodes.as_ptr(), root, xb.as_ptr(), nf, rows, s, acc);
                            continue;
                        }
                        if nan {
                            tree_scalar::<true>(nodes.as_ptr(), root, nf, rows, s, acc);
                        } else {
                            tree_scalar::<false>(nodes.as_ptr(), root, nf, rows, s, acc);
                        }
                    }
                }
            };
        predict_blocks(
            &meta,
            block,
            x,
            out,
            bm,
            threads,
            &|| Scratch::new(block, nf + 1),
            &fold_block,
        );
    }
}
