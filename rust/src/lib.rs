//! Exact evaluator of tree forests (xgboost, LightGBM, YDF) compiled to ONE flat file. It computes what the source library computes:
//! float32 thresholds, `value < threshold` goes left, NaN follows the node's default direction; for xgboost float32 sums in tree
//! order and a float32 softmax or sigmoid, for LightGBM and YDF double sums. A forest may hold several models (the folds of a k-fold
//! ensemble); their results are averaged. C ABI for ctypes; no dependencies.
//!
//! File (little endian). Header, 64 bytes:
//!   0 magic `TREEFLT1` | 8 u32 format: 4 holds what xgboost can express; 5 adds double sums, exp, identity with several groups,
//!   zero-as-missing nodes and categorical nodes; 6 adds linear leaves (formats 1 to 3 were written before the checksum covered the
//!   header: they are refused, the model must be compiled again) | 12 u32 n_features | 16 u32 n_groups (margin columns = output columns)
//!   20 u32 n_folds | 24 u32 n_trees | 28 u32 n_nodes | 32 u32 objective (0 identity, 1 sigmoid, 2 softmax, 3 exp; formats 5 and 6: identity and exp may have several groups)
//!   36 u32 average (0: mean of the folds' outputs, 1: mean of their margins, then the objective) | 40 u64 FNV-1a 64 of the WHOLE file,
//!   these eight bytes read as zero (so a damaged header is refused like a damaged body) | 48 u32 flags (0 in format 4): bit 0 margins
//!   summed, transformed and averaged in f64 (leaves stay f32); bit 1 extended nodes (below); bit 2 linear leaves (format 6); bit 3
//!   float32 sums with the base margin added late, in double
//!   52 u32 cat_words: length of the category table | 56..64: format 4 and 5, an f64 margin scale (0 means 1, NaN means models with
//!   different scales); format 6, u32 lin_leaves and u32 lin_coefs. The margin scale is the factor a caller's base margin is multiplied
//!   by (LightGBM binary `sigmoid:s`, folded into the leaves). The library reads it from the verified image (`tfl_margin_scale`); in a
//!   format 6 file it sits in a marked record that follows the last linear leaf record and no leaf refers to.
//! Then `n_folds` fold records of 8 bytes (u32 first tree, u32 number of trees), the base margins (`n_folds * n_groups` f32,
//! padded to 8 bytes), `n_trees` tree records of 8 bytes (u32 root node, u32 group), `n_nodes` nodes of 8 bytes. A node is one
//! u64: bits 0..32 the f32 threshold (the leaf value for a leaf), 32..48 the distance to the right child (0 for a leaf),
//! 48..63 the feature, bit 63 the default direction is left. Trees are laid out depth first: the left child is the next node.
//! Extended nodes (flag bit 1): the feature takes bits 48..61 (<= 8,191 features) and bit 61 marks a ZERO-AS-MISSING node (NaN and
//! |x| <= 1.0000000180025095e-35 go the default way, else `x < threshold`), bit 62 a CATEGORICAL node: `trunc(x)` is looked up in a bitset,
//! left when its bit is set, right for NaN, for values below 0 after truncation toward zero, and for absent categories; the threshold bits are then the u32 offset of the node's
//! bitset in the category table (`cat_words` u32 after the last node, padded to 8 bytes): [n_words, word 0, word 1, ...].
//! Linear leaves (format 6, flag bit 2, which needs bit 0: LightGBM `linear_tree`): header 56 u32 lin_leaves, 60 u32
//! lin_coefs; after the category table, `lin_leaves` records of 24 bytes (f64 constant, f64 fallback, u32 first coefficient, u32 count)
//! and `lin_coefs` entries of 16 bytes (u32 feature, u32 zero, f64 coefficient). A linear LEAF is a node with bit 63 set, bits 32..63
//! zero and bits 0..32 the index of its record: its value is `constant + sum coef * x[feature]` in f64, in table order, or the fallback
//! when any of those features is NaN (what LightGBM's predictor does); the other leaves keep their f32 value.

use std::ffi::{c_char, CStr};

mod finish;
mod wave;

use finish::{Acc, Out, EXP, MEAN_MARGIN, SIGMOID, SOFTMAX};
#[cfg(test)]
use finish::{IDENTITY, MEAN_OUTPUT};

#[cfg(not(target_endian = "little"))]
compile_error!("the flat format is little endian only");

const MAGIC: &[u8; 8] = b"TREEFLT1";
/// The format numbers (header byte 8). 1 to 3 were the first layout, whose checksum covered the body only.
const FORMAT_FIRST_LAYOUT_LAST: u32 = 3;
const FORMAT_XGB: u32 = 4;
const FORMAT_DOUBLE: u32 = 5;
const FORMAT_LINEAR: u32 = 6;
/// Bytes 40..48 of the header hold the checksum, which is computed with those eight bytes read as zero.
const CHECKSUM_AT: usize = 40;
/// A finite double no real leaf fallback is: the fallback of the record that holds the margin scale (backend_flat.py writes it).
const SCALE_MARKER_BITS: u64 = 0x5153_4341_4C45_5343;
const FLAG_SUM64: u32 = 1;
const FLAG_EXT: u32 = 2;
const FLAG_LINEAR: u32 = 4;
/// float32 sums whose base margin is added late, in double (YDF with float32 sums); needs !SUM64
const FLAG_LATE: u32 = 8;
const LIN_REC: usize = 24;
const COEF_REC: usize = 16;
/// LightGBM's kZeroThreshold as a float32
const ZERO_F32: f32 = 1e-35;
const HEADER: usize = 64;
const FOLD_REC: usize = 8;
const TREE_REC: usize = 8;
const MAX_FEATURES: usize = 0x7fff;
const MAX_GROUPS: usize = 1024;
/// Rows per block: a block's features stay in the L1/L2 cache while every tree is walked over them (measured on a 42-feature
/// 4,500-tree forest: 64 rows 4.6k rows/s, 256 5.4k, 1024 6.5k, 4096 6.3k).
const BLOCK: usize = 1024;
/// Rows walked in lock step through one tree: the walks are independent, so the CPU overlaps their memory waits.
const LANES: usize = 4;
#[cfg(test)]
const DEFAULT_LEFT: u64 = 1 << 63;

/// Engines (the `engine` argument of `tfl_predict_ex`): 0 the lock-step walker below, 1 and 2 the wavefront engines of wave.rs.
pub const ENGINE_WALKER: u32 = 0;

#[cfg(unix)]
mod sys {
    use std::ffi::c_void;
    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
    }
}

/// Which way a row goes at an extended node (1 left, 0 right). Plain nodes: `v < thr`, NaN the default way. Zero-as-missing nodes:
/// NaN and |v| <= ZERO_F32 the default way. Categorical nodes: trunc(v) looked up in the node's bitset.
///
/// SAFETY: `cats` is the category table of a file `from_storage` validated: every categorical node's set lies inside it.
#[inline(always)]
unsafe fn left_ext(w: u64, v: f32, thr: f32, cats: *const u32) -> u64 {
    let default_left = w >> 63;
    if (w >> 62) & 1 == 1 {
        // `as i64` truncates toward zero and saturates; NaN is handled first
        let c = v as i64;
        if v.is_nan() || c < 0 {
            return 0;
        }
        let off = (w as u32) as usize;
        let word = (c >> 5) as usize;
        let hit =
            word < *cats.add(off) as usize && (*cats.add(off + 1 + word) >> (c & 31)) & 1 == 1;
        return u64::from(hit);
    }
    if (w >> 61) & 1 == 1 && (v.is_nan() || v.abs() <= ZERO_F32) {
        return default_left;
    }
    u64::from(v < thr) | (u64::from(v.is_nan()) & default_left)
}

/// The file image: a read-only memory map (unix) or an owned, 8-byte aligned copy.
enum Storage {
    Owned(Vec<u64>, usize),
    #[cfg(unix)]
    Mapped(*mut u8, usize),
}

// SAFETY: the mapping is read-only and never mutated; the owned copy is plain data.
unsafe impl Send for Storage {}
// SAFETY: as above; shared references only ever read.
unsafe impl Sync for Storage {}

impl Storage {
    fn bytes(&self) -> &[u8] {
        // SAFETY: both variants point at `len` initialised bytes that live as long as `self`.
        match self {
            Storage::Owned(v, len) => unsafe {
                std::slice::from_raw_parts(v.as_ptr().cast(), *len)
            },
            #[cfg(unix)]
            Storage::Mapped(p, len) => unsafe { std::slice::from_raw_parts(*p, *len) },
        }
    }

    fn copy_of(b: &[u8]) -> Storage {
        let mut v = vec![0u64; b.len().div_ceil(8)];
        // SAFETY: `v` holds at least `b.len()` bytes; u64 has no invalid bit patterns.
        unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), b.len()) }
            .copy_from_slice(b);
        Storage::Owned(v, b.len())
    }

    #[cfg(unix)]
    fn map(path: &str) -> Result<Storage, String> {
        use std::os::fd::AsRawFd;
        // Look before opening: opening a FIFO with no writer blocks forever. (The file could still be swapped for a FIFO between the two
        // calls; the second check below covers what is then open.)
        if !std::fs::metadata(path)
            .map_err(|e| format!("{path}: {e}"))?
            .is_file()
        {
            return Err(format!("{path}: not a regular file"));
        }
        let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let meta = file.metadata().map_err(|e| format!("{path}: {e}"))?;
        if !meta.is_file() {
            return Err(format!("{path}: not a regular file"));
        }
        let len = usize::try_from(meta.len()).map_err(|_| format!("{path}: file too large"))?;
        if len == 0 {
            return Err(format!("{path}: empty file"));
        }
        // PROT_READ = 1, MAP_PRIVATE = 2 on Linux and macOS.
        // SAFETY: a fresh read-only private mapping of an open file; the result is checked.
        let p = unsafe { sys::mmap(std::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0) };
        if p as isize == -1 {
            return Err(format!("{path}: mmap failed"));
        }
        Ok(Storage::Mapped(p.cast(), len))
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Storage::Mapped(p, len) = self {
            // SAFETY: `p`/`len` come from a successful mmap that nothing else unmaps.
            unsafe { sys::munmap(p.cast(), *len) };
        }
    }
}

/// Pointers to the linear-leaf tables of a validated file (record: 3 u64, coefficient entry: 2 u64).
#[derive(Clone, Copy)]
struct Lin {
    recs: *const u64,
    coefs: *const u64,
}

pub struct Model {
    storage: Storage,
    n_features: usize,
    n_groups: usize,
    n_folds: usize,
    n_trees: usize,
    n_nodes: usize,
    objective: u32,
    average: u32,
    flags: u32,
    node_at: usize,
    /// byte offset of the category table
    cats_at: usize,
    /// byte offsets of the linear-leaf records and of the coefficient table (format 6; 0 otherwise)
    lin_at: usize,
    coef_at: usize,
    /// per fold: `n_groups` base margins
    base: Vec<f32>,
    /// per fold: (first tree, number of trees)
    folds: Vec<(usize, usize)>,
    /// per tree: (root node, group)
    trees: Vec<(usize, usize)>,
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;

/// FNV-1a 64 continued from the state `h`.
fn fnv1a_from(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_from(FNV_OFFSET, bytes)
}

/// The checksum a file carries: FNV-1a 64 of every byte of the file, the checksum field itself read as eight zero bytes. `b` holds at
/// least the 64-byte header.
fn file_checksum(b: &[u8]) -> u64 {
    let h = fnv1a_from(FNV_OFFSET, &b[..CHECKSUM_AT]);
    let h = fnv1a_from(h, &[0u8; 8]);
    fnv1a_from(h, &b[CHECKSUM_AT + 8..])
}

impl Model {
    /// Parses and VALIDATES a file image. Everything the hot loop relies on is checked here, once: header, sizes, checksum,
    /// group and feature indices, and for every tree that its nodes form a forward-pointing, in-range binary tree.
    fn from_storage(storage: Storage) -> Result<Model, String> {
        let b = storage.bytes();
        if b.len() < HEADER || &b[0..8] != MAGIC {
            return Err("not a treeflat file (magic)".into());
        }
        let (objective, average) = (u32_at(b, 32), u32_at(b, 36));
        let format = u32_at(b, 8);
        if (1..=FORMAT_FIRST_LAYOUT_LAST).contains(&format) {
            return Err(format!(
                "format {format} is the layout of an earlier version (its checksum left the header out): compile the model again"
            ));
        }
        let (flags, cat_words) = if (FORMAT_DOUBLE..=FORMAT_LINEAR).contains(&format) {
            (u32_at(b, 48), u32_at(b, 52) as usize)
        } else {
            (0, 0)
        };
        let (lin_leaves, lin_coefs) = if format == FORMAT_LINEAR {
            (u32_at(b, 56) as usize, u32_at(b, 60) as usize)
        } else {
            (0, 0)
        };
        let linear = flags & FLAG_LINEAR != 0;
        if !(FORMAT_XGB..=FORMAT_LINEAR).contains(&format)
            || objective > EXP
            || (format == FORMAT_XGB && objective > SOFTMAX)
            || average > MEAN_MARGIN
            || flags & !(FLAG_SUM64 | FLAG_EXT | FLAG_LINEAR | FLAG_LATE) != 0
            || (flags & FLAG_EXT == 0 && cat_words != 0)
            || linear != (format == FORMAT_LINEAR)
            || (linear && (flags & FLAG_SUM64 == 0 || lin_leaves == 0))
            || (flags & FLAG_LATE != 0 && flags & FLAG_SUM64 != 0)
        {
            return Err("unknown format, objective, flags or average rule (file written by a newer version?)".into());
        }
        let ext = flags & FLAG_EXT != 0;
        let (n_features, n_groups, n_folds, n_trees, n_nodes) = (
            u32_at(b, 12) as usize,
            u32_at(b, 16) as usize,
            u32_at(b, 20) as usize,
            u32_at(b, 24) as usize,
            u32_at(b, 28) as usize,
        );
        if n_features == 0
            || n_features > MAX_FEATURES
            || n_groups == 0
            || n_groups > MAX_GROUPS
            || (objective == SOFTMAX) != (n_groups > 1)
                && (format == FORMAT_XGB || objective == SOFTMAX)
            || (objective == SIGMOID && n_groups > 1)
            || n_folds == 0
            || n_trees == 0
            || n_nodes == 0
        {
            return Err("counts out of range".into());
        }
        let base_bytes = (n_folds * n_groups * 4).next_multiple_of(8);
        let tree_at = HEADER + n_folds * FOLD_REC + base_bytes;
        let node_at = tree_at + n_trees * TREE_REC;
        let cats_at = n_nodes
            .checked_mul(8)
            .and_then(|s| s.checked_add(node_at))
            .ok_or("sizes overflow")?;
        let lin_at = cat_words
            .checked_mul(4)
            .and_then(|s| s.checked_next_multiple_of(8))
            .and_then(|s| s.checked_add(cats_at))
            .ok_or("sizes overflow")?;
        let coef_at = lin_leaves
            .checked_mul(LIN_REC)
            .and_then(|s| s.checked_add(lin_at))
            .ok_or("sizes overflow")?;
        let size = lin_coefs
            .checked_mul(COEF_REC)
            .and_then(|s| s.checked_add(coef_at))
            .ok_or("sizes overflow")?;
        if b.len() != size || !(b.as_ptr() as usize + node_at).is_multiple_of(8) {
            return Err(format!("file is {} bytes, the header says {size}", b.len()));
        }
        if file_checksum(b)
            != u64::from_le_bytes(b[CHECKSUM_AT..CHECKSUM_AT + 8].try_into().unwrap())
        {
            return Err("checksum mismatch".into());
        }
        let mut folds = Vec::with_capacity(n_folds);
        let mut next_tree = 0usize;
        for f in 0..n_folds {
            let at = HEADER + f * FOLD_REC;
            let (first, count) = (u32_at(b, at) as usize, u32_at(b, at + 4) as usize);
            if first != next_tree || count == 0 || first + count > n_trees {
                return Err("fold records do not tile the trees".into());
            }
            next_tree += count;
            folds.push((first, count));
        }
        if next_tree != n_trees {
            return Err("fold records do not cover every tree".into());
        }
        let base_at = HEADER + n_folds * FOLD_REC;
        let base: Vec<f32> = (0..n_folds * n_groups)
            .map(|i| f32::from_bits(u32_at(b, base_at + 4 * i)))
            .collect();
        let mut trees: Vec<(usize, usize)> = Vec::with_capacity(n_trees);
        for t in 0..n_trees {
            let at = tree_at + t * TREE_REC;
            let (root, group) = (u32_at(b, at) as usize, u32_at(b, at + 4) as usize);
            if group >= n_groups || root >= n_nodes || trees.last().is_some_and(|&(r, _)| root <= r)
            {
                return Err(format!("tree {t}: bad root or group"));
            }
            trees.push((root, group));
        }
        if trees[0].0 != 0 {
            return Err("first tree does not start at node 0".into());
        }
        // SAFETY: `node_at` is 8-byte aligned (checked above) and `n_nodes * 8` bytes follow it (size check).
        let nodes: &[u64] =
            unsafe { std::slice::from_raw_parts(b.as_ptr().add(node_at).cast(), n_nodes) };
        for (t, &(root, _)) in trees.iter().enumerate() {
            let end = trees.get(t + 1).map_or(n_nodes, |&(r, _)| r);
            for (i, &w) in nodes.iter().enumerate().take(end).skip(root) {
                let right = ((w >> 32) & 0xffff) as usize;
                if right != 0 {
                    let feature = ((w >> 48) & if ext { 0x1fff } else { 0x7fff }) as usize;
                    if feature >= n_features || right < 2 || i + 1 >= end || i + right >= end {
                        return Err(format!("tree {t}, node {i}: child or feature out of range"));
                    }
                    if ext && (w >> 62) & 1 == 1 {
                        let off = (w as u32) as usize;
                        let n = if off < cat_words {
                            u32_at(b, cats_at + 4 * off) as usize
                        } else {
                            usize::MAX
                        };
                        if n > cat_words || off + 1 + n > cat_words {
                            return Err(format!(
                                "tree {t}, node {i}: category set outside the table"
                            ));
                        }
                    } else if f32::from_bits(w as u32).is_nan() {
                        return Err(format!("tree {t}, node {i}: NaN threshold"));
                    }
                } else if linear && w >> 63 == 1 {
                    // a linear leaf: bits 32..63 zero, bits 0..32 the record
                    if (w >> 32) & 0x7fff_ffff != 0 || ((w as u32) as usize) >= lin_leaves {
                        return Err(format!("tree {t}, node {i}: bad linear leaf"));
                    }
                } else if w >> 32 != 0 {
                    return Err(format!("tree {t}, node {i}: leaf with child fields"));
                }
            }
        }
        // SAFETY: the tables lie inside the file (size check) and start 8-byte aligned (`node_at` is, and every table size is a multiple of 8).
        let tab = |at: usize, words: usize| unsafe {
            std::slice::from_raw_parts(b.as_ptr().add(at).cast::<u64>(), words)
        };
        let coefs = tab(coef_at, lin_coefs * 2);
        for k in 0..lin_coefs {
            let (feature, coef) = (coefs[2 * k], f64::from_bits(coefs[2 * k + 1]));
            if (feature >> 32) != 0 || (feature as u32) as usize >= n_features {
                return Err(format!("linear coefficient {k}: feature out of range"));
            }
            if !coef.is_finite() {
                return Err(format!("linear coefficient {k}: not a finite number"));
            }
        }
        let recs = tab(lin_at, lin_leaves * 3);
        for k in 0..lin_leaves {
            let (off, cnt) = (
                recs[3 * k + 2] as u32 as usize,
                (recs[3 * k + 2] >> 32) as usize,
            );
            if !f64::from_bits(recs[3 * k]).is_finite()
                || !f64::from_bits(recs[3 * k + 1]).is_finite()
                || off + cnt > lin_coefs
            {
                return Err(format!(
                    "linear leaf record {k}: bad constant, fallback or coefficient range"
                ));
            }
        }
        Ok(Model {
            storage,
            n_features,
            n_groups,
            n_folds,
            n_trees,
            n_nodes,
            objective,
            average,
            flags,
            node_at,
            cats_at,
            lin_at,
            coef_at,
            base,
            folds,
            trees,
        })
    }

    pub fn from_bytes(b: &[u8]) -> Result<Model, String> {
        Model::from_storage(Storage::copy_of(b))
    }

    /// Memory-maps the file (unix) or reads it, then validates it.
    pub fn open(path: &str) -> Result<Model, String> {
        #[cfg(unix)]
        let storage = Storage::map(path)?;
        #[cfg(not(unix))]
        let storage = Storage::copy_of(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?);
        Model::from_storage(storage)
    }

    /// The factor a caller's base margin is multiplied by (header notes), read from the verified image: the f64 in header bytes 56..64
    /// (0 means 1), or, in a format 6 file, the constant of the marked record after the last linear leaf record (1 when that record is an
    /// ordinary one).
    fn margin_scale(&self) -> f64 {
        let b = self.storage.bytes();
        let word = |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
        if self.flags & FLAG_LINEAR == 0 {
            let s = f64::from_bits(word(56));
            return if s == 0.0 { 1.0 } else { s };
        }
        // a linear file has at least one record (checked in `from_storage`)
        let last = self.coef_at - LIN_REC;
        if word(last + 8) == SCALE_MARKER_BITS && word(last + 16) == 0 {
            f64::from_bits(word(last))
        } else {
            1.0
        }
    }

    fn nodes(&self) -> &[u64] {
        // SAFETY: alignment and length were checked in `from_storage`; the storage outlives the borrow.
        unsafe {
            std::slice::from_raw_parts(
                self.storage.bytes().as_ptr().add(self.node_at).cast(),
                self.n_nodes,
            )
        }
    }

    /// Walks `LANES` rows of one block through one tree in lock step; returns their leaf nodes (a plain leaf holds its f32 value in the
    /// low 32 bits, a linear leaf has bit 63 set and its record in the low 32 bits).
    ///
    /// SAFETY of the unchecked indexing: `from_storage` proved that every node index reachable from a tree's root stays in
    /// `[root, end)` of that tree (children are at `i + 1` and `i + right`, both inside the tree, both greater than `i`),
    /// that each feature index is < `n_features`, and that a leaf's step is 0, so a finished lane stays on its leaf. The
    /// caller guarantees `x` holds the `LANES` rows starting at `first_row`.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)] // the lane index also addresses the row's features
    unsafe fn walk<const EXT: bool>(
        &self,
        nodes: *const u64,
        cats: *const u32,
        root: usize,
        x: &[f32],
        first_row: usize,
    ) -> [u64; LANES] {
        let nf = self.n_features;
        let xp = x.as_ptr().add(first_row * nf);
        let mut cur = [root; LANES];
        loop {
            let mut any = 0u64;
            for l in 0..LANES {
                let w = *nodes.add(cur[l]);
                let right = (w >> 32) & 0xffff;
                let internal = u64::from(right != 0);
                let feature = ((w >> 48) & if EXT { 0x1fff } else { 0x7fff }) as usize;
                let v = *xp.add(l * nf + feature);
                let thr = f32::from_bits(w as u32);
                // Branch free on purpose: which way a row goes is a coin toss for the CPU's predictor, and a wrong guess costs more
                // than the whole step. `left` is 1 for `v < thr`, or, for a missing value, the node's default direction.
                let left = if EXT {
                    left_ext(w, v, thr, cats)
                } else {
                    u64::from(v < thr) | (u64::from(v.is_nan()) & (w >> 63))
                };
                // 1 (the next node) when going left, the distance to the right child otherwise; a leaf has right 0, internal 0
                cur[l] +=
                    right.wrapping_add(left.wrapping_mul(internal.wrapping_sub(right))) as usize;
                any |= internal;
            }
            if any == 0 {
                break;
            }
        }
        let mut out = [0u64; LANES];
        for l in 0..LANES {
            out[l] = *nodes.add(cur[l]);
        }
        out
    }

    /// A linear leaf's value for the row at `xrow` (its `n_features` f32): constant + sum coef * x in table order, in f64; the fallback
    /// when one of the leaf's features is NaN.
    ///
    /// SAFETY: `lin` points at the tables of a file `from_storage` validated; `rec` is a record index of it; `xrow` holds `n_features` f32.
    #[inline(always)]
    unsafe fn linear_value(lin: Lin, rec: usize, xrow: *const f32) -> f64 {
        let r = lin.recs.add(rec * 3);
        let (c, fallback, oc) = (f64::from_bits(*r), f64::from_bits(*r.add(1)), *r.add(2));
        let e = lin.coefs.add((oc as u32 as usize) * 2);
        let mut out = c;
        for k in 0..(oc >> 32) as usize {
            let v = *xrow.add(*e.add(2 * k) as u32 as usize);
            if v.is_nan() {
                return fallback;
            }
            out += f64::from_bits(*e.add(2 * k + 1)) * f64::from(v);
        }
        out
    }

    /// Averaged outputs for the rows `x` (row major, `n_features` f32 each) into `out` (`n_groups` per row).
    fn predict_rows<A: Acc, O: Out, const EXT: bool, const LIN: bool>(
        &self,
        x: &[f32],
        out: &mut [O],
        bm: Option<&[f64]>,
    ) {
        let (nf, g) = (self.n_features, self.n_groups);
        let n_rows = x.len() / nf;
        let nodes = self.nodes().as_ptr();
        // SAFETY: `cats_at` lies inside the storage (checked in `from_storage`); with no table the pointer is never read.
        let cats: *const u32 = unsafe { self.storage.bytes().as_ptr().add(self.cats_at).cast() };
        // SAFETY: both table offsets lie inside the storage (checked in `from_storage`); with no linear leaves they are never read.
        let lin = unsafe {
            let p = self.storage.bytes().as_ptr();
            Lin {
                recs: p.add(self.lin_at).cast(),
                coefs: p.add(self.coef_at).cast(),
            }
        };
        let mut acc = vec![A::default(); BLOCK * g];
        let mut margin = vec![A::default(); BLOCK * g];
        let mut row = vec![A::default(); g];
        let mut pad = vec![0f32; LANES * nf];
        // float32 sums with a base margin: the margin joins late, in double (see `finish::LateAcc`)
        let mut late = bm
            .filter(|_| self.flags & FLAG_LATE != 0)
            .map(|b| (finish::LateAcc::new(BLOCK, g), b));
        let mut start = 0;
        while start < n_rows {
            let rows = BLOCK.min(n_rows - start);
            let xb = &x[start * nf..(start + rows) * nf];
            acc[..rows * g].fill(A::default());
            if let Some((la, _)) = late.as_mut() {
                la.clear(rows, g);
            }
            let full = rows - rows % LANES;
            let used = rows - full;
            // the short tail of a block (fewer than LANES rows) is padded once by repeating its last row; the pads' results are dropped
            for l in 0..if used > 0 { LANES } else { 0 } {
                let src = full + l.min(used - 1);
                pad[l * nf..(l + 1) * nf].copy_from_slice(&xb[src * nf..(src + 1) * nf]);
            }
            for (f, &(first, count)) in self.folds.iter().enumerate() {
                for (i, m) in margin[..rows * g].chunks_exact_mut(g).enumerate() {
                    for (v, b) in m.iter_mut().zip(&self.base[f * g..(f + 1) * g]) {
                        *v = A::from_f32(*b);
                    }
                    if let (Some(bm), None) = (bm, &late) {
                        for (v, b) in m.iter_mut().zip(&bm[(start + i) * g..(start + i + 1) * g]) {
                            v.add_f64(*b);
                        }
                    }
                }
                for &(root, group) in &self.trees[first..first + count] {
                    for r in (0..full).step_by(LANES) {
                        // SAFETY: r + LANES <= full <= rows, so the lanes' rows lie inside `xb`; see `walk`.
                        let leaves = unsafe { self.walk::<EXT>(nodes, cats, root, xb, r) };
                        for (l, leaf) in leaves.iter().enumerate() {
                            let m = &mut margin[(r + l) * g + group];
                            if LIN && *leaf >> 63 == 1 {
                                // SAFETY: the record index was validated; the row `r + l` lies inside `xb`.
                                m.add_f64(unsafe {
                                    Model::linear_value(
                                        lin,
                                        *leaf as u32 as usize,
                                        xb.as_ptr().add((r + l) * nf),
                                    )
                                });
                            } else {
                                m.add_leaf(f32::from_bits(*leaf as u32));
                            }
                        }
                    }
                    if used > 0 {
                        // SAFETY: `pad` holds LANES rows of nf features.
                        let leaves = unsafe { self.walk::<EXT>(nodes, cats, root, &pad, 0) };
                        for (l, leaf) in leaves.iter().take(used).enumerate() {
                            let m = &mut margin[(full + l) * g + group];
                            if LIN && *leaf >> 63 == 1 {
                                // SAFETY: as above; `pad` holds LANES rows.
                                m.add_f64(unsafe {
                                    Model::linear_value(
                                        lin,
                                        *leaf as u32 as usize,
                                        pad.as_ptr().add(l * nf),
                                    )
                                });
                            } else {
                                m.add_leaf(f32::from_bits(*leaf as u32));
                            }
                        }
                    }
                }
                if let Some((la, b)) = late.as_mut() {
                    let raw = |r: usize, gi: usize| margin[r * g + gi].to_f32();
                    let mean_output = self.average == finish::MEAN_OUTPUT;
                    la.add_fold(
                        rows,
                        g,
                        self.objective,
                        mean_output,
                        raw,
                        &b[start * g..(start + rows) * g],
                    );
                    continue;
                }
                for (m, a) in margin[..rows * g]
                    .chunks_exact_mut(g)
                    .zip(acc[..rows * g].chunks_exact_mut(g))
                {
                    if self.average == finish::MEAN_OUTPUT {
                        A::transform(self.objective, m);
                    }
                    for (a, v) in a.iter_mut().zip(m.iter()) {
                        a.add(*v);
                    }
                }
            }
            let o = &mut out[start * g..(start + rows) * g];
            if let Some((la, _)) = late.as_mut() {
                la.write(
                    rows,
                    g,
                    self.objective,
                    self.average == MEAN_MARGIN,
                    self.n_folds,
                    o,
                );
                start += rows;
                continue;
            }
            for (m, a) in o.chunks_exact_mut(g).zip(acc[..rows * g].chunks_exact(g)) {
                for gi in 0..g {
                    row[gi] = a[gi].mean(self.n_folds);
                }
                if self.average == MEAN_MARGIN {
                    A::transform(self.objective, &mut row);
                }
                for gi in 0..g {
                    m[gi] = O::from_acc(row[gi]);
                }
            }
            start += rows;
        }
    }

    fn rows_dispatch<O: Out>(&self, x: &[f32], out: &mut [O], bm: Option<&[f64]>) {
        let (sum64, ext, lin) = (
            self.flags & FLAG_SUM64 != 0,
            self.flags & FLAG_EXT != 0,
            self.flags & FLAG_LINEAR != 0,
        );
        match (sum64, ext, lin) {
            (false, false, _) => self.predict_rows::<f32, O, false, false>(x, out, bm),
            (false, true, _) => self.predict_rows::<f32, O, true, false>(x, out, bm),
            (true, false, false) => self.predict_rows::<f64, O, false, false>(x, out, bm),
            (true, true, false) => self.predict_rows::<f64, O, true, false>(x, out, bm),
            (true, false, true) => self.predict_rows::<f64, O, false, true>(x, out, bm),
            (true, true, true) => self.predict_rows::<f64, O, true, true>(x, out, bm),
        }
    }

    /// Is this engine available on this CPU? (The AVX2 one is not where Downfall's microcode fix makes gathers slow; see `wave::avx2_ok`.)
    pub fn engine_ok(engine: u32) -> bool {
        match engine {
            ENGINE_WALKER | wave::ENGINE_WAVE => true,
            wave::ENGINE_AVX2 => wave::avx2_ok(),
            _ => false,
        }
    }

    /// Like `predict`, with an explicit engine. An unavailable AVX2 engine runs the scalar wavefront one (the caller can ask
    /// `engine_ok` first); an unknown engine is an error.
    pub fn predict_engine(
        &self,
        x: &[f32],
        out: &mut [f32],
        threads: usize,
        engine: u32,
    ) -> Result<(), String> {
        self.predict_engine_with(x, out, None, threads, engine)
    }

    /// Like `predict_engine`, with a base margin: `n_rows * n_groups` doubles (row major), added to the raw score of every fold before
    /// the objective's transform.
    pub fn predict_engine_with<O: Out>(
        &self,
        x: &[f32],
        out: &mut [O],
        bm: Option<&[f64]>,
        threads: usize,
        engine: u32,
    ) -> Result<(), String> {
        let (nf, g) = (self.n_features, self.n_groups);
        if !x.len().is_multiple_of(nf) || out.len() != x.len() / nf * g {
            return Err("x or out has the wrong length".into());
        }
        if bm.is_some_and(|b| b.len() != out.len()) {
            return Err("the base margin has the wrong length".into());
        }
        match engine {
            ENGINE_WALKER => self.predict_with(x, out, bm, threads),
            wave::ENGINE_WAVE | wave::ENGINE_AVX2 => {
                if self.flags & !FLAG_LATE != 0 {
                    return Err("the wavefront engines do not run double-sum, extended or linear-leaf (LightGBM / YDF) files; use engine 0".into());
                }
                self.predict_wave(x, out, bm, threads, engine);
                Ok(())
            }
            _ => Err(format!("unknown engine {engine}")),
        }
    }

    /// `threads` > 1 splits the rows into contiguous chunks of whole blocks, one scoped thread each.
    pub fn predict(&self, x: &[f32], out: &mut [f32], threads: usize) -> Result<(), String> {
        self.predict_with(x, out, None, threads)
    }

    /// `predict` with a base margin (see `predict_engine_with`).
    pub fn predict_with<O: Out>(
        &self,
        x: &[f32],
        out: &mut [O],
        bm: Option<&[f64]>,
        threads: usize,
    ) -> Result<(), String> {
        let (nf, g) = (self.n_features, self.n_groups);
        if !x.len().is_multiple_of(nf) || out.len() != x.len() / nf * g {
            return Err("x or out has the wrong length".into());
        }
        if bm.is_some_and(|b| b.len() != out.len()) {
            return Err("the base margin has the wrong length".into());
        }
        let n_rows = x.len() / nf;
        let threads = threads.clamp(1, 64).min(n_rows.div_ceil(BLOCK).max(1));
        if threads == 1 {
            self.rows_dispatch(x, out, bm);
            return Ok(());
        }
        let per = n_rows.div_ceil(threads).div_ceil(BLOCK) * BLOCK;
        std::thread::scope(|s| {
            for (i, (xc, oc)) in x.chunks(per * nf).zip(out.chunks_mut(per * g)).enumerate() {
                let bmc = bm.map(|b| &b[i * per * g..((i + 1) * per * g).min(b.len())]);
                s.spawn(move || self.rows_dispatch(xc, oc, bmc));
            }
        });
        Ok(())
    }
}

/// The caller's `x` (`n_rows * n_features` f32) and `out` (`n_rows * n_groups` T) as slices, or None when a size overflows, exceeds what
/// a slice may hold, or a needed pointer is null (the C functions answer 1, a length problem, instead of reading from nowhere).
///
/// # Safety
/// Whenever the sizes are sane and the pointers non-null, `x` and `out` must be valid for those lengths (the contract of the C functions).
unsafe fn io_slices<'a, T>(
    m: &Model,
    x: *const f32,
    n_rows: usize,
    out: *mut T,
) -> Option<(&'a [f32], &'a mut [T])> {
    let nx = n_rows.checked_mul(m.n_features)?;
    let no = n_rows.checked_mul(m.n_groups)?;
    let limit = isize::MAX as usize;
    if nx.checked_mul(4)? > limit || no.checked_mul(std::mem::size_of::<T>().max(1))? > limit {
        return None;
    }
    if (nx > 0 && x.is_null()) || (no > 0 && out.is_null()) {
        return None;
    }
    // SAFETY: sizes and null-ness were checked above; validity of the memory is the caller's contract.
    Some(if nx == 0 || no == 0 {
        (&[], &mut [])
    } else {
        (
            std::slice::from_raw_parts(x, nx),
            std::slice::from_raw_parts_mut(out, no),
        )
    })
}

/// The base margin (`n_rows * n_groups` f64) as a slice; `Ok(None)` for a null pointer, Err on an overflow.
///
/// # Safety
/// A non-null `bm` must be valid for `n_rows * n_groups` f64.
unsafe fn margin_slice<'a>(
    m: &Model,
    bm: *const f64,
    n_rows: usize,
) -> Result<Option<&'a [f64]>, ()> {
    if bm.is_null() {
        return Ok(None);
    }
    let n = n_rows.checked_mul(m.n_groups).ok_or(())?;
    if n.checked_mul(8).ok_or(())? > isize::MAX as usize {
        return Err(());
    }
    // SAFETY: non-null, size checked; validity is the caller's contract.
    Ok(Some(std::slice::from_raw_parts(bm, n)))
}

fn put_err(err: *mut c_char, cap: usize, msg: &str) {
    if err.is_null() || cap == 0 {
        return;
    }
    let n = msg.len().min(cap - 1);
    // SAFETY: the caller gives a writable buffer of `cap` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(msg.as_ptr(), err.cast::<u8>(), n);
        *err.add(n) = 0;
    }
}

/// Opens and validates a forest file. Returns null (and a message in `err`) on any problem.
///
/// # Safety
/// `path` is a NUL-terminated string; `err` is null or a writable buffer of `err_cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn tfl_open(
    path: *const c_char,
    err: *mut c_char,
    err_cap: usize,
) -> *mut Model {
    if path.is_null() {
        put_err(err, err_cap, "path is null");
        return std::ptr::null_mut();
    }
    let path = match CStr::from_ptr(path).to_str() {
        Ok(p) => p,
        Err(_) => {
            put_err(err, err_cap, "path is not UTF-8");
            return std::ptr::null_mut();
        }
    };
    match Model::open(path) {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(e) => {
            put_err(err, err_cap, &e);
            std::ptr::null_mut()
        }
    }
}

/// Writes [n_features, n_groups, n_folds, n_trees, n_nodes, objective, average] into `out` (7 u32).
///
/// # Safety
/// `m` comes from `tfl_open`; `out` has room for 7 u32.
#[no_mangle]
pub unsafe extern "C" fn tfl_info(m: *const Model, out: *mut u32) {
    if m.is_null() || out.is_null() {
        return;
    }
    let m = &*m;
    let info = [
        m.n_features,
        m.n_groups,
        m.n_folds,
        m.n_trees,
        m.n_nodes,
        m.objective as usize,
        m.average as usize,
    ];
    for (i, v) in info.iter().enumerate() {
        *out.add(i) = *v as u32;
    }
}

/// The factor a caller's base margin is multiplied by (LightGBM binary `sigmoid:s`, folded into the leaves; NaN or negative: the models of
/// a fold set disagree), read from the file image `tfl_open` verified. 1 for a null handle.
///
/// # Safety
/// `m` comes from `tfl_open`.
#[no_mangle]
pub unsafe extern "C" fn tfl_margin_scale(m: *const Model) -> f64 {
    if m.is_null() {
        return 1.0;
    }
    (*m).margin_scale()
}

/// Predicts `n_rows` rows (f32, row major) into `out` (`n_rows * n_groups` f32). Returns 0, or 1 on a length problem.
///
/// # Safety
/// `x` holds `n_rows * n_features` f32, `out` has room for `n_rows * n_groups` f32.
#[no_mangle]
pub unsafe extern "C" fn tfl_predict(
    m: *const Model,
    x: *const f32,
    n_rows: usize,
    out: *mut f32,
    threads: u32,
) -> i32 {
    if m.is_null() {
        return 1;
    }
    let m = &*m;
    let Some((x, out)) = io_slices(m, x, n_rows, out) else {
        return 1;
    };
    match m.predict(x, out, threads as usize) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Like `tfl_predict`, with an engine: 0 walker, 1 scalar wavefront, 2 AVX2 wavefront (the scalar one when the CPU lacks AVX2).
/// Returns 0, 1 on a length problem, 2 on an unknown engine.
///
/// # Safety
/// As `tfl_predict`.
#[no_mangle]
pub unsafe extern "C" fn tfl_predict_ex(
    m: *const Model,
    x: *const f32,
    n_rows: usize,
    out: *mut f32,
    threads: u32,
    engine: u32,
) -> i32 {
    if m.is_null() {
        return 1;
    }
    let m = &*m;
    let Some((x, out)) = io_slices(m, x, n_rows, out) else {
        return 1;
    };
    match m.predict_engine(x, out, threads as usize, engine) {
        Ok(()) => 0,
        Err(e) if e.starts_with("unknown engine") => 2,
        Err(_) => 1,
    }
}

/// Like `tfl_predict_ex`, plus a base margin: `n_rows * n_groups` f64 (row major, or null for none) added to every fold's raw score
/// before the objective's transform. Returns 0, 1 on a length problem, 2 on an unknown engine.
///
/// # Safety
/// As `tfl_predict_ex`; `base_margin` is null or holds `n_rows * n_groups` f64.
#[no_mangle]
pub unsafe extern "C" fn tfl_predict_margin(
    m: *const Model,
    x: *const f32,
    n_rows: usize,
    out: *mut f32,
    threads: u32,
    engine: u32,
    base_margin: *const f64,
) -> i32 {
    if m.is_null() {
        return 1;
    }
    let m = &*m;
    let Some((x, out)) = io_slices(m, x, n_rows, out) else {
        return 1;
    };
    let Ok(bm) = margin_slice(m, base_margin, n_rows) else {
        return 1;
    };
    match m.predict_engine_with(x, out, bm, threads as usize, engine) {
        Ok(()) => 0,
        Err(e) if e.starts_with("unknown engine") => 2,
        Err(_) => 1,
    }
}

/// Like `tfl_predict_margin`, with the answers as f64 (`out` holds `n_rows * n_groups` f64). A double-sum forest (LightGBM, YDF) gives
/// its sum without the final rounding to float32; a float32 forest (xgboost) gives its float32 result widened. Returns 0, 1 on a
/// length problem, 2 on an unknown engine.
///
/// # Safety
/// As `tfl_predict_margin`, with `out` an f64 array.
#[no_mangle]
pub unsafe extern "C" fn tfl_predict_f64(
    m: *const Model,
    x: *const f32,
    n_rows: usize,
    out: *mut f64,
    threads: u32,
    engine: u32,
    base_margin: *const f64,
) -> i32 {
    if m.is_null() {
        return 1;
    }
    let m = &*m;
    let Some((x, out)) = io_slices(m, x, n_rows, out) else {
        return 1;
    };
    let Ok(bm) = margin_slice(m, base_margin, n_rows) else {
        return 1;
    };
    match m.predict_engine_with(x, out, bm, threads as usize, engine) {
        Ok(()) => 0,
        Err(e) if e.starts_with("unknown engine") => 2,
        Err(_) => 1,
    }
}

/// 1 if this engine runs natively on this CPU, else 0.
#[no_mangle]
pub extern "C" fn tfl_engine_ok(engine: u32) -> i32 {
    i32::from(Model::engine_ok(engine))
}

/// The file checksum (FNV-1a 64), for the compiler, so that both sides use the same implementation.
///
/// # Safety
/// `data` holds `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tfl_fnv1a(data: *const u8, len: usize) -> u64 {
    if len == 0 || data.is_null() {
        return fnv1a(&[]);
    }
    fnv1a(std::slice::from_raw_parts(data, len))
}

/// # Safety
/// `m` comes from `tfl_open` and is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn tfl_close(m: *mut Model) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NF: usize = 300; // more than 255 features on purpose: the stumps split on features 200 + group
    const FOLDS: usize = 2;

    struct Stump {
        thr: f32,
        value: f32,
        default_left: bool,
    }

    fn stumps(groups: usize) -> Vec<Vec<Stump>> {
        (0..FOLDS)
            .map(|f| {
                (0..groups)
                    .map(|c| Stump {
                        thr: [0.5f32, 0.0, -1.0][(c + f) % 3],
                        value: [1.0f32, -0.5, 0.25][(c + 2 * f) % 3],
                        default_left: (c + f) % 2 == 0,
                    })
                    .collect()
            })
            .collect()
    }

    fn base(f: usize, c: usize) -> f32 {
        0.1 * (f + c + 1) as f32
    }

    /// Writes the checksum a file of this layout carries into its header.
    fn set_checksum(file: &mut [u8]) {
        let h = file_checksum(file).to_le_bytes();
        file[CHECKSUM_AT..CHECKSUM_AT + 8].copy_from_slice(&h);
    }

    fn image(objective: u32, average: u32, groups: usize, corrupt: bool, right: u64) -> Vec<u8> {
        let st = stumps(groups);
        let mut body = Vec::new();
        for f in 0..FOLDS {
            body.extend(((f * groups) as u32).to_le_bytes());
            body.extend((groups as u32).to_le_bytes());
        }
        for f in 0..FOLDS {
            for c in 0..groups {
                body.extend(base(f, c).to_le_bytes());
            }
        }
        while !body.len().is_multiple_of(8) {
            body.push(0);
        }
        for t in 0..FOLDS * groups {
            body.extend(((t * 3) as u32).to_le_bytes());
            body.extend(((t % groups) as u32).to_le_bytes());
        }
        for t in 0..FOLDS * groups {
            let s = &st[t / groups][t % groups];
            let node = u64::from(s.thr.to_bits())
                | right << 32
                | ((200 + t % groups) as u64) << 48
                | if s.default_left { DEFAULT_LEFT } else { 0 };
            for w in [
                node,
                u64::from(s.value.to_bits()),
                u64::from((-s.value).to_bits()),
            ] {
                body.extend(w.to_le_bytes());
            }
        }
        let mut file = Vec::new();
        file.extend(MAGIC);
        for v in [
            FORMAT_XGB,
            NF as u32,
            groups as u32,
            FOLDS as u32,
            (FOLDS * groups) as u32,
            (FOLDS * groups * 3) as u32,
            objective,
            average,
        ] {
            file.extend(v.to_le_bytes());
        }
        file.extend([0u8; 8 + 16]); // the checksum (set below), flags, cat_words and the margin scale
        file.extend(&body);
        set_checksum(&mut file);
        if corrupt {
            let last = file.len() - 1;
            file[last] ^= 1;
        }
        file
    }

    fn apply(objective: u32, m: &mut [f32]) {
        match objective {
            SOFTMAX => {
                let top = m.iter().cloned().fold(f32::MIN, f32::max);
                let e: Vec<f32> = m.iter().map(|v| (v - top).exp()).collect();
                let s = e.iter().map(|&v| f64::from(v)).sum::<f64>() as f32;
                m.iter_mut().zip(&e).for_each(|(v, e)| *v = e / s);
            }
            SIGMOID => m[0] = 1.0 / ((-m[0]).min(88.7).exp() + 1.0 + 1e-16),
            _ => {}
        }
    }

    /// xgboost's `common::Softmax` sums the exponentials in a double and divides by that sum rounded to float32. Many classes make
    /// the float32 sum differ from it, so this pins the double sum on margins no 3-class forest reaches.
    #[test]
    fn f32_softmax_normaliser_is_summed_in_double() {
        use crate::finish::Acc;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut differs_from_f32_sum = 0;
        for classes in [3usize, 30, 100, 300] {
            for _ in 0..40 {
                let margins: Vec<f32> = (0..classes).map(|_| 6.0 * next() - 3.0).collect();
                let top = margins.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = margins.iter().map(|v| (v - top).exp()).collect();
                let double = e.iter().map(|&v| f64::from(v)).sum::<f64>() as f32;
                let single = e.iter().fold(0f32, |a, &v| a + v);
                let want: Vec<f32> = e.iter().map(|v| v / double).collect();
                let mut got = margins.clone();
                <f32 as Acc>::transform(SOFTMAX, &mut got);
                assert_eq!(got, want, "{classes} classes");
                differs_from_f32_sum += usize::from(e.iter().any(|v| v / single != v / double));
            }
        }
        assert!(
            differs_from_f32_sum > 0,
            "the vectors cannot tell the two sums apart"
        );
    }

    /// xgboost's `common::Sigmoid`, written out: the exponent is clamped at 88.7f (`std::min(-x, 88.7f)`), 1e-16f is added.
    fn xgboost_sigmoid(m: f32) -> f32 {
        let x = -m;
        let x = if 88.7f32 < x { 88.7f32 } else { x };
        1.0f32 / (x.exp() + 1.0f32 + 1e-16f32)
    }

    fn same_f32(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// Margins from -1e4 to 1e4, plus infinity and NaN, which a float32 sum can reach: the f32 transforms are xgboost's own at both ends.
    #[test]
    fn f32_transforms_follow_xgboost_at_the_edges() {
        use crate::finish::Acc;
        let margins = [
            -1e4f32,
            -1000.0,
            -200.0,
            -100.0,
            -88.8,
            -88.7,
            -88.6,
            -50.0,
            0.0,
            50.0,
            88.6,
            88.7,
            88.8,
            100.0,
            1e4,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for m in margins {
            let mut got = [m];
            <f32 as Acc>::transform(SIGMOID, &mut got);
            assert!(
                same_f32(got[0], xgboost_sigmoid(m)),
                "sigmoid({m}) = {}",
                got[0]
            );
        }
        // below the clamp xgboost's answer is the same tiny number, not 0
        let mut low = [-1e4f32];
        <f32 as Acc>::transform(SIGMOID, &mut low);
        assert!(low[0] > 0.0 && low[0] < 1e-38, "{}", low[0]);
        let mut nan = [f32::NAN];
        <f32 as Acc>::transform(SIGMOID, &mut nan);
        assert!(nan[0].is_nan(), "a NaN margin stays NaN");
        // exp: plain `expf`, so overflow is inf and underflow 0
        let mut e = [1e4f32, -1e4, 88.8, f32::NAN];
        <f32 as Acc>::transform(EXP, &mut e);
        assert!(e[0].is_infinite() && e[1] == 0.0 && e[2].is_infinite() && e[3].is_nan());
        // softmax: the largest margin is subtracted first (no overflow), a NaN margin makes every answer NaN
        for (margins, want) in [
            (vec![1e4f32, 0.0, -1e4], vec![1.0, 0.0, 0.0]),
            (vec![-1e4, -1e4, -1e4], vec![1.0 / 3.0; 3]),
            (vec![3e38, 3e38, 0.0], vec![0.5, 0.5, 0.0]),
            (
                vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0],
                vec![0.0, 0.0, 1.0],
            ),
            (vec![f32::NAN, 0.0, 1.0], vec![f32::NAN; 3]),
            (vec![0.0, f32::NAN, 1.0], vec![f32::NAN; 3]),
            (vec![f32::INFINITY, 0.0, 1.0], vec![f32::NAN; 3]),
            (vec![f32::NEG_INFINITY; 3], vec![f32::NAN; 3]),
        ] {
            let mut got = margins.clone();
            <f32 as Acc>::transform(SOFTMAX, &mut got);
            assert!(
                got.iter().zip(&want).all(|(a, b)| same_f32(*a, *b)),
                "softmax({margins:?}) = {got:?}"
            );
        }
    }

    /// LightGBM and YDF sum in double and have no clamp: the f64 transforms are the plain formulas, with the same edges.
    #[test]
    fn f64_transforms_have_no_clamp_like_lightgbm() {
        use crate::finish::Acc;
        for m in [
            -1e4f64, -745.0, -709.0, -100.0, -30.0, 0.0, 30.0, 709.0, 1e4,
        ] {
            let mut got = [m];
            <f64 as Acc>::transform(SIGMOID, &mut got);
            assert_eq!(got[0], 1.0 / (1.0 + (-m).exp()), "sigmoid({m})");
        }
        let mut s = [-1e4f64, -709.0, 1e4, f64::NAN];
        for v in s.iter_mut() {
            <f64 as Acc>::transform(SIGMOID, std::slice::from_mut(v));
        }
        assert!(
            s[0] == 0.0 && s[1] > 0.0 && s[2] == 1.0 && s[3].is_nan(),
            "{s:?}"
        );
        let mut e = [1e4f64, -1e4, 709.0, -745.0, f64::NAN];
        <f64 as Acc>::transform(EXP, &mut e);
        assert!(
            e[0].is_infinite() && e[1] == 0.0 && e[2].is_finite() && e[3] > 0.0 && e[4].is_nan()
        );
        for (margins, want) in [
            (vec![1e4f64, 0.0, -1e4], vec![1.0, 0.0, 0.0]),
            (vec![-1e4, -1e4, -1e4], vec![1.0 / 3.0; 3]),
            (vec![f64::NAN, 0.0, 1.0], vec![f64::NAN; 3]),
            (vec![0.0, f64::NAN, 1.0], vec![f64::NAN; 3]),
            (vec![f64::INFINITY, 0.0, 1.0], vec![f64::NAN; 3]),
        ] {
            let mut got = margins.clone();
            <f64 as Acc>::transform(SOFTMAX, &mut got);
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
                "softmax({margins:?}) = {got:?}"
            );
        }
    }

    /// The forest's answer computed the plain way, in the same float32 operations and order.
    fn reference(objective: u32, average: u32, groups: usize, x: f32, bm: &[f64]) -> Vec<f32> {
        let st = stumps(groups);
        let mut acc = vec![0f32; groups];
        for (f, fold) in st.iter().enumerate() {
            let mut m: Vec<f32> = (0..groups)
                .map(|c| {
                    let s = &fold[c];
                    let left = if x.is_nan() {
                        s.default_left
                    } else {
                        x < s.thr
                    };
                    base(f, c) + bm[c] as f32 + if left { s.value } else { -s.value }
                })
                .collect();
            if average == MEAN_OUTPUT {
                apply(objective, &mut m);
            }
            acc.iter_mut().zip(&m).for_each(|(a, v)| *a += v);
        }
        let mut out: Vec<f32> = acc.iter().map(|a| a / FOLDS as f32).collect();
        if average == MEAN_MARGIN {
            apply(objective, &mut out);
        }
        out
    }

    #[test]
    fn matches_the_hand_computed_forest_for_every_objective_row_count_and_thread_count() {
        let values = [
            f32::NAN,
            -2.0,
            -1.0,
            -0.5,
            0.0,
            0.25,
            0.5,
            3.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            0.49999997,
        ];
        for (objective, groups) in [(SOFTMAX, 3), (SIGMOID, 1), (IDENTITY, 1)] {
            for average in [MEAN_OUTPUT, MEAN_MARGIN] {
                let model =
                    Model::from_bytes(&image(objective, average, groups, false, 2)).unwrap();
                for n in [1usize, 7, 8, 9, 11, 300, 1025] {
                    let rows: Vec<f32> = (0..n)
                        .flat_map(|i| {
                            let mut r = vec![9.0f32; NF];
                            r[200..203].fill(values[i % values.len()]);
                            r
                        })
                        .collect();
                    // a base margin that differs by row and group (row-major n x groups), and none at all
                    let margin: Vec<f64> = (0..n * groups)
                        .map(|k| 0.37 * (k % 7) as f64 - 1.1)
                        .collect();
                    for (threads, engine) in [(1, 0), (3, 0), (1, 1), (3, 1), (1, 2), (3, 2)] {
                        for use_margin in [false, true] {
                            let mut out = vec![0f32; n * groups];
                            model
                                .predict_engine_with(
                                    &rows,
                                    &mut out,
                                    use_margin.then_some(&margin[..]),
                                    threads,
                                    engine,
                                )
                                .unwrap();
                            for i in 0..n {
                                let bm = if use_margin {
                                    margin[i * groups..(i + 1) * groups].to_vec()
                                } else {
                                    vec![0.0; groups]
                                };
                                let want = reference(
                                    objective,
                                    average,
                                    groups,
                                    values[i % values.len()],
                                    &bm,
                                );
                                assert_eq!(
                                    &out[i * groups..(i + 1) * groups],
                                    &want[..],
                                    "row {i} of {n}, engine {engine}, margin {use_margin}"
                                );
                            }
                            // a float32 forest's f64 answer is its float32 answer widened
                            let mut out64 = vec![0f64; n * groups];
                            model
                                .predict_engine_with(
                                    &rows,
                                    &mut out64,
                                    use_margin.then_some(&margin[..]),
                                    threads,
                                    engine,
                                )
                                .unwrap();
                            for (a, b) in out.iter().zip(&out64) {
                                assert_eq!(
                                    f64::from(*a),
                                    *b,
                                    "engine {engine}, margin {use_margin}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_a_corrupt_or_malformed_file() {
        let msg = Model::from_bytes(&image(SOFTMAX, 0, 3, true, 2))
            .err()
            .unwrap();
        assert!(msg.contains("checksum"));
        assert!(Model::from_bytes(b"short").is_err());
        let mut bad = image(SOFTMAX, 0, 3, false, 2);
        bad.pop();
        assert!(Model::from_bytes(&bad).is_err());
        // a softmax forest with a single group is inconsistent
        assert!(Model::from_bytes(&image(SOFTMAX, 0, 1, false, 2)).is_err());
    }

    #[test]
    fn rejects_a_child_that_points_outside_its_tree() {
        let msg = Model::from_bytes(&image(SOFTMAX, 0, 3, false, 7))
            .err()
            .unwrap();
        assert!(msg.contains("out of range"));
    }

    /// A format-6 file: one fold, one group, one tree = [split on feature 2 at 0.5, linear leaf (left), plain leaf (right)], plus a
    /// second, one-node tree that is a linear leaf of no features (a stump). Records: 0 = 1.5 + 0.5 x0 - 2 x1 (fallback 7.25), 1 = constant 3.
    fn linear_image(corrupt: u32) -> Vec<u8> {
        let nf = 4usize;
        let mut body = Vec::new();
        body.extend(0u32.to_le_bytes());
        body.extend(2u32.to_le_bytes());
        body.extend(0.25f32.to_le_bytes()); // base margin
        body.extend([0u8; 4]);
        for (root, group) in [(0u32, 0u32), (3, 0)] {
            body.extend(root.to_le_bytes());
            body.extend(group.to_le_bytes());
        }
        let root = u64::from(0.5f32.to_bits()) | 2u64 << 32 | 2u64 << 48;
        let lin_leaf = |rec: u64| DEFAULT_LEFT | rec;
        let mut nodes = [root, lin_leaf(0), u64::from(10.0f32.to_bits()), lin_leaf(1)];
        if corrupt == 1 {
            nodes[1] = lin_leaf(9); // record out of range
        }
        if corrupt == 4 {
            nodes[1] = lin_leaf(2); // the first index past the two records
        }
        for w in nodes {
            body.extend(w.to_le_bytes());
        }
        // corrupt == 5: the second record claims one coefficient more than the table holds (offset 2 + count 1 > 2 entries)
        for (c, fb, off, cnt) in [
            (1.5f64, 7.25f64, 0u32, 2u32),
            (3.0, 3.0, 2, u32::from(corrupt == 5)),
        ] {
            body.extend(c.to_le_bytes());
            body.extend(fb.to_le_bytes());
            body.extend(off.to_le_bytes());
            body.extend(cnt.to_le_bytes());
        }
        // corrupt == 2: feature 9; corrupt == 3: feature 4 = n_features, the first one past the row
        for (f, c) in [
            (0u32, 0.5f64),
            (
                match corrupt {
                    2 => 9,
                    3 => nf as u32,
                    _ => 1,
                },
                -2.0,
            ),
        ] {
            body.extend(u64::from(f).to_le_bytes());
            body.extend(c.to_le_bytes());
        }
        let mut file = Vec::new();
        file.extend(MAGIC);
        for v in [FORMAT_LINEAR, nf as u32, 1, 1, 2, 4, IDENTITY, MEAN_OUTPUT] {
            file.extend(v.to_le_bytes());
        }
        file.extend([0u8; 8]); // the checksum, set below
        for v in [FLAG_SUM64 | FLAG_LINEAR, 0, 2, 2] {
            file.extend(v.to_le_bytes());
        }
        file.extend(&body);
        set_checksum(&mut file);
        file
    }

    #[test]
    fn linear_leaves_follow_lightgbm_nan_fallback_and_take_a_base_margin() {
        let model = Model::from_bytes(&linear_image(0)).unwrap();
        let nan = f32::NAN;
        // x0, x1, x2 (the split feature), x3
        let rows: [[f32; 4]; 6] = [
            [1.0, 2.0, 0.0, 9.0],    // left: 1.5 + 0.5 - 4 = -2
            [nan, 2.0, 0.0, 9.0],    // left, NaN in a leaf feature: the fallback 7.25
            [1.0, nan, 0.0, 9.0],    // the same through the second feature
            [1.0, 2.0, 0.7, 9.0],    // right: the plain leaf 10
            [1.0, 2.0, nan, nan], // NaN split value goes right (default right): 10; NaN in an unused feature
            [3e38, 1e-30, 0.1, 0.0], // large and tiny values stay finite in f64
        ];
        let want_tree0 = [
            1.5 + 0.5 * 1.0 - 2.0 * 2.0,
            7.25,
            7.25,
            10.0,
            10.0,
            1.5 + 0.5 * f64::from(3e38f32) - 2.0 * f64::from(1e-30f32),
        ];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let margin = [0.5f64, -1.0, 2.0, 0.0, 100.0, -3.0];
        for (threads, use_margin) in [(1, false), (1, true), (3, true)] {
            let mut out = vec![0f32; rows.len()];
            model
                .predict_engine_with(
                    &flat,
                    &mut out,
                    use_margin.then_some(&margin[..]),
                    threads,
                    0,
                )
                .unwrap();
            for i in 0..rows.len() {
                // base 0.25 (+ margin), then the two trees in order, in f64; the answer is rounded to f32
                let mut m = f64::from(0.25f32) + if use_margin { margin[i] } else { 0.0 };
                m += want_tree0[i];
                m += 3.0;
                assert_eq!(out[i], m as f32, "row {i}");
            }
            // the f64 answer is the sum itself, not rounded; the f32 answer is its rounding
            let mut out64 = vec![0f64; rows.len()];
            model
                .predict_engine_with(
                    &flat,
                    &mut out64,
                    use_margin.then_some(&margin[..]),
                    threads,
                    0,
                )
                .unwrap();
            for i in 0..rows.len() {
                let mut m = f64::from(0.25f32) + if use_margin { margin[i] } else { 0.0 };
                m += want_tree0[i];
                m += 3.0;
                assert_eq!(out64[i], m, "f64 row {i}");
                assert_eq!(out64[i] as f32, out[i]);
            }
        }
        // the wavefront engines refuse a linear-leaf file
        let mut out = vec![0f32; rows.len()];
        assert!(model.predict_engine(&flat, &mut out, 1, 1).is_err());
        // a margin of the wrong length is an error
        assert!(model
            .predict_engine_with(&flat, &mut out, Some(&margin[..3]), 1, 0)
            .is_err());
    }

    #[test]
    fn rejects_a_malformed_linear_table() {
        assert!(Model::from_bytes(&linear_image(0)).is_ok());
        for (k, what) in [(1, "bad linear leaf"), (2, "feature out of range")] {
            let msg = Model::from_bytes(&linear_image(k)).err().unwrap();
            assert!(msg.contains(what), "{msg}");
        }
        // the flag and the format go together, and the flag needs double sums
        let mut b = linear_image(0);
        b[8] = FORMAT_DOUBLE as u8;
        set_checksum(&mut b); // so that only the format rule can refuse it
        let msg = Model::from_bytes(&b).err().unwrap();
        assert!(msg.contains("unknown format"), "{msg}");
        let mut b = linear_image(0);
        b[48] = FLAG_LINEAR as u8;
        set_checksum(&mut b);
        let msg = Model::from_bytes(&b).err().unwrap();
        assert!(msg.contains("unknown format"), "{msg}");
    }

    /// The scale in a file of format 4 or 5 sits in header bytes 56..64; in a format 6 file it is the constant of a marked record after
    /// the last linear leaf record.
    #[test]
    fn the_margin_scale_is_read_from_the_verified_image() {
        let mut b = image(SIGMOID, 0, 1, false, 2);
        assert_eq!(Model::from_bytes(&b).unwrap().margin_scale(), 1.0); // 0 means 1
        for want in [1.7f64, 0.25, f64::NAN, -1.0] {
            b[56..64].copy_from_slice(&want.to_le_bytes());
            set_checksum(&mut b);
            let got = Model::from_bytes(&b).unwrap().margin_scale();
            assert_eq!(got.to_bits(), want.to_bits());
        }
        let plain = linear_image(0);
        assert_eq!(Model::from_bytes(&plain).unwrap().margin_scale(), 1.0);
        // append the marked record (constant 1.7, fallback = the marker, no coefficients) after the last record, before the coefficients
        let n = plain.len() - 2 * COEF_REC; // the two coefficient entries
        let mut marked = plain[..n].to_vec();
        marked.extend(1.7f64.to_le_bytes());
        marked.extend(SCALE_MARKER_BITS.to_le_bytes());
        marked.extend([0u8; 8]);
        marked.extend(&plain[n..]);
        marked[56..60].copy_from_slice(&3u32.to_le_bytes()); // lin_leaves 2 -> 3
        set_checksum(&mut marked);
        assert_eq!(Model::from_bytes(&marked).unwrap().margin_scale(), 1.7);
    }
    // ---- mutation test: a damaged file is rejected with a message, or loads and then predicts without touching memory it does not own.
    // Run in the DEBUG profile too (`cargo test`): there `from_raw_parts` / `get_unchecked` / `add` carry their precondition checks.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            if n == 0 {
                return 0;
            }
            (self.next() % n as u64) as usize
        }
    }

    const FIXTURES: [(&str, &[u8]); 9] = [
        ("lgb_cat", include_bytes!("../tests/data/lgb_cat.flat")),
        (
            "lgb_cat_f64",
            include_bytes!("../tests/data/lgb_cat_f64.flat"),
        ),
        (
            "lgb_linear",
            include_bytes!("../tests/data/lgb_linear.flat"),
        ),
        ("lgb_multi", include_bytes!("../tests/data/lgb_multi.flat")),
        ("xgb_bin", include_bytes!("../tests/data/xgb_bin.flat")),
        ("xgb_deep", include_bytes!("../tests/data/xgb_deep.flat")),
        ("xgb_folds", include_bytes!("../tests/data/xgb_folds.flat")),
        ("xgb_multi", include_bytes!("../tests/data/xgb_multi.flat")),
        ("ydf_f32", include_bytes!("../tests/data/ydf_f32.flat")),
    ];

    fn refix_checksum(b: &mut [u8]) {
        if b.len() >= HEADER {
            set_checksum(b);
        }
    }

    /// The walker's answers for `n` rows of `x`: f32 without a margin, then f64 with one. None when either call is refused.
    fn answers(model: &Model, x: &[f32], bm: &[f64], n: usize) -> Option<(Vec<f32>, Vec<f64>)> {
        let g = model.n_groups;
        let mut a = vec![0f32; n * g];
        model.predict_engine_with(x, &mut a, None, 1, 0).ok()?;
        let mut b = vec![0f64; n * g];
        model.predict_engine_with(x, &mut b, Some(bm), 1, 0).ok()?;
        Some((a, b))
    }

    fn same_bits_f32(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(p, q)| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan()))
    }

    fn same_bits_f64(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(p, q)| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan()))
    }

    fn mutate(b: &mut Vec<u8>, rng: &mut XorShift) {
        const WORDS: [u32; 12] = [
            0,
            1,
            2,
            3,
            0xFF,
            0x7FFF,
            0xFFFF,
            0x10000,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFFE,
            0xFFFF_FFFF,
        ];
        for _ in 0..1 + rng.below(3) {
            if b.is_empty() {
                return;
            }
            match rng.below(7) {
                0 => {
                    let i = rng.below(b.len());
                    b[i] ^= 1 << rng.below(8);
                }
                1 => {
                    let i = rng.below(b.len());
                    b[i] = rng.next() as u8;
                }
                2 => {
                    // a whole aligned word of the header or the tables
                    if b.len() < 4 {
                        continue;
                    }
                    let i = rng.below(b.len() / 4) * 4;
                    let w = if rng.below(4) == 0 {
                        0x7FC0_0000 // NaN
                    } else {
                        WORDS[rng.below(WORDS.len())]
                    };
                    b[i..i + 4].copy_from_slice(&w.to_le_bytes());
                }
                3 => {
                    // a header count
                    let i = 12 + 4 * rng.below(13);
                    let w = WORDS[rng.below(WORDS.len())];
                    if i + 4 <= b.len() {
                        b[i..i + 4].copy_from_slice(&w.to_le_bytes());
                    }
                }
                4 => {
                    let n = rng.below(b.len() + 1);
                    b.truncate(n);
                }
                5 => {
                    for _ in 0..1 + rng.below(16) {
                        b.push(rng.next() as u8);
                    }
                }
                _ => {
                    // a node word: random child distance / feature / threshold bits in the 8-byte-aligned node area
                    if b.len() < 8 {
                        continue;
                    }
                    let i = rng.below(b.len() / 8) * 8;
                    let w = rng.next().to_le_bytes();
                    let k = 1 + rng.below(8);
                    b[i..i + k].copy_from_slice(&w[..k]);
                }
            }
        }
    }

    #[test]
    fn damaged_files_are_rejected_or_predict_inside_their_own_memory() {
        let cases: usize = std::env::var("TREEFLAT_MUTATIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1500);
        // A mutant that loads is clean only when its answers are bit-identical to the undamaged file's. A mutant whose checksum was
        // repaired may load a DIFFERENT forest (that is what repairing it means): counted apart. A changed file that loads with the
        // original checksum would be a damaged file accepted: the test fails.
        let (mut loaded, mut rejected, mut identical, mut different) =
            (0usize, 0usize, 0usize, 0usize);
        for (name, base) in FIXTURES {
            let original = Model::from_bytes(base)
                .unwrap_or_else(|e| panic!("{name} must load unchanged: {e}"));
            let mut rng = XorShift(0x9E37_79B9_7F4A_7C15 ^ base.len() as u64);
            for case in 0..cases {
                let mut b = base.to_vec();
                mutate(&mut b, &mut rng);
                if rng.below(3) != 0 {
                    refix_checksum(&mut b);
                }
                let Ok(model) = Model::from_bytes(&b) else {
                    rejected += 1;
                    continue;
                };
                loaded += 1;
                let (nf, g) = (model.n_features, model.n_groups);
                let n = 21;
                let special = [
                    f32::NAN,
                    0.0,
                    -0.0,
                    1e-36,
                    -1.0,
                    3.0,
                    7.5,
                    1e30,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                ];
                let x: Vec<f32> = (0..n * nf)
                    .map(|i| special[(i * 7 + case) % special.len()])
                    .collect();
                let bm: Vec<f64> = (0..n * g).map(|k| 0.1 * (k % 5) as f64).collect();
                let mut want = vec![0f32; n * g];
                let walker = model.predict_engine_with(&x, &mut want, None, 1, 0);
                let same_answers = (nf, g) == (original.n_features, original.n_groups)
                    && match (answers(&model, &x, &bm, n), answers(&original, &x, &bm, n)) {
                        (Some(a), Some(o)) => {
                            same_bits_f32(&a.0, &o.0) && same_bits_f64(&a.1, &o.1)
                        }
                        _ => false,
                    };
                let repaired =
                    b[CHECKSUM_AT..CHECKSUM_AT + 8] != base[CHECKSUM_AT..CHECKSUM_AT + 8];
                assert!(
                    b == base || repaired,
                    "{name} case {case}: a changed file loaded with the original checksum"
                );
                if same_answers {
                    identical += 1;
                } else {
                    different += 1;
                }
                for engine in 0..3u32 {
                    for threads in [1usize, 3] {
                        for use_margin in [false, true] {
                            let mut out = vec![0f32; n * g];
                            let margin = use_margin.then_some(&bm[..]);
                            let r =
                                model.predict_engine_with(&x, &mut out, margin, threads, engine);
                            if r.is_ok() && !use_margin && walker.is_ok() {
                                let same = out.iter().zip(&want).all(|(a, b)| {
                                    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
                                });
                                assert!(same, "{name} case {case}: engine {engine} threads {threads} differs from the walker");
                            }
                        }
                    }
                }
                let mut out64 = vec![0f64; n * g];
                let _ = model.predict_engine_with(&x, &mut out64, Some(&bm), 2, 0);
            }
        }
        // the mutations must reach every outcome, or the test checks nothing
        println!("loaded {loaded} ({identical} bit-identical to the original, {different} a different forest), rejected {rejected}");
        assert!(
            loaded > 100 && rejected > 100 && identical > 0 && different > 100,
            "loaded {loaded} ({identical} identical, {different} different), rejected {rejected}"
        );
    }

    /// The checksum covers the header: no single flipped bit of the 64 bytes loads (it used to for the objective, the average rule, the
    /// flags and the margin scale, which are valid over a range of values).
    #[test]
    fn a_bit_flipped_anywhere_in_the_header_is_refused() {
        for (name, base) in FIXTURES {
            for byte in 0..HEADER {
                for bit in 0..8 {
                    let mut b = base.to_vec();
                    b[byte] ^= 1 << bit;
                    assert!(
                        Model::from_bytes(&b).is_err(),
                        "{name}: byte {byte} bit {bit} flipped and the file still loads"
                    );
                }
            }
        }
    }

    /// A file of the first layout (formats 1 to 3: the checksum covered the body only) says to compile the model again, whatever
    /// else is in it.
    #[test]
    fn a_file_of_the_first_layout_says_to_compile_again() {
        for (name, base) in FIXTURES {
            let mut b = base.to_vec();
            let new_format = u32_at(&b, 8);
            b[8..12].copy_from_slice(&(new_format - FORMAT_XGB + 1).to_le_bytes());
            let old_sum = fnv1a(&b[HEADER..]).to_le_bytes();
            b[CHECKSUM_AT..CHECKSUM_AT + 8].copy_from_slice(&old_sum);
            let msg = Model::from_bytes(&b).err().unwrap();
            assert!(
                msg.contains("compile the model again") && msg.contains("earlier version"),
                "{name}: {msg}"
            );
        }
    }

    /// A format-5 file with extended nodes, one fold, one group, one feature, three trees summed in double:
    /// tree 0 = a categorical root over set A = {0, 2, 5, 20} whose left child is categorical over set B = {5, 33} (leaves 10 / 20 / 30);
    /// tree 1 = a zero-as-missing node at 0.5, default RIGHT (leaves 0.01 / 0.02); tree 2 = a plain node at 0.5, default LEFT (0.0001 / 0.0002).
    /// Category table: [1, A, 2, B0, B1] (set B follows set A, so a read one word past A lands in B).
    /// corrupt == 1: set B's offset is the table length; corrupt == 2: set B claims 3 words, one more than the table has left.
    fn ext_image(corrupt: u32) -> Vec<u8> {
        let cats: Vec<u32> = vec![
            1,
            1 | 1 << 2 | 1 << 5 | 1 << 20,
            if corrupt == 2 { 3 } else { 2 },
            1 << 5,
            1 << 1,
        ];
        let mut body = Vec::new();
        body.extend(0u32.to_le_bytes()); // fold: first tree 0 ...
        body.extend(3u32.to_le_bytes()); // ... three trees
        body.extend(0.0f32.to_le_bytes()); // base margin
        body.extend([0u8; 4]);
        for root in [0u32, 5, 8] {
            body.extend(root.to_le_bytes());
            body.extend(0u32.to_le_bytes()); // group 0
        }
        let leaf = |v: f32| u64::from(v.to_bits());
        let (cat, zero_missing) = (1u64 << 62, 1u64 << 61);
        let half = u64::from(0.5f32.to_bits());
        let set_b = if corrupt == 1 { cats.len() as u64 } else { 2 };
        let nodes = [
            cat | 4 << 32, // tree 0 root: categorical, set A at offset 0, right child 4 nodes on
            cat | 2 << 32 | set_b, // its left child: categorical, set B
            leaf(10.0),
            leaf(20.0),
            leaf(30.0),
            zero_missing | half | 2 << 32, // tree 1: default right
            leaf(0.01),
            leaf(0.02),
            DEFAULT_LEFT | half | 2 << 32, // tree 2: default left
            leaf(0.0001),
            leaf(0.0002),
        ];
        for w in nodes {
            body.extend(w.to_le_bytes());
        }
        for w in &cats {
            body.extend(w.to_le_bytes());
        }
        while !body.len().is_multiple_of(8) {
            body.push(0);
        }
        let mut file = Vec::new();
        file.extend(MAGIC);
        for v in [
            FORMAT_DOUBLE,
            1,
            1,
            1,
            3,
            nodes.len() as u32,
            IDENTITY,
            MEAN_OUTPUT,
        ] {
            file.extend(v.to_le_bytes());
        }
        file.extend([0u8; 8]); // the checksum, set below
        for v in [FLAG_SUM64 | FLAG_EXT, cats.len() as u32] {
            file.extend(v.to_le_bytes());
        }
        file.extend(0f64.to_le_bytes()); // the margin scale: 0 means 1
        file.extend(&body);
        set_checksum(&mut file);
        file
    }

    /// Categorical bitsets, zero-as-missing and default directions of the extended walker, at their edges (LightGBM's rules): the Rust tests had
    /// no extended fixture, so a changed `<`, a dropped default, a wrong bit mask or a read past a set passed `cargo test`.
    #[test]
    fn extended_nodes_follow_lightgbm_at_their_edges() {
        let in_set = |set: &[i64], c: f32| !c.is_nan() && set.contains(&(c as i64)); // truncation toward zero; absent and unseen go right
        let want = |x: f32| -> f64 {
            let t0 = if in_set(&[0, 2, 5, 20], x) {
                if in_set(&[5, 33], x) {
                    10.0
                } else {
                    20.0
                }
            } else {
                30.0
            };
            let t1 = if x.is_nan() || x.abs() <= 1e-35 || x >= 0.5 {
                0.02
            } else {
                0.01
            };
            let t2 = if x.is_nan() || x < 0.5 {
                0.0001
            } else {
                0.0002
            };
            f64::from(t0 as f32) + f64::from(t1 as f32) + f64::from(t2 as f32)
        };
        let xs = [
            f32::NAN,
            -3.0,
            -0.5,
            0.0,
            0.25,
            0.5,
            1.0,
            2.9,
            5.0,
            20.0,
            31.0,
            32.0,
            33.0,
            33.9,
            64.0,
            1e-35,
            -1e-35,
            1.0000001e-35,
            1e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let model = Model::from_bytes(&ext_image(0)).unwrap();
        for threads in [1, 3] {
            let mut out = vec![0f64; xs.len()];
            model
                .predict_engine_with(&xs, &mut out, None, threads, 0)
                .unwrap();
            for (i, &x) in xs.iter().enumerate() {
                assert_eq!(out[i], want(x), "x = {x}, threads {threads}");
            }
        }
        // the wavefront engines refuse a file with extended nodes
        let mut out = vec![0f64; xs.len()];
        assert!(model
            .predict_engine_with(&xs, &mut out, None, 1, 1)
            .is_err());
        for corrupt in [1, 2] {
            let msg = Model::from_bytes(&ext_image(corrupt)).err().unwrap();
            assert!(
                msg.contains("category set outside the table"),
                "{corrupt}: {msg}"
            );
        }
    }

    /// Every bound `from_storage` checks, one step past the limit and with the checksum repaired, so that only that rule can refuse the file.
    /// The evaluator reads nodes, tables and rows through raw pointers once a file is accepted: an off-by-one in any of these is an
    /// out-of-bounds read (the fuzz harness cannot see it: a file that loads is expected to load).
    #[test]
    fn each_bound_of_the_loader_refuses_the_first_value_past_it() {
        let groups = 3usize;
        let good = image(SOFTMAX, 0, groups, false, 2);
        assert!(Model::from_bytes(&good).is_ok());
        let n_trees = FOLDS * groups;
        let n_nodes = n_trees * 3;
        let tree_at = HEADER + FOLDS * FOLD_REC + (FOLDS * groups * 4).next_multiple_of(8);
        let put32 = |at: usize, v: u32| {
            let mut b = good.clone();
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
            set_checksum(&mut b);
            b
        };
        let refused = |b: &[u8], rule: &str| {
            let msg = Model::from_bytes(b)
                .err()
                .unwrap_or_else(|| panic!("a file that breaks `{rule}` was accepted"));
            assert!(msg.contains(rule), "{rule}: {msg}");
        };
        // the right child of a three-node tree on the first node after the tree (distance 3), and on the left child's place (distance 1)
        refused(
            &image(SOFTMAX, 0, groups, false, 3),
            "child or feature out of range",
        );
        refused(
            &image(SOFTMAX, 0, groups, false, 1),
            "child or feature out of range",
        );
        // tree records: a group equal to n_groups, a root equal to n_nodes, a repeated root, a first root other than 0
        refused(&put32(tree_at + 4, groups as u32), "bad root or group");
        refused(
            &put32(tree_at + (n_trees - 1) * TREE_REC, n_nodes as u32),
            "bad root or group",
        );
        refused(&put32(tree_at + TREE_REC, 0), "bad root or group");
        refused(&put32(tree_at, 1), "first tree does not start at node 0");
        // one byte too many (the size is exact; the checksum is not what refuses it)
        let mut longer = good.clone();
        longer.push(0);
        refused(&longer, "the header says");
        // format 6: a coefficient on feature n_features, a linear leaf on record lin_leaves, a record reaching one coefficient past the table
        refused(&linear_image(3), "feature out of range");
        refused(&linear_image(4), "bad linear leaf");
        refused(&linear_image(5), "coefficient range");
    }
}
