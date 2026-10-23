//! Pieces shared by the interpreted engines (lib.rs, wave.rs) and by the per-model crates the `codegen` backend writes (that
//! backend copies this file next to the generated code): the objective's transform, and the block loop that turns per-fold
//! margins into the averaged outputs, in exactly the operations and order xgboost uses (float32 margins and exponentials, the
//! softmax normaliser summed in double; margin = base + the leaves in tree order; transform; folds summed in order; divided by
//! the number of folds). Only `std`.

pub const IDENTITY: u32 = 0;
pub const SIGMOID: u32 = 1;
pub const SOFTMAX: u32 = 2;
/// output = exp(margin) (LightGBM poisson / gamma / tweedie)
pub const EXP: u32 = 3;
pub const MEAN_OUTPUT: u32 = 0;
pub const MEAN_MARGIN: u32 = 1;

/// The number type the margins are summed and transformed in: `f32` is xgboost's own arithmetic (bit for bit); `f64` is for the
/// models whose libraries sum in double (LightGBM, YDF): leaves are still stored as f32, but the sums, the transform and the
/// averaging over folds are done in f64 and only the final answer is rounded to f32.
pub trait Acc: Copy + Default + Send + Sync + 'static {
    fn from_f32(v: f32) -> Self;
    /// adds a double (a base margin, or a linear leaf's value; float32 forests only ever see base margins, which are rounded to f32)
    fn add_f64(&mut self, v: f64);
    fn add_leaf(&mut self, leaf: f32);
    fn add(&mut self, v: Self);
    fn mean(self, n: usize) -> Self;
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
    fn transform(objective: u32, m: &mut [Self]);
}

impl Acc for f32 {
    fn from_f32(v: f32) -> f32 {
        v
    }
    fn add_f64(&mut self, v: f64) {
        *self += v as f32;
    }
    fn add_leaf(&mut self, leaf: f32) {
        *self += leaf;
    }
    fn add(&mut self, v: f32) {
        *self += v;
    }
    fn mean(self, n: usize) -> f32 {
        self / n as f32
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn transform(objective: u32, m: &mut [f32]) {
        match objective {
            SOFTMAX => {
                let top = m.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                // xgboost's `common::Softmax`: float32 exponentials, but the normaliser is summed in double and rounded to float32
                // once (`wsum` is a double, the division is by `static_cast<float>(wsum)`)
                let mut sum = 0f64;
                for v in m.iter_mut() {
                    *v = (*v - top).exp();
                    sum += f64::from(*v);
                }
                let sum = sum as f32;
                for v in m.iter_mut() {
                    *v /= sum;
                }
            }
            SIGMOID => {
                // xgboost's `common::Sigmoid`: `x = std::min(-x, 88.7f)` (so a NaN margin stays NaN: `std::min` returns its first argument
                // unless the second is smaller, which `f32::min` does not), then `1.0f / (expf(x) + 1.0f + 1e-16f)`
                let x = -m[0];
                let x = if 88.7f32 < x { 88.7f32 } else { x };
                m[0] = 1.0 / (x.exp() + 1.0 + 1e-16f32);
            }
            EXP => m.iter_mut().for_each(|v| *v = v.exp()),
            _ => debug_assert_eq!(objective, IDENTITY),
        }
    }
}

impl Acc for f64 {
    fn from_f32(v: f32) -> f64 {
        f64::from(v)
    }
    fn add_f64(&mut self, v: f64) {
        *self += v;
    }
    fn add_leaf(&mut self, leaf: f32) {
        *self += f64::from(leaf);
    }
    fn add(&mut self, v: f64) {
        *self += v;
    }
    fn mean(self, n: usize) -> f64 {
        self / n as f64
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn transform(objective: u32, m: &mut [f64]) {
        match objective {
            SOFTMAX => {
                let top = m.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0f64;
                for v in m.iter_mut() {
                    *v = (*v - top).exp();
                    sum += *v;
                }
                for v in m.iter_mut() {
                    *v /= sum;
                }
            }
            SIGMOID => m[0] = 1.0 / (1.0 + (-m[0]).exp()),
            EXP => m.iter_mut().for_each(|v| *v = v.exp()),
            _ => debug_assert_eq!(objective, IDENTITY),
        }
    }
}

/// The type of the answers written to the caller: `f32` (rounded from the accumulator, as always) or `f64` (the accumulator itself; a
/// float32 forest's float32 result widened, a double-sum forest's sum without the final rounding).
pub trait Out: Copy + Send + Sync + 'static {
    fn from_acc<A: Acc>(a: A) -> Self;
}

impl Out for f32 {
    fn from_acc<A: Acc>(a: A) -> f32 {
        a.to_f32()
    }
}

impl Out for f64 {
    fn from_acc<A: Acc>(a: A) -> f64 {
        a.to_f64()
    }
}

/// What the block loop needs to know about a forest.
pub struct Meta {
    pub n_features: usize,
    pub n_groups: usize,
    pub n_folds: usize,
    pub objective: u32,
    pub average: u32,
    /// per fold: `n_groups` base margins
    pub base: Vec<f32>,
    /// float32 sums whose base margin is added LATE, in double (YDF with `--float32-sums`): see `LateAcc`
    pub late: bool,
}

/// A float32-sum forest that gets a base margin: YDF's own predictor sums its leaves in float32 starting from its initial prediction, so
/// the margin cannot join the sum (it would move the whole float32 trajectory, ~2e-5 on a log-scale target around 12). Each fold's raw float32
/// score is widened, the margin is added in double, and the transform, the averaging over folds and the answer are double.
pub struct LateAcc {
    acc: Vec<f64>,
    row: Vec<f64>,
}

impl LateAcc {
    pub fn new(block: usize, g: usize) -> LateAcc {
        LateAcc {
            acc: vec![0.0; block * g],
            row: vec![0.0; g],
        }
    }

    pub fn clear(&mut self, rows: usize, g: usize) {
        self.acc[..rows * g].fill(0.0);
    }

    /// Adds one fold: `raw(r, gi)` is the float32 raw score of row `r`, group `gi`; `bm` the `rows * g` margins of the block.
    pub fn add_fold(
        &mut self,
        rows: usize,
        g: usize,
        objective: u32,
        transform_now: bool,
        raw: impl Fn(usize, usize) -> f32,
        bm: &[f64],
    ) {
        for r in 0..rows {
            for gi in 0..g {
                self.row[gi] = f64::from(raw(r, gi)) + bm[r * g + gi];
            }
            if transform_now {
                f64::transform(objective, &mut self.row);
            }
            for gi in 0..g {
                self.acc[r * g + gi] += self.row[gi];
            }
        }
    }

    /// Writes the averaged answers of the block's `rows` rows.
    pub fn write<O: Out>(
        &mut self,
        rows: usize,
        g: usize,
        objective: u32,
        transform_after: bool,
        n_folds: usize,
        out: &mut [O],
    ) {
        for r in 0..rows {
            for gi in 0..g {
                self.row[gi] = self.acc[r * g + gi].mean(n_folds);
            }
            if transform_after {
                f64::transform(objective, &mut self.row);
            }
            for gi in 0..g {
                out[r * g + gi] = O::from_acc(self.row[gi]);
            }
        }
    }
}

/// Predicts `x` (row major) into `out` in blocks of at most `block` rows. For each block and fold the margins are initialised
/// with the base and `fold_block(scratch, fold, rows_of_the_block, rows, margin)` must add the fold's leaves, tree by tree in
/// order; `margin` is laid out [group][block] (`block` floats per group). `threads` > 1 splits the rows in chunks of whole blocks.
/// `bm`: a base margin per row and group (`n_rows * n_groups` doubles, row major) added to every fold's margin right after the base
/// (before the leaves, so before the objective's transform); with `meta.late` it is added after the leaves, in double (`LateAcc`).
#[allow(clippy::too_many_arguments)]
pub fn predict_blocks<A: Acc, O: Out, S, M, F>(
    meta: &Meta,
    block: usize,
    x: &[f32],
    out: &mut [O],
    bm: Option<&[f64]>,
    threads: usize,
    make: &M,
    fold_block: &F,
) where
    M: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &[f32], usize, &mut [A]) + Sync,
{
    let (nf, g) = (meta.n_features, meta.n_groups);
    let n_rows = x.len() / nf;
    let threads = threads.clamp(1, 64).min(n_rows.div_ceil(block).max(1));
    if threads == 1 {
        chunk::<A, O, _, _, _>(meta, block, x, out, bm, make, fold_block);
        return;
    }
    let per = n_rows.div_ceil(threads).div_ceil(block) * block;
    std::thread::scope(|s| {
        for (i, (xc, oc)) in x.chunks(per * nf).zip(out.chunks_mut(per * g)).enumerate() {
            let bmc = bm.map(|b| &b[i * per * g..((i + 1) * per * g).min(b.len())]);
            s.spawn(move || chunk::<A, O, _, _, _>(meta, block, xc, oc, bmc, make, fold_block));
        }
    });
}

fn chunk<A: Acc, O: Out, S, M, F>(
    meta: &Meta,
    block: usize,
    x: &[f32],
    out: &mut [O],
    bm: Option<&[f64]>,
    make: &M,
    fold_block: &F,
) where
    M: Fn() -> S,
    F: Fn(&mut S, usize, &[f32], usize, &mut [A]),
{
    let (nf, g) = (meta.n_features, meta.n_groups);
    let n_rows = x.len() / nf;
    let mut scratch = make();
    let mut acc = vec![A::default(); block * g];
    let mut margin = vec![A::default(); block * g];
    let mut row = vec![A::default(); g];
    let mut late = bm
        .filter(|_| meta.late)
        .map(|b| (LateAcc::new(block, g), b));
    let mut start = 0;
    while start < n_rows {
        let rows = block.min(n_rows - start);
        let xb = &x[start * nf..(start + rows) * nf];
        acc[..rows * g].fill(A::default());
        if let Some((la, _)) = late.as_mut() {
            la.clear(rows, g);
        }
        for f in 0..meta.n_folds {
            for gi in 0..g {
                margin[gi * block..gi * block + rows].fill(A::from_f32(meta.base[f * g + gi]));
                if let (Some(bm), None) = (bm, &late) {
                    for (r, v) in margin[gi * block..gi * block + rows].iter_mut().enumerate() {
                        v.add_f64(bm[(start + r) * g + gi]);
                    }
                }
            }
            fold_block(&mut scratch, f, xb, rows, &mut margin);
            if let Some((la, b)) = late.as_mut() {
                let raw = |r: usize, gi: usize| margin[gi * block + r].to_f32();
                la.add_fold(
                    rows,
                    g,
                    meta.objective,
                    meta.average == MEAN_OUTPUT,
                    raw,
                    &b[start * g..(start + rows) * g],
                );
                continue;
            }
            for r in 0..rows {
                for gi in 0..g {
                    row[gi] = margin[gi * block + r];
                }
                if meta.average == MEAN_OUTPUT {
                    A::transform(meta.objective, &mut row);
                }
                for gi in 0..g {
                    acc[r * g + gi].add(row[gi]);
                }
            }
        }
        let o = &mut out[start * g..(start + rows) * g];
        if let Some((la, _)) = late.as_mut() {
            la.write(
                rows,
                g,
                meta.objective,
                meta.average == MEAN_MARGIN,
                meta.n_folds,
                o,
            );
            start += rows;
            continue;
        }
        for (m, a) in o.chunks_exact_mut(g).zip(acc[..rows * g].chunks_exact(g)) {
            for gi in 0..g {
                row[gi] = a[gi].mean(meta.n_folds);
            }
            if meta.average == MEAN_MARGIN {
                A::transform(meta.objective, &mut row);
            }
            for gi in 0..g {
                m[gi] = O::from_acc(row[gi]);
            }
        }
        start += rows;
    }
}
