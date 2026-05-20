//! Garbage polynomials `g_{ij}` and `h_{ij}` of Protocol 2, plus centered
//! base-`b` decomposition for the commit messages.
//!
//! `g_{ij} = ⟨w_i, w_j⟩` (upper-triangular).
//! `h_{ij} = (⟨φ_i, w_j⟩ + ⟨φ_j, w_i⟩) / 2`.
//! Each `g, h` is decomposed into `t` chunks with `‖chunk‖_∞ < base / 2`.

use crate::statement::{ring_inner_product, sparse_phi_inner_product};
use modring::{Modulus, Ring, RingElem, D};
use rayon::prelude::*;

/// Upper-triangular `g_{ij} = ⟨w_i, w_j⟩` for `i ≤ j`. Returned as a
/// `Vec<Vec<RingElem>>` of shape `[r][r]` with the lower triangle left as
/// `RingElem::zero()` (caller fills via symmetry if needed).
pub fn compute_g(ring: &Ring, ws: &[Vec<RingElem>]) -> Vec<Vec<RingElem>> {
    let r = ws.len();
    // r(r+1)/2 upper-triangle inner products, all independent.
    let pairs: Vec<(usize, usize)> = (0..r)
        .flat_map(|i| (i..r).map(move |j| (i, j)))
        .collect();
    let entries: Vec<(usize, usize, RingElem)> = pairs
        .par_iter()
        .map(|&(i, j)| (i, j, ring_inner_product(ring, &ws[i], &ws[j])))
        .collect();
    let mut g = vec![vec![RingElem::zero(); r]; r];
    for (i, j, v) in entries {
        g[i][j] = v;
    }
    g
}

/// `h_{ij} = (⟨φ_i, w_j⟩ + ⟨φ_j, w_i⟩) / 2`, upper-triangular.
/// `phis[i]` is the sparse `φ_i` for witness vector `i`.
pub fn compute_h(
    ring: &Ring,
    phis: &[Vec<(usize, RingElem)>],
    ws: &[Vec<RingElem>],
) -> Vec<Vec<RingElem>> {
    let m = &ring.m;
    let r = ws.len();
    let inv2 = m.inv(2);
    let pairs: Vec<(usize, usize)> = (0..r)
        .flat_map(|i| (i..r).map(move |j| (i, j)))
        .collect();
    let entries: Vec<(usize, usize, RingElem)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let a = sparse_phi_inner_product(ring, &phis[i], &ws[j]);
            let b = sparse_phi_inner_product(ring, &phis[j], &ws[i]);
            let sum = a.add(m, &b);
            (i, j, sum.scale(m, inv2))
        })
        .collect();
    let mut h = vec![vec![RingElem::zero(); r]; r];
    for (i, j, v) in entries {
        h[i][j] = v;
    }
    h
}

/// Lossless centered base-`b` decomposition of a `RingElem` into `parts`
/// chunks. The first `parts - 1` chunks have every coefficient in
/// `(−b/2, b/2]`; the LAST chunk absorbs the remaining
/// `(centered(x.c[k]) - lower) / b^{parts-1}`, which can lie outside that
/// range when `b^parts < q`.
///
/// `recompose(decompose(x), b) = x` exactly, regardless of `x`'s magnitude —
/// the recursion's fold step relies on this round-trip.
///
/// When `b^parts ≥ q` (the regime the paper's estimator targets for v/h
/// decompositions, where `b₁^{t₁} ≥ 2^q_bitlen`), the last chunk
/// automatically also fits in `(−b/2, b/2]`. For decompositions where the
/// estimator picks smaller `b^parts` (g via `(b₂, t₂)` sized for `σ_h`, and
/// z via `(b, 2)` sized for `σ_z`), the last chunk holds the overflow — this
/// is acceptable because the norm-bound check at the next iteration
/// (`‖z^(0)‖² + ‖z^(1)‖² + ‖ê‖² ≤ β'²`) is on `ℓ₂` of all chunks together,
/// not per-coefficient `ℓ_∞`.
pub fn decompose(x: &RingElem, m: &Modulus, base: u64, parts: usize) -> Vec<RingElem> {
    assert!(base >= 1, "base must be ≥ 1");
    assert!(parts >= 1, "parts must be ≥ 1");
    if base == 1 {
        // No decomposition possible — just return one chunk equal to x.
        let mut chunks = vec![RingElem::zero(); parts];
        chunks[0] = x.clone();
        return chunks;
    }
    let b = base as i128;
    let half = (b / 2) as i128;
    let mut chunks: Vec<RingElem> = vec![RingElem::zero(); parts];
    for k in 0..D {
        let mut v = m.centered(x.c[k]) as i128;
        for c in 0..(parts - 1) {
            let mut digit = v % b;
            if digit > half {
                digit -= b;
            } else if digit < -half {
                digit += b;
            }
            v = (v - digit) / b;
            chunks[c].c[k] = m.from_i64(digit as i64);
        }
        // Last chunk absorbs any residue exactly.
        chunks[parts - 1].c[k] = m.from_i64(v as i64);
    }
    chunks
}

/// Inverse of [`decompose`]: recompose the chunks into a single `RingElem`.
pub fn recompose(chunks: &[RingElem], m: &Modulus, base: u64) -> RingElem {
    let mut out = RingElem::zero();
    let b = base as i128;
    for k in 0..D {
        let mut v: i128 = 0;
        let mut scale: i128 = 1;
        for c in 0..chunks.len() {
            v += scale * (m.centered(chunks[c].c[k]) as i128);
            scale *= b;
        }
        out.c[k] = m.from_i64(v as i64);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use modring::{find_prime_5mod8, Modulus, Ring};

    fn ring() -> Ring {
        Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
    }

    #[test]
    fn decompose_recompose_round_trips_on_small_values() {
        let r = ring();
        let mut rng = modring::rng::SplitMix64::new(11);
        for _ in 0..50 {
            let mut x = RingElem::zero();
            for k in 0..D {
                // Bounded so 4 chunks of base 100 suffice.
                let v = (rng.next() % 200_000) as i64 - 100_000;
                x.c[k] = r.m.from_i64(v);
            }
            let chunks = decompose(&x, &r.m, 100, 4);
            for c in &chunks {
                for k in 0..D {
                    let centered = r.m.centered(c.c[k]).abs();
                    assert!(centered <= 50, "chunk coef magnitude {centered} > 50");
                }
            }
            let restored = recompose(&chunks, &r.m, 100);
            assert_eq!(restored, x);
        }
    }
}
