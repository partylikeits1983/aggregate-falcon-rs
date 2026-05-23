//! Johnson-Lindenstrauss projection (Lemma 2.2 / §2.5).
//!
//! `Π ∈ {−1, 0, +1}^{256 × (n · d)}` with `Pr[0] = 1/2`, `Pr[±1] = 1/4` each.
//! For a witness vector `w ∈ S^n` with centered ℤ-coefficient vector
//! `w̃ ∈ ℤ^{n·d}` (length-d integer slice per ring element), the projection
//! `p_j = Σ_i Π_{j, ·} · w̃_i` is an `i128` to avoid overflow.

use crate::statement::ConstTermConstraint;
use crate::transcript::Transcript;
use modring::{Modulus, RingElem, D};
use rayon::prelude::*;

pub const LAMBDA: usize = 128;
pub const PROJECTION_ROWS: usize = 2 * LAMBDA; // 256

/// Parse a sparse Π row (length `cols`) from `cols/4` bytes already drawn
/// from the transcript. 2 bits per coordinate: `(bit0=zero?, bit1=sign)`.
fn parse_pi_row(bytes: &[u8], cols: usize) -> Vec<(usize, i8)> {
    let mut out = Vec::new();
    for k in 0..cols {
        let byte = bytes[k / 4];
        let pair = (byte >> ((k % 4) * 2)) & 0b11;
        match pair {
            0b00 | 0b01 => {} // zero with probability 2/4
            0b10 => out.push((k, 1)),
            0b11 => out.push((k, -1)),
            _ => unreachable!(),
        }
    }
    out
}

/// Sample the full projection matrix `Π` as `PROJECTION_ROWS` sparse rows.
///
/// All 256 rows are drawn from a single SHAKE squeeze of
/// `PROJECTION_ROWS · ⌈cols/4⌉` bytes, then split per-row in parallel. This
/// replaces the previous 256 individual `challenge_bytes` calls per
/// projection (one per row) and parses the rows on all cores; the call was a
/// top wall-clock hotspot in the v2 prover and the fold replay
/// (~150–400 ms per call serially).
pub fn sample_projection(
    t: &mut Transcript,
    label: &[u8],
    cols: usize,
) -> Vec<Vec<(usize, i8)>> {
    let nbytes_per_row = (cols + 3) / 4;
    let bytes = t.challenge_bytes(label, nbytes_per_row * PROJECTION_ROWS);
    (0..PROJECTION_ROWS)
        .into_par_iter()
        .map(|j| {
            let s = j * nbytes_per_row;
            parse_pi_row(&bytes[s..s + nbytes_per_row], cols)
        })
        .collect()
}

/// Project a single witness vector `w ∈ S^n` using `pi` (sparse rows).
/// Returns `[i128; 256]` with each entry being a (possibly large) signed sum
/// over centered coefficients.
pub fn project_vector(
    pi: &[Vec<(usize, i8)>],
    w: &[RingElem],
    m: &Modulus,
) -> [i128; PROJECTION_ROWS] {
    // Flatten centered coefficients of w. The flatten cost is O(n·D) and
    // sequential here; the 256-row reduction below dominates.
    let mut wflat: Vec<i64> = Vec::with_capacity(w.len() * D);
    for e in w {
        for k in 0..D {
            wflat.push(m.centered(e.c[k]));
        }
    }
    let row_results: Vec<i128> = pi
        .par_iter()
        .map(|row| {
            let mut acc: i128 = 0;
            for &(pos, sign) in row {
                acc += (sign as i128) * (wflat[pos] as i128);
            }
            acc
        })
        .collect();
    let mut out = [0i128; PROJECTION_ROWS];
    for (j, v) in row_results.into_iter().enumerate() {
        out[j] = v;
    }
    out
}

/// `σ_{-1}(a) = a(X^{-1}) = a_0 - a_{D-1}·X - a_{D-2}·X² - ... - a_1·X^{D-1}` in S.
fn sigma_minus_one(s: &RingElem, m: &Modulus) -> RingElem {
    let mut r = RingElem::zero();
    r.c[0] = s.c[0];
    for j in 1..D {
        r.c[j] = m.neg(s.c[D - j]);
    }
    r
}

/// Build the `2λ` JL-projection constant-term constraints (Protocol 2 §B.6
/// Step 2). For each coordinate `j ∈ [2λ]`, the constraint is
///
/// `0 = ct(Σ_i ⟨σ_{-1}(π̂_i^(j)), w_i⟩) − p_j` (mod q')
///
/// where `π̂_i^(j) ∈ S^n` packs the sparse `{-1, 0, +1}` row `Π_i^(j)` so
/// `π̂_i^(j)[idx].c[k] = Π_i^(j)[idx·D + k]`. Once the verifier appends these
/// constraints to `F'` (between the `p`-absorb and the `ψ`-squeeze), the JL
/// projection vector `p` is tied to the witness `w` — a malicious prover
/// can't lie about `p` independently of `w`.
///
/// The constraints are emitted lazily as `ConstTermConstraint`s in the order
/// `j = 0, 1, …, 2λ − 1`. Each phi bucket holds the per-position non-zero
/// coefficients of `σ_{-1}(π̂_i^(j))` — typically about `D/2` coefficients
/// per witness position, all in `{-1, 0, +1}` (centered into `[0, q')`).
pub fn build_jl_constraints(
    pis: &[Vec<Vec<(usize, i8)>>],
    p: &[i128],
    n: usize,
    m: &Modulus,
) -> Vec<ConstTermConstraint> {
    assert_eq!(p.len(), PROJECTION_ROWS);
    let _ = n; // dims kept for API stability; no longer needed (no dense scratch)
    let neg_one = m.from_i64(-1);
    let plus_one = 1u64;
    let q = m.q as i128;
    // Parallelize over the 256 projection rows; each row is independent and the
    // result Vec stays in j-order (collect preserves index order).
    (0..PROJECTION_ROWS)
        .into_par_iter()
        .map(|j| {
            let mut phi: Vec<(usize, Vec<(usize, RingElem)>)> = Vec::with_capacity(pis.len());
            for (i, pi_i) in pis.iter().enumerate() {
                // `parse_pi_row` emits flat positions in ascending order, so the
                // sparse row is already grouped by `idx = flat_pos / D`. Stream
                // it: accumulate the coefficients of one ring element, apply
                // σ_{-1} once when `idx` advances, and push the non-zero result.
                // No dense `vec![zero; n]` scratch and no per-position scan.
                let row = &pi_i[j];
                let mut sparse_phi: Vec<(usize, RingElem)> = Vec::new();
                let mut cur_idx: usize = usize::MAX;
                let mut cur = RingElem::zero();
                for &(flat_pos, sign) in row {
                    let idx = flat_pos / D;
                    let k = flat_pos % D;
                    if idx != cur_idx {
                        if cur_idx != usize::MAX {
                            sparse_phi.push((cur_idx, sigma_minus_one(&cur, m)));
                        }
                        cur = RingElem::zero();
                        cur_idx = idx;
                    }
                    cur.c[k] = if sign == 1 { plus_one } else { neg_one };
                }
                if cur_idx != usize::MAX {
                    sparse_phi.push((cur_idx, sigma_minus_one(&cur, m)));
                }
                if !sparse_phi.is_empty() {
                    phi.push((i, sparse_phi));
                }
            }
            // b0: centered p_j mapped into Z_{q'}.
            let mut reduced = p[j] % q;
            if reduced < 0 {
                reduced += q;
            }
            ConstTermConstraint { a: Vec::new(), phi, b0: reduced as u64 }
        })
        .collect()
}

/// Sum projections over many witness vectors: `p_j = Σ_i ⟨π_j^{(i)}, w_i⟩`.
/// `pis[i]` is the projection matrix for witness vector `i` (each over its
/// own `n_i · d` columns).
pub fn project_combined(
    pis: &[Vec<Vec<(usize, i8)>>],
    ws: &[Vec<RingElem>],
    m: &Modulus,
) -> [i128; PROJECTION_ROWS] {
    pis.par_iter()
        .zip(ws.par_iter())
        .fold(
            || [0i128; PROJECTION_ROWS],
            |mut acc, (pi, w)| {
                let p = project_vector(pi, w, m);
                for j in 0..PROJECTION_ROWS {
                    acc[j] += p[j];
                }
                acc
            },
        )
        .reduce(
            || [0i128; PROJECTION_ROWS],
            |mut a, b| {
                for j in 0..PROJECTION_ROWS {
                    a[j] += b[j];
                }
                a
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use modring::{find_prime_5mod8, Modulus, Ring};

    fn ring() -> Ring {
        Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
    }

    #[test]
    fn hand_computed_projection_matches() {
        // Use a tiny witness and a hand-built Π.
        let r = ring();
        let m = &r.m;
        let mut w = vec![RingElem::zero(); 2];
        w[0].c[0] = 5;
        w[0].c[1] = m.from_i64(-3);
        w[1].c[0] = 2;
        // Flatten: [5, -3, 0, 0, ..., 0, 2, 0, ...] of length 2*D.
        // Hand row: Π_0 = [+1 at pos 0, -1 at pos 1, +1 at pos D]
        let pi = vec![vec![(0usize, 1i8), (1, -1), (D, 1)]];
        let mut pi_padded: Vec<Vec<(usize, i8)>> = pi.clone();
        // pad up to PROJECTION_ROWS rows of zeros so the type matches:
        while pi_padded.len() < PROJECTION_ROWS {
            pi_padded.push(vec![]);
        }
        let p = project_vector(&pi_padded, &w, m);
        // 5 - (-3) + 2 = 10
        assert_eq!(p[0], 10);
        for j in 1..PROJECTION_ROWS {
            assert_eq!(p[j], 0);
        }
    }

    #[test]
    fn distribution_is_approximately_50_25_25() {
        let mut t = Transcript::new(b"jl-dist");
        let cols = 10_000;
        let nbytes = (cols + 3) / 4;
        let bytes = t.challenge_bytes(b"r0", nbytes);
        let row = parse_pi_row(&bytes, cols);
        let total = row.len();
        let pos = row.iter().filter(|(_, s)| *s == 1).count();
        let neg = row.iter().filter(|(_, s)| *s == -1).count();
        // Expected: zeros ≈ cols/2, pos ≈ cols/4, neg ≈ cols/4.
        assert!((total as i64 - cols as i64 / 2).abs() < cols as i64 / 10);
        assert!((pos as i64 - cols as i64 / 4).abs() < cols as i64 / 10);
        assert!((neg as i64 - cols as i64 / 4).abs() < cols as i64 / 10);
    }
}
