//! Ajtai commitments and matrix expansion for LaBRADOR.
//!
//! Per Protocol 2 §G(1^λ): the public parameters consist of matrices
//! `A ∈ R_{q'}^{κ × n}`, `B_{ik}`, `C_{ijk}`, `D_{ijk}` whose entries are
//! deterministically expanded from a transcript seed and *never* serialised
//! — the verifier regenerates them from the same transcript.
//!
//! Inner commitment: `v_i = A · w_i` for each witness vector `w_i`.
//! Outer commitment: `u_1 = Σ B_{ik} v_i^{(k)} + Σ C_{ijk} g_{ij}^{(k)}` and
//! `u_2 = Σ D_{ijk} h_{ij}^{(k)}`.

use crate::transcript::Transcript;
use modring::{Ring, RingElem, D};

/// Expand a deterministic `rows × cols` matrix of `RingElem` from the current
/// transcript state, under `label`. Each coefficient is a uniform `Z_{q'}`
/// element drawn via rejection sampling.
pub fn expand_matrix(
    t: &mut Transcript,
    label: &[u8],
    rows: usize,
    cols: usize,
    ring: &Ring,
) -> Vec<Vec<RingElem>> {
    let q = ring.m.q;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            let mut elem = RingElem::zero();
            for k in 0..D {
                let sub_label = [
                    label,
                    b"|",
                    &(r as u64).to_le_bytes(),
                    &(c as u64).to_le_bytes(),
                    &(k as u64).to_le_bytes(),
                ]
                .concat();
                elem.c[k] = t.derive_below(&sub_label, q);
            }
            row.push(elem);
        }
        out.push(row);
    }
    out
}

/// `A · w` where `A: rows × cols` and `w: cols`. Result has length `rows`.
pub fn matmul(ring: &Ring, a: &[Vec<RingElem>], w: &[RingElem]) -> Vec<RingElem> {
    assert!(!a.is_empty());
    let cols = a[0].len();
    assert_eq!(w.len(), cols, "rank mismatch");
    let m = &ring.m;
    a.iter()
        .map(|row| {
            let mut acc = RingElem::zero();
            for (i, x) in w.iter().enumerate() {
                let p = ring.mul(&row[i], x);
                acc = acc.add(m, &p);
            }
            acc
        })
        .collect()
}

/// Inner-commitment helper: `A · w_i` for each witness vector `w_i`.
pub fn commit_inner(
    ring: &Ring,
    a: &[Vec<RingElem>],
    ws: &[Vec<RingElem>],
) -> Vec<Vec<RingElem>> {
    ws.iter().map(|w| matmul(ring, a, w)).collect()
}

/// Add two ring vectors elementwise.
pub fn add_vec(ring: &Ring, x: &[RingElem], y: &[RingElem]) -> Vec<RingElem> {
    assert_eq!(x.len(), y.len());
    let m = &ring.m;
    x.iter().zip(y.iter()).map(|(a, b)| a.add(m, b)).collect()
}

/// Subtract `y` from `x` elementwise.
pub fn sub_vec(ring: &Ring, x: &[RingElem], y: &[RingElem]) -> Vec<RingElem> {
    assert_eq!(x.len(), y.len());
    let m = &ring.m;
    x.iter().zip(y.iter()).map(|(a, b)| a.sub(m, b)).collect()
}

/// Scale a ring vector by a single `RingElem`.
pub fn scale_vec(ring: &Ring, s: &RingElem, x: &[RingElem]) -> Vec<RingElem> {
    x.iter().map(|e| ring.mul(s, e)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use modring::{find_prime_5mod8, Modulus};

    fn ring() -> Ring {
        Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
    }

    #[test]
    fn expand_matrix_is_deterministic_given_same_transcript() {
        let r = ring();
        let mut t1 = Transcript::new(b"unit");
        let mut t2 = Transcript::new(b"unit");
        t1.absorb(b"x", &[1, 2]);
        t2.absorb(b"x", &[1, 2]);
        let a = expand_matrix(&mut t1, b"A", 2, 3, &r);
        let b = expand_matrix(&mut t2, b"A", 2, 3, &r);
        assert_eq!(a, b);
    }

    #[test]
    fn commit_inner_is_linear() {
        let r = ring();
        let mut t = Transcript::new(b"unit");
        let a = expand_matrix(&mut t, b"A", 3, 4, &r);

        let mut rng = modring::rng::SplitMix64::new(7);
        let mk_vec = |rng: &mut modring::rng::SplitMix64, n: usize| {
            (0..n)
                .map(|_| {
                    let mut p = RingElem::zero();
                    for k in 0..D {
                        p.c[k] = rng.below(r.m.q);
                    }
                    p
                })
                .collect::<Vec<_>>()
        };
        let w1 = mk_vec(&mut rng, 4);
        let w2 = mk_vec(&mut rng, 4);
        let w12 = add_vec(&r, &w1, &w2);

        let v1 = matmul(&r, &a, &w1);
        let v2 = matmul(&r, &a, &w2);
        let v12 = matmul(&r, &a, &w12);
        let sum = add_vec(&r, &v1, &v2);
        for k in 0..v12.len() {
            assert_eq!(v12[k], sum[k]);
        }
    }
}
