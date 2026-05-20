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

/// Outer commitment over the inner-commitment chunks `v_i^{(k)}`.
///
/// `b_mats[i][k]` is the per-(witness-index, chunk-index) Ajtai matrix
/// `B_{i,k} ∈ R_{q'}^{κ₁ × κ}`. `v_chunks[i][k]` is the chunk `v_i^{(k)} ∈ R^κ`
/// produced by decomposing the inner commitment `v_i = A·w_i` into `t₁` chunks
/// w.r.t. base `b₁`.
///
/// Result: `u = Σ_{i,k} B_{i,k} · v_i^{(k)} ∈ R^{κ₁}`. (Protocol 2 Step 1.)
pub fn outer_commit_v(
    ring: &Ring,
    b_mats: &[Vec<Vec<Vec<RingElem>>>],
    v_chunks: &[Vec<Vec<RingElem>>],
) -> Vec<RingElem> {
    let r = v_chunks.len();
    assert_eq!(b_mats.len(), r, "b_mats outer rank must equal r");
    let t1 = if r == 0 { 0 } else { v_chunks[0].len() };
    let kappa1 = if r == 0 || t1 == 0 {
        0
    } else {
        b_mats[0][0].len()
    };
    let mut u = vec![RingElem::zero(); kappa1];
    let m = &ring.m;
    for i in 0..r {
        assert_eq!(v_chunks[i].len(), t1, "ragged v_chunks at i={i}");
        assert_eq!(b_mats[i].len(), t1, "ragged b_mats at i={i}");
        for k in 0..t1 {
            let contrib = matmul(ring, &b_mats[i][k], &v_chunks[i][k]);
            assert_eq!(contrib.len(), kappa1);
            for r1 in 0..kappa1 {
                u[r1] = u[r1].add(m, &contrib[r1]);
            }
        }
    }
    u
}

/// Outer commitment over upper-triangular garbage chunks `m_{ij}^{(k)}`.
///
/// `c_mats[i][j][k]` is the per-(i,j,chunk) Ajtai matrix `C_{i,j,k} ∈ R^{κ₁ × 1}`
/// (so it stores `κ₁` ring elements as a column; we represent it as a row of
/// length `κ₁` and treat the input as a scalar). `chunks[i][j][k]` is the
/// `k`-th decomposition chunk of `g_{ij}` (or `h_{ij}`), a single ring element.
///
/// Result: `u = Σ_{i ≤ j, k} C_{i,j,k} · m_{ij}^{(k)} ∈ R^{κ₁}`.
/// (Protocol 2 Step 1 for `g`, Step 4 for `h`.) Only upper triangle is summed
/// because `g`, `h` are symmetric and stored once.
pub fn outer_commit_sym(
    ring: &Ring,
    c_mats: &[Vec<Vec<Vec<RingElem>>>],
    chunks: &[Vec<Vec<RingElem>>],
) -> Vec<RingElem> {
    let r = chunks.len();
    assert_eq!(c_mats.len(), r, "c_mats outer rank must equal r");
    let kappa1 = if r == 0 {
        0
    } else if c_mats[0].is_empty() {
        0
    } else {
        let first_nonempty = c_mats[0]
            .iter()
            .find(|col| !col.is_empty())
            .map(|col| col[0].len())
            .unwrap_or(0);
        first_nonempty
    };
    let mut u = vec![RingElem::zero(); kappa1];
    let m = &ring.m;
    for i in 0..r {
        for j in i..r {
            let n_chunks = chunks[i][j].len();
            assert_eq!(
                c_mats[i][j].len(),
                n_chunks,
                "ragged c_mats at (i={i}, j={j}): {} vs {n_chunks} chunks",
                c_mats[i][j].len()
            );
            for k in 0..n_chunks {
                let cijk = &c_mats[i][j][k];
                let coef = &chunks[i][j][k];
                assert_eq!(cijk.len(), kappa1, "c_mats[{i}][{j}][{k}] is not length κ₁");
                for r1 in 0..kappa1 {
                    let prod = ring.mul(&cijk[r1], coef);
                    u[r1] = u[r1].add(m, &prod);
                }
            }
        }
    }
    u
}

/// Expand a per-(i,k) B matrix family `B_{i,k} ∈ R^{κ₁ × κ}` from a transcript
/// seed. Shape: `b_mats[i][k]` is a `κ₁ × κ` matrix.
pub fn expand_b_mats(
    t: &mut crate::transcript::Transcript,
    label: &[u8],
    r: usize,
    t1: usize,
    kappa1: usize,
    kappa: usize,
    ring: &Ring,
) -> Vec<Vec<Vec<Vec<RingElem>>>> {
    let mut out = Vec::with_capacity(r);
    for i in 0..r {
        let mut per_i = Vec::with_capacity(t1);
        for k in 0..t1 {
            let sublabel = [
                label,
                b"|i|",
                &(i as u64).to_le_bytes(),
                b"|k|",
                &(k as u64).to_le_bytes(),
            ]
            .concat();
            per_i.push(expand_matrix(t, &sublabel, kappa1, kappa, ring));
        }
        out.push(per_i);
    }
    out
}

/// Expand a per-(i, j, k) C matrix family `C_{i,j,k} ∈ R^{κ₁ × 1}` (i ≤ j) from
/// a transcript seed. Shape: `c_mats[i][j][k]` is a length-`κ₁` ring vector.
/// Entries with `j < i` are filled with empty vectors (never indexed).
pub fn expand_sym_mats(
    t: &mut crate::transcript::Transcript,
    label: &[u8],
    r: usize,
    t_chunks: usize,
    kappa1: usize,
    ring: &Ring,
) -> Vec<Vec<Vec<Vec<RingElem>>>> {
    let mut out = vec![vec![vec![vec![]; t_chunks]; r]; r];
    for i in 0..r {
        for j in i..r {
            for k in 0..t_chunks {
                let sublabel = [
                    label,
                    b"|i|",
                    &(i as u64).to_le_bytes(),
                    b"|j|",
                    &(j as u64).to_le_bytes(),
                    b"|k|",
                    &(k as u64).to_le_bytes(),
                ]
                .concat();
                // C_{i,j,k} is a column of κ₁ ring elements.
                let row = expand_matrix(t, &sublabel, 1, kappa1, ring);
                // row is [[e_0, e_1, ..., e_{κ₁-1}]] (1 row, κ₁ cols); flatten to a length-κ₁ vec.
                out[i][j][k] = row.into_iter().next().unwrap();
            }
        }
    }
    out
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

    fn mk_random_vec(ring: &Ring, n: usize, seed: u64) -> Vec<RingElem> {
        let mut rng = modring::rng::SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                let mut p = RingElem::zero();
                for k in 0..D {
                    p.c[k] = rng.below(ring.m.q);
                }
                p
            })
            .collect()
    }

    #[test]
    fn outer_commit_v_is_linear() {
        // u(v + v') = u(v) + u(v') when summed element-wise over witness index
        // and chunk index.
        let ring = ring();
        let m = &ring.m;
        let r = 2usize;
        let t1 = 3usize;
        let kappa = 2usize;
        let kappa1 = 4usize;

        let mut tr = Transcript::new(b"outer-commit-v");
        let b_mats = expand_b_mats(&mut tr, b"B", r, t1, kappa1, kappa, &ring);

        let v: Vec<Vec<Vec<RingElem>>> = (0..r)
            .map(|i| (0..t1).map(|k| mk_random_vec(&ring, kappa, (i * 17 + k) as u64 + 1)).collect())
            .collect();
        let v_prime: Vec<Vec<Vec<RingElem>>> = (0..r)
            .map(|i| (0..t1).map(|k| mk_random_vec(&ring, kappa, (i * 31 + k) as u64 + 100)).collect())
            .collect();

        let u_v = outer_commit_v(&ring, &b_mats, &v);
        let u_vp = outer_commit_v(&ring, &b_mats, &v_prime);
        let sum: Vec<Vec<Vec<RingElem>>> = (0..r)
            .map(|i| {
                (0..t1)
                    .map(|k| {
                        (0..kappa)
                            .map(|c| v[i][k][c].add(m, &v_prime[i][k][c]))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let u_sum = outer_commit_v(&ring, &b_mats, &sum);

        for r1 in 0..kappa1 {
            assert_eq!(u_sum[r1], u_v[r1].add(m, &u_vp[r1]), "linearity at row {r1}");
        }
    }

    #[test]
    fn outer_commit_sym_only_touches_upper_triangle() {
        // Constructing a chunks tensor with garbage in the strict-lower triangle
        // must not affect the output (we don't read i > j).
        let ring = ring();
        let r = 3usize;
        let t_chunks = 2usize;
        let kappa1 = 3usize;

        let mut tr = Transcript::new(b"outer-commit-sym");
        let c_mats = expand_sym_mats(&mut tr, b"C", r, t_chunks, kappa1, &ring);

        // Build chunks: upper triangle deterministic; lower triangle GARBAGE.
        let mut chunks = vec![vec![vec![RingElem::zero(); t_chunks]; r]; r];
        for i in 0..r {
            for j in i..r {
                for k in 0..t_chunks {
                    let seed = (i * 100 + j * 10 + k) as u64 + 1;
                    chunks[i][j][k] = mk_random_vec(&ring, 1, seed).into_iter().next().unwrap();
                }
            }
        }
        let u_clean = outer_commit_sym(&ring, &c_mats, &chunks);

        // Pollute the strict-lower triangle with random nonsense.
        let mut polluted = chunks.clone();
        for i in 1..r {
            for j in 0..i {
                for k in 0..t_chunks {
                    polluted[i][j][k] = mk_random_vec(&ring, 1, 9999 + (i * 100 + j * 10 + k) as u64)
                        .into_iter()
                        .next()
                        .unwrap();
                }
            }
        }
        let u_polluted = outer_commit_sym(&ring, &c_mats, &polluted);

        assert_eq!(u_clean, u_polluted, "strict-lower triangle must be ignored");
    }

    #[test]
    fn outer_commit_v_is_deterministic() {
        // Same transcript → same B_mats → same u.
        let ring = ring();
        let r = 2usize;
        let t1 = 2usize;
        let kappa = 2usize;
        let kappa1 = 3usize;

        let v: Vec<Vec<Vec<RingElem>>> = (0..r)
            .map(|i| (0..t1).map(|k| mk_random_vec(&ring, kappa, (i + k * 7) as u64 + 1)).collect())
            .collect();

        let mut t1 = Transcript::new(b"det");
        let b1 = expand_b_mats(&mut t1, b"B", r, 2, kappa1, kappa, &ring);
        let mut t2 = Transcript::new(b"det");
        let b2 = expand_b_mats(&mut t2, b"B", r, 2, kappa1, kappa, &ring);
        assert_eq!(b1, b2);

        let u1 = outer_commit_v(&ring, &b1, &v);
        let u2 = outer_commit_v(&ring, &b2, &v);
        assert_eq!(u1, u2);
    }
}
