//! Subring embedding of the Falcon ring into the LaBRADOR ring.
//!
//! Following [LNPS21] §2.8 (used in §6.4 of the aggregation paper), the
//! Falcon ring `R = Z[X]/(X^512+1)` is a free `S`-module of rank `c = 8`
//! over the LaBRADOR ring `S = Z[Y]/(Y^64+1)`, via `Y = X^8`. Concretely,
//! every `a(X) ∈ R` decomposes uniquely as
//!
//! ```text
//! a(X) = Σ_{k=0}^{7} X^k · a_k(X^8),    a_k ∈ S.
//! ```
//!
//! Coefficient-wise this is just a permutation: `a_k.coeffs[j] = a.coeffs[k + 8j]`.
//! The map is norm-preserving and product-compatible — multiplication in `R`
//! becomes the explicit bilinear product on `S^8` given by [`mul_subring`].
//!
//! The aggregation construction lifts integer-valued Falcon polynomials
//! (`s1, s2, h, c, …`) from `Z_{12289}` to the larger LaBRADOR modulus
//! `Z_{q'}`. We therefore take *centered* Falcon coefficients (small signed
//! values) and reduce them mod `q'` before placing them into the slots.

use crate::falcon_ring::{FPoly, FALCON_N};
use modring::{Modulus, Ring, RingElem, D};

/// The number of LaBRADOR-ring slots a Falcon-ring element decomposes into:
/// `c = deg(R) / deg(S) = 512 / 64 = 8`.
pub const C: usize = 8;

const _: () = assert!(FALCON_N == C * D);

/// Embed centered Falcon coefficients into the LaBRADOR ring as an `S^c` tuple.
///
/// `coeffs[i]` is taken as the centered representative of the `i`-th Falcon
/// coefficient (typically a small signed integer for `s1`, `s2`, `ε`, `v`).
/// It is reduced mod the LaBRADOR modulus and placed into slot `i mod C`,
/// position `i div C`.
pub fn embed_signed(coeffs: &[i64; FALCON_N], m: &Modulus) -> [RingElem; C] {
    let mut out = [
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
    ];
    for i in 0..FALCON_N {
        let k = i % C; // slot
        let j = i / C; // position within slot
        out[k].c[j] = m.from_i64(coeffs[i]);
    }
    out
}

/// Embed an `FPoly` (Falcon residue representation) using centered coefficients.
pub fn embed_fpoly_centered(f: &FPoly, m: &Modulus) -> [RingElem; C] {
    let mut coeffs = [0i64; FALCON_N];
    for i in 0..FALCON_N {
        coeffs[i] = f.centered(i) as i64;
    }
    embed_signed(&coeffs, m)
}

/// Inverse of [`embed_signed`], returning the centered Falcon coefficients
/// (as `i64`). Used in tests and for inspecting decoded witnesses.
pub fn extract_centered(slots: &[RingElem; C], m: &Modulus) -> [i64; FALCON_N] {
    let mut out = [0i64; FALCON_N];
    for k in 0..C {
        for j in 0..D {
            out[k + C * j] = m.centered(slots[k].c[j]);
        }
    }
    out
}

/// Multiply two elements of `R` represented in `S^c` form: the image of the
/// Falcon-ring product `a · b ∈ R` under the subring decomposition.
///
/// Derivation: with `Y = X^c`,
/// `a · b = Σ_{i,j} X^{i+j} a_i(Y) b_j(Y)`. For `i + j < c` this contributes to
/// slot `i + j`; for `c ≤ i + j < 2c − 1` the factor `X^{i+j} = X^{(i+j)−c}·Y`
/// contributes to slot `(i+j) − c` with one extra factor of `Y` (which is
/// just `RingElem::monomial(1)` in `S`).
pub fn mul_subring(a: &[RingElem; C], b: &[RingElem; C], ring: &Ring) -> [RingElem; C] {
    let m = &ring.m;
    let y = RingElem::monomial(m, 1);
    let mut out = [
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
    ];
    for i in 0..C {
        for j in 0..C {
            let p = ring.mul(&a[i], &b[j]);
            let s = i + j;
            if s < C {
                out[s] = out[s].add(m, &p);
            } else {
                let py = ring.mul(&y, &p);
                out[s - C] = out[s - C].add(m, &py);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use modring::{find_prime_5mod8, rng::SplitMix64};

    fn ring() -> Ring {
        Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
    }

    /// A degree-512 negacyclic polynomial mod the LaBRADOR modulus,
    /// used purely as a reference oracle for the subring multiplication.
    struct BigPoly {
        c: [u64; FALCON_N],
    }

    impl BigPoly {
        fn zero() -> Self {
            BigPoly { c: [0u64; FALCON_N] }
        }

        fn random(m: &Modulus, max_centered: i64, rng: &mut SplitMix64) -> Self {
            let mut p = BigPoly::zero();
            let span = (2 * max_centered + 1) as u64;
            for i in 0..FALCON_N {
                let r = (rng.next() % span) as i64 - max_centered;
                p.c[i] = m.from_i64(r);
            }
            p
        }

        fn mul(&self, m: &Modulus, other: &BigPoly) -> BigPoly {
            let mut out = BigPoly::zero();
            for i in 0..FALCON_N {
                if self.c[i] == 0 {
                    continue;
                }
                for j in 0..FALCON_N {
                    let p = m.mul(self.c[i], other.c[j]);
                    let k = i + j;
                    if k < FALCON_N {
                        out.c[k] = m.add(out.c[k], p);
                    } else {
                        out.c[k - FALCON_N] = m.sub(out.c[k - FALCON_N], p);
                    }
                }
            }
            out
        }

        fn embed(&self, m: &Modulus) -> [RingElem; C] {
            let mut signed = [0i64; FALCON_N];
            for i in 0..FALCON_N {
                signed[i] = m.centered(self.c[i]);
            }
            embed_signed(&signed, m)
        }
    }

    #[test]
    fn embed_extract_round_trip() {
        let rg = ring();
        let mut rng = SplitMix64::new(7);
        let a = BigPoly::random(&rg.m, 1_000, &mut rng);
        let slots = a.embed(&rg.m);
        let back = extract_centered(&slots, &rg.m);
        for i in 0..FALCON_N {
            assert_eq!(back[i], rg.m.centered(a.c[i]));
        }
    }

    #[test]
    fn mul_subring_matches_bigpoly_multiplication() {
        let rg = ring();
        // Keep coefficient magnitudes modest so the BigPoly mul oracle stays
        // exact under u128 intermediates (q ≈ 2^44, |coef| ≤ 50_000 → product
        // sum < 2^44 * 2 * 50_000^2 * 512 ≈ 2^81 fits u128).
        for seed in 0..6 {
            let mut rng = SplitMix64::new(100 + seed);
            let a = BigPoly::random(&rg.m, 50_000, &mut rng);
            let b = BigPoly::random(&rg.m, 50_000, &mut rng);

            let c_direct = a.mul(&rg.m, &b);
            let c_subring = mul_subring(&a.embed(&rg.m), &b.embed(&rg.m), &rg);
            assert_eq!(extract_centered(&c_subring, &rg.m),
                       extract_centered(&c_direct.embed(&rg.m), &rg.m),
                       "subring mul disagrees with degree-512 negacyclic mul (seed {seed})");
        }
    }

    #[test]
    fn embedding_preserves_norm() {
        let rg = ring();
        let mut rng = SplitMix64::new(99);
        let a = BigPoly::random(&rg.m, 5_000, &mut rng);
        let slots = a.embed(&rg.m);

        let direct_norm: i128 = (0..FALCON_N)
            .map(|i| {
                let v = rg.m.centered(a.c[i]) as i128;
                v * v
            })
            .sum();
        let slot_norm: i128 = slots.iter().map(|s| s.norm_sq(&rg.m)).sum();
        assert_eq!(direct_norm, slot_norm);
    }
}
