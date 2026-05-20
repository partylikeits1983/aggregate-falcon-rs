//! Two-splitting CRT decomposition of the LaBRADOR ring.
//!
//! With `Q ≡ 5 (mod 8)` the modulus `X^64 + 1` factors mod `Q` as
//!
//! ```text
//! X^64 + 1 = (X^32 - r) * (X^32 + r),   where r^2 = -1 (mod Q).
//! ```
//!
//! The CRT isomorphism therefore maps `R_Q` to a pair of degree-32 rings
//! `Z_Q[X]/(X^32 - r)` and `Z_Q[X]/(X^32 + r)`. Multiplication becomes a
//! pair of independent degree-32 products — the "NTT" of the two-splitting
//! parameter set. (There is no radix-2 NTT here; the C `ntt4`/`ntt8` code in
//! the reference repo targets a deeper splitting and is deliberately unused.)

use crate::modulus::Modulus;
use crate::poly::{RingElem, D};

/// Half-degree of each CRT factor.
pub const H: usize = D / 2; // 32

/// The LaBRADOR ring together with the constant `r = sqrt(-1)` that drives
/// the two-splitting CRT.
#[derive(Clone, Copy, Debug)]
pub struct Ring {
    /// Coefficient field.
    pub m: Modulus,
    /// A fixed square root of `-1` mod `q`.
    pub r: u64,
}

/// CRT representation of a ring element: residues modulo each factor.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CrtRepr {
    /// Residue modulo `X^32 - r`.
    pub h0: [u64; H],
    /// Residue modulo `X^32 + r`.
    pub h1: [u64; H],
}

impl Ring {
    /// Build the ring context for modulus `m`, computing `r = sqrt(-1)`.
    /// Panics if `q ≢ 1 (mod 4)` (no square root of `-1`).
    pub fn new(m: Modulus) -> Self {
        let r = m
            .sqrt_minus_one()
            .expect("q must be 1 mod 4 for the two-splitting ring");
        Ring { m, r }
    }

    /// Map a ring element to its CRT residues.
    ///
    /// Writing `a = a_lo + X^32 * a_hi`, reduction by `X^32 = ±r` gives
    /// `h0 = a_lo + r*a_hi` and `h1 = a_lo - r*a_hi`.
    pub fn split(&self, a: &RingElem) -> CrtRepr {
        let m = &self.m;
        let mut h0 = [0u64; H];
        let mut h1 = [0u64; H];
        for i in 0..H {
            let lo = a.c[i];
            let hi = a.c[i + H];
            let rhi = m.mul(self.r, hi);
            h0[i] = m.add(lo, rhi);
            h1[i] = m.sub(lo, rhi);
        }
        CrtRepr { h0, h1 }
    }

    /// Inverse of [`Ring::split`].
    ///
    /// From `h0 = a_lo + r*a_hi` and `h1 = a_lo - r*a_hi`:
    /// `a_lo = (h0 + h1) / 2` and `a_hi = (h0 - h1) / (2r)`.
    pub fn combine(&self, c: &CrtRepr) -> RingElem {
        let m = &self.m;
        let inv2 = m.inv(2);
        let inv2r = m.inv(m.mul(2, self.r));
        let mut a = RingElem::zero();
        for i in 0..H {
            a.c[i] = m.mul(m.add(c.h0[i], c.h1[i]), inv2);
            a.c[i + H] = m.mul(m.sub(c.h0[i], c.h1[i]), inv2r);
        }
        a
    }

    /// Degree-32 schoolbook product in `Z_Q[X]/(X^32 - s)`.
    fn mul_half(&self, a: &[u64; H], b: &[u64; H], s: u64) -> [u64; H] {
        let m = &self.m;
        let mut out = [0u64; H];
        for i in 0..H {
            if a[i] == 0 {
                continue;
            }
            for j in 0..H {
                let p = m.mul(a[i], b[j]);
                let k = i + j;
                if k < H {
                    out[k] = m.add(out[k], p);
                } else {
                    // X^32 = s in this factor.
                    out[k - H] = m.add(out[k - H], m.mul(s, p));
                }
            }
        }
        out
    }

    /// Multiply two ring elements via the CRT path.
    ///
    /// Must agree with [`RingElem::mul`] — this is the central invariant
    /// guarding the two-splitting decomposition.
    pub fn mul(&self, a: &RingElem, b: &RingElem) -> RingElem {
        let ca = self.split(a);
        let cb = self.split(b);
        let neg_r = self.m.neg(self.r);
        let h0 = self.mul_half(&ca.h0, &cb.h0, self.r);
        let h1 = self.mul_half(&ca.h1, &cb.h1, neg_r);
        self.combine(&CrtRepr { h0, h1 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulus::find_prime_5mod8;
    use crate::rng::SplitMix64;

    fn ring(bits: u32) -> Ring {
        Ring::new(Modulus::new(find_prime_5mod8(1 << bits)))
    }

    fn uniform(m: &Modulus, rng: &mut SplitMix64) -> RingElem {
        let mut p = RingElem::zero();
        for i in 0..D {
            p.c[i] = rng.below(m.q);
        }
        p
    }

    #[test]
    fn split_combine_round_trip() {
        for bits in [39u32, 44, 50] {
            let rg = ring(bits);
            let mut rng = SplitMix64::new(bits as u64);
            for _ in 0..200 {
                let a = uniform(&rg.m, &mut rng);
                assert_eq!(rg.combine(&rg.split(&a)), a);
            }
        }
    }

    #[test]
    fn crt_mul_matches_schoolbook() {
        for bits in [39u32, 44, 50] {
            let rg = ring(bits);
            let mut rng = SplitMix64::new(1000 + bits as u64);
            for _ in 0..300 {
                let a = uniform(&rg.m, &mut rng);
                let b = uniform(&rg.m, &mut rng);
                assert_eq!(rg.mul(&a, &b), a.mul(&rg.m, &b));
            }
        }
    }

    #[test]
    fn crt_mul_handles_monomials() {
        let rg = ring(44);
        // X^32 * X^32 = X^64 = -1
        let x32 = RingElem::monomial(&rg.m, 32);
        assert_eq!(
            rg.mul(&x32, &x32),
            RingElem::constant(&rg.m, rg.m.neg(1))
        );
    }

    #[test]
    fn r_squares_to_minus_one() {
        let rg = ring(44);
        assert_eq!(rg.m.mul(rg.r, rg.r), rg.m.q - 1);
    }
}
