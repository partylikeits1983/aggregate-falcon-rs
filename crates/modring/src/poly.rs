//! Elements of the LaBRADOR ring `R_Q = Z_Q[X] / (X^64 + 1)`.
//!
//! Coefficients are canonical residues in `[0, q)`. Multiplication here is
//! the straightforward negacyclic schoolbook product; it is intentionally
//! the *reference oracle* against which the faster CRT path in [`crate::crt`]
//! is tested.

use crate::modulus::Modulus;
use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

/// Degree of the LaBRADOR ring (`X^64 + 1`).
pub const D: usize = 64;

/// An element of `R_Q`, in coefficient representation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingElem {
    /// Coefficients `c[i]` of `X^i`, each a canonical residue in `[0, q)`.
    pub c: [u64; D],
}

impl Serialize for RingElem {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut tup = s.serialize_tuple(D)?;
        for v in &self.c {
            tup.serialize_element(v)?;
        }
        tup.end()
    }
}

impl<'de> Deserialize<'de> for RingElem {
    fn deserialize<De: Deserializer<'de>>(d: De) -> Result<Self, De::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = RingElem;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "RingElem with {D} u64 coefficients")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<RingElem, A::Error> {
                let mut c = [0u64; D];
                for slot in &mut c {
                    *slot = seq.next_element()?.ok_or_else(|| {
                        serde::de::Error::invalid_length(D, &"RingElem requires D elements")
                    })?;
                }
                Ok(RingElem { c })
            }
        }
        d.deserialize_tuple(D, V)
    }
}

impl RingElem {
    /// The zero polynomial.
    pub fn zero() -> Self {
        RingElem { c: [0u64; D] }
    }

    /// The constant polynomial `a` (reduced mod `q`).
    pub fn constant(m: &Modulus, a: u64) -> Self {
        let mut p = Self::zero();
        p.c[0] = a % m.q;
        p
    }

    /// The monomial `X^k` for any `k >= 0`, reduced via `X^64 = -1`.
    pub fn monomial(m: &Modulus, k: usize) -> Self {
        let mut p = Self::zero();
        let sign_neg = (k / D) % 2 == 1;
        let idx = k % D;
        p.c[idx] = if sign_neg { m.neg(1) } else { 1 };
        p
    }

    /// Build from signed coefficients (centered representatives).
    pub fn from_i64(m: &Modulus, coeffs: &[i64; D]) -> Self {
        let mut p = Self::zero();
        for i in 0..D {
            p.c[i] = m.from_i64(coeffs[i]);
        }
        p
    }

    /// True if every coefficient is zero.
    pub fn is_zero(&self) -> bool {
        self.c.iter().all(|&x| x == 0)
    }

    /// Coefficient-wise sum.
    pub fn add(&self, m: &Modulus, other: &RingElem) -> RingElem {
        let mut r = RingElem::zero();
        for i in 0..D {
            r.c[i] = m.add(self.c[i], other.c[i]);
        }
        r
    }

    /// Coefficient-wise difference.
    pub fn sub(&self, m: &Modulus, other: &RingElem) -> RingElem {
        let mut r = RingElem::zero();
        for i in 0..D {
            r.c[i] = m.sub(self.c[i], other.c[i]);
        }
        r
    }

    /// Coefficient-wise negation.
    pub fn neg(&self, m: &Modulus) -> RingElem {
        let mut r = RingElem::zero();
        for i in 0..D {
            r.c[i] = m.neg(self.c[i]);
        }
        r
    }

    /// Multiply every coefficient by a scalar in `[0, q)`.
    pub fn scale(&self, m: &Modulus, s: u64) -> RingElem {
        let mut r = RingElem::zero();
        for i in 0..D {
            r.c[i] = m.mul(self.c[i], s);
        }
        r
    }

    /// Negacyclic schoolbook product in `R_Q` (the reference multiplication).
    ///
    /// For `i + j >= 64` the term wraps with a sign flip, since `X^64 = -1`.
    pub fn mul(&self, m: &Modulus, other: &RingElem) -> RingElem {
        let mut r = RingElem::zero();
        for i in 0..D {
            if self.c[i] == 0 {
                continue;
            }
            for j in 0..D {
                let p = m.mul(self.c[i], other.c[j]);
                let k = i + j;
                if k < D {
                    r.c[k] = m.add(r.c[k], p);
                } else {
                    r.c[k - D] = m.sub(r.c[k - D], p);
                }
            }
        }
        r
    }

    /// Squared Euclidean norm of the centered coefficient vector, as `i128`.
    pub fn norm_sq(&self, m: &Modulus) -> i128 {
        self.c
            .iter()
            .map(|&x| {
                let v = m.centered(x) as i128;
                v * v
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulus::find_prime_5mod8;
    use crate::rng::SplitMix64;

    fn modulus() -> Modulus {
        Modulus::new(find_prime_5mod8(1 << 44))
    }

    fn uniform(m: &Modulus, rng: &mut SplitMix64) -> RingElem {
        let mut p = RingElem::zero();
        for i in 0..D {
            p.c[i] = rng.next() % m.q;
        }
        p
    }

    #[test]
    fn add_sub_inverse() {
        let m = modulus();
        let mut rng = SplitMix64::new(1);
        for _ in 0..100 {
            let a = uniform(&m, &mut rng);
            let b = uniform(&m, &mut rng);
            assert_eq!(a.add(&m, &b).sub(&m, &b), a);
        }
    }

    #[test]
    fn mul_is_commutative_and_distributive() {
        let m = modulus();
        let mut rng = SplitMix64::new(2);
        for _ in 0..50 {
            let a = uniform(&m, &mut rng);
            let b = uniform(&m, &mut rng);
            let c = uniform(&m, &mut rng);
            assert_eq!(a.mul(&m, &b), b.mul(&m, &a));
            // a*(b+c) == a*b + a*c
            let lhs = a.mul(&m, &b.add(&m, &c));
            let rhs = a.mul(&m, &b).add(&m, &a.mul(&m, &c));
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn monomial_wraps_with_sign() {
        let m = modulus();
        // X^64 == -1
        assert_eq!(RingElem::monomial(&m, 64), RingElem::constant(&m, m.neg(1)));
        // X^64 == X^0 * (-1): multiplying X^1 by X^63 gives -1.
        let x = RingElem::monomial(&m, 1);
        let x63 = RingElem::monomial(&m, 63);
        assert_eq!(x.mul(&m, &x63), RingElem::constant(&m, m.neg(1)));
    }

    #[test]
    fn one_is_multiplicative_identity() {
        let m = modulus();
        let mut rng = SplitMix64::new(3);
        let one = RingElem::constant(&m, 1);
        for _ in 0..50 {
            let a = uniform(&m, &mut rng);
            assert_eq!(a.mul(&m, &one), a);
        }
    }
}
