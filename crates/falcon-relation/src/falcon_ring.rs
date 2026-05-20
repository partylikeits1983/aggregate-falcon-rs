//! Arithmetic in the Falcon-512 ring `R_f = Z_12289[X] / (X^512 + 1)`.
//!
//! This is used only to *check* and *encode* signatures — never to produce
//! them — so a straightforward schoolbook negacyclic product is enough; no
//! Falcon-style NTT is needed here.
//!
//! Coefficients are kept as canonical residues in `[0, 12289)`. Falcon's
//! short vectors `s1, s2` are small and signed; use [`FPoly::from_i32`] to
//! enter them and [`FPoly::centered`] / [`FPoly::norm_sq`] to read them back.

/// The Falcon modulus.
pub const FALCON_Q: u32 = 12289;

/// The Falcon-512 ring degree.
pub const FALCON_N: usize = 512;

/// Falcon's acceptance bound on `||(s1, s2)||^2` for the 512 parameter set.
/// (`beta` in `proof_size_estimate.py` is `5834`; this is `beta^2`.)
pub const FALCON_BETA_SQ: i64 = 5834 * 5834;

/// An element of `R_f`, in coefficient representation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FPoly {
    /// Coefficients `c[i]` of `X^i`, each a canonical residue in `[0, q)`.
    pub c: [u32; FALCON_N],
}

#[inline]
fn add_q(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= FALCON_Q {
        s - FALCON_Q
    } else {
        s
    }
}

#[inline]
fn sub_q(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + FALCON_Q - b
    }
}

impl FPoly {
    /// The zero polynomial.
    pub fn zero() -> Self {
        FPoly { c: [0u32; FALCON_N] }
    }

    /// Build from residues already in `[0, q)`.
    pub fn from_residues(c: [u32; FALCON_N]) -> Self {
        for &x in &c {
            debug_assert!(x < FALCON_Q);
        }
        FPoly { c }
    }

    /// Build from signed (centered) coefficients, reducing mod `q`.
    pub fn from_i32(coeffs: &[i32; FALCON_N]) -> Self {
        let mut p = Self::zero();
        for i in 0..FALCON_N {
            p.c[i] = coeffs[i].rem_euclid(FALCON_Q as i32) as u32;
        }
        p
    }

    /// The centered representative of coefficient `i`, in `(-q/2, q/2]`.
    #[inline]
    pub fn centered(&self, i: usize) -> i32 {
        let x = self.c[i] as i32;
        if x > FALCON_Q as i32 / 2 {
            x - FALCON_Q as i32
        } else {
            x
        }
    }

    /// Coefficient-wise sum.
    pub fn add(&self, other: &FPoly) -> FPoly {
        let mut r = FPoly::zero();
        for i in 0..FALCON_N {
            r.c[i] = add_q(self.c[i], other.c[i]);
        }
        r
    }

    /// Coefficient-wise difference.
    pub fn sub(&self, other: &FPoly) -> FPoly {
        let mut r = FPoly::zero();
        for i in 0..FALCON_N {
            r.c[i] = sub_q(self.c[i], other.c[i]);
        }
        r
    }

    /// Coefficient-wise negation.
    pub fn neg(&self) -> FPoly {
        let mut r = FPoly::zero();
        for i in 0..FALCON_N {
            r.c[i] = if self.c[i] == 0 {
                0
            } else {
                FALCON_Q - self.c[i]
            };
        }
        r
    }

    /// Negacyclic schoolbook product in `R_f` (`X^512 = -1`).
    pub fn mul(&self, other: &FPoly) -> FPoly {
        let mut acc = [0i64; FALCON_N];
        let q = FALCON_Q as i64;
        for i in 0..FALCON_N {
            let ai = self.c[i] as i64;
            if ai == 0 {
                continue;
            }
            for j in 0..FALCON_N {
                let p = ai * other.c[j] as i64;
                let k = i + j;
                if k < FALCON_N {
                    acc[k] += p;
                } else {
                    acc[k - FALCON_N] -= p;
                }
            }
        }
        let mut r = FPoly::zero();
        for i in 0..FALCON_N {
            r.c[i] = acc[i].rem_euclid(q) as u32;
        }
        r
    }

    /// Squared Euclidean norm of the centered coefficient vector.
    pub fn norm_sq(&self) -> i64 {
        (0..FALCON_N)
            .map(|i| {
                let v = self.centered(i) as i64;
                v * v
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modring::rng::SplitMix64;

    fn uniform(rng: &mut SplitMix64) -> FPoly {
        let mut p = FPoly::zero();
        for i in 0..FALCON_N {
            p.c[i] = rng.below(FALCON_Q as u64) as u32;
        }
        p
    }

    fn monomial(k: usize) -> FPoly {
        let mut p = FPoly::zero();
        let neg = (k / FALCON_N) % 2 == 1;
        p.c[k % FALCON_N] = if neg { FALCON_Q - 1 } else { 1 };
        p
    }

    #[test]
    fn add_sub_inverse() {
        let mut rng = SplitMix64::new(1);
        for _ in 0..50 {
            let a = uniform(&mut rng);
            let b = uniform(&mut rng);
            assert_eq!(a.add(&b).sub(&b), a);
        }
    }

    #[test]
    fn mul_commutative_and_distributive() {
        let mut rng = SplitMix64::new(2);
        for _ in 0..20 {
            let a = uniform(&mut rng);
            let b = uniform(&mut rng);
            let c = uniform(&mut rng);
            assert_eq!(a.mul(&b), b.mul(&a));
            assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
        }
    }

    #[test]
    fn x_pow_512_is_minus_one() {
        // X^256 * X^256 = X^512 = -1
        let x256 = monomial(256);
        let mut neg_one = FPoly::zero();
        neg_one.c[0] = FALCON_Q - 1;
        assert_eq!(x256.mul(&x256), neg_one);
    }

    #[test]
    fn identity_and_norm() {
        let mut rng = SplitMix64::new(3);
        let mut one = FPoly::zero();
        one.c[0] = 1;
        let a = uniform(&mut rng);
        assert_eq!(a.mul(&one), a);

        // Norm of a known small vector.
        let mut s = [0i32; FALCON_N];
        s[0] = 3;
        s[1] = -4;
        assert_eq!(FPoly::from_i32(&s).norm_sq(), 25);
    }
}
