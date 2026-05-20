//! Prime field `F_Q` for the LaBRADOR modulus `Q`.
//!
//! `Q` is chosen dynamically (it grows with the number of aggregated
//! signatures). Phase-0 measurement of the paper's estimator showed `Q`
//! stays below `2^50` for `N` up to 4096, so a `u64` representation with
//! `u128` intermediates is exact: a product of two residues is at most
//! `(2^50)^2 = 2^100 < 2^128`. We therefore use plain `u128 % q` reduction
//! rather than Montgomery form — clarity over speed for this reference.
//!
//! All field elements are stored as canonical residues in `[0, q)`.

/// A prime modulus together with the field operations on `[0, q)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Modulus {
    /// The prime `q`.
    pub q: u64,
}

impl Modulus {
    /// Build a modulus. Panics if `q` is not an odd prime, or `q >= 2^62`
    /// (beyond which a sum of two residues could overflow `u64`).
    pub fn new(q: u64) -> Self {
        assert!(q >= 3, "modulus must be at least 3");
        assert!(q < (1u64 << 62), "modulus too large for the u64 backend");
        assert!(is_prime(q), "modulus must be prime");
        Modulus { q }
    }

    /// `a + b mod q`, assuming `a, b < q`.
    #[inline]
    pub fn add(&self, a: u64, b: u64) -> u64 {
        let s = a + b;
        if s >= self.q {
            s - self.q
        } else {
            s
        }
    }

    /// `a - b mod q`, assuming `a, b < q`.
    #[inline]
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            a + self.q - b
        }
    }

    /// `-a mod q`, assuming `a < q`.
    #[inline]
    pub fn neg(&self, a: u64) -> u64 {
        if a == 0 {
            0
        } else {
            self.q - a
        }
    }

    /// `a * b mod q`, assuming `a, b < q`.
    #[inline]
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % self.q as u128) as u64
    }

    /// Reduce an arbitrary `u128` into `[0, q)`.
    #[inline]
    pub fn reduce(&self, x: u128) -> u64 {
        (x % self.q as u128) as u64
    }

    /// Map a signed integer into `[0, q)`.
    #[inline]
    pub fn from_i64(&self, x: i64) -> u64 {
        x.rem_euclid(self.q as i64) as u64
    }

    /// The centered representative of `a` in `(-q/2, q/2]`.
    #[inline]
    pub fn centered(&self, a: u64) -> i64 {
        debug_assert!(a < self.q);
        if a > self.q / 2 {
            a as i64 - self.q as i64
        } else {
            a as i64
        }
    }

    /// `a^e mod q`.
    pub fn pow(&self, mut a: u64, mut e: u64) -> u64 {
        let mut r = 1u64;
        a %= self.q;
        while e > 0 {
            if e & 1 == 1 {
                r = self.mul(r, a);
            }
            a = self.mul(a, a);
            e >>= 1;
        }
        r
    }

    /// Multiplicative inverse via Fermat's little theorem. Panics on `0`.
    pub fn inv(&self, a: u64) -> u64 {
        assert!(a % self.q != 0, "0 has no inverse");
        self.pow(a, self.q - 2)
    }

    /// A square root of `a`, or `None` if `a` is a non-residue.
    /// Tonelli–Shanks; `0` maps to `0`.
    pub fn sqrt(&self, a: u64) -> Option<u64> {
        let q = self.q;
        if a == 0 {
            return Some(0);
        }
        // Euler's criterion.
        if self.pow(a, (q - 1) / 2) != 1 {
            return None;
        }
        if q % 4 == 3 {
            return Some(self.pow(a, (q + 1) / 4));
        }
        // Write q - 1 = m * 2^s with m odd.
        let mut s = 0u32;
        let mut m = q - 1;
        while m & 1 == 0 {
            m >>= 1;
            s += 1;
        }
        // Find a non-residue z.
        let mut z = 2u64;
        while self.pow(z, (q - 1) / 2) != q - 1 {
            z += 1;
        }
        let mut c = self.pow(z, m);
        let mut t = self.pow(a, m);
        let mut r = self.pow(a, (m + 1) / 2);
        let mut s = s;
        loop {
            if t == 1 {
                return Some(r);
            }
            // Smallest i with t^(2^i) == 1.
            let mut i = 0u32;
            let mut t2 = t;
            while t2 != 1 {
                t2 = self.mul(t2, t2);
                i += 1;
            }
            let b = self.pow(c, 1u64 << (s - i - 1));
            r = self.mul(r, b);
            c = self.mul(b, b);
            t = self.mul(t, c);
            s = i;
        }
    }

    /// A square root of `-1`, i.e. an element `r` with `r^2 = q - 1`.
    /// Exists exactly when `q ≡ 1 (mod 4)`.
    pub fn sqrt_minus_one(&self) -> Option<u64> {
        self.sqrt(self.q - 1)
    }
}

/// Deterministic Miller–Rabin primality test, exact for all `u64`.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    let mut d = n - 1;
    let mut r = 0u32;
    while d & 1 == 0 {
        d >>= 1;
        r += 1;
    }
    let mulmod = |a: u64, b: u64| -> u64 { (a as u128 * b as u128 % n as u128) as u64 };
    let powmod = |mut a: u64, mut e: u64| -> u64 {
        let mut x = 1u64;
        while e > 0 {
            if e & 1 == 1 {
                x = mulmod(x, a);
            }
            a = mulmod(a, a);
            e >>= 1;
        }
        x
    };
    // These 12 witnesses are a proven-correct set for all n < 2^64.
    'next: for &a in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod(a % n, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mulmod(x, x);
            if x == n - 1 {
                continue 'next;
            }
        }
        return false;
    }
    true
}

/// Smallest prime `>= min_value` congruent to `5 (mod 8)`.
///
/// `q ≡ 5 (mod 8)` guarantees `q ≡ 1 (mod 4)` (so `sqrt(-1)` exists) while
/// `-1` is not itself a square's square — exactly the condition under which
/// `X^64 + 1` factors into two irreducible degree-32 polynomials mod `q`,
/// i.e. the two-splitting case targeted by this implementation.
pub fn find_prime_5mod8(min_value: u64) -> u64 {
    let mut c = min_value + ((5 + 8 - min_value % 8) % 8);
    loop {
        if is_prime(c) {
            return c;
        }
        c += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primality_known_values() {
        assert!(is_prime(2));
        assert!(is_prime(12289)); // Falcon modulus
        assert!(!is_prime(1));
        assert!(!is_prime(12288));
        assert!(is_prime((1 << 31) - 1)); // Mersenne prime
        assert!(!is_prime(((1u64 << 31) - 1) * ((1u64 << 31) - 1)));
    }

    #[test]
    fn prime_search_is_5_mod_8_and_prime() {
        for bits in [39u32, 44, 50] {
            let q = find_prime_5mod8(1u64 << bits);
            assert!(is_prime(q));
            assert_eq!(q % 8, 5);
            assert!(q >= 1u64 << bits);
        }
    }

    #[test]
    fn sqrt_minus_one_squares_to_minus_one() {
        for bits in [39u32, 44, 50] {
            let m = Modulus::new(find_prime_5mod8(1u64 << bits));
            let r = m.sqrt_minus_one().expect("q = 5 mod 8 has sqrt(-1)");
            assert_eq!(m.mul(r, r), m.q - 1);
        }
    }

    #[test]
    fn inverse_round_trips() {
        let m = Modulus::new(find_prime_5mod8(1 << 44));
        for a in [1u64, 2, 3, 100, m.q - 1, m.q / 2] {
            assert_eq!(m.mul(a, m.inv(a)), 1);
        }
    }

    #[test]
    fn centered_round_trips() {
        let m = Modulus::new(find_prime_5mod8(1 << 44));
        for a in [0u64, 1, 7, m.q / 2, m.q / 2 + 1, m.q - 1] {
            assert_eq!(m.from_i64(m.centered(a)), a);
        }
    }
}
