//! Negacyclic NTT over `Z_q[X]/(X^D + 1)`.
//!
//! Requires `q ≡ 1 (mod 2D)` so that `F_q` admits a primitive `2D`-th root of
//! unity `ψ` with `ψ^D = -1`. The forward NTT evaluates a polynomial `a` at
//! the `D` points `ψ^{2·br(i)+1}` for `i ∈ [0, D)` (where `br` is the
//! `log₂(D)`-bit reversal); the inverse NTT undoes this.
//!
//! Layout is the standard Cooley–Tukey (forward) / Gentleman–Sande (inverse)
//! radix-2 butterfly with bit-reversed twiddles, mirroring the well-tested
//! CRYSTALS-Kyber reference implementation. The forward NTT takes natural
//! order to bit-reversed order; the inverse takes bit-reversed back to
//! natural. Pointwise multiplication in bit-reversed order is the standard
//! `(a₀, a₁) * (b₀, b₁) = (a₀b₀, a₁b₁)` for each adjacent pair under the
//! `(X² - ψ²ⁱ⁺¹)` factorisation, but for our D=64 and depth `log₂(D)=6` the
//! NTT factors the ring down to D linear factors so each pair degenerates to
//! a single coefficient-wise multiplication.
//!
//! The cost per NTT is `D/2 · log₂(D)` modular multiplications; for D=64
//! that's 192 vs the 4096 of the schoolbook product. The full multiplication
//! (forward × 2, pointwise D, inverse, post-scale D) totals roughly
//! `3·192 + 2·D = 704` modular multiplications — a 5–6× win over schoolbook.

use crate::modulus::Modulus;
use crate::poly::D;

/// log₂(D). With D = 64, log_d = 6.
const LOG_D: u32 = D.trailing_zeros();

/// Reverse the lowest `bits` bits of `x`.
const fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    let mut i = 0u32;
    while i < bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
        i += 1;
    }
    r
}

/// Precomputed twiddle tables for the negacyclic NTT.
///
/// `fwd[i] = ψ^{br(i)} mod q` and `inv[i] = ψ^{-br(i)} mod q` for
/// `i ∈ [1, D)`. Slot `0` is unused (set to 1 for safety). The inverse table
/// is intentionally separate from `fwd` so that the inverse-NTT post-scale
/// step (multiplication by `D^{-1}`) can be folded into a single coefficient
/// pass.
#[derive(Clone, Copy, Debug)]
pub struct NttTables {
    pub fwd: [u64; D],
    pub inv: [u64; D],
    pub n_inv: u64,
}

impl NttTables {
    /// Build twiddles from a primitive `2D`-th root of unity `psi` in `F_q`.
    pub fn from_psi(m: &Modulus, psi: u64) -> Self {
        debug_assert_eq!(m.pow(psi, (2 * D) as u64), 1, "ψ is not a 2D-th root");
        debug_assert_eq!(
            m.pow(psi, D as u64),
            m.q - 1,
            "ψ^D must be -1 for the negacyclic NTT"
        );
        let psi_inv = m.inv(psi);
        let mut fwd = [1u64; D];
        let mut inv = [1u64; D];
        for i in 1..D {
            let e = bit_reverse(i, LOG_D) as u64;
            fwd[i] = m.pow(psi, e);
            inv[i] = m.pow(psi_inv, e);
        }
        let n_inv = m.inv(D as u64);
        NttTables { fwd, inv, n_inv }
    }
}

/// In-place forward negacyclic NTT (natural → bit-reversed order).
#[inline]
pub fn ntt_in_place(m: &Modulus, a: &mut [u64; D], t: &NttTables) {
    let mut len = D / 2;
    let mut k: usize = 1;
    while len >= 1 {
        let mut start = 0;
        while start < D {
            let zeta = t.fwd[k];
            k += 1;
            let mut j = start;
            while j < start + len {
                let v = m.mul(zeta, a[j + len]);
                let u = a[j];
                a[j] = m.add(u, v);
                a[j + len] = m.sub(u, v);
                j += 1;
            }
            start = j + len;
        }
        len /= 2;
    }
}

/// In-place inverse negacyclic NTT (bit-reversed → natural order). Includes
/// the `D^{-1}` post-scale.
///
/// Levels run from the leaf (len=1) up to the root (len=D/2), i.e. in reverse
/// order of the forward pass. Within each level, k indexes through the same
/// twiddle range the forward used at that level — `[groups, 2·groups)` where
/// `groups = D / (2·len)` — but reading from `inv` (the table of `ψ^{-br(i)}`).
#[inline]
pub fn intt_in_place(m: &Modulus, a: &mut [u64; D], t: &NttTables) {
    let mut len: usize = 1;
    while len < D {
        let groups = D / (2 * len);
        for group_idx in 0..groups {
            let zeta = t.inv[groups + group_idx];
            let start = group_idx * 2 * len;
            for j in start..(start + len) {
                let u = a[j];
                let v = a[j + len];
                a[j] = m.add(u, v);
                a[j + len] = m.mul(zeta, m.sub(u, v));
            }
        }
        len *= 2;
    }
    // Post-scale by D^{-1}.
    let n_inv = t.n_inv;
    for slot in a.iter_mut() {
        *slot = m.mul(*slot, n_inv);
    }
}

/// Negacyclic ring multiplication via forward NTT → pointwise → inverse NTT.
pub fn mul_ntt(m: &Modulus, a: &[u64; D], b: &[u64; D], t: &NttTables) -> [u64; D] {
    let mut ah = *a;
    let mut bh = *b;
    ntt_in_place(m, &mut ah, t);
    ntt_in_place(m, &mut bh, t);
    for i in 0..D {
        ah[i] = m.mul(ah[i], bh[i]);
    }
    intt_in_place(m, &mut ah, t);
    ah
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulus::find_prime_ntt_friendly_below;
    use crate::poly::RingElem;
    use crate::rng::SplitMix64;

    fn ntt_ring(bits: u32) -> (Modulus, NttTables) {
        let q = find_prime_ntt_friendly_below(1u64 << bits, 2 * D as u64);
        let m = Modulus::new(q);
        let psi = m.primitive_root_of_unity(2 * D as u64);
        (m, NttTables::from_psi(&m, psi))
    }

    fn uniform(m: &Modulus, rng: &mut SplitMix64) -> [u64; D] {
        let mut p = [0u64; D];
        for slot in p.iter_mut() {
            *slot = rng.below(m.q);
        }
        p
    }

    #[test]
    fn ntt_round_trip() {
        let (m, t) = ntt_ring(44);
        let mut rng = SplitMix64::new(7);
        for _ in 0..200 {
            let a = uniform(&m, &mut rng);
            let mut ah = a;
            ntt_in_place(&m, &mut ah, &t);
            intt_in_place(&m, &mut ah, &t);
            for i in 0..D {
                assert_eq!(a[i], ah[i], "round trip at i={i}");
            }
        }
    }

    #[test]
    fn ntt_mul_matches_schoolbook() {
        for bits in [39u32, 44, 50] {
            let (m, t) = ntt_ring(bits);
            let mut rng = SplitMix64::new(1000 + bits as u64);
            for _ in 0..200 {
                let a = uniform(&m, &mut rng);
                let b = uniform(&m, &mut rng);
                let c_ntt = mul_ntt(&m, &a, &b, &t);
                let aa = RingElem { c: a };
                let bb = RingElem { c: b };
                let c_sb = aa.mul(&m, &bb);
                assert_eq!(c_ntt, c_sb.c, "ntt mul vs schoolbook at bits={bits}");
            }
        }
    }
}
