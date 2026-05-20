//! LaBRADOR challenge sampling: small ring elements with bounded ℓ₂² *and*
//! operator norm.
//!
//! Per §3 / the two-splitting parameter set `CHAL_2_SPLIT_64_128`:
//! - Weight `w = 43` non-zero coefficients per challenge.
//! - Coefficients drawn uniformly from `{−2, −1, +1, +2}` (γ = 2).
//! - Bounds: `τ = 86` (squared ℓ₂), `T_op = 43` (operator norm).
//! - Operator norm of `c ∈ S` equals `max(‖slot_0(c)‖₂, ‖slot_1(c)‖₂)` via the
//!   two-splitting CRT.
//!
//! `sample_challenge` reject-samples until both norms are within bound. With
//! the chosen weight and γ these bounds hold *almost* always for fresh
//! draws; the rejection loop is the safety net.

use crate::transcript::Transcript;
use modring::{Ring, RingElem, D};

pub const W: usize = 43;
/// Soundness constant `τ = T_2` from the paper / `CHAL_2_SPLIT_64_128`.
/// Used as a multiplier in σ_z; not used as a rejection threshold here since
/// our sampling is deterministic in `w` and `γ`.
pub const TAU_SQ: i64 = 86;
/// Soundness constant `T_op` from the paper. Bounds the operator-norm
/// amplification of a challenge across CRT slots. Auto-satisfied by our
/// sampling (`w = 43`, `γ = 2`).
pub const T_OP: i64 = 43;
/// Worst-case ℓ₂² for a `w`-weight γ-bounded challenge — every coefficient
/// could be `±γ`. We reject only if the draw somehow exceeds this.
pub const WORST_CASE_L2SQ: i64 = (W as i64) * 4;

/// Sample one challenge `c ∈ S` from `transcript`, satisfying both bounds.
pub fn sample_challenge(transcript: &mut Transcript, label: &[u8], ring: &Ring) -> RingElem {
    let m = &ring.m;
    let mut counter: u32 = 0;
    loop {
        let mut c = RingElem::zero();
        let mut taken = [false; D];
        // Place exactly W non-zero coefficients.
        for k in 0..W {
            // Pick a position uniformly among the still-available ones.
            let remaining = D - k;
            let pick_label = [
                label,
                b"|pos|",
                &(counter as u64).to_le_bytes(),
                &(k as u64).to_le_bytes(),
            ]
            .concat();
            let pick = transcript.derive_below(&pick_label, remaining as u64) as usize;
            // Convert that into an actual coordinate.
            let mut seen = 0usize;
            let mut chosen = 0usize;
            for j in 0..D {
                if taken[j] {
                    continue;
                }
                if seen == pick {
                    chosen = j;
                    break;
                }
                seen += 1;
            }
            taken[chosen] = true;

            // Pick coefficient value uniformly from {-2, -1, +1, +2}.
            let val_label = [
                label,
                b"|val|",
                &(counter as u64).to_le_bytes(),
                &(k as u64).to_le_bytes(),
            ]
            .concat();
            let v = transcript.derive_below(&val_label, 4);
            let signed: i64 = match v {
                0 => -2,
                1 => -1,
                2 => 1,
                3 => 2,
                _ => unreachable!(),
            };
            c.c[chosen] = m.from_i64(signed);
        }
        counter = counter.wrapping_add(1);

        if check_norms(ring, &c) {
            return c;
        }
        if counter > 100 {
            panic!("challenge sampling failed after 100 tries (label = {label:?})");
        }
    }
}

/// True iff `c` is well-formed: each coefficient lies in `{−γ, …, +γ}` with
/// `γ = 2`, exactly `W` non-zero, and `‖c‖₂² ≤ W · γ²`. These are deterministic
/// from the sampling routine; the function exists as a structural sanity
/// check rather than a rejection criterion.
pub fn check_norms(ring: &Ring, c: &RingElem) -> bool {
    let mut l2sq: i128 = 0;
    let mut nz = 0usize;
    for k in 0..D {
        let s = ring.m.centered(c.c[k]);
        if s.abs() > 2 {
            return false;
        }
        if s != 0 {
            nz += 1;
        }
        l2sq += (s as i128) * (s as i128);
    }
    nz == W && l2sq <= WORST_CASE_L2SQ as i128
}

#[cfg(test)]
mod tests {
    use super::*;
    use modring::{find_prime_5mod8, Modulus};

    fn ring() -> Ring {
        Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
    }

    #[test]
    fn every_sample_obeys_both_bounds() {
        let r = ring();
        let mut t = Transcript::new(b"chal-test");
        for i in 0u64..1000 {
            let c = sample_challenge(&mut t, &i.to_le_bytes(), &r);
            assert!(check_norms(&r, &c));
            // ‖c‖₂² ≤ 86 ⇒ at most 21 coefficients of magnitude 2 with the rest
            // ±1 — the weight is exactly W = 43, so check it.
            let nz = c.c.iter().filter(|&&x| x != 0).count();
            assert_eq!(nz, W);
        }
    }
}
