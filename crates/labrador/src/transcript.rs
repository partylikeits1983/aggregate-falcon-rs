//! Fiat-Shamir transcript backing the LaBRADOR Protocols 2 & 3.
//!
//! # Soundness invariant
//!
//! Every prover message must be *absorbed* into the transcript before any
//! verifier challenge derived from it is *squeezed* out. The transcript order
//! below is the canonical contract; if either side deviates, soundness fails
//! silently — round-trip tests do not catch ordering bugs.
//!
//! Order per LaBRADOR iteration (Protocols 2 & 3 of the paper):
//!
//! 1. **Setup absorbs.** `iter_index, n, r, q_'`, and the verifier statement
//!    digest (`a_{i,j}^{(k)}`, `φ^{(k)}`, `b^{(k)}`, `a'^{(l)}`, `φ'^{(l)}`,
//!    `b'_0^{(l)}` for all `k, l`). The "public-input digest" is one absorb
//!    of a SHAKE256 hash of all those values.
//! 2. **Commit (prover).** Absorb the outer commitment `u_1`.
//! 3. **Squeeze projection matrices** `Π_i` (`i ∈ [r]`).
//! 4. **Commit (prover).** Absorb the projection vector `p`.
//! 5. **Squeeze aggregation scalars** `ψ^{(k)} ∈ Z_{q'}^{|F'|}`,
//!    `ω^{(k)} ∈ Z_{q'}^{2λ}` for each `k ∈ [K'']`.
//! 6. **Commit (prover).** Absorb the aggregated full polynomials `b''^{(k)}`
//!    for `k ∈ [K'']`.
//! 7. **Squeeze full-aggregation scalars** `α ∈ R^K`, `β ∈ R^{K''}`.
//! 8. **Commit (prover).** Absorb the second outer commitment `u_2`.
//! 9. **Squeeze amortizing challenges** `c_i ∈ C` for `i ∈ [r]` (each via
//!    rejection sampling — see `challenge.rs`).
//!
//! For recursion, the next iteration's transcript is *forked* from this one
//! after the last message of the current iteration is absorbed — that's done
//! by simply continuing to use the same `Transcript` instance.

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

/// Fiat-Shamir transcript. The internal state is a SHAKE256 absorb/squeeze
/// machine: every `absorb` re-initialises the squeeze reader (so we can't
/// accidentally read stale challenges), and every `challenge_bytes` updates
/// the internal hash with a domain-separator label so subsequent challenges
/// remain bound to the call site.
#[derive(Clone)]
pub struct Transcript {
    hasher: Shake256,
}

impl Transcript {
    /// A fresh transcript tagged with `domain` (e.g., `b"labrador-falcon-512-v1"`).
    pub fn new(domain: &[u8]) -> Self {
        let mut hasher = Shake256::default();
        hasher.update(b"transcript-domain:");
        hasher.update(&(domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        Self { hasher }
    }

    /// Absorb `bytes` under `label`. The label disambiguates call sites so a
    /// reordering of absorbs by the verifier produces a different challenge
    /// stream than the prover used.
    pub fn absorb(&mut self, label: &[u8], bytes: &[u8]) {
        self.hasher.update(b"a:");
        self.hasher.update(&(label.len() as u64).to_le_bytes());
        self.hasher.update(label);
        self.hasher.update(&(bytes.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
    }

    /// Squeeze `n` deterministic bytes for challenge `label`. After this call
    /// the squeeze is committed into the hash state, so subsequent absorbs
    /// still chain correctly.
    pub fn challenge_bytes(&mut self, label: &[u8], n: usize) -> Vec<u8> {
        // Bind the squeeze to this label, then finalise the hasher into a
        // fresh reader of `n` bytes, and feed the result back into the hash
        // so future challenges chain (otherwise reading would not advance).
        self.hasher.update(b"c:");
        self.hasher.update(&(label.len() as u64).to_le_bytes());
        self.hasher.update(label);
        self.hasher.update(&(n as u64).to_le_bytes());

        let snapshot = self.hasher.clone();
        let mut reader = snapshot.finalize_xof();
        let mut buf = vec![0u8; n];
        reader.read(&mut buf);

        self.hasher.update(b"chal-out:");
        self.hasher.update(&buf);
        buf
    }

    /// Derive a `u64` challenge.
    pub fn derive_u64(&mut self, label: &[u8]) -> u64 {
        let b = self.challenge_bytes(label, 8);
        u64::from_le_bytes(b.try_into().unwrap())
    }

    /// Derive a uniform `u64` in `[0, bound)` by rejection sampling.
    pub fn derive_below(&mut self, label: &[u8], bound: u64) -> u64 {
        assert!(bound > 0, "bound must be positive");
        if bound == 1 {
            return 0;
        }
        // Largest multiple of `bound` that fits in u64.
        let cap = u64::MAX - (u64::MAX % bound);
        let mut counter: u32 = 0;
        loop {
            let bytes = self.challenge_bytes(label, 8);
            let v = u64::from_le_bytes(bytes.try_into().unwrap());
            counter = counter.wrapping_add(1);
            if v < cap {
                return v % bound;
            }
            if counter > 1000 {
                panic!("transcript derive_below stuck (bound = {bound})");
            }
        }
    }

    /// Derive a uniform field element `< q'`.
    pub fn derive_field(&mut self, label: &[u8], q: u64) -> u64 {
        self.derive_below(label, q)
    }

    /// Derive `n` uniform field elements `< q` from a *single* SHAKE squeeze.
    ///
    /// Equivalent in distribution to calling `derive_field(label_i, q)` `n`
    /// times with disjoint labels, but ~`n`× fewer SHAKE permutations. Used by
    /// `sample_ring_element` (D=64 coefficients) and similar bulk samplers
    /// that previously dominated wall-clock through per-coefficient absorbs.
    pub fn derive_field_array(&mut self, label: &[u8], n: usize, q: u64) -> Vec<u64> {
        assert!(q > 1, "q must be > 1");
        // Bind the label + (n, q) into the transcript so the squeeze is unique.
        self.hasher.update(b"cm:");
        self.hasher.update(&(label.len() as u64).to_le_bytes());
        self.hasher.update(label);
        self.hasher.update(&(n as u64).to_le_bytes());
        self.hasher.update(&q.to_le_bytes());

        let snapshot = self.hasher.clone();
        let mut reader = snapshot.finalize_xof();
        let cap = u64::MAX - (u64::MAX % q);
        let mut out = Vec::with_capacity(n);
        let mut buf = [0u8; 8];
        // Stream 8-byte chunks; rejection sampling drops any v ≥ cap.
        // For practical q (random prime near 2^k), rejection rate is small.
        let mut guard: u32 = 0;
        while out.len() < n {
            reader.read(&mut buf);
            let v = u64::from_le_bytes(buf);
            if v < cap {
                out.push(v % q);
            }
            guard = guard.wrapping_add(1);
            if guard > 1_000_000 + (n as u32 * 4) {
                panic!("derive_field_array rejection-sampling exceeded guard");
            }
        }

        // Bind the squeezed `n` count back so subsequent challenges chain.
        // We don't need to absorb the values themselves — the label already
        // disambiguates this call site from any other, and (n, q) above already
        // committed to the shape. Absorb a sentinel marking the call's end.
        self.hasher.update(b"chal-array-out:");
        self.hasher.update(&(n as u64).to_le_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_inputs_same_outputs() {
        let mut a = Transcript::new(b"test");
        let mut b = Transcript::new(b"test");
        a.absorb(b"x", &[1, 2, 3]);
        b.absorb(b"x", &[1, 2, 3]);
        assert_eq!(a.challenge_bytes(b"c1", 32), b.challenge_bytes(b"c1", 32));
        a.absorb(b"y", &[7]);
        b.absorb(b"y", &[7]);
        assert_eq!(a.derive_u64(b"u"), b.derive_u64(b"u"));
    }

    #[test]
    fn different_labels_diverge() {
        let mut a = Transcript::new(b"test");
        let mut b = Transcript::new(b"test");
        a.absorb(b"x", &[1]);
        b.absorb(b"y", &[1]);
        assert_ne!(a.challenge_bytes(b"c", 16), b.challenge_bytes(b"c", 16));
    }

    #[test]
    fn derive_below_is_uniform_enough() {
        let mut t = Transcript::new(b"u");
        let n = 10_000;
        let mut counts = [0u32; 7];
        for _ in 0..n {
            counts[t.derive_below(b"x", 7) as usize] += 1;
        }
        let expect = n / 7;
        for c in counts {
            let dev = (c as i64 - expect as i64).abs();
            assert!(dev < (expect / 2) as i64, "skew: {counts:?}");
        }
    }
}
