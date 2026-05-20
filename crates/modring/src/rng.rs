//! A tiny deterministic PRNG used for test vectors and randomized property
//! checks. This is **not** cryptographically secure and must never be used
//! for protocol randomness — Fiat–Shamir challenges are SHAKE-derived.

/// SplitMix64 — a fast, well-distributed deterministic generator.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Next 64-bit output.
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `[0, bound)`.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}
