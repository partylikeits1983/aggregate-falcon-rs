//! `modring` — base field and polynomial ring arithmetic for the LaBRADOR
//! proof system used to aggregate Falcon-512 signatures.
//!
//! The LaBRADOR ring is `R_Q = Z_Q[X] / (X^64 + 1)` for a dynamically chosen
//! prime `Q ≡ 5 (mod 8)`. This crate is pure algebra: it has no knowledge of
//! the proof protocol, Falcon, commitments, or challenges.
//!
//! Layers:
//! - [`modulus`] — the prime field `F_Q` and prime/`sqrt` utilities.
//! - [`poly`] — ring elements with reference (schoolbook) multiplication.
//! - [`crt`] — the two-splitting CRT decomposition and fast multiplication.
//! - [`rng`] — a deterministic PRNG for tests only.

pub mod crt;
pub mod modulus;
pub mod ntt;
pub mod poly;
pub mod rng;

pub use crt::{CrtRepr, Ring, H};
pub use modulus::{
    find_prime_5mod8, find_prime_5mod8_below, find_prime_ntt_friendly_below, is_prime, Modulus,
};
pub use poly::{RingElem, D};
