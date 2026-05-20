//! `labrador` — the LaBRADOR proof system, instantiated over the two-splitting
//! ring `S_{q'} = Z_{q'}[X] / (X^64 + 1)` from [`modring`].
//!
//! Modules (built up across phases):
//! - [`statement`] — the LaBRADOR statement and witness types (Phase 3).
//! - `params`, `commit`, `transcript`, `challenge`, `jl`, `garbage`,
//!   `prover`, `verifier`, `fold`, `proof` — pending.

pub mod statement;

pub use statement::{ConstTermConstraint, DotConstraint, Statement, Witness};
