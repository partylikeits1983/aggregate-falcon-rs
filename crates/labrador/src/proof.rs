//! Single-iteration LaBRADOR proof messages.
//!
//! The structure follows Protocols 2 & 3 (paper pp. 39-40) but with the
//! engineering simplifications listed in `prover.rs` and `verifier.rs`:
//! commitments are transcript-absorbed rather than outer-committed, and the
//! v1 implementation runs a single iteration (no recursive folding).

use modring::RingElem;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationProof {
    /// Inner commitments `v_i = A · w_i` for each witness vector.
    /// Outer shape: `[r][kappa]` ring elements.
    pub v: Vec<Vec<RingElem>>,
    /// Aggregated full polynomials `b''^{(k)}` for `k ∈ [K'']` — each
    /// `RingElem` whose constant term is the aggregated `b''_0^{(k)}`.
    pub b_double_prime: Vec<RingElem>,
    /// Amortized opening `z = Σ c_i w_i ∈ S^n`.
    pub z: Vec<RingElem>,
    /// Quadratic garbage `g_{ij} = ⟨w_i, w_j⟩` (upper triangle `i ≤ j`).
    /// Stored as a flat `[r][r]` matrix; lower triangle filled by symmetry.
    pub g: Vec<Vec<RingElem>>,
    /// Linear garbage `h_{ij} = (⟨φ_i^{agg}, w_j⟩ + ⟨φ_j^{agg}, w_i⟩)/2`
    /// (upper triangle).
    pub h: Vec<Vec<RingElem>>,
}

/// v2 iteration proof: paper-correct single-iteration shape from Protocols
/// 2 & 3. The wire contents are
/// `(u_1, p, b'', u_2, z^{(0)}, z^{(1)}, v, g, h)` where `v, g, h` are sent in
/// full and the verifier decomposes them on its side to check `u_1, u_2`
/// (deterministic centered-base-b decomposition).
///
/// Decomposition bases and chunk counts come from
/// `Params::for_n(N).iterations[0]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationProofV2 {
    /// Outer commitment to inner-commitment + g chunks. Length `κ₁`.
    pub u1: Vec<RingElem>,
    /// JL projection vector `p = Σ Π_i τ(w_i) ∈ Z^{2λ}`.
    pub p: Vec<i128>,
    /// Constant-term aggregation polynomials `b''^{(k)}`, length `K''`.
    pub b_double_prime: Vec<RingElem>,
    /// Outer commitment to `h` chunks. Length `κ₁`.
    pub u2: Vec<RingElem>,
    /// Amortized opening, decomposed as `z = z^{(0)} + b · z^{(1)}`. Length `n`.
    pub z0: Vec<RingElem>,
    pub z1: Vec<RingElem>,
    /// Inner commitments `v_i ∈ R^κ`: outer `[r]`, inner `[κ]`.
    pub v: Vec<Vec<RingElem>>,
    /// Quadratic garbage `g_{ij}` for `i ≤ j` — upper-triangular `[r][r]`
    /// with entries `g[i][j]` populated only for `i ≤ j` (lower triangle
    /// kept as `RingElem::zero()`).
    pub g: Vec<Vec<RingElem>>,
    /// Linear garbage `h_{ij}` for `i ≤ j`, same upper-tri convention as `g`.
    pub h: Vec<Vec<RingElem>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregateProof {
    /// LaBRADOR modulus `q'`, recorded so the verifier can rebuild the ring
    /// independently and detect tampering.
    pub q_prime: u64,
    /// Number of aggregated Falcon-512 signatures.
    pub n_sigs: usize,
    /// Per-iteration messages. v1 has exactly one entry.
    pub iterations: Vec<IterationProof>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    ModulusMismatch { got: u64, want: u64 },
    SigCountMismatch { got: usize, want: usize },
    StatementRankMismatch,
    ConstTermMismatch(usize),
    InnerCommitmentOpening,
    QuadraticOpening,
    LinearOpening,
    AggregatedRelation,
    NormBound,
    OuterCommitment(&'static str),
    JlBound,
    BadIterationStage,
    ProofShape(&'static str),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModulusMismatch { got, want } => write!(f, "modulus mismatch: proof q' = {got}, expected {want}"),
            Self::SigCountMismatch { got, want } => write!(f, "sig-count mismatch: proof = {got}, expected {want}"),
            Self::StatementRankMismatch => write!(f, "statement-rank mismatch"),
            Self::ConstTermMismatch(k) => write!(f, "constant-term aggregation mismatch at k = {k}"),
            Self::InnerCommitmentOpening => write!(f, "inner-commitment opening A·z ≠ Σ c_i v_i"),
            Self::QuadraticOpening => write!(f, "quadratic opening ⟨z,z⟩ ≠ Σ c_i c_j g_ij"),
            Self::LinearOpening => write!(f, "linear opening Σ ⟨φ_i, z⟩ c_i ≠ Σ c_i c_j h_ij"),
            Self::AggregatedRelation => write!(f, "aggregated relation Σ a g + Σ h_ii ≠ b"),
            Self::NormBound => write!(f, "witness norm bound exceeded"),
            Self::OuterCommitment(label) => write!(f, "outer-commitment opening mismatch ({label})"),
            Self::JlBound => write!(f, "JL projection norm exceeds bound"),
            Self::BadIterationStage => write!(f, "iteration stage chain is malformed"),
            Self::ProofShape(s) => write!(f, "proof shape invalid: {s}"),
        }
    }
}

impl std::error::Error for VerifyError {}
