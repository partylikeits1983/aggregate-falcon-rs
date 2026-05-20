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
/// `last_msg` is `Some` for the final iteration of an aggregate proof
/// (verified directly by `verify_v2`) and `None` for intermediate iterations
/// (folded into the next iteration's statement — the openings become the next
/// iteration's witness, so they're not sent on the wire). This is what makes
/// the recursive aggregate proof smaller than naive concatenation.
///
/// Decomposition bases and chunk counts come from
/// `Params::for_n(N).iterations[k]`.
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
    /// Last-message openings — present for the final iteration, absent for
    /// intermediates (their openings become the next iteration's witness).
    pub last_msg: Option<IterationLastMsg>,
}

/// Step-5 "last message" of an iteration: the openings that the verifier
/// directly checks. For an intermediate iteration these are encoded into the
/// next iteration's witness via `fold`, so they don't appear on the wire.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationLastMsg {
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

impl IterationProofV2 {
    /// Construct an intermediate-iteration proof (no last message — the
    /// openings will be folded into the next statement's witness).
    pub fn into_intermediate(mut self) -> Self {
        self.last_msg = None;
        self
    }
}

/// A multi-iteration LaBRADOR aggregate proof.
///
/// `intermediate` has `depth − 1` entries (each with `last_msg = None`).
/// `final_iter` is the depth-th iteration (`Stage::SecLast`) and carries the
/// only `last_msg` on the wire. The verifier walks the iterations forward,
/// folding each intermediate's statement before reaching `final_iter`'s
/// statement and running the single concluding `verify_v2`.
///
/// The serde implementation routes through `crate::wire`, which emits a
/// tightly bit-packed encoding (one `q_bitlen`-bit field per ring coefficient,
/// Golomb-Rice for the Gaussian fields `z0/z1/g`, upper-triangle only for
/// `g/h`) instead of the bincode-default `Vec<u64>` layout.
#[derive(Clone, Debug)]
pub struct AggregateProofV2 {
    pub q_prime: u64,
    pub n_sigs: usize,
    pub beta_sq: i128,
    pub intermediate: Vec<IterationProofV2>,
    pub final_iter: IterationProofV2,
}

impl serde::Serialize for AggregateProofV2 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // The example calls `bincode::serialize(&proof)`; bincode encodes
        // `Vec<u8>` as `len_u64 || bytes`. We emit a single Vec<u8> via
        // `serialize_bytes` so we don't pay a `Vec<u8>`-as-tuple cost on top.
        let bytes = crate::wire::pack(self);
        s.serialize_bytes(&bytes)
    }
}

impl<'de> serde::Deserialize<'de> for AggregateProofV2 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = AggregateProofV2;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a tightly bit-packed AggregateProofV2 byte stream")
            }
            fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
                crate::wire::unpack(bytes).map_err(E::custom)
            }
            fn visit_byte_buf<E: serde::de::Error>(self, bytes: Vec<u8>) -> Result<Self::Value, E> {
                crate::wire::unpack(&bytes).map_err(E::custom)
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                // bincode 1's default `deserialize_bytes` dispatches to
                // `visit_seq` for `Vec<u8>`; fall back to assembling the byte
                // buffer manually so we round-trip with `bincode::serialize`.
                let mut buf: Vec<u8> = Vec::new();
                while let Some(b) = seq.next_element::<u8>()? {
                    buf.push(b);
                }
                crate::wire::unpack(&buf).map_err(serde::de::Error::custom)
            }
        }
        d.deserialize_byte_buf(V)
    }
}

/// Per-field byte breakdown of a serialized [`AggregateProofV2`]. Useful for
/// understanding where proof bytes go (encoding work in PERF_PLAN.md "Tier 5"
/// targets specific fields).
///
/// Counts are produced by `bincode::serialized_size(&field)` — they match the
/// bincode wire layout exactly, so the per-field sum is the total proof size
/// up to a few bytes of struct framing.
#[derive(Clone, Debug, Default)]
pub struct ProofBreakdown {
    /// Sum across all intermediate iterations.
    pub intermediate_u1: usize,
    pub intermediate_u2: usize,
    pub intermediate_p: usize,
    pub intermediate_bpp: usize,
    /// Final (deepest) iteration commitments.
    pub final_u1: usize,
    pub final_u2: usize,
    pub final_p: usize,
    pub final_bpp: usize,
    /// Final iteration last-message openings (the dominant wire cost).
    pub final_z0: usize,
    pub final_z1: usize,
    pub final_v: usize,
    pub final_g: usize,
    pub final_h: usize,
}

impl ProofBreakdown {
    pub fn total(&self) -> usize {
        self.intermediate_u1
            + self.intermediate_u2
            + self.intermediate_p
            + self.intermediate_bpp
            + self.final_u1
            + self.final_u2
            + self.final_p
            + self.final_bpp
            + self.final_z0
            + self.final_z1
            + self.final_v
            + self.final_g
            + self.final_h
    }
}

impl AggregateProofV2 {
    /// Per-field byte breakdown of this proof under the actual on-wire
    /// (bit-packed) encoding. Implemented by writing each field through the
    /// same encoder `crate::wire::pack` uses and measuring its output.
    pub fn breakdown(&self) -> ProofBreakdown {
        crate::wire::pack_breakdown(self)
    }
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
