//! End-to-end aggregation and verification of Falcon-512 signatures via
//! LaBRADOR.
//!
//! Public surface:
//! - [`FalconInstance`] — one signature ready to be aggregated.
//! - [`aggregate`] — takes `N` instances, produces a single
//!   [`AggregateProof`].
//! - [`verify`] — takes the `(pk, message)` pairs and the proof, checks the
//!   aggregated proof of "all `N` signatures are valid Falcon signatures of
//!   their respective messages under their respective public keys".
//!
//! The proof's soundness rests on:
//! 1. The Falcon-eq constraints, which pin the witness `y_{·,j}` to the real
//!    `s_{i,1}, s_{i,2}` such that `s_1 + h·s_2 + q·v = c` in `Z_q[X]`.
//! 2. The four-square constraints, which (via the σ₋₁ˢ ties and padding form
//!    constraints) measure `‖s_{i,1}‖² + ‖s_{i,2}‖² + ‖ε_i‖² = β²`, i.e.,
//!    `‖s_i‖² ≤ β²` — Falcon's acceptance bound.
//! 3. The LaBRADOR amortized opening: a single random linear combination of
//!    the witness vectors is enough (with the binding `A · z = Σ c_i · v_i`
//!    check) to soundly attest the witness satisfies the aggregated
//!    constraint.

use falcon_relation::parse::{decode_instance, decode_public_key, hash_to_point, FalconSig, NONCE_LEN};
use falcon_relation::relation::build_falcon_statement;
use labrador::aggregate_v2::{
    prove_aggregate_with_progress, verify_aggregate_with_progress, ProgressSink,
};
use labrador::params::Params;
use labrador::proof::AggregateProofV2;
pub use labrador::proof::ProofBreakdown;
use modring::{Modulus, Ring};
use serde::{Deserialize, Serialize};

const TRANSCRIPT_DOMAIN: &[u8] = b"aggregate-falcon/v1";

/// One Falcon-512 signature ready to aggregate.
#[derive(Clone, Debug)]
pub struct FalconInstance {
    pub public_key: Vec<u8>,
    pub message: Vec<u8>,
    pub signature: Vec<u8>,
}

/// Public-side view of a Falcon instance — what the verifier sees.
pub type PublicPair = (Vec<u8>, Vec<u8>);

/// Progress callback for `aggregate_with_progress`. Each LaBRADOR iteration
/// emits one `iter_start` before its work begins and one `iter_done` after
/// the matching `fold` (or — for the final iteration — after the proof
/// completes). `depth` is the total iteration count (= `Params::for_n(N).depth`).
///
/// The default `()` impl is a no-op, so [`aggregate`] is just
/// `aggregate_with_progress(sigs, &mut ())`.
pub trait Progress {
    fn iter_start(&mut self, k: usize, depth: usize, label: &str);
    fn iter_done(&mut self, k: usize);
}

impl Progress for () {
    fn iter_start(&mut self, _: usize, _: usize, _: &str) {}
    fn iter_done(&mut self, _: usize) {}
}

/// Adapter that forwards `aggregate_falcon::Progress` events through the
/// `labrador::aggregate_v2::ProgressSink` trait so the prover can stay
/// agnostic of this crate.
struct SinkAdapter<'a, P: Progress + ?Sized>(&'a mut P);

impl<P: Progress + ?Sized> ProgressSink for SinkAdapter<'_, P> {
    fn iter_start(&mut self, k: usize, depth: usize, label: &str) {
        self.0.iter_start(k, depth, label);
    }
    fn iter_done(&mut self, k: usize) {
        self.0.iter_done(k);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregateProof {
    /// Per-signature nonces (needed by the verifier to reconstruct `c`).
    /// Each inner `Vec` is exactly `NONCE_LEN = 40` bytes.
    pub nonces: Vec<Vec<u8>>,
    /// Witness-side norm bound used at prove time (lifted to the verifier
    /// so it can rebuild the same statement deterministically).
    pub beta_sq: i128,
    /// The recursive LaBRADOR proof — the multi-iteration v2 path. Sessions
    /// C/D land the intermediate-iteration stripping that makes this shrink
    /// below `N · sizeof(sig)`.
    pub labrador: AggregateProofV2,
}

impl AggregateProof {
    /// Per-field byte breakdown of the inner labrador proof. Useful for the
    /// roundtrip example and for tracking where proof bytes go as encoding
    /// work lands. Does not include the outer-struct framing (nonces,
    /// beta_sq) which are small and constant.
    pub fn breakdown(&self) -> ProofBreakdown {
        self.labrador.breakdown()
    }
}

#[derive(Debug, Clone)]
pub enum AggregateError {
    EmptyInput,
    DecodeFailed(String),
}

impl std::fmt::Display for AggregateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "no signatures provided"),
            Self::DecodeFailed(msg) => write!(f, "Falcon decode failed: {msg}"),
        }
    }
}

impl std::error::Error for AggregateError {}

#[derive(Debug, Clone)]
pub enum VerifyError {
    SigCountMismatch { got: usize, want: usize },
    DecodeFailed(String),
    Labrador(labrador::proof::VerifyError),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SigCountMismatch { got, want } => {
                write!(f, "sig-count mismatch: pairs={got}, proof={want}")
            }
            Self::DecodeFailed(msg) => write!(f, "Falcon pk decode failed: {msg}"),
            Self::Labrador(e) => write!(f, "labrador verification failed: {e}"),
        }
    }
}

impl std::error::Error for VerifyError {}

pub fn aggregate(sigs: &[FalconInstance]) -> Result<AggregateProof, AggregateError> {
    aggregate_with_progress(sigs, &mut ())
}

/// Aggregate, reporting progress to `progress`. See [`Progress`] for the
/// callback shape. Use this from CLI/example code to drive a progress bar.
pub fn aggregate_with_progress<P: Progress>(
    sigs: &[FalconInstance],
    progress: &mut P,
) -> Result<AggregateProof, AggregateError> {
    if sigs.is_empty() {
        return Err(AggregateError::EmptyInput);
    }

    let params = Params::for_n(sigs.len());
    let q_prime = params.select_modulus();
    let ring = Ring::new(Modulus::new(q_prime));

    // Decode every signature into FalconSig. The result includes (h, c,
    // s1, s2, nonce); the secret s1, s2 are used by the prover's witness
    // build, then discarded — never serialised.
    let mut parsed: Vec<FalconSig> = Vec::with_capacity(sigs.len());
    for s in sigs {
        let p = decode_instance(&s.public_key, &s.signature, &s.message)
            .map_err(|e| AggregateError::DecodeFailed(format!("{e:?}")))?;
        parsed.push(p);
    }

    let beta_sq = params.beta_init_sq();
    let (statement, witness, _layout) = build_falcon_statement(&parsed, &ring, beta_sq);

    let mut adapter = SinkAdapter(progress);
    let labrador =
        prove_aggregate_with_progress(&statement, &witness, &params, TRANSCRIPT_DOMAIN, &mut adapter);
    let nonces: Vec<Vec<u8>> = parsed.iter().map(|p| p.nonce.to_vec()).collect();

    Ok(AggregateProof {
        nonces,
        beta_sq,
        labrador,
    })
}

/// Return the number of LaBRADOR iterations that `aggregate` of `n_sigs`
/// signatures will run. Useful for sizing a `ProgressBar` before kicking off
/// the prover.
pub fn aggregate_depth(n_sigs: usize) -> usize {
    Params::for_n(n_sigs).depth
}

pub fn verify(pairs: &[PublicPair], proof: &AggregateProof) -> Result<(), VerifyError> {
    verify_with_progress(pairs, proof, &mut ())
}

/// Verify, reporting per-iteration progress to `progress`. See [`Progress`] for
/// the callback shape. Use this from CLI/example code to drive a progress bar
/// during the LaBRADOR verifier's depth-K fold loop and final `verify_v2`.
pub fn verify_with_progress<P: Progress>(
    pairs: &[PublicPair],
    proof: &AggregateProof,
    progress: &mut P,
) -> Result<(), VerifyError> {
    if pairs.len() != proof.nonces.len() {
        return Err(VerifyError::SigCountMismatch {
            got: pairs.len(),
            want: proof.nonces.len(),
        });
    }

    let params = Params::for_n(pairs.len());
    let q_prime = params.select_modulus();
    if q_prime != proof.labrador.q_prime {
        return Err(VerifyError::Labrador(labrador::proof::VerifyError::ModulusMismatch {
            got: proof.labrador.q_prime,
            want: q_prime,
        }));
    }
    let ring = Ring::new(Modulus::new(q_prime));

    let stmt = build_verifier_statement(pairs, &proof.nonces, proof.beta_sq, &ring)?;
    let mut adapter = SinkAdapter(progress);
    verify_aggregate_with_progress(&stmt, &proof.labrador, &params, TRANSCRIPT_DOMAIN, &mut adapter)
        .map_err(VerifyError::Labrador)
}

/// Build the verifier-side LaBRADOR `Statement` from public-only inputs
/// `(pk, msg, nonce)`. The witness components `s1, s2` of every reconstructed
/// `FalconSig` are zeroed because the constraint builders only read `h` and
/// `c` from each sig (Falcon-eq uses `h, c`; four-square and form
/// constraints use neither). The
/// `crates/aggregate-falcon/tests/roundtrip.rs::prover_and_verifier_statements_match`
/// test pins this invariant — if a future constraint builder starts reading
/// `s1` or `s2`, that test will fail loudly rather than silently producing a
/// different statement on the verifier side.
pub fn build_verifier_statement(
    pairs: &[PublicPair],
    nonces: &[Vec<u8>],
    beta_sq: i128,
    ring: &Ring,
) -> Result<labrador::statement::Statement, VerifyError> {
    use falcon_relation::relation::{
        add_falcon_eq_constraints, add_form_constraints_e, add_form_constraints_ep,
        add_form_constraints_y_padding, add_form_constraints_yp, add_four_square_constraints,
        WitnessLayout,
    };
    use labrador::statement::Statement;

    assert_eq!(pairs.len(), nonces.len(), "pairs/nonces length mismatch");
    let mut sigs_view: Vec<FalconSig> = Vec::with_capacity(pairs.len());
    for ((pk_bytes, message), nonce_vec) in pairs.iter().zip(nonces.iter()) {
        if nonce_vec.len() != NONCE_LEN {
            return Err(VerifyError::DecodeFailed(format!(
                "nonce length {} != {NONCE_LEN}",
                nonce_vec.len()
            )));
        }
        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(nonce_vec);
        let h = decode_public_key(pk_bytes).map_err(|e| VerifyError::DecodeFailed(format!("{e:?}")))?;
        let c = hash_to_point(&nonce, message);
        sigs_view.push(FalconSig {
            h,
            c,
            s1: falcon_relation::falcon_ring::FPoly::zero(),
            s2: falcon_relation::falcon_ring::FPoly::zero(),
            nonce,
        });
    }

    let layout = WitnessLayout::new(sigs_view.len());
    let mut stmt = Statement {
        ring: *ring,
        n: layout.n_s(),
        r: layout.r(),
        full: vec![],
        const_term: vec![],
        beta_sq,
    };
    add_falcon_eq_constraints(&mut stmt, &layout, &sigs_view, ring);
    add_four_square_constraints(
        &mut stmt,
        &layout,
        falcon_relation::falcon_ring::FALCON_BETA_SQ,
        ring,
    );
    add_form_constraints_y_padding(&mut stmt, &layout, ring);
    add_form_constraints_yp(&mut stmt, &layout, ring);
    add_form_constraints_e(&mut stmt, &layout, ring);
    add_form_constraints_ep(&mut stmt, &layout, ring);
    Ok(stmt)
}
