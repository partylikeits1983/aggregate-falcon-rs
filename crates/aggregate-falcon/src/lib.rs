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
use labrador::proof::IterationProof;
use labrador::prover::prove;
use labrador::transcript::Transcript;
use labrador::verifier::verify as verify_iteration;
use modring::{find_prime_5mod8, Modulus, Ring};
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregateProof {
    /// LaBRADOR modulus `q'`, lifted from Falcon's `q = 12289`.
    pub q_prime: u64,
    /// Per-signature nonces (needed by the verifier to reconstruct `c`).
    /// Each inner `Vec` is exactly `NONCE_LEN = 40` bytes.
    pub nonces: Vec<Vec<u8>>,
    /// Witness-side norm bound used at prove time (lifted to the verifier
    /// so it can rebuild the same statement deterministically).
    pub beta_sq: i128,
    /// The (single) iteration's prover messages.
    pub iteration: IterationProof,
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

/// Aggregate-time LaBRADOR modulus selection. Use a 44-bit prime ≡ 5 (mod 8)
/// — comfortably above the wrap-around bounds for `N ≤ 100` per the Phase 0
/// estimator. For larger `N`, the bit-length should scale with the witness
/// norm bound; v1 hard-codes 44 bits for simplicity.
fn select_qprime() -> u64 {
    find_prime_5mod8(1 << 44)
}

fn select_beta_sq() -> i128 {
    // Generous bound — the form constraints already pin the structure of the
    // witness; this just ensures `z` after amortization fits.
    1i128 << 60
}

pub fn aggregate(sigs: &[FalconInstance]) -> Result<AggregateProof, AggregateError> {
    if sigs.is_empty() {
        return Err(AggregateError::EmptyInput);
    }

    let q_prime = select_qprime();
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

    let beta_sq = select_beta_sq();
    let (statement, witness, _layout) = build_falcon_statement(&parsed, &ring, beta_sq);

    let mut transcript = Transcript::new(TRANSCRIPT_DOMAIN);
    let iteration = prove(&statement, &witness, &mut transcript);

    let nonces: Vec<Vec<u8>> = parsed.iter().map(|p| p.nonce.to_vec()).collect();

    Ok(AggregateProof {
        q_prime,
        nonces,
        beta_sq,
        iteration,
    })
}

pub fn verify(pairs: &[PublicPair], proof: &AggregateProof) -> Result<(), VerifyError> {
    if pairs.len() != proof.nonces.len() {
        return Err(VerifyError::SigCountMismatch {
            got: pairs.len(),
            want: proof.nonces.len(),
        });
    }

    let ring = Ring::new(Modulus::new(proof.q_prime));

    // Reconstruct "public" FalconSig views (h from pk; c from nonce + msg).
    // s1, s2 are set to zero — they're never read by the constraint builders
    // (only Falcon-eq uses h and c; four-square + form constraints use
    // neither).
    let mut sigs_view: Vec<FalconSig> = Vec::with_capacity(pairs.len());
    for ((pk_bytes, message), nonce_vec) in pairs.iter().zip(proof.nonces.iter()) {
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

    // The statement-builder also uses `build_honest_witness` to compute a
    // throwaway witness — we don't use the returned witness on the verifier
    // side. But `build_honest_witness` calls `compute_v_signed` which uses
    // s1, s2, h, c — with our zero s1, s2, the integer check
    // `(c - 0 - h·0) % q == 0` would fail. So we need a statement builder
    // that DOESN'T require a witness. Let's call only the constraint
    // builders directly.

    use falcon_relation::relation::{
        add_falcon_eq_constraints, add_form_constraints_e, add_form_constraints_ep,
        add_form_constraints_y_padding, add_form_constraints_yp, add_four_square_constraints,
        WitnessLayout,
    };
    use labrador::statement::Statement;

    let layout = WitnessLayout::new(sigs_view.len());
    let mut stmt = Statement {
        ring,
        n: layout.n_s(),
        r: layout.r(),
        full: vec![],
        const_term: vec![],
        beta_sq: proof.beta_sq,
    };
    add_falcon_eq_constraints(&mut stmt, &layout, &sigs_view, &ring);
    add_four_square_constraints(
        &mut stmt,
        &layout,
        falcon_relation::falcon_ring::FALCON_BETA_SQ,
        &ring,
    );
    add_form_constraints_y_padding(&mut stmt, &layout, &ring);
    add_form_constraints_yp(&mut stmt, &layout, &ring);
    add_form_constraints_e(&mut stmt, &layout, &ring);
    add_form_constraints_ep(&mut stmt, &layout, &ring);

    let mut transcript = Transcript::new(TRANSCRIPT_DOMAIN);
    verify_iteration(&stmt, &proof.iteration, &mut transcript).map_err(VerifyError::Labrador)?;

    Ok(())
}
