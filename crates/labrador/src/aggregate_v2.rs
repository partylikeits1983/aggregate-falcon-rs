//! Multi-iteration recursive prover and verifier driving the v2 single-
//! iteration prover/verifier through Sessions C and D of Phase 6.
//!
//! For an initial statement at depth 0, the prover runs:
//! ```text
//!   prove_v2 → fold → prove_v2 → fold → ... → prove_v2 (SecLast)
//! ```
//! Each intermediate iteration's last-message openings (`z, v, g, h`) become
//! the next iteration's witness via [`fold::fold`]; the intermediate proof on
//! the wire only carries the commitments `(u_1, p, b'', u_2)`.
//!
//! The verifier mirror walks the iterations forward via
//! [`fold::fold_statement`] (no openings required) and runs `verify_v2` once
//! on the deepest folded statement against the final iteration's proof.
//!
//! Soundness follows from the recursive translation: the fold's emitted
//! constraints are exactly what `verify_v2` would have checked on the
//! intermediate iteration. Accepting the deepest `verify_v2` implies all
//! intermediate checks pass.

use crate::fold::{fold, fold_statement, FoldOutput};
use crate::params::Params;
use crate::proof::{AggregateProofV2, IterationProofV2, VerifyError};
use crate::prover_v2::prove_v2;
use crate::statement::{Statement, Witness};
use crate::transcript::Transcript;
use crate::verifier_v2::verify_v2;

/// Run the multi-iteration recursive prover.
pub fn prove_aggregate(
    stmt: &Statement,
    witness: &Witness,
    params: &Params,
    transcript_seed: &[u8],
) -> AggregateProofV2 {
    let depth = params.depth;
    assert!(depth >= 1, "depth must be ≥ 1");

    let mut cur_stmt = stmt.clone();
    let mut cur_witness = witness.clone();
    let mut t_prove = Transcript::new(transcript_seed);
    let mut intermediate: Vec<IterationProofV2> = Vec::with_capacity(depth - 1);

    for k in 0..(depth - 1) {
        // Snapshot the transcript BEFORE prove_v2; fold needs the same state.
        let mut t_fold = t_prove.clone();
        let proof_k = prove_v2(&cur_stmt, &cur_witness, &params.iterations[k], &mut t_prove);

        // Fold using params.iterations[k+1].(prev_nu, prev_mu) — those are the
        // folding parameters that produced iter[k+1]'s rank.
        let nu = params.iterations[k + 1].prev_nu as usize;
        let mu = params.iterations[k + 1].prev_mu as usize;
        let FoldOutput { statement, witness } =
            fold(&cur_stmt, &proof_k, &params.iterations[k], nu, mu, &mut t_fold);

        intermediate.push(proof_k.into_intermediate());
        cur_stmt = statement;
        cur_witness = witness;
    }

    let final_iter = prove_v2(
        &cur_stmt,
        &cur_witness,
        &params.iterations[depth - 1],
        &mut t_prove,
    );

    AggregateProofV2 {
        q_prime: stmt.ring.m.q,
        n_sigs: params.num_sigs,
        beta_sq: stmt.beta_sq,
        intermediate,
        final_iter,
    }
}

/// Run the multi-iteration recursive verifier.
pub fn verify_aggregate(
    stmt: &Statement,
    proof: &AggregateProofV2,
    params: &Params,
    transcript_seed: &[u8],
) -> Result<(), VerifyError> {
    let depth = params.depth;
    if proof.intermediate.len() != depth.saturating_sub(1) {
        return Err(VerifyError::BadIterationStage);
    }

    if stmt.ring.m.q != proof.q_prime {
        return Err(VerifyError::ModulusMismatch { got: proof.q_prime, want: stmt.ring.m.q });
    }

    let mut cur_stmt = stmt.clone();
    let mut t = Transcript::new(transcript_seed);

    for (k, inter_proof) in proof.intermediate.iter().enumerate() {
        let nu = params.iterations[k + 1].prev_nu as usize;
        let mu = params.iterations[k + 1].prev_mu as usize;
        let next_stmt = fold_statement(&cur_stmt, inter_proof, &params.iterations[k], nu, mu, &mut t);
        cur_stmt = next_stmt;
    }

    verify_v2(&cur_stmt, &proof.final_iter, &params.iterations[depth - 1], &mut t)
}
