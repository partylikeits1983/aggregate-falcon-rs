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

use crate::fold::{fold_statement, fold_with_replay, FoldOutput};
use crate::params::Params;
use crate::proof::{AggregateProofV2, IterationProofV2, VerifyError};
use crate::prover_v2::{prove_v2, prove_v2_with_replay};
use crate::stage_timing;
use crate::statement::{Statement, Witness};
use crate::transcript::Transcript;
use crate::verifier_v2::verify_v2;
use std::time::Instant;

/// Minimal progress callback for the multi-iteration prover. Each LaBRADOR
/// iteration emits one `iter_start` before its work begins and one `iter_done`
/// after the matching `fold` finishes (the final iteration has no fold).
///
/// The trait is implemented for `()` as a no-op, so callers that don't care
/// about progress can keep using [`prove_aggregate`] unchanged.
pub trait ProgressSink {
    fn iter_start(&mut self, k: usize, depth: usize, label: &str);
    fn iter_done(&mut self, k: usize);
}

impl ProgressSink for () {
    fn iter_start(&mut self, _: usize, _: usize, _: &str) {}
    fn iter_done(&mut self, _: usize) {}
}

/// Run the multi-iteration recursive prover. Thin wrapper that discards
/// progress events; see [`prove_aggregate_with_progress`] for the live-tracking
/// variant the example driver uses.
pub fn prove_aggregate(
    stmt: &Statement,
    witness: &Witness,
    params: &Params,
    transcript_seed: &[u8],
) -> AggregateProofV2 {
    prove_aggregate_with_progress(stmt, witness, params, transcript_seed, &mut ())
}

/// Run the multi-iteration recursive prover, emitting per-iteration progress
/// to `sink`. Identical to [`prove_aggregate`] otherwise.
pub fn prove_aggregate_with_progress<P: ProgressSink>(
    stmt: &Statement,
    witness: &Witness,
    params: &Params,
    transcript_seed: &[u8],
    sink: &mut P,
) -> AggregateProofV2 {
    let depth = params.depth;
    assert!(depth >= 1, "depth must be ≥ 1");

    let mut cur_stmt = stmt.clone();
    let mut cur_witness = witness.clone();
    let mut t_prove = Transcript::new(transcript_seed);
    let mut intermediate: Vec<IterationProofV2> = Vec::with_capacity(depth - 1);

    for k in 0..(depth - 1) {
        sink.iter_start(k, depth, "prove_v2 + fold");
        let ts = Instant::now();
        let (proof_k, replay) =
            prove_v2_with_replay(&cur_stmt, &cur_witness, &params.iterations[k], &mut t_prove);
        stage_timing::record("prove_v2_total", ts.elapsed());

        // Fold using params.iterations[k+1].(prev_nu, prev_mu) — those are the
        // folding parameters that produced iter[k+1]'s rank. The replay we
        // received from prove_v2_with_replay lets us skip fold's own
        // (otherwise redundant) transcript walk.
        let nu = params.iterations[k + 1].prev_nu as usize;
        let mu = params.iterations[k + 1].prev_mu as usize;
        let ts = Instant::now();
        let FoldOutput { statement, witness } =
            fold_with_replay(&cur_stmt, &proof_k, &params.iterations[k], nu, mu, &replay);
        stage_timing::record("fold_total", ts.elapsed());
        stage_timing::mark("iter-end");

        intermediate.push(proof_k.into_intermediate());
        cur_stmt = statement;
        cur_witness = witness;
        sink.iter_done(k);
    }

    sink.iter_start(depth - 1, depth, "prove_v2 (final)");
    let ts = Instant::now();
    let final_iter = prove_v2(
        &cur_stmt,
        &cur_witness,
        &params.iterations[depth - 1],
        &mut t_prove,
    );
    stage_timing::record("prove_v2_total", ts.elapsed());
    stage_timing::mark("iter-end");
    sink.iter_done(depth - 1);

    AggregateProofV2 {
        q_prime: stmt.ring.m.q,
        n_sigs: params.num_sigs,
        beta_sq: stmt.beta_sq,
        intermediate,
        final_iter,
    }
}

/// Run the multi-iteration recursive verifier. Thin wrapper that discards
/// progress events; see [`verify_aggregate_with_progress`] for the live-tracking
/// variant the example driver uses.
pub fn verify_aggregate(
    stmt: &Statement,
    proof: &AggregateProofV2,
    params: &Params,
    transcript_seed: &[u8],
) -> Result<(), VerifyError> {
    verify_aggregate_with_progress(stmt, proof, params, transcript_seed, &mut ())
}

/// Run the multi-iteration recursive verifier, emitting per-iteration progress
/// to `sink`. Identical to [`verify_aggregate`] otherwise.
pub fn verify_aggregate_with_progress<P: ProgressSink>(
    stmt: &Statement,
    proof: &AggregateProofV2,
    params: &Params,
    transcript_seed: &[u8],
    sink: &mut P,
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
        sink.iter_start(k, depth, "fold_statement");
        let nu = params.iterations[k + 1].prev_nu as usize;
        let mu = params.iterations[k + 1].prev_mu as usize;
        let ts = Instant::now();
        let next_stmt = fold_statement(&cur_stmt, inter_proof, &params.iterations[k], nu, mu, &mut t);
        stage_timing::record("verify_fold_statement", ts.elapsed());
        stage_timing::mark("verify-iter-end");
        cur_stmt = next_stmt;
        sink.iter_done(k);
    }

    sink.iter_start(depth - 1, depth, "verify_v2 (final)");
    let ts = Instant::now();
    let res = verify_v2(&cur_stmt, &proof.final_iter, &params.iterations[depth - 1], &mut t);
    stage_timing::record("verify_v2_final", ts.elapsed());
    stage_timing::mark("verify-iter-end");
    sink.iter_done(depth - 1);
    res
}
