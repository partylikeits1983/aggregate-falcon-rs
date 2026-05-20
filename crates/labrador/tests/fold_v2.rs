//! Phase 6 Session C gate: `fold` converts a finished v2 iteration into a
//! satisfiable next-iteration statement that the honest witness opens.

use falcon_relation::parse::decode_instance;
use falcon_relation::relation::build_falcon_statement;
use labrador::fold::{fold, FoldOutput};
use labrador::params::Params;
use labrador::prover_v2::prove_v2;
use labrador::statement::satisfies;
use labrador::transcript::Transcript;
use labrador::verifier_v2::verify_v2;
use modring::{Modulus, Ring};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn ring_for(n_sigs: usize) -> Ring {
    let params = Params::for_n(n_sigs);
    Ring::new(Modulus::new(params.select_modulus()))
}

fn fresh_sigs(n: usize) -> Vec<falcon_relation::parse::FalconSig> {
    (0..n)
        .map(|i| {
            let msg = format!("phase 6 sess C msg #{}", i).into_bytes();
            let (pk, sk) = falcon512::keypair();
            let sig = falcon512::detached_sign(&msg, &sk);
            decode_instance(pk.as_bytes(), sig.as_bytes(), &msg).unwrap()
        })
        .collect()
}

fn run_iter0(n_sigs: usize, seed: &[u8]) -> (
    labrador::statement::Statement,
    labrador::proof::IterationProofV2,
    Transcript,
    labrador::params::Params,
) {
    let ring = ring_for(n_sigs);
    let sigs = fresh_sigs(n_sigs);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(n_sigs);
    let it_params = &params.iterations[0];

    let mut tp = Transcript::new(seed);
    let proof = prove_v2(&stmt, &witness, it_params, &mut tp);

    // For fold we need a fresh transcript at the same state as before prove_v2.
    let tf = Transcript::new(seed);
    (stmt, proof, tf, params)
}

#[test]
fn debug_check3_vs_verifier_n4() {
    use labrador::commit::matmul;
    use labrador::fold::replay_iteration;
    use labrador::garbage::recompose;
    use modring::RingElem;
    let ring = ring_for(4);
    let sigs = fresh_sigs(4);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(4);
    let it_params = &params.iterations[0];
    let seed: &[u8] = b"fold-debug-n4";

    let mut tp = Transcript::new(seed);
    let proof = prove_v2(&stmt, &witness, it_params, &mut tp);

    // Replay separately to get challenges.
    let mut tr = Transcript::new(seed);
    let replay = replay_iteration(&stmt, &proof, it_params, &mut tr);

    // Compute A·z and Σ c_i v_i[k] DIRECTLY (mirror verify_v2 Check 1).
    let n = stmt.n;
    let kappa = it_params.kappa as usize;
    let b = it_params.b;
    let m = &ring.m;
    // Recompose z.
    let mut z: Vec<RingElem> = vec![RingElem::zero(); n];
    for k in 0..n {
        let last = proof.last_msg.as_ref().unwrap();
        let chunks = vec![last.z0[k].clone(), last.z1[k].clone()];
        z[k] = recompose(&chunks, m, b);
    }
    let az = matmul(&ring, &replay.a_mat, &z);
    let mut expected = vec![RingElem::zero(); kappa];
    for i in 0..stmt.r {
        for k in 0..kappa {
            let p = ring.mul(&replay.cs[i], &proof.last_msg.as_ref().unwrap().v[i][k]);
            expected[k] = expected[k].add(m, &p);
        }
    }
    for k in 0..kappa {
        assert_eq!(az[k], expected[k], "verify_v2 Check 1 fails at row {k}");
    }
    println!("verify_v2 Check 1 OK for all {} rows", kappa);

    // Now compute the SAME thing using the folded witness and our Check 3 builder.
    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;

    let mut tf = Transcript::new(seed);
    let out = fold(&stmt, &proof, it_params, nu, mu, &mut tf);
    // Sanity: replay state and fold state should match.
    let after_replay = tr.challenge_bytes(b"probe", 8);
    let after_fold = tf.challenge_bytes(b"probe", 8);
    assert_eq!(after_replay, after_fold, "replay vs fold transcript drift");

    // Eval Check 3 row 0.
    let c0 = &out.statement.full[0];
    let res = labrador::statement::eval_full(c0, &out.statement.ring, &out.witness);
    if !res.is_zero() {
        // For row 0: dump structure.
        let mut phi_total = 0;
        for (_wi, phi_i) in &c0.phi {
            phi_total += phi_i.len();
        }
        panic!("Check 3 row 0 residual = {} (nonzero); phi entries: {} buckets, total {} positions",
               res.c[0], c0.phi.len(), phi_total);
    }
    println!("Check 3 row 0 OK");
}

#[test]
fn fold_witness_satisfies_new_statement_n4() {
    let (stmt, proof, mut t_fold, params) = run_iter0(4, b"fold-test/n4");
    let it_params = &params.iterations[0];

    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;
    let FoldOutput { statement: new_stmt, witness: new_witness } =
        fold(&stmt, &proof, it_params, nu, mu, &mut t_fold);

    // Identify which constraint fails to ease debugging.
    let kappa = it_params.kappa as usize;
    let kappa1 = it_params.kappa1 as usize;
    for (idx, c) in new_stmt.full.iter().enumerate() {
        let r = labrador::statement::eval_full(c, &new_stmt.ring, &new_witness);
        if !r.is_zero() {
            let label = match idx {
                k if k < kappa => format!("Check 3 row {k}"),
                k if k == kappa => "Check 4".to_string(),
                k if k == kappa + 1 => "Check 5".to_string(),
                k if k == kappa + 2 => "Check 6".to_string(),
                k if k < kappa + 3 + kappa1 => format!("Check 8 row {}", k - (kappa + 3)),
                k => format!("Check 9 row {}", k - (kappa + 3 + kappa1)),
            };
            panic!("constraint {idx} ({label}) non-zero — first coef: {}", r.c[0]);
        }
    }
    let norm_ok = new_witness.norm_sq(&new_stmt.ring) <= new_stmt.beta_sq;
    assert!(norm_ok, "norm bound violated");
    assert!(satisfies(&new_stmt, &new_witness), "folded witness must satisfy folded statement");
}

#[test]
fn fold_norm_bound_holds_n4() {
    let (stmt, proof, mut t_fold, params) = run_iter0(4, b"fold-test/n4-norm");
    let it_params = &params.iterations[0];
    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;
    let FoldOutput { statement: new_stmt, witness: new_witness } =
        fold(&stmt, &proof, it_params, nu, mu, &mut t_fold);

    let norm_sq = new_witness.norm_sq(&new_stmt.ring);
    assert!(
        norm_sq <= new_stmt.beta_sq,
        "norm² {} exceeds β'² {}",
        norm_sq, new_stmt.beta_sq,
    );
}

#[test]
fn fold_shape_matches_params_n4() {
    let (_stmt, proof, mut t_fold, params) = run_iter0(4, b"fold-test/n4-shape");
    let it_params = &params.iterations[0];
    let next_it = &params.iterations[1];
    let nu = next_it.prev_nu as usize;
    let mu = next_it.prev_mu as usize;
    let stmt = run_iter0(4, b"fold-test/n4-shape").0; // not great but cheap; the function is deterministic per seed
    let FoldOutput { statement: new_stmt, .. } =
        fold(&stmt, &proof, it_params, nu, mu, &mut t_fold);

    assert_eq!(new_stmt.n, next_it.n, "new statement rank n' must match next iteration's n");
    let expected_r: usize = next_it.r_list.iter().sum::<u64>() as usize;
    assert_eq!(new_stmt.r, expected_r, "new statement multiplicity r' must match Σ next.r_list");
}

#[test]
fn fold_constraints_count_matches_paper_n8() {
    let (stmt, proof, mut t_fold, params) = run_iter0(8, b"fold-test/n8-constraints");
    let it_params = &params.iterations[0];
    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;
    let FoldOutput { statement: new_stmt, .. } =
        fold(&stmt, &proof, it_params, nu, mu, &mut t_fold);

    // From HANDOFF.md: κ + 1 + 1 + 1 + κ₁ + κ₁ full constraints.
    let kappa = it_params.kappa as usize;
    let kappa1 = it_params.kappa1 as usize;
    let expected = kappa + 3 + 2 * kappa1;
    assert_eq!(new_stmt.full.len(), expected);
    assert_eq!(new_stmt.const_term.len(), 0);
}

#[test]
fn tampering_z0_breaks_fold_consistency() {
    // If the prover's z0 is tampered, fold-with-the-tampered-proof produces a
    // witness that fails one of checks 3-6 (because the verifier's challenges
    // were derived from the same transcript and won't match the new opening).
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let it_params = &params.iterations[0];
    let seed = b"fold-test/n2-tamper-z0";

    let mut tp = Transcript::new(seed);
    let mut proof = prove_v2(&stmt, &witness, it_params, &mut tp);
    // Mutate z0 AFTER the transcript has been advanced; fold will sample the
    // same challenges, but the witness built from the tampered proof won't
    // satisfy the constraints.
    let v = proof.last_msg.as_ref().unwrap().z0[0].c[0];
    proof.last_msg.as_mut().unwrap().z0[0].c[0] = ring.m.add(v, 1);

    let mut tf = Transcript::new(seed);
    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;
    let FoldOutput { statement: new_stmt, witness: new_witness } =
        fold(&stmt, &proof, it_params, nu, mu, &mut tf);
    assert!(
        !satisfies(&new_stmt, &new_witness),
        "tampered z0 must break folded satisfaction",
    );
}

#[test]
fn tampering_v_breaks_fold_consistency() {
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let it_params = &params.iterations[0];
    let seed = b"fold-test/n2-tamper-v";

    let mut tp = Transcript::new(seed);
    let mut proof = prove_v2(&stmt, &witness, it_params, &mut tp);
    let v = proof.last_msg.as_ref().unwrap().v[0][0].c[0];
    proof.last_msg.as_mut().unwrap().v[0][0].c[0] = ring.m.add(v, 1);

    let mut tf = Transcript::new(seed);
    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;
    let FoldOutput { statement: new_stmt, witness: new_witness } =
        fold(&stmt, &proof, it_params, nu, mu, &mut tf);
    assert!(
        !satisfies(&new_stmt, &new_witness),
        "tampered v must break folded satisfaction",
    );
}

#[test]
fn fold_replay_matches_verify_v2_transcript_state() {
    // After fold finishes, the transcript should be at the same state as
    // after verify_v2 finishes. Compare by deriving the same challenge from
    // both — they should agree.
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let it_params = &params.iterations[0];
    let seed = b"fold-test/n2-replay";

    let mut tp = Transcript::new(seed);
    let proof = prove_v2(&stmt, &witness, it_params, &mut tp);
    let after_prove = tp.challenge_bytes(b"probe", 16);

    let mut tv = Transcript::new(seed);
    verify_v2(&stmt, &proof, it_params, &mut tv).unwrap();
    let after_verify = tv.challenge_bytes(b"probe", 16);
    assert_eq!(after_prove, after_verify, "verify_v2 must end at same transcript state as prove_v2");

    let mut tf = Transcript::new(seed);
    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;
    let _ = fold(&stmt, &proof, it_params, nu, mu, &mut tf);
    let after_fold = tf.challenge_bytes(b"probe", 16);
    assert_eq!(after_prove, after_fold, "fold must end at same transcript state as prove_v2");
}
