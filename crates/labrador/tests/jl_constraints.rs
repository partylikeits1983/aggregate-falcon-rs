//! Phase 6 Session E gate: JL projection constraints tie `p` to `w`.

use falcon_relation::parse::decode_instance;
use falcon_relation::relation::build_falcon_statement;
use labrador::aggregate_v2::{prove_aggregate, verify_aggregate};
use labrador::jl::{build_jl_constraints, sample_projection, project_combined};
use labrador::params::Params;
use labrador::prover_v2::prove_v2;
use labrador::statement::eval_const_term;
use labrador::transcript::Transcript;
use modring::{Modulus, Ring, D};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn ring_for(n_sigs: usize) -> Ring {
    let params = Params::for_n(n_sigs);
    Ring::new(Modulus::new(params.select_modulus()))
}

fn fresh_sigs(n: usize) -> Vec<falcon_relation::parse::FalconSig> {
    (0..n)
        .map(|i| {
            let msg = format!("jl msg #{}", i).into_bytes();
            let (pk, sk) = falcon512::keypair();
            let sig = falcon512::detached_sign(&msg, &sk);
            decode_instance(pk.as_bytes(), sig.as_bytes(), &msg).unwrap()
        })
        .collect()
}

#[test]
fn honest_witness_satisfies_jl_constraints_n2() {
    // Build a statement, project the witness, re-derive Π and the JL
    // constraints, and assert each constraint's constant term vanishes on
    // the honest witness.
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    let m = &ring.m;
    let n = stmt.n;
    let r = stmt.r;

    // Bind statement to a fresh transcript and replay the prover's transcript
    // steps up to the Π squeeze + p commitment.
    let mut t = Transcript::new(b"jl-honest");
    labrador::prover::bind_statement(&mut t, &stmt);
    // Skip A, B, C, u_1 (we just need Π).
    let _ = labrador::commit::expand_matrix(&mut t, b"labrador.A", 1, 1, &ring);
    // Reusing the prover's transcript order isn't necessary for THIS test
    // — we only need a Π derived deterministically from SOME transcript and
    // the corresponding p. Use a fresh transcript here so the constraints
    // are independent of the rest of the proof.
    let mut t = Transcript::new(b"jl-honest-direct");
    let pis: Vec<_> = (0..r)
        .map(|i| {
            let label = [b"labrador.Pi".as_ref(), &(i as u64).to_le_bytes()].concat();
            sample_projection(&mut t, &label, n * D)
        })
        .collect();
    let p_arr = project_combined(&pis, &witness.w, m);
    let p: Vec<i128> = p_arr.iter().copied().collect();

    let jl = build_jl_constraints(&pis, &p, n, m);
    assert_eq!(jl.len(), 256, "expect 2λ = 256 JL constraints");
    for (j, c) in jl.iter().enumerate() {
        let r = eval_const_term(c, &ring, &witness);
        assert_eq!(r, 0, "JL constraint {j} does not vanish on honest witness");
    }
}

#[test]
fn tampering_p_breaks_jl_constraint_n2() {
    // Build the JL constraints from honest Π but a TAMPERED p — at least one
    // constraint must fail.
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (_stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let m = &ring.m;
    let n = witness.w[0].len();
    let r = witness.r();

    let mut t = Transcript::new(b"jl-tamper");
    let pis: Vec<_> = (0..r)
        .map(|i| {
            let label = [b"labrador.Pi".as_ref(), &(i as u64).to_le_bytes()].concat();
            sample_projection(&mut t, &label, n * D)
        })
        .collect();
    let p_arr = project_combined(&pis, &witness.w, m);
    let mut p: Vec<i128> = p_arr.iter().copied().collect();
    p[0] = p[0].wrapping_add(1);

    let jl = build_jl_constraints(&pis, &p, n, m);
    // Constraint 0 used the tampered p[0], so it must fail.
    let r0 = eval_const_term(&jl[0], &ring, &witness);
    assert_ne!(r0, 0, "tampered p[0] must break JL constraint 0");
}

#[test]
fn tampered_jl_proof_p_rejects_at_n2() {
    // End-to-end: aggregate, then flip a byte of the FINAL iteration's p,
    // assert verify rejects. The JL constraint embeds p into F' so the
    // ConstTermMismatch check catches this (or the JL norm bound).
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);

    let mut proof = prove_aggregate(&stmt, &witness, &params, b"jl-e2e");
    proof.final_iter.p[0] = proof.final_iter.p[0].wrapping_add(1);
    assert!(verify_aggregate(&stmt, &proof, &params, b"jl-e2e").is_err());
}

#[test]
fn tampered_intermediate_p_rejects_at_n2() {
    // Flip a byte of an INTERMEDIATE iteration's p. The next-iteration
    // verifier rebuilds the JL constraints from this p, which doesn't match
    // the honest witness, so the ConstTermMismatch / aggregation chain breaks.
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);

    let mut proof = prove_aggregate(&stmt, &witness, &params, b"jl-e2e-mid");
    proof.intermediate[0].p[0] = proof.intermediate[0].p[0].wrapping_add(1);
    assert!(verify_aggregate(&stmt, &proof, &params, b"jl-e2e-mid").is_err());
}

#[test]
fn prove_v2_first_iter_passes_jl_constraints_n2() {
    // Sanity check: prove_v2 on the iter[0] statement WITHOUT additional
    // const_term should succeed even though the JL constraints are added
    // inside. The single-iteration test (single_iteration_v2.rs) covers this
    // implicitly — assert here that the workspace round-trip continues to
    // work with JL constraints enabled.
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let it_params = &params.iterations[0];

    let mut t = Transcript::new(b"jl-single-iter");
    let proof = prove_v2(&stmt, &witness, it_params, &mut t);
    let mut tv = Transcript::new(b"jl-single-iter");
    labrador::verifier_v2::verify_v2(&stmt, &proof, it_params, &mut tv).expect("verify with JL");
}
