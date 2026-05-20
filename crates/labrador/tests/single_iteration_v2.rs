//! Phase 6 Session B gate: paper-correct v2 single-iteration prover/verifier
//! round-trips on a real Falcon-512 statement, and tampering each prover
//! message rejects.

use falcon_relation::parse::decode_instance;
use falcon_relation::relation::build_falcon_statement;
use labrador::params::Params;
use labrador::prover_v2::prove_v2;
use labrador::transcript::Transcript;
use labrador::verifier_v2::verify_v2;
use modring::{find_prime_5mod8, Modulus, Ring};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn ring() -> Ring {
    Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
}

fn fresh_sigs(n: usize) -> Vec<falcon_relation::parse::FalconSig> {
    (0..n)
        .map(|i| {
            let msg = format!("phase 6 sess B msg #{}", i).into_bytes();
            let (pk, sk) = falcon512::keypair();
            let sig = falcon512::detached_sign(&msg, &sk);
            decode_instance(pk.as_bytes(), sig.as_bytes(), &msg).unwrap()
        })
        .collect()
}

#[test]
fn honest_proof_v2_round_trips_for_n4() {
    let ring = ring();
    let sigs = fresh_sigs(4);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(4);
    let it_params = &params.iterations[0];

    let mut tp = Transcript::new(b"v2-e2e");
    let proof = prove_v2(&stmt, &witness, it_params, &mut tp);

    let mut tv = Transcript::new(b"v2-e2e");
    verify_v2(&stmt, &proof, it_params, &mut tv).expect("honest v2 proof verifies");
}

#[test]
fn tampering_u1_rejects_v2() {
    let ring = ring();
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let it_params = &params.iterations[0];

    let mut tp = Transcript::new(b"v2-e2e");
    let mut proof = prove_v2(&stmt, &witness, it_params, &mut tp);
    proof.u1[0].c[0] = ring.m.add(proof.u1[0].c[0], 1);

    let mut tv = Transcript::new(b"v2-e2e");
    assert!(verify_v2(&stmt, &proof, it_params, &mut tv).is_err());
}

#[test]
fn tampering_u2_rejects_v2() {
    let ring = ring();
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let it_params = &params.iterations[0];

    let mut tp = Transcript::new(b"v2-e2e");
    let mut proof = prove_v2(&stmt, &witness, it_params, &mut tp);
    proof.u2[0].c[0] = ring.m.add(proof.u2[0].c[0], 1);

    let mut tv = Transcript::new(b"v2-e2e");
    assert!(verify_v2(&stmt, &proof, it_params, &mut tv).is_err());
}

#[test]
fn tampering_z0_rejects_v2() {
    let ring = ring();
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let it_params = &params.iterations[0];

    let mut tp = Transcript::new(b"v2-e2e");
    let mut proof = prove_v2(&stmt, &witness, it_params, &mut tp);
    proof.last_msg.as_mut().unwrap().z0[0].c[0] = ring.m.add(proof.last_msg.as_ref().unwrap().z0[0].c[0], 1);

    let mut tv = Transcript::new(b"v2-e2e");
    assert!(verify_v2(&stmt, &proof, it_params, &mut tv).is_err());
}

#[test]
fn tampering_v_rejects_v2() {
    let ring = ring();
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let it_params = &params.iterations[0];

    let mut tp = Transcript::new(b"v2-e2e");
    let mut proof = prove_v2(&stmt, &witness, it_params, &mut tp);
    proof.last_msg.as_mut().unwrap().v[0][0].c[0] = ring.m.add(proof.last_msg.as_ref().unwrap().v[0][0].c[0], 1);

    let mut tv = Transcript::new(b"v2-e2e");
    assert!(verify_v2(&stmt, &proof, it_params, &mut tv).is_err());
}
