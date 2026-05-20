//! Phase 5 gate: a single LaBRADOR iteration round-trips on a real Falcon-512
//! statement, and tampering each prover message rejects.

use falcon_relation::parse::decode_instance;
use falcon_relation::relation::build_falcon_statement;
use labrador::prover::prove;
use labrador::transcript::Transcript;
use labrador::verifier::verify;
use modring::{find_prime_5mod8, Modulus, Ring};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn ring() -> Ring {
    Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
}

fn fresh_sigs(n: usize) -> Vec<falcon_relation::parse::FalconSig> {
    (0..n)
        .map(|i| {
            let msg = format!("phase 5 message #{}", i).into_bytes();
            let (pk, sk) = falcon512::keypair();
            let sig = falcon512::detached_sign(&msg, &sk);
            decode_instance(pk.as_bytes(), sig.as_bytes(), &msg).unwrap()
        })
        .collect()
}

#[test]
fn honest_proof_round_trips_for_n4() {
    let ring = ring();
    let sigs = fresh_sigs(4);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    let mut tp = Transcript::new(b"e2e-test");
    let proof = prove(&stmt, &witness, &mut tp);

    let mut tv = Transcript::new(b"e2e-test");
    verify(&stmt, &proof, &mut tv).expect("honest proof verifies");
}

#[test]
fn tampering_z_rejects() {
    let ring = ring();
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    let mut tp = Transcript::new(b"e2e-test");
    let mut proof = prove(&stmt, &witness, &mut tp);
    proof.z[0].c[0] = ring.m.add(proof.z[0].c[0], 1);

    let mut tv = Transcript::new(b"e2e-test");
    assert!(verify(&stmt, &proof, &mut tv).is_err());
}

#[test]
fn tampering_v_rejects() {
    let ring = ring();
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    let mut tp = Transcript::new(b"e2e-test");
    let mut proof = prove(&stmt, &witness, &mut tp);
    proof.v[0][0].c[0] = ring.m.add(proof.v[0][0].c[0], 1);

    let mut tv = Transcript::new(b"e2e-test");
    assert!(verify(&stmt, &proof, &mut tv).is_err());
}

#[test]
fn tampering_g_rejects() {
    let ring = ring();
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    let mut tp = Transcript::new(b"e2e-test");
    let mut proof = prove(&stmt, &witness, &mut tp);
    proof.g[0][0].c[0] = ring.m.add(proof.g[0][0].c[0], 1);

    let mut tv = Transcript::new(b"e2e-test");
    assert!(verify(&stmt, &proof, &mut tv).is_err());
}

#[test]
fn tampering_bpp_rejects() {
    let ring = ring();
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    let mut tp = Transcript::new(b"e2e-test");
    let mut proof = prove(&stmt, &witness, &mut tp);
    if !proof.b_double_prime.is_empty() {
        proof.b_double_prime[0].c[0] = ring.m.add(proof.b_double_prime[0].c[0], 1);
    }

    let mut tv = Transcript::new(b"e2e-test");
    assert!(verify(&stmt, &proof, &mut tv).is_err());
}
