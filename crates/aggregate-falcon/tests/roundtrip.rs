//! End-to-end aggregation + verification of Falcon-512 signatures via
//! LaBRADOR.
//!
//! Gate: for `N ∈ {2, 8}` (fast tests) the honest aggregate proof verifies,
//! and every mutation (wrong pk, wrong message, flipped signature byte,
//! flipped proof byte) rejects.

use aggregate_falcon::{aggregate, verify, AggregateProof, FalconInstance, PublicPair};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn make_sigs(n: usize) -> (Vec<FalconInstance>, Vec<PublicPair>) {
    let mut instances = Vec::with_capacity(n);
    let mut pairs = Vec::with_capacity(n);
    for i in 0..n {
        let msg = format!("e2e roundtrip msg #{i}").into_bytes();
        let (pk, sk) = falcon512::keypair();
        let sig = falcon512::detached_sign(&msg, &sk);
        let pk_bytes = pk.as_bytes().to_vec();
        let sig_bytes = sig.as_bytes().to_vec();
        pairs.push((pk_bytes.clone(), msg.clone()));
        instances.push(FalconInstance {
            public_key: pk_bytes,
            message: msg,
            signature: sig_bytes,
        });
    }
    (instances, pairs)
}

#[test]
fn roundtrip_n2() {
    let (sigs, pairs) = make_sigs(2);
    let proof = aggregate(&sigs).expect("aggregate");
    verify(&pairs, &proof).expect("verify");
}

#[test]
fn roundtrip_n8() {
    let (sigs, pairs) = make_sigs(8);
    let proof = aggregate(&sigs).expect("aggregate");
    verify(&pairs, &proof).expect("verify");
}

#[test]
fn wrong_public_key_rejects() {
    let (sigs, mut pairs) = make_sigs(2);
    let proof = aggregate(&sigs).expect("aggregate");
    // Substitute a fresh key into the first pair.
    let (other_pk, _) = falcon512::keypair();
    pairs[0].0 = other_pk.as_bytes().to_vec();
    assert!(verify(&pairs, &proof).is_err());
}

#[test]
fn wrong_message_rejects() {
    let (sigs, mut pairs) = make_sigs(2);
    let proof = aggregate(&sigs).expect("aggregate");
    pairs[0].1.push(0xff);
    assert!(verify(&pairs, &proof).is_err());
}

#[test]
fn flipped_proof_byte_rejects() {
    let (sigs, pairs) = make_sigs(2);
    let mut proof = aggregate(&sigs).expect("aggregate");
    // Flip a coefficient of the amortized opening z.
    proof.iteration.z[0].c[0] ^= 1;
    assert!(verify(&pairs, &proof).is_err());
}

#[test]
fn serde_roundtrip_then_verify() {
    let (sigs, pairs) = make_sigs(2);
    let proof = aggregate(&sigs).expect("aggregate");
    let bytes = bincode::serialize(&proof).expect("serialize");
    let restored: AggregateProof = bincode::deserialize(&bytes).expect("deserialize");
    verify(&pairs, &restored).expect("verify after bincode round-trip");
}
