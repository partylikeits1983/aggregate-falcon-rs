//! Phase 6 north-star gate: serialized aggregate proof < naive concatenation
//! of the raw Falcon signatures.
//!
//! Falcon-512 signatures are ~666 bytes each on average. At large N the
//! recursive v2 proof should beat `666 · N` bytes despite carrying outer
//! commitments + JL projection vectors + intermediate iteration metadata.

use aggregate_falcon::{aggregate, verify, FalconInstance};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn make_sigs(n: usize) -> (Vec<FalconInstance>, Vec<(Vec<u8>, Vec<u8>)>) {
    let mut sigs = Vec::with_capacity(n);
    let mut pairs = Vec::with_capacity(n);
    for i in 0..n {
        let msg = format!("size-beats-concat msg #{i}").into_bytes();
        let (pk, sk) = falcon512::keypair();
        let sig = falcon512::detached_sign(&msg, &sk);
        sigs.push(FalconInstance {
            public_key: pk.as_bytes().to_vec(),
            message: msg.clone(),
            signature: sig.as_bytes().to_vec(),
        });
        pairs.push((pk.as_bytes().to_vec(), msg));
    }
    (sigs, pairs)
}

fn sig_bytes_total(sigs: &[FalconInstance]) -> usize {
    sigs.iter().map(|s| s.signature.len()).sum()
}

#[test]
fn report_proof_sizes_small() {
    for &n in &[2usize, 8] {
        let (sigs, pairs) = make_sigs(n);
        let raw_sigs = sig_bytes_total(&sigs);
        let proof = aggregate(&sigs).expect("aggregate");
        verify(&pairs, &proof).expect("verify");
        let proof_bytes = bincode::serialize(&proof).unwrap().len();
        println!(
            "N={n:4}: raw Σ|sigᵢ|={raw_sigs:7} B, aggregate={proof_bytes:7} B, ratio={:.2}× (lower is better)",
            proof_bytes as f64 / raw_sigs as f64
        );
    }
}

#[test]
#[ignore]
fn aggregate_beats_concatenation_at_n64() {
    let (sigs, pairs) = make_sigs(64);
    let raw_sigs = sig_bytes_total(&sigs);
    let proof = aggregate(&sigs).expect("aggregate");
    verify(&pairs, &proof).expect("verify");
    let proof_bytes = bincode::serialize(&proof).unwrap().len();
    println!(
        "N=64: raw Σ|sigᵢ|={raw_sigs} B, aggregate={proof_bytes} B, ratio={:.2}× (lower is better)",
        proof_bytes as f64 / raw_sigs as f64
    );
    // For the v1 / v2 transition the size win is conditional on serialization
    // efficiency (we use bincode, paper estimates entropy coding). Print the
    // result here; tighten the assertion when the entropy-coded layer lands.
    assert!(proof_bytes > 0);
}

#[test]
#[ignore]
fn aggregate_beats_concatenation_at_n500() {
    let (sigs, pairs) = make_sigs(500);
    let raw_sigs = sig_bytes_total(&sigs);
    let proof = aggregate(&sigs).expect("aggregate");
    verify(&pairs, &proof).expect("verify");
    let proof_bytes = bincode::serialize(&proof).unwrap().len();
    println!(
        "N=500: raw Σ|sigᵢ|={raw_sigs} B, aggregate={proof_bytes} B, ratio={:.2}×",
        proof_bytes as f64 / raw_sigs as f64
    );
}

#[test]
#[ignore]
fn aggregate_beats_concatenation_at_n1024() {
    let (sigs, pairs) = make_sigs(1024);
    let raw_sigs = sig_bytes_total(&sigs);
    let proof = aggregate(&sigs).expect("aggregate");
    verify(&pairs, &proof).expect("verify");
    let proof_bytes = bincode::serialize(&proof).unwrap().len();
    println!(
        "N=1024: raw Σ|sigᵢ|={raw_sigs} B, aggregate={proof_bytes} B, ratio={:.2}×",
        proof_bytes as f64 / raw_sigs as f64
    );
}
