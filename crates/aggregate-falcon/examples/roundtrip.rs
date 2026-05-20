//! Driver example: aggregate N Falcon-512 signatures, verify, report sizes.
//!
//!   cargo run --release --example roundtrip [N]

use aggregate_falcon::{aggregate, verify, FalconInstance, PublicPair};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};
use std::env;
use std::time::Instant;

fn main() {
    let n: usize = env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    println!("Aggregating {n} Falcon-512 signatures…");

    let mut instances: Vec<FalconInstance> = Vec::with_capacity(n);
    let mut pairs: Vec<PublicPair> = Vec::with_capacity(n);
    let t0 = Instant::now();
    for i in 0..n {
        let msg = format!("example msg #{i}").into_bytes();
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
    println!("  keygen + sign: {:.2?}", t0.elapsed());

    let raw_total: usize = instances
        .iter()
        .map(|i| i.public_key.len() + i.message.len() + i.signature.len())
        .sum();
    println!(
        "  raw inputs (pk + msg + sig per instance): {} KB",
        raw_total / 1024
    );

    let t1 = Instant::now();
    let proof = aggregate(&instances).expect("aggregate");
    println!("  aggregate: {:.2?}", t1.elapsed());

    let proof_bytes = bincode::serialize(&proof).expect("serialize");
    println!("  proof size (bincode): {} KB", proof_bytes.len() / 1024);

    let t2 = Instant::now();
    verify(&pairs, &proof).expect("verify");
    println!("  verify: {:.2?}  ✓", t2.elapsed());

    println!("\nDone.");
}
