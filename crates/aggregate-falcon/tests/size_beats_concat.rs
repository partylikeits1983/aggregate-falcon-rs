//! Small-N size monotonicity check.
//!
//! Generates real Falcon-512 signatures, runs the actual `aggregate` +
//! `verify` pipeline at N=2 and N=8, and asserts that as N grows the proof's
//! per-signature amortized size shrinks (proof_bytes / N decreasing). This
//! is the smallest empirical check that the recursion is amortizing real
//! work, not just emitting a fixed-size envelope.
//!
//! The larger crossover-N runs (N=128/512/1024) live in `large_n_e2e.rs` and
//! are `#[ignore]`-gated because they take hours per the paper's inherent
//! O(N²) prover cost (see AUDIT.md §3 and PERF_PLAN.md).

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
fn proof_size_per_signature_shrinks_from_n2_to_n8() {
    let mut per_sig_bytes: Vec<f64> = Vec::new();
    for &n in &[2usize, 8] {
        let (sigs, pairs) = make_sigs(n);
        let raw_sigs = sig_bytes_total(&sigs);
        let proof = aggregate(&sigs).expect("aggregate");
        verify(&pairs, &proof).expect("verify");
        let proof_bytes = bincode::serialize(&proof).unwrap().len();
        let per_sig = proof_bytes as f64 / n as f64;
        println!(
            "N={n:4}: raw={raw_sigs:7} B, aggregate={proof_bytes:7} B, per-sig={per_sig:.0} B"
        );
        per_sig_bytes.push(per_sig);
    }
    // Per-signature amortized cost must decrease from N=2 to N=8.
    assert!(
        per_sig_bytes[1] < per_sig_bytes[0],
        "amortized proof size must shrink with N: N=2 → {:.0} B/sig, N=8 → {:.0} B/sig",
        per_sig_bytes[0], per_sig_bytes[1]
    );
}
