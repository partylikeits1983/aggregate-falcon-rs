//! Real Falcon-512 aggregate-then-verify at the canonical sizes.
//!
//! Each test:
//! 1. Generates N real Falcon-512 keypairs and signs a distinct message per pair
//!    using `pqcrypto-falcon` (FFI to the audited C reference).
//! 2. Calls the real `aggregate()` to produce a LaBRADOR-folded proof.
//! 3. Calls `verify()` on the honest `(pk, message)` pairs and asserts it
//!    accepts.
//! 4. Mutates the message of one pair and asserts `verify()` rejects.
//! 5. At N ≥ 1024 also asserts the proof is smaller than the naive sig-byte
//!    concatenation (the size north-star).
//!
//! All tests are `#[ignore]`-gated because the LaBRADOR prover is
//! single-threaded and inherently O(N²) per paper §F.1. Extrapolating the
//! observed ~50s at N=8: N=128 ≈ 3.5h, N=512 ≈ 57h, N=1024 ≈ 230h. Run
//! deliberately:
//!
//! ```text
//! cargo test -p aggregate-falcon --test large_n_e2e --release \
//!     -- --ignored --nocapture roundtrip_n128
//! ```
//!
//! Replace `roundtrip_n128` with `roundtrip_n512` / `roundtrip_n1024` for the
//! larger sizes. See PERF_PLAN.md for the optimization roadmap that will
//! eventually make these tractable.

use aggregate_falcon::{aggregate, verify, FalconInstance, PublicPair};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn make_sigs(n: usize) -> (Vec<FalconInstance>, Vec<PublicPair>) {
    let mut instances = Vec::with_capacity(n);
    let mut pairs = Vec::with_capacity(n);
    for i in 0..n {
        let msg = format!("large-N e2e msg #{i} of {n}").into_bytes();
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

fn roundtrip_at(n: usize) {
    let (sigs, pairs) = make_sigs(n);
    let raw = sigs.iter().map(|s| s.signature.len()).sum::<usize>();

    let proof = aggregate(&sigs).expect("aggregate must succeed on honest inputs");
    verify(&pairs, &proof).expect("verify must accept the honest aggregate");

    let mut tampered = pairs.clone();
    tampered[0].1.push(0xff);
    assert!(
        verify(&tampered, &proof).is_err(),
        "verify must reject a tampered message at N={n}"
    );

    let bytes = bincode::serialize(&proof).expect("serialize proof").len();
    println!(
        "N={n}: raw=Σ|sig|={raw} B, proof={bytes} B, ratio={:.2}×",
        bytes as f64 / raw as f64
    );
    if n >= 1024 {
        assert!(
            bytes < raw,
            "north star: aggregate proof must beat naive concatenation at N={n} \
             (got {bytes} B vs {raw} B raw)"
        );
    }
}

#[test]
#[ignore]
fn roundtrip_n128() {
    roundtrip_at(128);
}

#[test]
#[ignore]
fn roundtrip_n512() {
    roundtrip_at(512);
}

#[test]
#[ignore]
fn roundtrip_n1024() {
    roundtrip_at(1024);
}
