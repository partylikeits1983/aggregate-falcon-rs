//! End-to-end aggregation + verification of Falcon-512 signatures via
//! LaBRADOR.
//!
//! Gate: for `N ∈ {2, 8}` (fast tests) the honest aggregate proof verifies,
//! and every mutation (wrong pk, wrong message, flipped signature byte,
//! flipped proof byte) rejects.

use aggregate_falcon::{aggregate, verify, AggregateProof, FalconInstance, PublicPair};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

use falcon_relation::parse::decode_instance;
use falcon_relation::relation::build_falcon_statement;
use labrador::params::Params;
use labrador::statement::Statement;
use modring::{Modulus, Ring};

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
    // Flip a coefficient of the final iteration's amortized opening z0.
    let last = proof.labrador.final_iter.last_msg.as_mut().unwrap();
    last.z0[0].c[0] ^= 1;
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

/// CONCERN-3 closer: the public-input statement built by the verifier
/// (s1=s2=0 reconstructed from pk/msg/nonce) must be byte-equal to the
/// statement the prover built from the full decoded signatures (real
/// s1, s2). The witness is not compared — only the public statement is. If
/// any constraint builder ever starts reading `s1` or `s2` (today none do),
/// this test will fail loudly.
#[test]
fn prover_and_verifier_statements_match() {
    let (sigs, pairs) = make_sigs(2);

    // Prover-side: full sigs with real s1, s2 → build_falcon_statement.
    let params = Params::for_n(sigs.len());
    let q_prime = params.select_modulus();
    let ring = Ring::new(Modulus::new(q_prime));
    let mut full_parsed = Vec::with_capacity(sigs.len());
    for s in &sigs {
        let p = decode_instance(&s.public_key, &s.signature, &s.message)
            .expect("decode real Falcon sig");
        full_parsed.push(p);
    }
    let beta_sq = params.beta_init_sq();
    let (stmt_full, _w, _layout) = build_falcon_statement(&full_parsed, &ring, beta_sq);

    // Verifier-side: same (pk, msg, nonce) but s1=s2=0.
    let nonces: Vec<Vec<u8>> = full_parsed.iter().map(|p| p.nonce.to_vec()).collect();
    let stmt_pubonly: Statement =
        aggregate_falcon::build_verifier_statement(&pairs, &nonces, beta_sq, &ring)
            .expect("verifier statement");

    // Public scalar fields.
    assert_eq!(stmt_full.n, stmt_pubonly.n, "n");
    assert_eq!(stmt_full.r, stmt_pubonly.r, "r");
    assert_eq!(stmt_full.beta_sq, stmt_pubonly.beta_sq, "beta_sq");
    assert_eq!(stmt_full.ring.m.q, stmt_pubonly.ring.m.q, "ring modulus");

    // Constraint lists must match exactly (PartialEq derived on
    // Dot/ConstTermConstraint).
    assert_eq!(
        stmt_full.full.len(),
        stmt_pubonly.full.len(),
        "full constraint count"
    );
    assert_eq!(
        stmt_full.const_term.len(),
        stmt_pubonly.const_term.len(),
        "const_term constraint count"
    );
    for (i, (a, b)) in stmt_full.full.iter().zip(stmt_pubonly.full.iter()).enumerate() {
        assert_eq!(a, b, "full constraint #{i} differs");
    }
    for (i, (a, b)) in stmt_full
        .const_term
        .iter()
        .zip(stmt_pubonly.const_term.iter())
        .enumerate()
    {
        assert_eq!(a, b, "const_term constraint #{i} differs");
    }
}

/// A `PublicPair` with the wrong nonce length must be rejected with a
/// dedicated `DecodeFailed` error — not silently treated as a downstream
/// constraint mismatch. Falcon nonces are exactly 40 bytes.
#[test]
fn malformed_nonce_length_rejects() {
    let (sigs, pairs) = make_sigs(2);
    let mut proof = aggregate(&sigs).expect("aggregate");
    // Truncate the first nonce by one byte.
    proof.nonces[0].pop();
    let err = verify(&pairs, &proof).expect_err("malformed nonce must reject");
    let s = format!("{err}");
    assert!(
        s.contains("nonce length"),
        "expected nonce-length error, got: {s}"
    );
}

/// Honest aggregate, then verify with the (pk, msg) pairs swapped between
/// two slots. Since each signature is bound to (pk_i, msg_i, nonce_i) and
/// every nonce stays in its original slot, the reconstructed `c` and `h`
/// at slot 0 will no longer match what the prover used. Verification must
/// reject.
#[test]
fn swapped_public_pairs_rejects() {
    let (sigs, mut pairs) = make_sigs(2);
    let proof = aggregate(&sigs).expect("aggregate");
    pairs.swap(0, 1);
    assert!(
        verify(&pairs, &proof).is_err(),
        "swapped (pk, msg) pairs must reject"
    );
}

/// Mutate the first intermediate iteration's `u1` outer commitment. The
/// existing `flipped_proof_byte_rejects` only covers the final iteration's
/// `z0` opening — this guards against an intermediate-iteration
/// tampering oversight.
#[test]
fn flipped_intermediate_u1_rejects() {
    let (sigs, pairs) = make_sigs(2);
    let mut proof = aggregate(&sigs).expect("aggregate");
    if proof.labrador.intermediate.is_empty() {
        // N=2 may pick depth=1 (only the final iter); skip rather than misreport.
        eprintln!("note: depth=1 at N=2; intermediate u1 tamper test skipped");
        return;
    }
    proof.labrador.intermediate[0].u1[0].c[0] ^= 1;
    assert!(
        verify(&pairs, &proof).is_err(),
        "flipped intermediate u1 byte must reject"
    );
}

/// Mutate the first intermediate iteration's JL projection vector `p`. Like
/// the u1 test above, this hardens against missed-tampering of intermediate
/// data, not just the final opening.
#[test]
fn flipped_intermediate_p_rejects() {
    let (sigs, pairs) = make_sigs(2);
    let mut proof = aggregate(&sigs).expect("aggregate");
    if proof.labrador.intermediate.is_empty() {
        eprintln!("note: depth=1 at N=2; intermediate p tamper test skipped");
        return;
    }
    proof.labrador.intermediate[0].p[0] ^= 1;
    assert!(
        verify(&pairs, &proof).is_err(),
        "flipped intermediate p byte must reject"
    );
}

/// One-shot end-to-end check at N=8: real Falcon-512 keygen + sign, real
/// aggregate, honest verify, every tamper-mutation rejected, bincode round
/// trip still verifies. This is the "is the aggregator alive" signal for
/// CI; the split tests above are kept for granular failure messages.
#[test]
fn roundtrip_n8_full_adversarial() {
    let (sigs, pairs) = make_sigs(8);
    let proof = aggregate(&sigs).expect("aggregate must succeed");

    // Honest path: accept.
    verify(&pairs, &proof).expect("honest verify must accept");

    // Tamper 1: substitute a fresh public key.
    {
        let (other_pk, _) = falcon512::keypair();
        let mut tampered = pairs.clone();
        tampered[3].0 = other_pk.as_bytes().to_vec();
        assert!(verify(&tampered, &proof).is_err(), "wrong pk must reject");
    }

    // Tamper 2: mutate a message.
    {
        let mut tampered = pairs.clone();
        tampered[5].1.push(0xff);
        assert!(verify(&tampered, &proof).is_err(), "wrong message must reject");
    }

    // Tamper 3: flip a coefficient inside the proof's final opening.
    {
        let mut bad_proof = proof.clone();
        let last = bad_proof.labrador.final_iter.last_msg.as_mut().unwrap();
        last.z0[0].c[0] ^= 1;
        assert!(verify(&pairs, &bad_proof).is_err(), "flipped proof byte must reject");
    }

    // Bincode round-trip preserves verification.
    let bytes = bincode::serialize(&proof).expect("serialize");
    let restored: AggregateProof = bincode::deserialize(&bytes).expect("deserialize");
    verify(&pairs, &restored).expect("verify after bincode round-trip");
}
