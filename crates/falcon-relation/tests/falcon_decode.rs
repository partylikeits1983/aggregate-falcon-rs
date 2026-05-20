//! End-to-end validation of the Falcon-512 decoder against the audited
//! `pqcrypto-falcon` implementation.
//!
//! The decisive oracle is the *norm bound*: `s1` is recovered as
//! `c - s2·h`, so `s1 + s2·h = c` holds by construction and proves nothing.
//! But `||(s1, s2)||^2 <= beta^2` holds **only if** every decoding step —
//! public-key unpacking, signature decompression, and especially
//! `HashToPoint` — is correct. A wrong message (hence wrong `c`) makes the
//! recovered `s1` large, which the negative test confirms.

use falcon_relation::falcon_ring::FALCON_BETA_SQ;
use falcon_relation::parse::decode_instance;
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn norm_sq(sig: &falcon_relation::FalconSig) -> i64 {
    sig.s1.norm_sq() + sig.s2.norm_sq()
}

#[test]
fn genuine_signatures_decode_within_norm_bound() {
    let messages: [&[u8]; 4] = [
        b"",
        b"hello falcon",
        b"a slightly longer message for aggregation testing",
        &[0xABu8; 137],
    ];
    for (k, msg) in messages.iter().enumerate() {
        let (pk, sk) = falcon512::keypair();
        let sig = falcon512::detached_sign(msg, &sk);

        // Sanity: the audited implementation accepts its own signature.
        assert!(
            falcon512::verify_detached_signature(&sig, msg, &pk).is_ok(),
            "pqcrypto verify failed for message {k}"
        );

        let decoded = decode_instance(pk.as_bytes(), sig.as_bytes(), msg)
            .expect("decode_instance should succeed on a genuine signature");

        // Verification equation holds (trivially, by construction of s1).
        assert_eq!(decoded.c, decoded.s1.add(&decoded.s2.mul(&decoded.h)));

        // The real oracle: a correct decode + HashToPoint yields a short s1.
        let n = norm_sq(&decoded);
        assert!(
            n <= FALCON_BETA_SQ,
            "message {k}: norm^2 {n} exceeds bound {FALCON_BETA_SQ} \
             — decoder or HashToPoint is wrong"
        );
    }
}

#[test]
fn wrong_message_breaks_the_norm_bound() {
    let (pk, sk) = falcon512::keypair();
    let sig = falcon512::detached_sign(b"the real message", &sk);

    let decoded = decode_instance(pk.as_bytes(), sig.as_bytes(), b"a different message")
        .expect("decoding still structurally succeeds");

    // With the wrong message, c is wrong, so s1 = c - s2*h is large.
    assert!(
        norm_sq(&decoded) > FALCON_BETA_SQ,
        "wrong message should blow past the norm bound"
    );
}

#[test]
fn corrupt_headers_are_rejected() {
    let (pk, sk) = falcon512::keypair();
    let sig = falcon512::detached_sign(b"m", &sk);

    let mut bad_pk = pk.as_bytes().to_vec();
    bad_pk[0] ^= 0xFF;
    assert!(decode_instance(&bad_pk, sig.as_bytes(), b"m").is_err());

    let mut bad_sig = sig.as_bytes().to_vec();
    bad_sig[0] ^= 0xFF;
    assert!(decode_instance(pk.as_bytes(), &bad_sig, b"m").is_err());

    assert!(decode_instance(&pk.as_bytes()[..10], sig.as_bytes(), b"m").is_err());
}
