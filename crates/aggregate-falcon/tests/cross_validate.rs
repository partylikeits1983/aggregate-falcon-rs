//! Cross-validation: signatures produced by `pqcrypto-falcon` (FFI to the
//! audited Falcon-512 C reference impl) must independently verify under
//! `falcon-rust` v0.1.2 (pure-Rust Falcon-512 by aszepieniec). If both
//! independent implementations agree, the bytes we feed into our aggregation
//! pipeline are genuine Falcon-512 signatures — not just numbers that satisfy
//! our own decoder.
//!
//! # Encoding bridge
//!
//! pqcrypto-falcon emits the variable-length "compressed" Falcon-512 detached
//! signature format (header 0x39, ~653 B). falcon-rust expects the
//! fixed-length "standard" format (header 0x59, exactly 666 B with trailing
//! bit-zero padding). Both use the same per-coefficient compression
//! (sign-bit | 7 low bits | unary high bits), so the conversion is purely a
//! header byte swap plus zero-padding. We do NOT touch the cryptographic
//! payload; falcon-rust runs its own math against the decompressed s2.

use pqcrypto_falcon::falcon512 as pqf;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey as PqPublicKey};

const FR_SIG_BYTELEN: usize = 666;

/// Re-encode a pqcrypto-falcon detached sig (variable-length, header 0x39)
/// into falcon-rust's fixed-length 666-byte standard format (header 0x59).
fn pq_to_fr_sig_bytes(pq_sig: &[u8]) -> Vec<u8> {
    assert!(
        pq_sig.len() <= FR_SIG_BYTELEN,
        "pqcrypto-falcon sig of {} B exceeds falcon-rust's 666 B buffer",
        pq_sig.len()
    );
    let mut out = vec![0u8; FR_SIG_BYTELEN];
    out[..pq_sig.len()].copy_from_slice(pq_sig);
    out[0] = 0x59;
    out
}

/// Verify a pqcrypto-falcon sig under falcon-rust. Returns true iff
/// falcon-rust's `verify()` accepts.
pub fn falcon_rust_verify(pk_bytes: &[u8], sig_bytes: &[u8], msg: &[u8]) -> bool {
    use falcon_rust::falcon512 as fr;
    let pk = match fr::PublicKey::from_bytes(pk_bytes) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let converted = pq_to_fr_sig_bytes(sig_bytes);
    let sig = match fr::Signature::from_bytes(&converted) {
        Ok(s) => s,
        Err(_) => return false,
    };
    fr::verify(msg, &sig, &pk)
}

#[test]
fn every_pqcrypto_sig_is_accepted_by_falcon_rust() {
    // Try several N to catch any state-leak / nonce-handling issues.
    for n in [1usize, 2, 8] {
        for i in 0..n {
            let msg = format!("cross-validate msg #{i} of {n}").into_bytes();
            let (pk, sk) = pqf::keypair();
            let sig = pqf::detached_sign(&msg, &sk);

            // Self-check: pqcrypto-falcon accepts its own signature.
            assert!(
                pqf::verify_detached_signature(&sig, &msg, &pk).is_ok(),
                "pqcrypto-falcon failed to verify its own sig (n={n}, i={i})"
            );

            // Cross-check: an independent Rust impl also accepts it.
            assert!(
                falcon_rust_verify(pk.as_bytes(), sig.as_bytes(), &msg),
                "falcon-rust REJECTED a genuine pqcrypto-falcon sig (n={n}, i={i}) \
                 — the two impls disagree on validity, which means our \
                 aggregation inputs are not what we think they are"
            );
        }
    }
}

#[test]
fn tampered_sig_is_rejected_by_both_impls() {
    let msg = b"cross-validate tamper-test message".to_vec();
    let (pk, sk) = pqf::keypair();
    let sig = pqf::detached_sign(&msg, &sk);

    // Sanity: honest sig is accepted by both.
    assert!(pqf::verify_detached_signature(&sig, &msg, &pk).is_ok());
    assert!(falcon_rust_verify(pk.as_bytes(), sig.as_bytes(), &msg));

    // Flip a byte in the compressed s2 region (well past the nonce, which
    // ends at byte 41) — both verifiers should reject.
    let mut bad = sig.as_bytes().to_vec();
    bad[100] ^= 1;
    let bad_sig = pqf::DetachedSignature::from_bytes(&bad)
        .expect("DetachedSignature::from_bytes accepts arbitrary bytes of correct length");

    let pq_accepts = pqf::verify_detached_signature(&bad_sig, &msg, &pk).is_ok();
    let fr_accepts = falcon_rust_verify(pk.as_bytes(), &bad, &msg);

    assert!(
        !pq_accepts && !fr_accepts,
        "tampered sig must be rejected by both impls, but pq={pq_accepts} fr={fr_accepts}"
    );
}

#[test]
fn tampered_message_is_rejected_by_falcon_rust() {
    // A different message → wrong `c = HashToPoint(salt||msg')` → falcon-rust
    // rejects. Confirms falcon-rust runs the real verification math, not just
    // a structural check on the bytes.
    let msg = b"the real message".to_vec();
    let (pk, sk) = pqf::keypair();
    let sig = pqf::detached_sign(&msg, &sk);

    assert!(falcon_rust_verify(pk.as_bytes(), sig.as_bytes(), &msg));
    let wrong = b"a different message".to_vec();
    assert!(!falcon_rust_verify(pk.as_bytes(), sig.as_bytes(), &wrong));
}
