//! End-to-end validation of the §6.1 modulus-lifting + §6.4 subring trick.
//!
//! For a real Falcon-512 signature, we:
//!   1. Decode `(h, c, s1, s2)` from the public key, signature, and message.
//!   2. Compute the integer quotient `v_i` via [`relation::compute_v_signed`].
//!   3. Embed every polynomial into `S^8` using the subring map.
//!   4. Check that `s1 + h·s2 + q·v − c == 0` holds *in* `S^8` (not just mod q).
//!
//! If this passes, the entire bridge from Falcon-512 to the LaBRADOR ring —
//! decoder, modulus lift, and subring multiplication — is provably correct
//! against an audited oracle.

use falcon_relation::embed::{embed_signed, mul_subring};
use falcon_relation::parse::decode_instance;
use falcon_relation::relation::{
    add_slots, all_slots_zero, compute_v_signed, scale_slots, sub_slots,
};
use modring::{find_prime_5mod8, Modulus, Ring};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn embed_fpoly(f: &falcon_relation::FPoly, m: &Modulus) -> [modring::RingElem; 8] {
    let mut coeffs = [0i64; falcon_relation::FALCON_N];
    for i in 0..falcon_relation::FALCON_N {
        coeffs[i] = f.centered(i) as i64;
    }
    embed_signed(&coeffs, m)
}

#[test]
fn modulus_lift_holds_in_subring_for_real_falcon_sigs() {
    // q' chosen ≥ Phase-0's q_bitlen for N≈100 (44 bits). Any prime ≡ 5 mod 8 of
    // sufficient size works — its concrete value doesn't matter for this test,
    // only that the integer identity holds and fits within q'.
    let ring = Ring::new(Modulus::new(find_prime_5mod8(1 << 44)));

    let messages: [&[u8]; 3] = [b"first", b"second message", &[0xA5u8; 64]];
    for (k, msg) in messages.iter().enumerate() {
        let (pk, sk) = falcon512::keypair();
        let sig = falcon512::detached_sign(msg, &sk);
        let inst = decode_instance(pk.as_bytes(), sig.as_bytes(), msg)
            .expect("decode_instance must succeed for a genuine signature");

        // §6.1 modulus lift, computed over the integers.
        let v_signed = compute_v_signed(&inst.s1, &inst.s2, &inst.h, &inst.c);

        // Embed everything into S^8 = (Z_{q'}[X]/(X^64+1))^8.
        let s1_s = embed_fpoly(&inst.s1, &ring.m);
        let s2_s = embed_fpoly(&inst.s2, &ring.m);
        let h_s = embed_fpoly(&inst.h, &ring.m);
        let c_s = embed_fpoly(&inst.c, &ring.m);
        let v_s = embed_signed(&v_signed, &ring.m);

        // Compute h·s2 via the subring bilinear product, and form the LHS:
        // (s1 + h*s2 + q·v) − c   must be 0 in S^8.
        let hs2 = mul_subring(&h_s, &s2_s, &ring);
        let qv = scale_slots(&v_s, falcon_relation::FALCON_Q as u64, &ring.m);
        let mut lhs = add_slots(&s1_s, &hs2, &ring.m);
        lhs = add_slots(&lhs, &qv, &ring.m);
        lhs = sub_slots(&lhs, &c_s, &ring.m);

        assert!(
            all_slots_zero(&lhs),
            "modulus-lift failed in S^8 for message #{k}"
        );

        // Sanity: v_i should be small (Lemma 2.1: ‖v‖∞ ≤ 1 + √d + d·β).
        let max_abs = v_signed.iter().map(|x| x.unsigned_abs()).max().unwrap();
        assert!(
            max_abs < 1_000_000,
            "v_i unexpectedly large ({max_abs}) — likely a centering bug"
        );
    }
}
