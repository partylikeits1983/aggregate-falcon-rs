//! Driver example: aggregate N Falcon-512 signatures, verify, report sizes.
//!
//! Independently verifies every signature under two implementations
//! (`pqcrypto-falcon` C ref impl via FFI, and `falcon-rust` pure-Rust) before
//! aggregating, then reports whether the aggregate proof actually beats naive
//! concatenation of the raw signatures.
//!
//!     cargo run --release --example roundtrip [N]

use aggregate_falcon::{aggregate, verify, FalconInstance, PublicPair};
use pqcrypto_falcon::falcon512 as pqf;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey as PqPublicKey};
use std::env;
use std::time::Instant;

const FR_SIG_BYTELEN: usize = 666;

/// Convert pqcrypto-falcon's variable-length detached sig (header 0x39) to
/// falcon-rust's fixed-length standard sig (header 0x59, padded to 666 B).
fn pq_to_fr_sig_bytes(pq_sig: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; FR_SIG_BYTELEN];
    out[..pq_sig.len()].copy_from_slice(pq_sig);
    out[0] = 0x59;
    out
}

fn falcon_rust_verify(pk_bytes: &[u8], sig_bytes: &[u8], msg: &[u8]) -> bool {
    use falcon_rust::falcon512 as fr;
    let Ok(pk) = fr::PublicKey::from_bytes(pk_bytes) else {
        return false;
    };
    let converted = pq_to_fr_sig_bytes(sig_bytes);
    let Ok(sig) = fr::Signature::from_bytes(&converted) else {
        return false;
    };
    fr::verify(msg, &sig, &pk)
}

fn fmt_kb(n: usize) -> String {
    format!("{} B ({:.2} KB)", n, n as f64 / 1024.0)
}

fn main() {
    let n: usize = env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    println!("Aggregating N={n} Falcon-512 signatures…\n");

    // --- keygen + sign ---
    let t_sign = Instant::now();
    let mut instances: Vec<FalconInstance> = Vec::with_capacity(n);
    let mut pairs: Vec<PublicPair> = Vec::with_capacity(n);
    for i in 0..n {
        let msg = format!("example msg #{i}").into_bytes();
        let (pk, sk) = pqf::keypair();
        let sig = pqf::detached_sign(&msg, &sk);
        let pk_bytes = pk.as_bytes().to_vec();
        let sig_bytes = sig.as_bytes().to_vec();
        pairs.push((pk_bytes.clone(), msg.clone()));
        instances.push(FalconInstance {
            public_key: pk_bytes,
            message: msg,
            signature: sig_bytes,
        });
    }
    let sign_elapsed = t_sign.elapsed();

    // --- size accounting ---
    let pk_total: usize = instances.iter().map(|i| i.public_key.len()).sum();
    let msg_total: usize = instances.iter().map(|i| i.message.len()).sum();
    let sig_total: usize = instances.iter().map(|i| i.signature.len()).sum();
    let raw_total = pk_total + msg_total + sig_total;
    let pk_one = instances[0].public_key.len();
    let sig_avg = sig_total as f64 / n as f64;

    println!("per-item sizes:");
    println!("  pk:        {} B   (Falcon-512: header 0x09 + 896-byte 14-bit packed h)", pk_one);
    println!("  sig:       avg {:.1} B   (Falcon-512 compressed detached, variable)", sig_avg);
    println!("  msg:       {} B   (test fixture)", instances[0].message.len());

    println!("\ntotals over N={n}:");
    println!("  Σ|pkᵢ|:                {}", fmt_kb(pk_total));
    println!("  Σ|sigᵢ|:               {}   ← what aggregation replaces", fmt_kb(sig_total));
    println!("  Σ|msgᵢ|:               {}", fmt_kb(msg_total));
    println!("  Σ|pkᵢ|+|msgᵢ|+|sigᵢ|: {}", fmt_kb(raw_total));

    println!("\nkeygen + sign:           {:.2?}", sign_elapsed);

    // --- independent verifications (per sig) ---
    let t_pq = Instant::now();
    let mut pq_ok = 0usize;
    for inst in &instances {
        let sig = pqf::DetachedSignature::from_bytes(&inst.signature).expect("decode pq sig");
        let pk = pqf::PublicKey::from_bytes(&inst.public_key).expect("decode pq pk");
        if pqf::verify_detached_signature(&sig, &inst.message, &pk).is_ok() {
            pq_ok += 1;
        }
    }
    let pq_elapsed = t_pq.elapsed();
    println!(
        "  {} {}/{} sigs verified by pqcrypto-falcon (audited C ref impl)  [{:.2?}]",
        if pq_ok == n { "✓" } else { "✗" },
        pq_ok,
        n,
        pq_elapsed
    );

    let t_fr = Instant::now();
    let mut fr_ok = 0usize;
    for inst in &instances {
        if falcon_rust_verify(&inst.public_key, &inst.signature, &inst.message) {
            fr_ok += 1;
        }
    }
    let fr_elapsed = t_fr.elapsed();
    println!(
        "  {} {}/{} sigs verified by falcon-rust v0.1.2 (independent Rust impl)  [{:.2?}]",
        if fr_ok == n { "✓" } else { "✗" },
        fr_ok,
        n,
        fr_elapsed
    );

    if pq_ok != n || fr_ok != n {
        eprintln!(
            "\n  ⚠ at least one impl rejected a signature — aborting before aggregation."
        );
        std::process::exit(1);
    }

    // --- aggregate + verify ---
    println!();
    let t_agg = Instant::now();
    let proof = aggregate(&instances).expect("aggregate");
    let agg_elapsed = t_agg.elapsed();
    println!("aggregate:               {:.2?}", agg_elapsed);

    let proof_bytes = bincode::serialize(&proof).expect("serialize");
    let proof_size = proof_bytes.len();
    println!("aggregate proof:         {}", fmt_kb(proof_size));

    let t_ver = Instant::now();
    verify(&pairs, &proof).expect("verify");
    let ver_elapsed = t_ver.elapsed();
    println!("verify:                  {:.2?}  ✓", ver_elapsed);

    // --- size analysis: does aggregation actually save bytes? ---
    println!("\nsize analysis (the headline question — does aggregation beat concatenation?):");
    let ratio = proof_size as f64 / sig_total as f64;
    let delta_sigs = proof_size as i64 - sig_total as i64;
    let delta_pct = (delta_sigs as f64 / sig_total as f64) * 100.0;
    println!(
        "  proof / Σ|sigᵢ|:           {:.2}× ({}{:.1}% vs concatenated sigs)",
        ratio,
        if delta_pct >= 0.0 { "+" } else { "" },
        delta_pct
    );
    let bytes_saved = -delta_sigs;
    if bytes_saved > 0 {
        println!(
            "  bytes saved vs raw sigs:  +{} B   ({:.1}× smaller)",
            bytes_saved,
            sig_total as f64 / proof_size as f64
        );
        println!("  EFFICIENCY:                ✓ aggregation beats naive concatenation");
    } else {
        println!(
            "  bytes WASTED vs raw sigs: {} B (proof is {}× LARGER than just shipping the sigs)",
            -bytes_saved, ratio.round() as u64
        );
        println!(
            "  EFFICIENCY:                ✗ aggregation is currently WORSE than concatenation."
        );
        println!(
            "                             v1 omits recursive folding (Phase 6 in HANDOFF.md);"
        );
        println!(
            "                             proof grows ~linearly in N. Folding is needed for the"
        );
        println!(
            "                             aggregate to be smaller than Σ|sigᵢ|."
        );
    }

    // Effective break-even: at what hypothetical proof size would aggregation
    // become strictly better than concatenation? Just a friendly reference.
    println!(
        "\n  for reference: at this N, the proof would need to be ≤ {} ({:.2} KB) to beat\n  naive sig concatenation; the v1 proof is {:.2} KB.",
        sig_total,
        sig_total as f64 / 1024.0,
        proof_size as f64 / 1024.0
    );

    println!("\nDone.");
}
