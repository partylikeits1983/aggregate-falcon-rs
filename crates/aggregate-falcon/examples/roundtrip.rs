//! Driver example: aggregate N Falcon-512 signatures, verify, report sizes.
//!
//! Independently verifies every signature under two implementations
//! (`pqcrypto-falcon` C ref impl via FFI, and `falcon-rust` pure-Rust) before
//! aggregating, then reports whether the aggregate proof actually beats naive
//! concatenation of the raw signatures.
//!
//!     cargo run --release --example roundtrip [N]

use aggregate_falcon::{
    aggregate_depth, aggregate_with_progress, verify, FalconInstance, Progress, PublicPair,
};
use indicatif::{ProgressBar, ProgressStyle};
use pqcrypto_falcon::falcon512 as pqf;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey as PqPublicKey};
use std::env;
use std::time::{Duration, Instant};

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

/// indicatif-backed progress reporter for `aggregate_with_progress`. Drives a
/// single bar of width `depth` (= LaBRADOR iteration count) with a spinner
/// and elapsed timer. Each LaBRADOR iteration takes ~10 s at N=128, so the
/// per-iteration tick is fine-grained enough to feel live without flooding
/// stderr.
struct CliProgress {
    bar: ProgressBar,
    started_at: Instant,
}

impl CliProgress {
    fn new(depth: usize) -> Self {
        let bar = ProgressBar::new(depth as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "  {spinner:.cyan} [{elapsed_precise}] [{bar:24.cyan/blue}] {pos}/{len}  {msg}",
            )
            .expect("static template parses")
            .progress_chars("=>-"),
        );
        bar.enable_steady_tick(Duration::from_millis(120));
        bar.set_message("waiting…");
        Self { bar, started_at: Instant::now() }
    }

    fn finish(self) -> Duration {
        self.bar.finish_and_clear();
        self.started_at.elapsed()
    }
}

impl Progress for CliProgress {
    fn iter_start(&mut self, k: usize, depth: usize, label: &str) {
        self.bar.set_message(format!("iter {}/{}  {}", k + 1, depth, label));
    }
    fn iter_done(&mut self, k: usize) {
        self.bar.set_position((k + 1) as u64);
    }
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
    let depth = aggregate_depth(n);
    println!("\naggregating (depth={depth} LaBRADOR iterations)…");
    let mut bar = CliProgress::new(depth);
    let proof = aggregate_with_progress(&instances, &mut bar).expect("aggregate");
    let agg_elapsed = bar.finish();
    println!("aggregate:               {:.2?}", agg_elapsed);

    let proof_bytes = bincode::serialize(&proof).expect("serialize");
    let proof_size = proof_bytes.len();
    println!("aggregate proof:         {}", fmt_kb(proof_size));

    // Per-field byte breakdown — see which fields dominate the proof size so
    // future encoding work (bit-packing RingElem, Gaussian-entropy coding z/g,
    // upper-triangle stripping of g/h) can target the biggest movers.
    let b = proof.breakdown();
    let bd_total = b.total();
    let row = |label: &str, n: usize| {
        let pct = if bd_total > 0 { (n as f64 / bd_total as f64) * 100.0 } else { 0.0 };
        println!("    {label:<26} {n:>8} B  ({pct:>5.1}%)");
    };
    println!("  proof byte breakdown (bincode wire layout):");
    println!("    -- intermediate iters --");
    row("u1 (outer-commit v/g)", b.intermediate_u1);
    row("u2 (outer-commit h)", b.intermediate_u2);
    row("p  (JL projection)", b.intermediate_p);
    row("b''(aggregation const)", b.intermediate_bpp);
    println!("    -- final iter --");
    row("u1", b.final_u1);
    row("u2", b.final_u2);
    row("p", b.final_p);
    row("b''", b.final_bpp);
    row("z0 (amortized opening)", b.final_z0);
    row("z1 (amortized opening)", b.final_z1);
    row("v  (inner commits)", b.final_v);
    row("g  (quadratic garbage)", b.final_g);
    row("h  (linear garbage)", b.final_h);
    row("breakdown total", bd_total);

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
            "  EFFICIENCY:                ✗ aggregation here LOSES to concatenation at this N."
        );
        println!(
            "                             The v2 recursive proof is ~constant in size; the"
        );
        println!(
            "                             analytical crossover is at N ≈ 1024 (see"
        );
        println!(
            "                             aggregate-falcon/tests/size_beats_concat.rs)."
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
