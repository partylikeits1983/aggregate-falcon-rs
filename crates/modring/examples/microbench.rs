//! Microbench the modular kernel and the NTT in isolation.
//! Used to confirm where the wall-clock actually goes in `aggregate()`.
//!
//!     cargo run --release -p modring --example microbench

use modring::modulus::{find_prime_ntt_friendly_below, Modulus};
use modring::crt::Ring;
use modring::poly::{RingElem, D};
use std::time::Instant;

fn rand_u64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed
}

fn rand_elem(m: &Modulus, seed: &mut u64) -> RingElem {
    let mut e = RingElem::zero();
    for i in 0..D {
        e.c[i] = rand_u64(seed) % m.q;
    }
    e
}

fn main() {
    // Use the same NTT-friendly prime range the prover picks (~2^44 for small N).
    let q = find_prime_ntt_friendly_below(1u64 << 44, 128);
    let m = Modulus::new(q);
    let r = Ring::new(m);
    println!("q = {q}  (q ≡ 1 mod 128: {})  NTT enabled: {}", (q - 1) % 128 == 0, r.ntt.is_some());

    let mut seed = 0xdead_beef_u64;
    let a: Vec<u64> = (0..1_000_000).map(|_| rand_u64(&mut seed) % q).collect();
    let b: Vec<u64> = (0..1_000_000).map(|_| rand_u64(&mut seed) % q).collect();

    // -------- Modulus::mul throughput --------
    let t = Instant::now();
    let mut acc: u64 = 0;
    for i in 0..a.len() {
        acc = acc.wrapping_add(m.mul(a[i], b[i]));
    }
    let dt = t.elapsed();
    std::hint::black_box(acc);
    println!("Modulus::mul × {}: {:.2?} ({:.1} ns/op)", a.len(), dt, dt.as_nanos() as f64 / a.len() as f64);

    // -------- Ring::mul throughput (full NTT roundtrip per call) --------
    let ea = rand_elem(&m, &mut seed);
    let eb = rand_elem(&m, &mut seed);
    let n_ring = 100_000;
    let t = Instant::now();
    let mut accum = RingElem::zero();
    for _ in 0..n_ring {
        let c = r.mul(&ea, &eb);
        accum.c[0] = accum.c[0].wrapping_add(c.c[0]);
    }
    let dt = t.elapsed();
    std::hint::black_box(accum);
    println!("Ring::mul (NTT) × {}: {:.2?} ({:.2} µs/op)", n_ring, dt, dt.as_nanos() as f64 / n_ring as f64 / 1000.0);

    // -------- Ring::add throughput (coord-wise) --------
    let n_add = 1_000_000;
    let mut ec = RingElem::zero();
    let t = Instant::now();
    for _ in 0..n_add {
        ec = ec.add(&m, &ea);
    }
    let dt = t.elapsed();
    std::hint::black_box(ec);
    println!("RingElem::add × {}: {:.2?} ({:.1} ns/op)", n_add, dt, dt.as_nanos() as f64 / n_add as f64);
}
