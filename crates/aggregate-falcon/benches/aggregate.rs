//! Stable wall-clock anchor for the aggregator across N.
//!
//! By default benches N ∈ {8, 32}. Set `LABRADOR_BENCH_N` to a comma-separated
//! list to override, e.g. `LABRADOR_BENCH_N=8,32,64,128`. Sample size is held
//! to the criterion minimum (10) and measurement time stretched so the larger
//! sizes don't trigger criterion's "could not complete" warning.
//!
//! Usage:
//!   cargo bench -p aggregate-falcon -- --save-baseline pre_t1
//!   cargo bench -p aggregate-falcon -- --baseline pre_t1

use aggregate_falcon::{aggregate, verify, FalconInstance, PublicPair};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pqcrypto_falcon::falcon512 as pqf;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey as PqPublicKey};
use std::time::Duration;

fn make_instances(n: usize) -> (Vec<FalconInstance>, Vec<PublicPair>) {
    let mut instances = Vec::with_capacity(n);
    let mut pairs = Vec::with_capacity(n);
    for i in 0..n {
        let msg = format!("bench msg #{i}").into_bytes();
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
    (instances, pairs)
}

fn bench_sizes() -> Vec<usize> {
    if let Ok(s) = std::env::var("LABRADOR_BENCH_N") {
        return s
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect();
    }
    vec![8, 32]
}

fn bench_aggregate(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregate");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(3));

    for n in bench_sizes() {
        // Larger N needs more measurement headroom — proof time grows as O(N²).
        if n >= 64 {
            group.measurement_time(Duration::from_secs(600));
        }
        let (instances, _pairs) = make_instances(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("aggregate", n), &instances, |b, sigs| {
            b.iter(|| {
                let proof = aggregate(sigs).expect("aggregate");
                criterion::black_box(proof);
            });
        });
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for n in bench_sizes() {
        let (instances, pairs) = make_instances(n);
        let proof = aggregate(&instances).expect("aggregate");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("verify", n), &(pairs, proof), |b, (p, pr)| {
            b.iter(|| {
                verify(p, pr).expect("verify");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_aggregate, bench_verify);
criterion_main!(benches);
