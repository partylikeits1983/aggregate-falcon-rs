use labrador::params::Params;
fn main() {
    for n in &[2usize, 4, 8] {
        let p = Params::for_n(*n);
        println!(
            "=== N={n} depth={} q_bitlen={} q_estimator={} (2^{:.2}) ===",
            p.depth,
            p.q_bitlen,
            p.q_value_estimator,
            (p.q_value_estimator as f64).log2()
        );
        for (k, it) in p.iterations.iter().enumerate() {
            let b1t1 = (it.b1 as f64).log2() * it.t1 as f64;
            println!(
                "iter[{k}]: stage={:?} n={} r_list={:?} b={} t={} b1={} t1={} (b1^t1≈2^{:.0}) b2={} t2={} kappa={} kappa1={} prevnu={} prevmu={} logq={}",
                it.stage, it.n, it.r_list, it.b, it.t,
                it.b1, it.t1, b1t1,
                it.b2, it.t2, it.kappa, it.kappa1, it.prev_nu, it.prev_mu, it.logq
            );
        }
    }
}
