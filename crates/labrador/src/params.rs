//! LaBRADOR aggregate-proof parameter table for Falcon-512 / two-splitting /
//! 128-bit security. Ported from `tools/phase0_params.py` line-by-line so the
//! Rust output matches the paper estimator's golden values.
//!
//! Entry point: [`Params::for_n`]. Each `Params` lists the iterations the
//! prover and verifier walk through in lock-step, with the per-iteration
//! decomposition bases (`b, b1, b2`), part counts (`t, t1, t2`), Ajtai ranks
//! (`κ, κ₁`), Gaussian widths (`σz, σh`), folding parameters from the
//! previous iteration (`prev_nu, prev_mu`), and the next iteration's norm
//! bound (`next_beta_list`).

use self::falcon::{FALCON_64_128, CHAL_2_SPLIT_64_128};

const SCAL: usize = 8;
const MAX_DEPTH: usize = 15;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    First,
    Mid,
    SecLast,
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::First => "FIRST",
            Stage::Mid => "MID",
            Stage::SecLast => "SECLAST",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Iteration {
    pub stage: Stage,
    pub n: usize,
    pub r_list: Vec<u64>,
    pub beta_list: Vec<f64>,
    pub beta: f64,
    pub logq: u32,
    pub q_: u64,
    pub b: u64,
    pub t: u64,
    pub b1: u64,
    pub t1: u64,
    pub b2: u64,
    pub t2: u64,
    pub kappa: u64,
    pub kappa1: u64,
    pub m: f64,
    pub sigs: Vec<f64>,
    pub sigz: f64,
    pub sigh: f64,
    pub prev_nu: u64,
    pub prev_mu: u64,
    pub next_beta_list: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub num_sigs: usize,
    pub q_bitlen: u32,
    pub q_value_estimator: u64,
    pub parallel_reps: u32,
    pub depth: usize,
    pub iterations: Vec<Iteration>,
}

impl Params {
    /// Build the per-`N` parameter table by searching depth ∈ [1, MAX_DEPTH]
    /// for the proof-size minimum, exactly as the Python estimator does.
    pub fn for_n(num_sigs: usize) -> Self {
        let falcon = FALCON_64_128;
        let chal = CHAL_2_SPLIT_64_128;
        let (q_, n, r_list, beta_list) = get_initial_params(num_sigs, &falcon, &chal, SCAL);
        let it0 = Iteration::new(
            q_,
            falcon.d,
            falcon.jl_slack,
            n,
            r_list,
            beta_list,
            &chal,
            falcon.secparam,
            &falcon.kappa_lim,
            Stage::First,
            1,
            1,
        );

        let mut best_iters: Option<(Vec<Iteration>, f64, usize)> = None;
        for depth in 1..=MAX_DEPTH {
            let (iters, total) = recursion_to_depth_no_last_opt(it0.clone(), depth, &chal, &falcon);
            match &best_iters {
                None => best_iters = Some((iters, total, depth)),
                Some((_, best_total, _)) if total < *best_total => {
                    best_iters = Some((iters, total, depth));
                }
                _ => {}
            }
        }
        let (iters, _, depth) = best_iters.expect("at least one depth attempted");

        let q_bitlen = ceil_log2(q_);
        let parallel_reps: u32 = if chal.rep {
            // Mirror python's `parallel_reps`: smallest k with secparam < log2(k/d) + k * log2(q_).
            let mut found = 21u32;
            for k in 1..20 {
                let lhs = falcon.secparam as f64;
                let rhs =
                    (k as f64 / falcon.d as f64).log2() + (k as f64) * (q_ as f64).log2();
                if lhs < rhs {
                    found = k as u32;
                    break;
                }
            }
            found
        } else {
            1
        };

        Self {
            num_sigs,
            q_bitlen,
            q_value_estimator: q_,
            parallel_reps,
            depth,
            iterations: iters,
        }
    }

    /// Choose the LaBRADOR ring modulus `q'` for this `N`.
    ///
    /// Returns the largest prime `≡ 5 (mod 8)` strictly below `2^q_bitlen`,
    /// which guarantees each iteration's centered base-`b₁`/`b₂` decomposition
    /// satisfies `b^t ≥ 2^q_bitlen > q'` — lossless, as required for the
    /// recursion's fold to express `v_i = Σ_l b₁^l · v_i^(l)` as a linear
    /// constraint over the new witness.
    pub fn select_modulus(&self) -> u64 {
        modring::find_prime_5mod8_below(1u64 << self.q_bitlen)
    }

    /// Paper-correct initial norm bound `β² = β_init²`.
    ///
    /// Replaces the conservative `1<<60` hardcode used during early phases:
    /// once JL projection constraints tie `p` to `w`, this is the actual
    /// bound the verifier checks against.
    pub fn beta_init_sq(&self) -> i128 {
        let b = self.iterations[0].beta_list[0];
        (b * b).ceil() as i128
    }
}

impl Iteration {
    pub fn new(
        q_: u64,
        d: u32,
        slack: f64,
        n: usize,
        r_list: Vec<u64>,
        beta_list: Vec<f64>,
        chal: &Chal,
        secparam: u32,
        kappa_lim: &[i64],
        stage: Stage,
        prev_nu: u64,
        prev_mu: u64,
    ) -> Self {
        let mut it = Iteration {
            stage,
            n,
            r_list,
            beta_list,
            beta: 0.0,
            logq: 0,
            q_,
            b: 0,
            t: 0,
            b1: 0,
            t1: 0,
            b2: 0,
            t2: 0,
            kappa: 0,
            kappa1: 0,
            m: 0.0,
            sigs: vec![],
            sigz: 0.0,
            sigh: 0.0,
            prev_nu,
            prev_mu,
            next_beta_list: vec![0.0, 0.0],
        };
        it.init_normal(d, slack, chal, secparam, kappa_lim);
        it
    }

    fn init_normal(
        &mut self,
        d: u32,
        slack: f64,
        chal: &Chal,
        _secparam: u32,
        kappa_lim: &[i64],
    ) {
        self.beta = l2norm(&self.beta_list);
        self.logq = ceil_log2(self.q_);

        let d_f = d as f64;
        let n_f = self.n as f64;
        self.sigs = (0..self.r_list.len())
            .map(|i| {
                self.beta_list[i] / (self.r_list[i] as f64 * n_f * d_f).sqrt()
            })
            .collect();

        let tau = chal.tau as f64;
        let sigs0 = self.sigs[0];
        let r0 = self.r_list[0] as f64;
        let mut sigz_sq = sigs0 * sigs0 * (1.0 + (r0 - 1.0) * tau);
        for i in 1..self.r_list.len() {
            sigz_sq += self.sigs[i] * self.sigs[i] * (self.r_list[i] as f64) * tau;
        }
        self.sigz = sigz_sq.sqrt();

        let max_sig = self.sigs.iter().cloned().fold(0.0_f64, f64::max);
        self.sigh = (2.0 * n_f * d_f).sqrt() * max_sig * max_sig;

        if self.stage == Stage::SecLast {
            self.t = 1;
            self.b = 1;
        } else {
            self.t = 2;
            self.b = (12.0_f64.sqrt() * self.sigz).sqrt().round() as u64;
        }

        let logq_f = self.logq as f64;
        let sqrt12 = 12.0_f64.sqrt();
        let mut t1 = (logq_f / (sqrt12 * self.sigz / self.b as f64).log2()).round() as i64;
        if t1 < 2 {
            t1 = 2;
        }
        if t1 > 14 {
            t1 = 14;
        }
        self.t1 = t1 as u64;
        self.b1 = (2.0_f64.powf(logq_f / self.t1 as f64)).ceil() as u64;

        let mut t2 = ((sqrt12 * self.sigh).ln()
            / (sqrt12 * self.sigz / self.b as f64).ln())
        .round() as i64;
        if t2 < 1 {
            t2 = 1;
        }
        self.t2 = t2 as u64;
        self.b2 = (sqrt12 * self.sigh).powf(1.0 / self.t2 as f64).ceil() as u64;

        let sumr: u64 = self.r_list.iter().sum();
        let sumr_f = sumr as f64;

        let nextbeta0 = self.sigz / self.b as f64 * (self.t as f64 * n_f * d_f).sqrt();
        self.next_beta_list[0] = nextbeta0;

        let b1_f = self.b1 as f64;
        let b2_f = self.b2 as f64;
        let t1_f = self.t1 as f64;
        let t2_f = self.t2 as f64;
        let beta_now = self.beta;
        let new_beta1_fun = |kappa: f64| -> f64 {
            (b1_f * b1_f / 12.0 * t1_f * sumr_f * kappa * d_f
                + (b1_f * b1_f * t1_f + b2_f * b2_f * t2_f) / 12.0
                    * (sumr_f * sumr_f + sumr_f)
                    / 2.0
                    * d_f)
                .sqrt()
        };
        let new_beta_fun =
            |kappa: f64| -> f64 { (nextbeta0 * nextbeta0 + new_beta1_fun(kappa).powi(2)).sqrt() };
        let kappa_norm = |kappa: f64| -> f64 {
            (6.0 * chal.t_op as f64 * self.b as f64 * slack * new_beta_fun(kappa))
                .max(2.0 * self.b as f64 * slack * new_beta_fun(kappa) + 4.0 * chal.t_op as f64 * slack * beta_now)
        };
        self.kappa = get_kappa(&kappa_norm, self.q_, kappa_lim);
        self.next_beta_list[1] = new_beta1_fun(self.kappa as f64);

        let kappa1_norm = |kappa: f64| -> f64 { 2.0 * slack * new_beta_fun(kappa) };
        self.kappa1 = get_kappa(&kappa1_norm, self.q_, kappa_lim);

        self.m = t1_f * sumr_f * self.kappa as f64
            + (t1_f + t2_f) * (sumr_f * sumr_f + sumr_f) / 2.0;
    }

    /// Build the next iteration's parameters per `next_it` in the Python script.
    pub fn next_it(
        &self,
        nu: u64,
        mu: u64,
        next_stage: Stage,
        d: u32,
        slack: f64,
        chal: &Chal,
        secparam: u32,
        kappa_lim: &[i64],
    ) -> Self {
        let reps = 1u64; // chal.rep is false for our parameter set.
        let n_new = ((reps as f64 * self.n as f64) / nu as f64).ceil() as usize;
        let m_new = (self.m / mu as f64).ceil() as usize;
        let n_ = n_new.max(m_new);
        let r_ = vec![self.t * nu, mu];
        Iteration::new(
            self.q_,
            d,
            slack,
            n_,
            r_,
            self.next_beta_list.clone(),
            chal,
            secparam,
            kappa_lim,
            next_stage,
            nu,
            mu,
        )
    }

    /// Mirror Python's `size_step()` for the proof-size estimator.
    fn size_step(&self, secparam: u32, chal_rep: bool) -> f64 {
        let numproj: f64 = if self.stage == Stage::First { 2.0 } else { 1.0 };
        let parallel_reps = if chal_rep { /* not used here */ 1.0 } else { 1.0 };
        let jl_proj = numproj * 2.0 * secparam as f64 * gaussian_entropy(self.beta / 2.0_f64.sqrt());
        let _ = parallel_reps;

        let jl_proof = ((secparam as f64 / self.logq as f64).ceil()) * 64.0 * self.logq as f64;
        // The estimator uses self.d (= 64 here) and self.logq.

        let numouter: f64 = if self.stage == Stage::SecLast { 0.0 } else { 2.0 };
        let outer_com = numouter * self.kappa1 as f64 * 64.0 * self.logq as f64;
        jl_proj + jl_proof + outer_com
    }

    fn size_ti(&self) -> f64 {
        let sumr: u64 = self.r_list.iter().sum();
        sumr as f64 * self.kappa as f64 * 64.0 * self.logq as f64
    }

    fn size_gij(&self) -> f64 {
        let r0 = self.r_list[0] as f64;
        (r0 * r0 + r0) / 2.0 * 64.0 * gaussian_entropy(self.sigh)
    }

    fn size_hij(&self) -> f64 {
        let r0 = self.r_list[0] as f64;
        (r0 * r0 + r0) / 2.0 * 64.0 * self.logq as f64
    }

    fn size_z(&self) -> f64 {
        self.n as f64 * 64.0 * gaussian_entropy(self.sigz)
    }

    fn size_lastmsg(&self) -> f64 {
        self.size_ti() + self.size_gij() + self.size_hij() + self.size_z()
    }
}

pub struct Chal {
    pub rho: u32,
    pub tau: u32,
    pub t_op: u32,
    pub rep: bool,
}

pub struct Falcon {
    pub d: u32,
    pub _q: u32,
    pub beta: u32,
    pub _bit_len: u32,
    pub secparam: u32,
    pub jl_const: u32,
    pub jl_slack: f64,
    pub kappa_lim: &'static [i64],
}

mod falcon {
    use super::{Chal, Falcon};

    pub const CHAL_2_SPLIT_64_128: Chal = Chal {
        rho: 2,
        tau: 86,       // ceil(172/2) = ceil(omega·gamma²/2) = ceil(43·4/2)
        t_op: 43,      // ceil(86/2) = ceil(omega·gamma/2) = ceil(43·2/2)
        rep: false,
    };

    pub const FALCON_64_128: Falcon = Falcon {
        d: 64,
        _q: 12289,
        beta: 5834,
        _bit_len: 5328 - 320,
        secparam: 128,
        jl_const: 120,
        jl_slack: KAPPA_SLACK_64,
        kappa_lim: &KAPPA_LIM_64_128,
    };

    // sqrt(128/30) precomputed to f64 — used as `slack` in init_normal.
    const KAPPA_SLACK_64: f64 = 2.066_398_059_435_344_5_f64;

    pub static KAPPA_LIM_64_128: [i64; 36] = [
        -1, 271, 2687, 16383, 73727, 524287, 917503, 2621439, 7340031,
        18874367, 50331647, 117440511, 251658239, 1073741823, 1207959551,
        2684354559, 5368709119, 10737418239, 21474836479, 68719476735,
        137438953471, 137438953471, 240518168575, 549755813887, 755914244095,
        1374389534719, 4398046511103, 4398046511103, 8796093022207,
        13194139533311, 35184372088831, 35184372088831, 52776558133247,
        87960930222079, 140737488355326, 140737488355326,
    ];
}

fn l2norm(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn ceil_log2(x: u64) -> u32 {
    if x <= 1 {
        return 0;
    }
    (x as f64).log2().ceil() as u32
}

/// Mirror `get_kappa(beta_fun, q_, kappa_lim)` from the Python estimator:
/// find the smallest index `i` such that `beta_fun(i) ≤ kappa_lim[i]`.
fn get_kappa(beta_fun: &dyn Fn(f64) -> f64, q_: u64, kappa_lim: &[i64]) -> u64 {
    let mut kappa: u64 = 0;
    let mut beta: f64 = 0.0;
    for i in 1..kappa_lim.len() {
        beta = beta_fun(i as f64);
        if beta <= kappa_lim[i] as f64 {
            kappa = i as u64;
            break;
        }
    }
    if beta >= q_ as f64 {
        panic!("Beta must be smaller than the LaBRADOR modulus.");
    }
    if kappa == 0 {
        panic!(
            "kappa_lim has no kappa for beta = {beta}; q_ = {q_}, table len = {}",
            kappa_lim.len()
        );
    }
    kappa
}

/// Mirror `get_initial_params` (returns `q_`, `n`, `r_list`, `beta_list`).
fn get_initial_params(
    num_sigs: usize,
    falcon: &Falcon,
    chal: &Chal,
    scal: usize,
) -> (u64, usize, Vec<u64>, Vec<f64>) {
    let n_sigs = num_sigs as f64;
    let fb = falcon.beta as f64;
    let fd = falcon.d as f64;

    let beta_sig_ell2 = n_sigs.sqrt() * 2.0 * fb;
    let beta_quotient_ell2 = n_sigs.sqrt() * (fb * fd + fd.sqrt() + 1.0);
    let beta_labrador = beta_sig_ell2 + beta_quotient_ell2;

    let sp = (falcon.secparam as f64).sqrt();
    let jl = falcon.jl_const as f64;
    let bound_cond_1_sig = sp * beta_sig_ell2 * jl;
    let bound_cond_1_quo = sp * beta_quotient_ell2 * jl;
    let bound_cond_2_sig = falcon.jl_slack.powi(2) * beta_sig_ell2.powi(2) * 4.0 * (fd + 2.0);
    let bound_cond_2_quo = falcon.jl_slack * beta_quotient_ell2 * 6.0 * falcon._q as f64;

    let q_bitlen = bound_cond_1_sig
        .log2()
        .max(bound_cond_1_quo.log2())
        .max(bound_cond_2_sig.log2())
        .max(bound_cond_2_quo.log2())
        .ceil() as u32;
    let q_: u64 = (1u128 << q_bitlen).saturating_sub(1) as u64;

    let n = scal * num_sigs;
    let r = 6 * (n_sigs.sqrt().ceil() as u64) + 1;
    let _ = chal;
    (q_, n, vec![r], vec![beta_labrador, 0.0])
}

/// Search for the `(nu, mu)` minimising `parallel_reps · 2 · n_new + m_new`,
/// matching `recursion_strategy_4` from the Python source.
fn recursion_strategy_4(
    it: &Iteration,
    next_stage: Stage,
    chal: &Chal,
    falcon: &Falcon,
) -> (u64, u64) {
    let res = 50u64;
    let mut best_nu = 1u64;
    let mut best_mu = 1u64;
    let candidate = it.next_it(1, 1, next_stage, falcon.d, falcon.jl_slack, chal, falcon.secparam, falcon.kappa_lim);
    let mut best_size = 1.0 * 2.0 * candidate.n as f64 + candidate.m;
    for nu in 1..res {
        for mu in 1..res {
            let cand = it.next_it(nu, mu, next_stage, falcon.d, falcon.jl_slack, chal, falcon.secparam, falcon.kappa_lim);
            let cand_size = 1.0 * 2.0 * cand.n as f64 + cand.m;
            if cand_size < best_size {
                best_size = cand_size;
                best_nu = nu;
                best_mu = mu;
            }
        }
    }
    (best_nu, best_mu)
}

fn recursion_to_depth_no_last_opt(
    initial_it: Iteration,
    depth: usize,
    chal: &Chal,
    falcon: &Falcon,
) -> (Vec<Iteration>, f64) {
    let mut iters = vec![initial_it.clone()];
    let mut it = initial_it;
    let mid_strategies = if depth <= 2 { 0 } else { depth - 2 };
    for _ in 0..mid_strategies {
        let (nu, mu) = recursion_strategy_4(&it, Stage::Mid, chal, falcon);
        it = it.next_it(nu, mu, Stage::Mid, falcon.d, falcon.jl_slack, chal, falcon.secparam, falcon.kappa_lim);
        iters.push(it.clone());
    }
    let (nu, mu) = recursion_strategy_4(&it, Stage::SecLast, chal, falcon);
    let seclast = it.next_it(nu, mu, Stage::SecLast, falcon.d, falcon.jl_slack, chal, falcon.secparam, falcon.kappa_lim);
    iters.push(seclast);

    let mut total: f64 = 0.0;
    for it in &iters {
        total += it.size_step(falcon.secparam, chal.rep);
    }
    total += iters.last().unwrap().size_lastmsg();
    (iters, total)
}

/// Mirror `gaussianentropy(sig)`: bits per Gaussian coefficient.
fn gaussian_entropy(sig: f64) -> f64 {
    let mut sig = sig;
    let mut a = 1.0_f64;
    if sig >= 4.0 {
        a = (sig / 2.0).floor();
        sig /= a;
    }
    let d_inv = 1.0 / (2.0 * sig * sig);
    let max_i = (15.0 * sig).ceil() as i64;
    let mut n: f64 = 0.0;
    for i in -max_i..0 {
        n += (-(i as f64) * (i as f64) * d_inv).exp();
    }
    n = 2.0 * n + 1.0;
    let logn = n.ln();
    let mut e: f64 = 0.0;
    for i in -max_i..0 {
        let f = (-(i as f64) * (i as f64) * d_inv).exp();
        e += f * (f.ln() - logn);
    }
    e = (-2.0 * e + logn) / (n * 2.0_f64.ln());
    e + a.log2()
}
