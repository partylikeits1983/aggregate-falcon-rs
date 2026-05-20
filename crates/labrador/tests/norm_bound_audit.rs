//! Per-iteration measurement of the honest prover's decomposed-norm² vs
//! `stmt.beta_sq` (loose; currently carried forward through fold) vs
//! `fold::beta_prime_sq` (tight; paper's estimator value).
//!
//! Drives `prove_v2 + fold` manually so we can see each iteration's openings.
//! Asserts the loose bound is met (a regression guard) and prints the tight
//! bound for visual comparison. Whether tight-bound enforcement is safe today
//! is decided after looking at the printed ratios:
//!   - If `measured ≤ tight` for every iter, the verifier's `next_beta_sq`
//!     could be tightened to `beta_prime_sq(it_params)`.
//!   - If `measured > tight` at any iter, the lossless overflow-into-last-chunk
//!     decomposition is still producing chunks above the estimator — wait for
//!     Session E's rejection sampling before tightening.

use falcon_relation::parse::decode_instance;
use falcon_relation::relation::build_falcon_statement;
use labrador::fold::{beta_prime_sq, fold_with_replay, FoldOutput};
use labrador::garbage::decompose;
use labrador::params::Params;
use labrador::prover_v2::prove_v2_with_replay;
use labrador::transcript::Transcript;
use modring::{Modulus, Ring};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

const N_SIGS: usize = 8;

#[test]
fn honest_proof_norm_bounds_loose_and_tight() {
    let params = Params::for_n(N_SIGS);
    let ring = Ring::new(Modulus::new(params.select_modulus()));
    let sigs: Vec<_> = (0..N_SIGS)
        .map(|i| {
            let msg = format!("norm-audit msg #{i}").into_bytes();
            let (pk, sk) = falcon512::keypair();
            let sig = falcon512::detached_sign(&msg, &sk);
            decode_instance(pk.as_bytes(), sig.as_bytes(), &msg)
                .expect("decode real Falcon sig")
        })
        .collect();

    let beta_sq = params.beta_init_sq();
    let (stmt0, witness0, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    let mut t = Transcript::new(b"norm-audit/v1");
    let mut cur_stmt = stmt0;
    let mut cur_witness = witness0;
    let depth = params.depth;
    eprintln!(
        "norm-bound audit at N={N_SIGS}, depth={depth} (beta_init_sq={beta_sq})"
    );
    eprintln!(
        "    {:>5}  {:>14}  {:>14}  {:>10}  {:>14}  {:>10}",
        "iter", "stmt.beta_sq", "measured", "/ loose", "beta_prime²", "/ tight"
    );

    for k in 0..depth {
        let it_params = &params.iterations[k];

        let (proof_k, replay_k) =
            prove_v2_with_replay(&cur_stmt, &cur_witness, it_params, &mut t);
        let last_msg = proof_k
            .last_msg
            .as_ref()
            .expect("prove_v2 must emit last_msg");

        // Decompose v, g, h exactly as the verifier does (verifier_v2.rs lines
        // 95-111 and 251-256). The chunks' centered coeffs are what u_1 / u_2
        // commit to and what the norm check sums.
        let m = &cur_stmt.ring.m;
        let r = cur_stmt.r;
        let n = cur_stmt.n;
        let kappa = it_params.kappa as usize;
        let b1 = it_params.b1;
        let b2 = it_params.b2;
        let t1 = it_params.t1 as usize;
        let t2 = it_params.t2 as usize;

        let mut measured: i128 = 0;
        // z0, z1
        for p in 0..n {
            measured += last_msg.z0[p].norm_sq(m);
            measured += last_msg.z1[p].norm_sq(m);
        }
        // v decomposed into t1 chunks base b1
        for i in 0..r {
            for p in 0..kappa {
                let chunks = decompose(&last_msg.v[i][p], m, b1, t1);
                for c in &chunks {
                    measured += c.norm_sq(m);
                }
            }
        }
        // g decomposed into t2 chunks base b2, h into t1 chunks base b1.
        for i in 0..r {
            for j in i..r {
                let g_chunks = decompose(&last_msg.g[i][j], m, b2, t2);
                for c in &g_chunks {
                    measured += c.norm_sq(m);
                }
                let h_chunks = decompose(&last_msg.h[i][j], m, b1, t1);
                for c in &h_chunks {
                    measured += c.norm_sq(m);
                }
            }
        }

        let loose = cur_stmt.beta_sq;
        let tight = beta_prime_sq(it_params);
        eprintln!(
            "    {:>5}  {:>14}  {:>14}  {:>10.3}  {:>14}  {:>10.3}",
            k,
            loose,
            measured,
            measured as f64 / loose as f64,
            tight,
            measured as f64 / tight.max(1) as f64,
        );

        // Regression guard: honest proofs must always satisfy the bound the
        // verifier actually checks against.
        assert!(
            measured <= loose,
            "iter {k}: honest measured norm² {measured} exceeds stmt.beta_sq {loose}"
        );

        // Advance to next iter unless this is the final one.
        if k + 1 < depth {
            let nu = params.iterations[k + 1].prev_nu as usize;
            let mu = params.iterations[k + 1].prev_mu as usize;
            let FoldOutput { statement, witness } =
                fold_with_replay(&cur_stmt, &proof_k, it_params, nu, mu, &replay_k);
            cur_stmt = statement;
            cur_witness = witness;
        }
    }
}
