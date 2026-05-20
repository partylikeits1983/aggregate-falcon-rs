//! Phase 6 Session D gate: recursion driver — `prove_aggregate` /
//! `verify_aggregate` round-trip through the full depth-K LaBRADOR pipeline.

use falcon_relation::parse::decode_instance;
use falcon_relation::relation::build_falcon_statement;
use labrador::aggregate_v2::{prove_aggregate, verify_aggregate};
use labrador::params::Params;
use modring::{Modulus, Ring};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn ring_for(n_sigs: usize) -> Ring {
    let params = Params::for_n(n_sigs);
    Ring::new(Modulus::new(params.select_modulus()))
}

fn fresh_sigs(n: usize) -> Vec<falcon_relation::parse::FalconSig> {
    (0..n)
        .map(|i| {
            let msg = format!("rec-v2 msg #{}", i).into_bytes();
            let (pk, sk) = falcon512::keypair();
            let sig = falcon512::detached_sign(&msg, &sk);
            decode_instance(pk.as_bytes(), sig.as_bytes(), &msg).unwrap()
        })
        .collect()
}

#[test]
fn iter1_prover_satisfies_check1_at_n2() {
    use labrador::commit::matmul;
    use labrador::fold::{fold, replay_iteration};
    use labrador::garbage::recompose;
    use labrador::prover_v2::prove_v2;
    use labrador::transcript::Transcript;
    use modring::RingElem;

    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let seed = b"check1-iter1/n2";

    // Run iter[0] prove_v2 + fold to get stmt_1, witness_1.
    let mut t_prove = Transcript::new(seed);
    let mut t_fold = t_prove.clone();
    let proof0 = prove_v2(&stmt, &witness, &params.iterations[0], &mut t_prove);
    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;
    let out0 = fold(&stmt, &proof0, &params.iterations[0], nu, mu, &mut t_fold);

    // Run iter[1] prove_v2.
    let proof1 = prove_v2(&out0.statement, &out0.witness, &params.iterations[1], &mut t_prove);

    // Replay iter[1]'s challenges with a fresh transcript matching the state
    // after fold0.
    let mut t_replay = t_fold.clone();
    let replay1 = replay_iteration(&out0.statement, &proof1, &params.iterations[1], &mut t_replay);

    // Recompose z and verify A·z = Σ c_i v_i at iter[1].
    let it1 = &params.iterations[1];
    let m = &ring.m;
    let n = out0.statement.n;
    let r = out0.statement.r;
    let kappa = it1.kappa as usize;
    let b = it1.b;
    let last = proof1.last_msg.as_ref().unwrap();
    let mut z: Vec<RingElem> = vec![RingElem::zero(); n];
    for k in 0..n {
        z[k] = recompose(&[last.z0[k].clone(), last.z1[k].clone()], m, b);
    }
    // Verify z = Σ c_i w_i directly (prover invariant).
    let cs = &replay1.cs;
    for k in 0..n {
        let mut want = RingElem::zero();
        for i in 0..r {
            let p = ring.mul(&cs[i], &out0.witness.w[i][k]);
            want = want.add(m, &p);
        }
        assert_eq!(z[k], want, "z = Σ c_i w_i fails at iter[1] position {k}");
    }
    println!("iter[1] prover-invariant z = Σ c_i w_i holds for all n={} positions", n);

    let az = matmul(&ring, &replay1.a_mat, &z);
    for k in 0..kappa {
        let mut want = RingElem::zero();
        for i in 0..r {
            let p = ring.mul(&cs[i], &last.v[i][k]);
            want = want.add(m, &p);
        }
        assert_eq!(az[k], want, "A·z = Σ c_i v_i fails at iter[1] row {k}");
    }
    println!("iter[1] verifier-Check1 A·z = Σ c_i v_i holds for all κ={} rows", kappa);
}

#[test]
fn g_decomposition_is_lossless_at_iter1_n2() {
    use labrador::fold::fold;
    use labrador::garbage::{decompose, recompose};
    use labrador::prover_v2::prove_v2;
    use labrador::transcript::Transcript;

    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let seed = b"loss-g/n2";

    let mut t_prove = Transcript::new(seed);
    let mut t_fold = t_prove.clone();
    let proof0 = prove_v2(&stmt, &witness, &params.iterations[0], &mut t_prove);
    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;
    let out = fold(&stmt, &proof0, &params.iterations[0], nu, mu, &mut t_fold);
    let proof1 = prove_v2(&out.statement, &out.witness, &params.iterations[1], &mut t_prove);

    let m = &ring.m;
    let it1 = &params.iterations[1];
    let last = proof1.last_msg.as_ref().unwrap();

    let mut g_miss = 0usize;
    for i in 0..out.statement.r {
        for j in i..out.statement.r {
            let chunks = decompose(&last.g[i][j], m, it1.b2, it1.t2 as usize);
            let restored = recompose(&chunks, m, it1.b2);
            if restored != last.g[i][j] {
                g_miss += 1;
            }
        }
    }
    let mut h_miss = 0usize;
    for i in 0..out.statement.r {
        for j in i..out.statement.r {
            let chunks = decompose(&last.h[i][j], m, it1.b1, it1.t1 as usize);
            let restored = recompose(&chunks, m, it1.b1);
            if restored != last.h[i][j] {
                h_miss += 1;
            }
        }
    }
    println!("iter[1]: b2={}, t2={} -> g mismatches {}/{}", it1.b2, it1.t2, g_miss, out.statement.r * (out.statement.r + 1) / 2);
    println!("iter[1]: b1={}, t1={} -> h mismatches {}/{}", it1.b1, it1.t1, h_miss, out.statement.r * (out.statement.r + 1) / 2);
    assert_eq!(g_miss, 0, "g decomposition must be lossless");
    assert_eq!(h_miss, 0, "h decomposition must be lossless");
}

#[test]
fn decomposition_is_lossless_at_iter1_n2() {
    use labrador::fold::fold;
    use labrador::garbage::{decompose, recompose};
    use labrador::prover_v2::prove_v2;
    use labrador::transcript::Transcript;

    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let seed = b"loss-check/n2";

    println!("q'={} (2^{:.4})", ring.m.q, (ring.m.q as f64).log2());

    let mut t_prove = Transcript::new(seed);
    let mut t_fold = t_prove.clone();
    let proof0 = prove_v2(&stmt, &witness, &params.iterations[0], &mut t_prove);
    let nu = params.iterations[1].prev_nu as usize;
    let mu = params.iterations[1].prev_mu as usize;
    let out = fold(&stmt, &proof0, &params.iterations[0], nu, mu, &mut t_fold);

    // Now run prove_v2 at iter[1] on the folded (stmt, witness).
    let proof1 = prove_v2(&out.statement, &out.witness, &params.iterations[1], &mut t_prove);

    // For each v[i] in proof1, decompose then recompose and compare.
    let m = &ring.m;
    let it1 = &params.iterations[1];
    let b1 = it1.b1;
    let t1 = it1.t1 as usize;
    let v_last = &proof1.last_msg.as_ref().unwrap().v;
    println!("iter[1]: b1={b1}, t1={t1}, b1^t1≈2^{:.2}", (b1 as f64).log2() * t1 as f64);
    let mut mismatches = 0usize;
    for (i, vi) in v_last.iter().enumerate() {
        for (k, e) in vi.iter().enumerate() {
            let chunks = decompose(e, m, b1, t1);
            let restored = recompose(&chunks, m, b1);
            if restored != *e {
                if mismatches < 3 {
                    eprintln!("v[{i}][{k}] decompose-recompose mismatch: original={:?}, restored={:?}", &e.c[..4], &restored.c[..4]);
                }
                mismatches += 1;
            }
        }
    }
    println!("iter[1] v-decomposition mismatches: {} out of {} ring elements", mismatches, v_last.iter().map(|v| v.len()).sum::<usize>());
    if mismatches > 0 {
        panic!("decomposition is lossy at iter[1]");
    }
}

#[test]
fn fold_satisfies_all_depths_n2() {
    use labrador::fold::fold;
    use labrador::prover_v2::prove_v2;
    use labrador::statement::satisfies;
    use labrador::transcript::Transcript;

    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let seed = b"depth-walk/n2";

    let mut cur_stmt = stmt.clone();
    let mut cur_witness = witness.clone();
    let mut t_prove = Transcript::new(seed);
    for k in 0..(params.depth - 1) {
        let mut t_fold = t_prove.clone();
        let proof = prove_v2(&cur_stmt, &cur_witness, &params.iterations[k], &mut t_prove);
        let nu = params.iterations[k + 1].prev_nu as usize;
        let mu = params.iterations[k + 1].prev_mu as usize;
        let out = fold(&cur_stmt, &proof, &params.iterations[k], nu, mu, &mut t_fold);
        let sat = satisfies(&out.statement, &out.witness);
        // Identify failing constraint when sat is false.
        if !sat {
            let kappa = params.iterations[k].kappa as usize;
            let kappa1 = params.iterations[k].kappa1 as usize;
            for (idx, c) in out.statement.full.iter().enumerate() {
                let r = labrador::statement::eval_full(c, &out.statement.ring, &out.witness);
                if !r.is_zero() {
                    let label = match idx {
                        z if z < kappa => format!("Check3 row {z}"),
                        z if z == kappa => "Check4".to_string(),
                        z if z == kappa + 1 => "Check5".to_string(),
                        z if z == kappa + 2 => "Check6".to_string(),
                        z if z < kappa + 3 + kappa1 => format!("Check8 row {}", z - (kappa + 3)),
                        z => format!("Check9 row {}", z - (kappa + 3 + kappa1)),
                    };
                    panic!("fold at depth {k}→{}: stmt constraint {idx} ({label}) non-zero (c[0]={})",
                           k + 1, r.c[0]);
                }
            }
            panic!("fold at depth {k}→{}: norm bound violated; norm_sq={}, beta_sq={}",
                   k + 1, out.witness.norm_sq(&out.statement.ring), out.statement.beta_sq);
        }
        cur_stmt = out.statement;
        cur_witness = out.witness;
    }
    println!("All {} folds satisfied for N=2", params.depth - 1);
}

#[test]
fn recursion_round_trips_n2() {
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let seed = b"rec-v2/n2";

    let proof = prove_aggregate(&stmt, &witness, &params, seed);
    assert_eq!(proof.intermediate.len(), params.depth - 1);
    for inter in &proof.intermediate {
        assert!(inter.last_msg.is_none(), "intermediate must NOT carry openings");
    }
    assert!(proof.final_iter.last_msg.is_some(), "final iter must carry openings");

    verify_aggregate(&stmt, &proof, &params, seed).expect("recursive proof must verify");
}

#[test]
fn recursion_round_trips_n8() {
    let ring = ring_for(8);
    let sigs = fresh_sigs(8);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(8);
    let seed = b"rec-v2/n8";

    let proof = prove_aggregate(&stmt, &witness, &params, seed);
    verify_aggregate(&stmt, &proof, &params, seed).expect("recursive proof must verify");
}

#[test]
#[ignore]
fn recursion_round_trips_n64() {
    let ring = ring_for(64);
    let sigs = fresh_sigs(64);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(64);
    let seed = b"rec-v2/n64";

    let proof = prove_aggregate(&stmt, &witness, &params, seed);
    verify_aggregate(&stmt, &proof, &params, seed).expect("recursive proof must verify");
}

#[test]
fn tampering_intermediate_u1_rejects() {
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let seed = b"rec-v2/n2-tamper-u1";

    let mut proof = prove_aggregate(&stmt, &witness, &params, seed);
    // Flip a byte in the first intermediate iteration's u_1.
    proof.intermediate[0].u1[0].c[0] = ring.m.add(proof.intermediate[0].u1[0].c[0], 1);

    assert!(verify_aggregate(&stmt, &proof, &params, seed).is_err());
}

#[test]
fn tampering_intermediate_u2_rejects() {
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let seed = b"rec-v2/n2-tamper-u2";

    let mut proof = prove_aggregate(&stmt, &witness, &params, seed);
    proof.intermediate[0].u2[0].c[0] = ring.m.add(proof.intermediate[0].u2[0].c[0], 1);
    assert!(verify_aggregate(&stmt, &proof, &params, seed).is_err());
}

#[test]
fn tampering_final_z0_rejects() {
    let ring = ring_for(2);
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(2);
    let seed = b"rec-v2/n2-tamper-final-z0";

    let mut proof = prove_aggregate(&stmt, &witness, &params, seed);
    let last = proof.final_iter.last_msg.as_mut().unwrap();
    last.z0[0].c[0] = ring.m.add(last.z0[0].c[0], 1);
    assert!(verify_aggregate(&stmt, &proof, &params, seed).is_err());
}

#[test]
fn intermediate_proof_strip_excludes_openings() {
    use bincode;
    let ring = ring_for(8);
    let sigs = fresh_sigs(8);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    let params = Params::for_n(8);
    let seed = b"rec-v2/n8-size";

    let proof = prove_aggregate(&stmt, &witness, &params, seed);
    let intermediate_bytes: Vec<usize> = proof
        .intermediate
        .iter()
        .map(|p| bincode::serialize(p).unwrap().len())
        .collect();
    let final_bytes = bincode::serialize(&proof.final_iter).unwrap().len();
    let total_bytes = bincode::serialize(&proof).unwrap().len();
    println!(
        "N=8 recursion: depth={}, intermediates (bytes)={:?}, final={} bytes, total={} bytes",
        params.depth, intermediate_bytes, final_bytes, total_bytes
    );
    // Sanity: every intermediate must be MUCH smaller than its analogous full
    // iteration would be (no openings on the wire). Empirically each is on
    // the order of κ₁ + κ₁ + 2λ + K'' RingElems ≪ the openings.
    for (k, sz) in intermediate_bytes.iter().enumerate() {
        assert!(*sz < final_bytes, "intermediate[{k}]={sz} >= final={final_bytes}");
    }
}
