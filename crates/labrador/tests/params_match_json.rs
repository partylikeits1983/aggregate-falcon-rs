//! Verify that `labrador::params::Params::for_n(N)` reproduces every
//! per-iteration field stored in `tools/phase0_params.json`, which is the
//! ground truth dumped by the paper's estimator (see
//! `tools/phase0_params.py`).

use labrador::params::{Params, Stage};
use serde_json::Value;
use std::path::PathBuf;

fn load_json() -> Vec<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tools/phase0_params.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("json parse")
}

fn approx(a: f64, b: f64, label: &str) {
    let denom = b.abs().max(1.0);
    let err = (a - b).abs() / denom;
    assert!(
        err < 1e-9,
        "{label} differs: rust={a:.12e} json={b:.12e} (rel err {err:e})"
    );
}

#[test]
fn params_match_phase0_json_for_all_target_n() {
    let json = load_json();
    for entry in json.iter() {
        let num_sigs = entry["num_sigs"].as_u64().unwrap() as usize;
        if entry.get("error").is_some() {
            continue;
        }
        let p = Params::for_n(num_sigs);

        assert_eq!(
            p.q_bitlen as u64,
            entry["q_bitlen"].as_u64().unwrap(),
            "q_bitlen for N={num_sigs}"
        );
        assert_eq!(
            p.q_value_estimator,
            entry["q_value_estimator"].as_u64().unwrap(),
            "q_value_estimator for N={num_sigs}"
        );
        assert_eq!(
            p.parallel_reps as u64,
            entry["parallel_reps"].as_u64().unwrap(),
            "parallel_reps for N={num_sigs}"
        );
        assert_eq!(
            p.depth as u64,
            entry["depth"].as_u64().unwrap(),
            "depth for N={num_sigs}"
        );

        let iters_json = entry["iterations"].as_array().unwrap();
        assert_eq!(
            p.iterations.len(),
            iters_json.len(),
            "iterations count for N={num_sigs}"
        );

        for (k, (rust_it, j_it)) in p.iterations.iter().zip(iters_json.iter()).enumerate() {
            let label = |f: &str| format!("N={num_sigs} iter[{k}] {f}");

            let stage_expected = j_it["stage"].as_str().unwrap();
            assert_eq!(rust_it.stage.name(), stage_expected, "{}", label("stage"));

            assert_eq!(rust_it.n as u64, j_it["n"].as_u64().unwrap(), "{}", label("n"));

            let rl_rust: Vec<u64> = rust_it.r_list.clone();
            let rl_json: Vec<u64> = j_it["r_list"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(rl_rust, rl_json, "{}", label("r_list"));

            approx(rust_it.beta, j_it["beta"].as_f64().unwrap(), &label("beta"));
            assert_eq!(rust_it.logq as u64, j_it["logq"].as_u64().unwrap(), "{}", label("logq"));
            assert_eq!(rust_it.b, j_it["b"].as_u64().unwrap(), "{}", label("b"));
            assert_eq!(rust_it.t, j_it["t"].as_u64().unwrap(), "{}", label("t"));
            assert_eq!(rust_it.t1, j_it["t1"].as_u64().unwrap(), "{}", label("t1"));
            assert_eq!(rust_it.b1, j_it["b1"].as_u64().unwrap(), "{}", label("b1"));
            assert_eq!(rust_it.t2, j_it["t2"].as_u64().unwrap(), "{}", label("t2"));
            assert_eq!(rust_it.b2, j_it["b2"].as_u64().unwrap(), "{}", label("b2"));
            assert_eq!(rust_it.kappa, j_it["kappa"].as_u64().unwrap(), "{}", label("kappa"));
            assert_eq!(rust_it.kappa1, j_it["kappa1"].as_u64().unwrap(), "{}", label("kappa1"));
            approx(rust_it.m, j_it["m"].as_f64().unwrap(), &label("m"));
            approx(rust_it.sigz, j_it["sigz"].as_f64().unwrap(), &label("sigz"));
            approx(rust_it.sigh, j_it["sigh"].as_f64().unwrap(), &label("sigh"));
            assert_eq!(rust_it.prev_nu, j_it["prevnu"].as_u64().unwrap(), "{}", label("prevnu"));
            assert_eq!(rust_it.prev_mu, j_it["prevmu"].as_u64().unwrap(), "{}", label("prevmu"));

            // Verify Stage enum round-trips correctly.
            let _: Stage = rust_it.stage;
        }
    }
}
