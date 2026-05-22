# Handoff: diagnose & speed up aggregate-signature verification

Branch: `perf/verify-diagnosis` (this commit). Resume here on the VPS.

## Task
`cargo run --release --example roundtrip -p aggregate-falcon -- 140` shows
**verify ≈ 29s**, almost equal to prove (~29s). Goal: understand why, then make
the verifier faster. Diagnosis is DONE; fixes are NOT yet started.

## What's already done
Added env-gated fine-grained timers to the verifier fold path in
`crates/labrador/src/fold.rs` (`replay_iteration` + `fold_statement_with_replay`),
using the existing `stage_timing` facility. Run with:

```
STAGE_TIMING=1 cargo run --release --example roundtrip -p aggregate-falcon -- 140
```

## Diagnosis (confirmed with data — do NOT re-litigate)
Verify time is concentrated in the FIRST fold (cost falls ~3× per iteration as
the statement shrinks):

| iter | verify |
|---|---|
| 0 | 18.96s |
| 1 | 7.07s |
| 2 | 1.25s |
| 3–5 | <0.9s total |

Iter-0 sub-stage breakdown (of 18.96s):
- `constraint_agg` **8.87s** — the `(0..k_pp).into_par_iter()` loop in
  `replay_iteration` (`fold.rs:~268`). Iterates n_fp=214,876 constraints but
  parallelizes only over **k_pp=3**, so 9 of 12 cores idle.
- `jl_build` **5.97s** — `build_jl_constraints` (`crates/labrador/src/jl.rs:115`)
  is **fully sequential** (`for j in 0..256`) and allocates a dense
  `vec![RingElem::zero(); n]` for each of 256×r=256×73=18,688 (row,witness) pairs.
- `jl_sample` 1.8s — r=73 sequential SHAKE squeezes (`sample_projection`).
- matrix expansion (A/B/C/D) is only ~160ms — **the original "redundant matrix
  re-expansion / prover replay-reuse" hypothesis is DEBUNKED**.

Iter-0 dims: `n=1120 r=73 kappa=22 kappa1=7 t1=5 t2=3 k_pp=3 n_fp=214876
(const_term=214620)`. After iter 0, const_term drops to 0 (Falcon constraints
folded away); iters 1+ have n_fp=256 (JL only). The 214,620 Falcon const-term
constraints in iter 0 are the root driver — the verifier reconstructs the full
aggregated statement, the same work the prover does (hence prove≈verify).

## Proposed fixes (safe perf wins — implement & measure next)
All accumulation is modular addition (commutative/associative), so re-ordering /
re-parallelizing is mathematically identical. Do NOT change transcript
derivation order (psis, alphas, betas, challenges) — that would break soundness.

1. **Re-granularize `constraint_agg`** (`fold.rs` `replay_iteration`, and the
   mirrors in `crates/labrador/src/verifier_v2.rs:~186` and
   `crates/labrador/src/prover_v2.rs:~189`). Parallelize over chunks of the
   214k constraints (k_pp×partitions) and reduce partial (a_pp,phi_pp) per k,
   instead of only over k_pp=3. Expect ~3–4× on this stage.
2. **Parallelize + de-allocate `build_jl_constraints`** (`jl.rs:115`):
   `into_par_iter()` over the 256 rows; drop the dense `vec![RingElem::zero(); n]`
   intermediate (group nonzeros by `idx=flat_pos/D` directly, or reuse a cleared
   buffer). Expect large win on the 5.97s.
3. (Lower priority) `jl_sample`: internal row parsing is already parallel; the
   r=73 calls are transcript-sequential so limited gain.

`build_jl_constraints` and the per_k aggregation are shared by prover and
verifier, so fixing them speeds up BOTH.

## Verification after each fix
1. `cargo test --workspace` MUST stay green — esp. soundness/replay tests
   (commit e53d66b "Soundness audit pass") and fold-consistency tests. A valid
   proof must still verify; tampered proofs must still be rejected.
2. Re-run the STAGE_TIMING roundtrip at N=140; confirm verify total drops and
   the targeted sub-stage shrinks. Sanity-check N=8 and N=512 too.
3. Optional ground-truth: `samply` flamegraph of the verify path.

## Cleanup before merge
The `[dims]` eprintln and the extra-fine sub-timers are diagnostic scaffolding.
Keep the useful stage timers; remove the `[dims]` eprintln (or gate it more
quietly) before a final PR.

Full plan: `/Users/fermat/.claude/plans/make-a-plan-to-sorted-reddy.md`.
