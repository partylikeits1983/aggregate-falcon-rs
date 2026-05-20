# Audit: `aggregate-falcon-rs`

**Audited at**: commit `c7c5374`, 2026-05-20
**Status update**: 2026-05-20 audit-and-perf pass — CONCERN-1 already closed by `e7fbe17` (the prover now uses `Params::beta_init_sq()`, not the `1<<60` hardcode). CONCERN-3 closed by `prover_and_verifier_statements_match` test in `tests/roundtrip.rs`. CONCERN-2 partially addressed: Rayon is now used widely (see §6); the prover/fold replay duplication is eliminated via `prove_v2_with_replay` + `fold_with_replay`. Norm-bound recursion measurement is in §7.
**Paper**: `labrador.pdf` (sha256 prefix `2b18db11cc45fe0b`), *Aggregating Falcon Signatures with LaBRADOR*, Aardal–Aranha–Boudgoust–Kolby–Takahashi, CRYPTO 2024

## TL;DR

- Real Falcon-512 signatures are produced by `pqcrypto-falcon` (FFI to the audited C reference) and double-checked at every test invocation against the independent `falcon-rust` pure-Rust implementation. The bytes flowing into `aggregate()` are genuine Falcon-512.
- The aggregation implements the LaBRADOR scheme of the paper above. Every math operation we looked at is sited at a specific file:line that cites the corresponding paper section, and the in-source constants match the paper's stated values.
- The prover is single-threaded, has no SIMD/NTT, and clones liberally on hot paths. The asymptotic cost (O(N²) prover) is inherent per the paper (§F.1); the wall-clock cost on this machine is not — see `PERF_PLAN.md` for the optimization roadmap.
- Two issues surface as `CONCERN` below and are tracked for follow-up.

## 1. Falcon authenticity

| Item | Source |
|---|---|
| External Falcon impl (production sigs) | `pqcrypto-falcon = "0.4"` — FFI to the audited Falcon-512 C reference (`crates/aggregate-falcon/Cargo.toml:16`) |
| External Falcon impl (cross-check) | `falcon-rust = "0.1.2"` — independent pure-Rust Falcon-512 (`crates/aggregate-falcon/Cargo.toml:18`) |
| Cross-check tests | `crates/aggregate-falcon/tests/cross_validate.rs:53` (every honest sig accepted by both impls) and `:79` (every tampered sig rejected by both) |
| In-tree Falcon math | **none.** No hand-rolled signing or key-generation in the repo. `crates/falcon-relation/src/parse.rs` only *decodes* the published byte formats. |

**Verdict: PASS.** The signatures fed into the LaBRADOR pipeline are real Falcon-512 outputs.

## 2. Math cross-check vs the paper

For each math claim implemented in the repo, the table lists the paper section, the code site, and a verdict:

- **PASS-cited**: in-source citation matches the paper section, constants match the paper's stated values, sample-based unit tests in `#[cfg(test)]` confirm the expected shape (e.g. JL distribution test, four-square round-trip test).
- **CONCERN**: a known divergence from the paper-correct path; described in §3 below.
- **UNVERIFIED**: the shape is reasonable and the in-source comment names the paper section, but the audit did not work the algebra line-by-line against the paper text. A future pass with the paper open should walk these.

| # | Claim | Paper | Code (file:line) | Verdict |
|---|---|---|---|---|
| 1 | JL projection distribution `Π ∈ {−1,0,+1}^{256×(nD)}`, `Pr[0] = ½`, `Pr[±1] = ¼` each | Lemma 2.2 / §2.5 | `crates/labrador/src/jl.rs:18-34` (`sample_pi_row`) — implements `(bit0=zero, bit1=sign)`; unit test `jl.rs:200-211` empirically confirms 50/25/25 over 10k samples | PASS-cited |
| 2 | JL norm-bound constraint `ct(Σᵢ⟨σ₋₁(π̂ᵢ), wᵢ⟩) − pⱼ ≡ 0 (mod q′)` for `j ∈ [2λ]` | Protocol 2 step 2 / Protocol 3 step 2 | `crates/labrador/src/jl.rs:101-145` (`build_jl_constraints`), `λ = 128`, `PROJECTION_ROWS = 256` at `jl.rs:12-13` | UNVERIFIED |
| 3 | σ₋₁ on `S = Z_{q'}[X]/(X^d+1)`: `σ₋₁(a) = a₀ − a_{d−1}X − a_{d−2}X² − ⋯ − a₁X^{d−1}` | §2.5 background | `crates/labrador/src/jl.rs:77-84` and `crates/falcon-relation/src/relation.rs:184-191` (two identical defs) | PASS-cited |
| 4 | Challenge set `S`: weight `w = 43`, coefficients in `{−2,−1,+1,+2}` (γ = 2), `τ² = 86`, `T_op = 43` | §3 / `CHAL_2_SPLIT_64_128` parameter set | `crates/labrador/src/challenge.rs:18-29` constants; sampler at `:32-92`; unit test `:124-135` confirms shape across 1000 draws | PASS-cited |
| 5 | Falcon modulus lift: `s₁ + h·s₂ + q·v − c = 0` in `ℤ[X]/(X^512+1)`, with `q = 12289` | §6.1 eq. (6) | `crates/falcon-relation/src/relation.rs:74-98` (`compute_v_signed`), uses *centered* Falcon coefficients so `\|v\|` is minimized. `debug_assert!` at `:90` enforces exact divisibility. | PASS-cited |
| 6 | Lagrange four-square: `β² − ‖s₁‖² − ‖s₂‖² = ε₀² + ε₁² + ε₂² + ε₃²` | §6.2 | `crates/falcon-relation/src/relation.rs:148-171` (`four_square`); round-trip unit test at `:748-753` | PASS-cited |
| 7 | Subring embedding `R = ℤ[X]/(X^512+1)` as 8-rank `S`-module via `Y = X^8`: `a(X) = Σ_{k=0}^{7} X^k aₖ(X^8)` | [LNPS21]§2.8, used in §6.4 | `crates/falcon-relation/src/embed.rs:5-10` doc; `embed_signed` `:36-47` (permutation `i → (i mod C, i div C)`); inverse `extract_centered` `:60-68` | PASS-cited |
| 8 | Subring product `mul_subring`: image of `R`-product under the slot decomposition | §6.4 derivation | `crates/falcon-relation/src/embed.rs:78-` | UNVERIFIED |
| 9 | Padded witness `r = 3·⌈N/ρ⌉ + 3·ρ + 1`, S-rank `8N`, with `ρ = round(√N)` | §F.1 | `crates/falcon-relation/src/relation.rs:217-290` (`WitnessLayout`) | PASS-cited |
| 10 | Falcon-eq constraints (per slot `k`): linear `DotConstraint` `slot_k(s₁ + h·s₂ + q·v − c) = 0` | §F.2 | `crates/falcon-relation/src/relation.rs:355-399` (`add_falcon_eq_constraints`) | UNVERIFIED |
| 11 | Four-square constraints (per sig): `ct(⟨y, y'⟩) = β²` via σ₋₁ tie | §F.2 | `crates/falcon-relation/src/relation.rs:410-440` (`add_four_square_constraints`); coefficient `½` used so symmetric doubling recovers 1 — see `:425-428` | PASS-cited |
| 12 | Form constraints (padding + σ₋₁ ties for `y'`, `e'`) | §F.2 | `crates/falcon-relation/src/relation.rs:448-708` (four `add_form_*` functions) | UNVERIFIED |
| 13 | Ajtai/Module-SIS inner commit `vᵢ = A·wᵢ`; outer commits `u₁ = Σ B·v⁽ᵏ⁾ + Σ C·g⁽ᵏ⁾`, `u₂ = Σ D·h⁽ᵏ⁾` | Protocol 2 step 1 / §G / §B.6 | `crates/labrador/src/commit.rs:1-10` doc; `expand_matrix` `:18-47`; `matmul` `:50-65`; `commit_inner` `:68-74`; outer commits exercised in `prover_v2.rs:96-100` | PASS-cited |
| 14 | Recursive fold of verifier checks 3–9 into new dot-product constraints over `(z⁽⁰⁾, z⁽¹⁾, ê)` with `ê = v ‖ g_chunks ‖ h_chunks` | §B.6 step 5 | `crates/labrador/src/fold.rs:1-15` doc; `ELayout` `:58-106`; `FoldedLayout` `:130-` (folding params `ν, μ` produce `r' = 2ν + μ`, `n' = max(⌈n/ν⌉, ⌈m/μ⌉)`); main `fold` function `:200-438` and `fold_statement` `:647-808` | UNVERIFIED |
| 15 | Per-N parameter table (depth, b/b₁/b₂, t/t₁/t₂, κ/κ₁, σ_z, σ_h) | Estimator (§F.1 + `tools/phase0_params.py`) | `crates/labrador/src/params.rs:70-131` (`Params::for_n`) — runtime search over depths replicating the Python estimator; locked-in golden values at `tools/phase0_params.json`. Verified by `crates/labrador/tests/params_match_json.rs`. | PASS-cited |
| 16 | Modulus `q' ≡ 5 (mod 8)`, largest prime below `2^q_bitlen`, lossless `bᵗ ≥ 2^q_bitlen > q'` | §F.1 / §G | `crates/labrador/src/params.rs:140-142` (`select_modulus`) | PASS-cited |
| 17 | Initial norm bound `β² = β_init²` | §6.5 | `crates/labrador/src/params.rs:149-152` (`beta_init_sq`) vs **`crates/aggregate-falcon/src/lib.rs:101-103` which currently returns `1 << 60`** — see CONCERN-1 | **CONCERN** |
| 18 | Prover complexity O(N²) by §F.1 ("the recursion does not start from a balanced state") | §F.1 ("Impact on Runtime and Proof Size") | inherent shape of the `r² · n · D` loops in `prover_v2.rs:88-93` (g computation) and `fold.rs:679-754` (Check-4/5/6 constraint building) | PASS as designed |
| 19 | Fiat-Shamir via SHAKE256 | §2 / standard | `crates/labrador/src/transcript.rs` (uses `sha3::Shake256`); domain separation by per-call labels (e.g. `LABEL_A`, `LABEL_PI` in `fold.rs:28-40`) | PASS-cited |

The `UNVERIFIED` rows are the ones a future audit pass should walk against the paper text — each names the paper section so a reader can compare side-by-side.

## 3. Concerns

### CONCERN-1: `select_beta_sq` does not use `Params::beta_init_sq` — **CLOSED (`e7fbe17`)**

The prover now uses `params.beta_init_sq()` at `crates/aggregate-falcon/src/lib.rs:166`. The `select_beta_sq()` hardcode is gone. No code change as part of this audit pass.

Earlier text (kept for historical context): the bound used to be hard-coded to `1<<60`, weakening the soundness margin. The two LaBRADOR modules already expose the paper-correct value at `crates/labrador/src/params.rs:149-152`.

### CONCERN-2: Single-threaded prover with quadratic-shaped hot loops — **Largely addressed**

The "no rayon" claim is no longer accurate. Rayon now lives in `commit.rs` (`par_iter` in `expand_matrix`, `matmul`, `commit_inner`, both `outer_commit_*`), `jl.rs` (`sample_projection`, `project_vector`, `project_combined`), `garbage.rs` (`compute_g`, `compute_h`), `prover_v2.rs` (decompose loops, `compute_g`, `k_pp` per-constraint aggregation, z amortize), `verifier_v2.rs` (`k_pp` aggregation), and `fold.rs` (replay's `k_pp`, build_check rows). The constraint-aggregation hot loop is parallel across `k_pp` tasks.

Two further wins landed in the 2026-05-20 pass:
- `prove_v2_with_replay` + `fold_with_replay` (`prover_v2.rs`, `fold.rs`): the multi-iteration prover used to run `prove_v2` and then `fold` on the same iteration, with each independently walking the transcript and re-running the `k_pp` aggregation. At N=8 this duplication was costing ~550ms per intermediate iter (about half the iter total). The new replay-share path makes `fold` consume the prover's already-derived matrices and `aggregate_full` output; `fold_total` drops from ~552ms to ~10ms at iter 0 on N=8.
- Clone reduction in `fold::build_folded_witness`: removed the parallel `Vec<(usize, usize, RingElem)>` collect-then-apply pattern at `fold.rs:553-606` that doubled peak memory before serial application. Now a single serial pass writes directly into `w`, eliminating the transient that was a likely contributor to the OOM kills the user reported at N=256 on the x86 box.

**Severity**: still high asymptotically (the protocol is O(N²) per §F.1), but the recent commits + this pass have brought N=128 from "doesn't finish" to ~1 min (`6932ea4`) and N=256 to ~127s aggregate / ~57s verify on the user's reference machine.

### CONCERN-3: Verifier statement reconstruction with `s1, s2 = 0` — **CLOSED**

Closed by `prover_and_verifier_statements_match` in `crates/aggregate-falcon/tests/roundtrip.rs`: builds the prover-side statement from real Falcon sigs and the verifier-side statement from public-only `(pk, msg, nonce)` views with `s1=s2=0`, then asserts every public field of `Statement` (and every constraint in `full` / `const_term`) is structurally equal. Test passes today. The extracted helper `aggregate_falcon::build_verifier_statement` is what the test calls — it is the same code the production `verify()` path runs, factored out so the test can reach it.

The constraint builders read only `(h, c)` from each `FalconSig`; the test makes that invariant load-bearing — any future change that starts reading `s1` or `s2` from a constraint builder will fail the test loudly.

## 4. Tests

After this audit's cleanup pass:

- `crates/aggregate-falcon/tests/cross_validate.rs` — Falcon authenticity (every honest+tampered case agrees between pqcrypto-falcon and falcon-rust). 3 tests.
- `crates/aggregate-falcon/tests/roundtrip.rs` — honest aggregate + verify + tamper-rejection at N ∈ {2, 8}. 6 split tests plus one consolidated `roundtrip_n8_full_adversarial`.
- `crates/aggregate-falcon/tests/size_beats_concat.rs` — small-N size monotonicity check (rewritten to assert, not print).
- `crates/aggregate-falcon/tests/large_n_e2e.rs` — `#[ignore]`-gated real-aggregation runs at N ∈ {128, 512, 1024} for overnight / dedicated-CI use.

Removed in this pass (purely analytical, no real proof generation):
- `crossover_n_estimate`
- `aggregate_beats_concatenation_analytically_at_n1024`
- `aggregate_beats_concatenation_analytically_at_n4096`
- `analytical_size_matches_actual_at_n8`
- the `analytical_proof_bytes` helper
- the three `#[ignore]`'d weak tests at N ∈ {64, 500, 1024} that printed ratios without asserting

## 5. Out of scope for this audit

- Hand-walking each `UNVERIFIED` math row against the paper text (needs a reader with the paper open).
- Performance work beyond the two wins listed under CONCERN-2 (see `PERF_PLAN.md`).
- Tight norm-bound enforcement — see §7. Honest proofs currently exceed `beta_prime_sq` at later iterations; deferred until rejection sampling lands.

## 6. New tests added in the 2026-05-20 audit pass

In `crates/aggregate-falcon/tests/roundtrip.rs`:
- `prover_and_verifier_statements_match` — closes CONCERN-3.
- `malformed_nonce_length_rejects` — wrong nonce length must surface a `DecodeFailed("nonce length …")` error, not a downstream constraint mismatch.
- `swapped_public_pairs_rejects` — swapping `(pk, msg)` between two slots must reject (each slot binds (pk_i, msg_i, nonce_i)).
- `flipped_intermediate_u1_rejects` and `flipped_intermediate_p_rejects` — tamper the first intermediate iteration's u1/p, not just the final z0. Extends `flipped_proof_byte_rejects` to intermediate-iteration data.

In `crates/falcon-relation/src/relation.rs` (unit):
- `compute_v_signed_panics_on_non_divisible_input` — the `debug_assert_eq!` divisibility check has been promoted to a release-active `assert!`, so any future caller that fabricates `(s1, s2, h, c)` outside the Falcon verification invariant panics rather than silently truncating to a wrong `v`.

In `crates/labrador/tests/norm_bound_audit.rs` (new):
- `honest_proof_norm_bounds_loose_and_tight` — see §7.

## 7. Norm-bound recursion audit

`fold_statement` currently propagates `stmt.beta_sq` from one iteration to the next unchanged (`fold.rs:472`); the paper's tight estimator value `beta_prime_sq(it_params)` exists at `fold.rs:485` but is not enforced. The new test `tests/norm_bound_audit.rs` measures the honest prover's decomposed-norm² at every iteration and compares to both bounds.

At N=8, depth=4, with `beta_init_sq = 1186126502473` carried as the loose bound:

| iter | stmt.beta_sq (loose) | measured        | / loose | beta_prime² (tight) | / tight |
|------|----------------------|-----------------|---------|---------------------|---------|
| 0    | 1,186,126,502,473    | 20,346,134,099  | 0.017   | 26,432,495,951      | 0.770   |
| 1    | 1,186,126,502,473    | 561,294,776     | 0.000   | 565,936,178         | 0.992   |
| 2    | 1,186,126,502,473    | 32,880,718      | 0.000   | 29,170,777          | **1.127** |
| 3    | 1,186,126,502,473    | 6,767,941,886   | 0.006   | 5,266,740,191       | **1.285** |

**Finding.** Honest proofs exceed the paper's tight bound at iter 2 (by 12.7%) and iter 3 (by 28.5%). The loose `stmt.beta_sq` has massive headroom and is satisfied at every iter. **Tightening `fold_statement` to use `beta_prime_sq` today would reject honest proofs**, so the carry-through stays as-is. The in-source comment at `fold.rs:460-473` captures this; the test pins the loose bound as a regression guard and prints the tight ratios so the next pass (post Session E rejection sampling) can re-evaluate.
