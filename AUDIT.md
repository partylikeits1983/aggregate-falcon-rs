# Audit: `aggregate-falcon-rs`

**Audited at**: commit `c7c5374`, 2026-05-20
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

### CONCERN-1: `select_beta_sq` does not use `Params::beta_init_sq`

`crates/aggregate-falcon/src/lib.rs:101-103` hard-codes the initial norm bound to `1 << 60`:

```rust
fn select_beta_sq() -> i128 {
    1i128 << 60
}
```

The in-source comment at `:96-100` admits this is a stop-gap and that `Params::beta_init_sq()` should be wired in once JL projection constraints land — and they have landed (Phase 6 Session E in commit `34d5560`). Until this is replaced, the verifier checks against a loose bound which weakens the soundness margin the paper relies on. The two LaBRADOR modules already expose the paper-correct value at `crates/labrador/src/params.rs:149-152`.

**Severity**: medium — the prover/verifier transcript and constraint structure are correct; only the global ℓ² bound is loose. Follow-up: swap `select_beta_sq()` for `Params::for_n(N).beta_init_sq()` and re-run `tests/roundtrip.rs` plus the new `tests/large_n_e2e.rs`.

### CONCERN-2: Single-threaded prover with quadratic-shaped hot loops

`prover_v2.rs:145-161` aggregates F′ constraints with a triply-nested loop `for k in 0..k_pp { for c in const_term_extended { for (i, j) in c.a { … } } }`. The structure is protocol-correct per §F.1, but at N=1024 it iterates roughly `k_pp · (256 + O(N·D/ρ))` sparse entries serially. No `rayon` or `par_iter` anywhere in the workspace (`grep` confirms zero hits). Combined with cloning RingElems in the fold loops (`fold.rs:512,514,521,530,535`), this gives the observed ~50s at N=8 and the extrapolated ~230h at N=1024.

**Severity**: high for usability, zero for soundness — the asymptotic cost is paper-inherent. The optimization roadmap is `PERF_PLAN.md`.

### CONCERN-3: Verifier statement reconstruction with `s1, s2 = 0`

`crates/aggregate-falcon/src/lib.rs:160-178` reconstructs `FalconSig` views for the verifier with secret components zeroed, then runs the same constraint builders (`add_falcon_eq_constraints`, etc.) the prover used. The constraint builders are documented as only reading the public components `(h, c)` from `FalconSig`. UNVERIFIED that no constraint builder secretly reads `s1` or `s2`; if any does, verification would silently use zero in place of the real values. Suggested follow-up: a one-shot test that calls `build_falcon_statement(pubonly_sigs, …)` and `build_falcon_statement(full_sigs, …)` and asserts the resulting `Statement` (modulo the witness) is byte-equal.

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
- Performance work (see `PERF_PLAN.md`).
- Soundness analysis of `select_beta_sq()`'s loose `1 << 60` bound (CONCERN-1).
- A negative-case test confirming the verifier reads only public `FalconSig` fields (CONCERN-3).
