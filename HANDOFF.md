# Handoff prompt — continue building LaBRADOR Falcon-512 aggregation in Rust

Copy everything between the `---` markers as the prompt for the new agent.

---

You are continuing a from-scratch Rust implementation of the protocol from
the paper *"Aggregating Falcon Signatures with LaBRADOR"* (Aardal, Aranha,
Boudgoust, Kolby, Takahashi — CRYPTO 2024). When complete, the program must
take N Falcon-512 signatures and produce one short LaBRADOR aggregate proof,
and verify it.

## Read these first, in order

1. The approved plan **including the "Continuation guide" section**, which is
   the canonical source of truth for what to build and in what order:
   `/Users/fermat/.claude/plans/make-a-plan-to-cozy-yeti.md`
2. The paper PDF: `/Users/fermat/Downloads/2024-311 (1).pdf`
   (text-extracted page-by-page at `/tmp/falcon-paper/p{01..70}.txt`).
   The sections that matter for code:
   - **§B.6 (pages 36–38) + Protocols 2 & 3 (pages 39–40)** — full LaBRADOR
     prover/verifier pseudocode. Phase 5 transcribes this.
   - **§F.1–F.2 (pages 47–50)** — exact per-signature constraint set. Phase 3c
     transcribes this.
   - **§6 (pages 22–25)** — the Falcon-specific modifications.
   - **§2.5 (pages 10–11) + Lemma 2.2** — JL projection (Phase 4).
3. The golden parameter table:
   `aggregate-falcon-rs/tools/phase0_params.json` — per-N values of
   `q_bitlen`, depth, and every iteration's `n, r, b, b1, b2, t, t1, t2, κ, κ₁, ν, μ`.
   Your `params.rs` must reproduce these.

## What's already built (32 tests passing — do not touch unless extending)

Run `cd aggregate-falcon-rs && cargo test --workspace` to confirm.

| Crate | What's in it |
|---|---|
| `crates/modring` | `F_Q` field, `Z_Q[X]/(X^64+1)` ring, two-splitting CRT. `Ring`, `Modulus`, `RingElem`, `find_prime_5mod8`, `SplitMix64`. CRT-mul ≡ schoolbook-mul verified. |
| `crates/falcon-relation` | `falcon_ring.rs` (Falcon ring arithmetic); `parse.rs` (pubkey/sig decode + HashToPoint, validated vs `pqcrypto-falcon`); `embed.rs` (`embed_signed`, `embed_fpoly_centered`, `extract_centered`, `mul_subring`, verified against a degree-512 oracle); `relation.rs` (`compute_v_signed` modulus lift, `four_square` Lagrange, `add_slots`/`sub_slots`/`scale_slots`/`all_slots_zero` helpers). **Decisive test:** `tests/modulus_lift.rs` proves `s1 + h·s2 + q·v − c = 0` in `S^8` for real Falcon signatures. |
| `crates/labrador` | `statement.rs` — `Statement`, `Witness`, `DotConstraint`, `ConstTermConstraint`, `ring_inner_product`, `eval_full`, `eval_const_term`, `satisfies`. |

Workspace `Cargo.toml` includes all three crates. The crate `aggregate-falcon`
(top-level API) doesn't exist yet.

## Your job

Complete Phases 3c → 7 from the Continuation guide. Build one phase at a
time. After each phase, ALL workspace tests must pass — `cargo test --workspace`
green is the gate.

**Phase 3c (next, ~500 LOC):** Add to `falcon-relation/relation.rs`:
- `WitnessLayout { N, rho, num_y, num_yp, … }` with `index(i)`,
  `index_prime(i)` matching paper §F.1.
- `build_honest_witness(N, sigs, m_labrador) -> Witness` producing
  `Vec<Vec<RingElem>>` of length `r` (paper's `3⌈N/ρ⌉ + 3ρ + 1`), each inner
  vec of length `8N` (post-subring S-rank). Use `embed_fpoly_centered`,
  `compute_v_signed`, `four_square`, and a `sigma_minus_one` helper (the
  coefficient permutation `a(X) ↦ a_0 − a_{d−1}X − … − a_1 X^{d−1}`).
- Constraint builders, each adding `DotConstraint` or `ConstTermConstraint`
  to a `Statement`. **One R-constraint expands to either 8 S-full-constraints
  (full) or 1 S-const-term-constraint on slot 0 (const-term).** Start with the
  Falcon eq and four-square; add form constraints after the Falcon-eq test
  passes.
- **Gate test:** `satisfies(stmt, witness)` returns true for the honest witness
  built from real `pqcrypto-falcon` signatures. Mutating any byte of any
  signature must flip it to false.

**Phases 4–7:** see the Continuation guide. Build `crates/labrador/`
modules (`params`, `transcript`, `commit`, `challenge`, `jl`, `garbage`,
`prover`, `verifier`, `fold`, `proof`) then the top-level `aggregate-falcon`
crate. The guide gives per-phase test milestones; do not move on without
them.

## Critical gotchas (each one of these would silently break the protocol)

1. **Sign of `v`.** Paper eq (6) is `s1 + h·s2 + q·v − c = 0`. Therefore
   `v = (c − s1 − h·s2)/q`, **not** `(s1 + h·s2 − c)/q`. This was an actual
   bug fixed last session; don't reintroduce it.
2. **Centered coefficients.** Every conversion from a `u64` residue to a
   signed integer (for norms, embedding, etc.) MUST go through
   `Modulus::centered`. Never cast `u64 -> i64` directly.
3. **`τ` vs `T` naming.** In code: `tau = 86 = T_2` (ℓ₂² bound on
   challenges); `T = 43 = T_op` (operator-norm bound). The paper symbols are
   `T_2` and `T_op`; the Python estimator's `tau` and `T` correspond exactly.
   The `CHAL_2_SPLIT_64_128` halving (`/2`) is already baked in.
4. **The subring is `S = Z_{q'}[X]/(X^64+1)` — not Falcon's ring.** `modring`
   is `S`. `FALCON_Q = 12289` is Falcon's modulus, distinct from `q'`.
   `Ring::new(Modulus::new(q'))` builds `S` with `r = √(-1) mod q'`.
5. **Witness rank is `8N` post-subring.** Every R-witness-vector of length
   `N` becomes an S-vector of length `8N`. R-position-`i` occupies S-positions
   `8(i-1)..8i-1` (0-indexed: `8·idx .. 8·idx+7` where `idx = i-1`).
6. **Operator norm uses the CRT split.** `‖c‖_op` for `c ∈ S` is the max over
   the two CRT slots (`Ring::split`) of the slot's coefficient ℓ₂-norm.
   Challenge sampling rejects until both norms hold.
7. **Fiat-Shamir absorb-before-squeeze ordering is the soundness invariant.**
   Every prover message must be absorbed into the transcript before any
   challenge derived from it is squeezed. Match the paper's transcript order
   *exactly*. Document the order in `transcript.rs`. Round-trip tests do NOT
   detect ordering bugs — they detect *correctness* but not *soundness*.
8. **`q' < 2^50`** for N up to 4096 (verified in Phase 0). The `u64`+`u128`
   backend is exact — do not add `crypto-bigint`. `Modulus::new` panics if
   `q ≥ 2^62`.
9. **Parameters are dynamic, not hard-coded.** Recompute `q'` (a real prime
   ≡ 5 mod 8 of `q_bitlen` bits) at aggregate-time from N; the verifier
   re-derives it from `public_inputs.len()`. Stash `q'` and `r = √(-1)` in
   the proof header for robustness.
10. **Don't write decorative comments.** Per the project's style (CLAUDE-style
    rules in this repo), only comment the non-obvious *why*. Never explain
    what well-named code already says. Never add file/section banners that
    just restate the filename.

## Style and discipline

- Match existing style: terse top-of-file rustdoc explaining intent and
  citing the paper section; method docstrings only where non-obvious.
- Tests live alongside the code they test; integration tests in `tests/`.
- Use `pqcrypto-falcon` (already a `dev-dependency` of `falcon-relation`)
  for generating test signatures. Never use it in non-test code.
- Use `proptest` only when randomized properties materially help. Deterministic
  `SplitMix64` tests are preferred for reproducibility.
- Run `cargo test --workspace` after every meaningful change. Don't proceed
  past a failing test; diagnose root cause.
- Don't refactor the existing crates unless you find an actual bug. The
  substrate is verified.

## How to verify the final result

When all phases are done, this end-to-end flow must work:

```rust
let pairs: Vec<(FalconPublicKey, Vec<u8>)> = /* pk + msg for N signatures */;
let sigs: Vec<FalconInstance> = /* full instances incl. signatures */;
let proof = aggregate_falcon::aggregate(&sigs).unwrap();
assert!(aggregate_falcon::verify(&pairs, &proof).is_ok());
// Mutating any pk, msg, or proof byte → verify returns Err.
```

Test it for `N ∈ {2, 8, 64, 500, 1024}`. The serialized proof size should
trend with the Python estimator's predictions
(`aggregate-falcon-rs/tools/phase0_params.json`); an exact match isn't
expected since v1 uses `bincode` rather than entropy coding.

## Bounds on what you decide alone

Make and document any reasonable engineering call needed to keep moving.
But stop and ask the user before:
- Adding any new external crate that isn't in the Continuation guide.
- Changing the workspace structure or the public API of an existing crate.
- Diverging from the paper's protocol order or constraint shapes.
- Removing or skipping any of the §F.2 form constraints (they look tedious
  but each one is part of soundness).

Start at Phase 3c. Reach `cargo test --workspace` green before moving on.
Report progress after each phase.

---
