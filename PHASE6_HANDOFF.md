# Phase 6 handoff — recursive folding for LaBRADOR Falcon-512 aggregation

## What you're picking up

You're implementing **Session B** of Phase 6 (recursive folding). The goal of
Phase 6 is to make the aggregate proof smaller than naive sig concatenation
at large N. The paper estimator predicts ≈ 80 KB at N=1024 (vs 666 KB of raw
sigs — an 8× win). Today's v1 single-iteration proof is **2.86 MB at N=64
and grows linearly**, so it loses to concatenation at every tested N. Phase 6
is the fix.

## Read these first, in order

1. **This doc** — sets the immediate scope.
2. **`HANDOFF.md`** — the original Phases 0-5+7 handoff. The "Critical
   gotchas" section near the bottom is mandatory reading. The phase status
   table at the top is now stale (Phases 0-5+7 are done; cross-validation
   work shipped after that).
3. **`labrador.pdf`** §B.6 (pp. 36-38), Protocol 2 (p. 39), Protocol 3 (p. 40).
   Protocol 2/3 is the spec for what Session B needs to produce.
4. **`tools/phase0_params.json`** — golden per-iteration parameters. Your
   `prove_v2`/`verify_v2` must produce a proof whose shape matches the
   iter[0] entry for the relevant N.

## Repo state right now

- Branch: `main`. Last commits:
  - `92f1b01` Phase 6 Session A: outer-commit helpers + 3 unit tests.
  - `25cc318` Cross-validate Falcon sigs against falcon-rust + e2e example.
  - `e13856f` Strengthen e' form constraint.
- **`cargo test --workspace --release`**: 67/67 pass. Don't push a change
  that breaks any of these.
- v1 prover/verifier still in place at `crates/labrador/src/{prover,verifier}.rs`.
  v1's `IterationProof` shape (in `proof.rs`) is unchanged.

## Session B scope

Add `prove_v2`/`verify_v2` as a parallel implementation **alongside** the
existing v1. Do not delete v1 in this session — migration of
`crates/aggregate-falcon/src/lib.rs` to call v2 is a later session.

`prove_v2` runs **one iteration** of the paper-correct Protocol 2 (no
recursion yet). It:

1. **Step 1 (Commit).** For each `w_i`:
   - Compute `v_i = A · w_i ∈ R^κ` (already done by `commit_inner`).
   - Decompose `v_i` into `t₁` chunks base `b₁`: `v_i^{(0)}, …, v_i^{(t₁-1)}`
     using existing `garbage::decompose` (centered base-b expansion).
   - Compute `g_{ij} = ⟨w_i, w_j⟩` for `i ≤ j` (use existing
     `garbage::compute_g`), decompose each into `t₂` chunks base `b₂`.
   - Expand `B` matrices via `commit::expand_b_mats` (Session A).
   - Expand `C` matrices via `commit::expand_sym_mats` (Session A).
   - Compute `u_1 = outer_commit_v(B, v_chunks) + outer_commit_sym(C, g_chunks)`.
   - Absorb `u_1` into the transcript.

2. **Step 2 (Project).** Sample per-witness JL matrices `Π_i` via
   `jl::sample_projection(transcript, cols=n·D)`. Compute the combined
   integer projection `p = jl::project_combined(Πs, ws) ∈ Z^{2λ=256}`.
   Send `p` (absorb into transcript).
   - **Defer JL constraints to Session E.** For Session B, just emit `p` and
     check `‖p‖² ≤ λ·β²` on the verifier side. Don't extend `F'` with the
     2λ projection constraints yet.

3. **Step 3 (Aggregate F').** Same as v1: sample `ψ^{(k)}`, compute `b''^{(k)}`,
   absorb. Existing v1 code in `prover.rs:71-163` is correct; lift it.

4. **Step 4 (Aggregate F + F'').** Same as v1: sample `α, β`, compute the
   aggregated `(a_agg, phi_agg)` (existing `prover::aggregate_full`).
   - **NEW**: compute `h_{ij}` for `i ≤ j` (existing `garbage::compute_h`),
     decompose into `t₁` chunks base `b₁`.
   - Expand `D` matrices via `commit::expand_sym_mats`.
   - Compute `u_2 = outer_commit_sym(D, h_chunks)`. Absorb.

5. **Step 5 (Amortize).** Sample challenges `c_i ∈ C` (existing
   `challenge::sample_challenge`). Compute `z = Σ c_i w_i`. Decompose
   `z` into 2 chunks base `b` (the witness-level decomposition, not `b₁/b₂`):
   `z = z^{(0)} + b · z^{(1)}`.
   - The final message of Session B's v2 is
     `(u_1, p, b'', u_2, z^{(0)}, z^{(1)}, v_chunks, g_chunks, h_chunks)`.

`verify_v2` mirrors the prover:

1. Re-derive `A` (as v1).
2. Recompose `v_i = Σ b₁^k · v_i^{(k)}` (via `garbage::recompose`). Recompose
   `g, h` similarly.
3. Re-derive `B, C, D` matrices.
4. Re-derive `Π_i`.
5. Check **‖p‖² ≤ λ·β²** (JL bound). Where `p = Σ Π_i · centered(w_i)`. The
   prover sent `p`; the verifier doesn't have `w_i`, so it can't recompute
   `p` directly — but the verifier CAN check the bound on the sent `p`.
6. Re-derive `ψ`, run constant-term check (as v1).
7. Re-derive `α, β`, run the four amortized identity checks (Checks 4-7 of
   Protocol 3, lines 164-240 of v1 verifier).
8. **NEW Check 7** (norm of decomposed pieces):
   `‖z^{(0)}‖² + ‖z^{(1)}‖² + Σ‖v_i^{(k)}‖² + Σ‖g_{ij}^{(k)}‖² + Σ‖h_{ij}^{(k)}‖² ≤ β'²`
   where `β'` is the next-iteration norm bound (from `params.rs`).
9. **NEW Check 8** (outer-commit openings):
   `u_1 = outer_commit_v(B, v_chunks) + outer_commit_sym(C, g_chunks)`,
   `u_2 = outer_commit_sym(D, h_chunks)`.
   Use `commit::outer_commit_v` / `commit::outer_commit_sym` — they're
   already tested.

## Per-iteration parameters

Pull from `Params::for_n(N).iterations[0]`:

```rust
let p = labrador::params::Params::for_n(n_sigs);
let it = &p.iterations[0];  // Session B only uses iteration 0.
let b   = it.b   as u64;   // z decomposition base
let t   = it.t   as usize; // z chunk count (always 2 except SecLast)
let b1  = it.b1  as u64;   // v, h decomposition base
let t1  = it.t1  as usize; // v, h chunk count
let b2  = it.b2  as u64;   // g decomposition base
let t2  = it.t2  as usize; // g chunk count
let kappa  = it.kappa  as usize;   // A matrix rank
let kappa1 = it.kappa1 as usize;   // outer-commit matrix rank
```

Verify your params lookup matches `tests/params_match_json.rs` (already
passes against the JSON for all target N).

## New types

Add a new file `crates/labrador/src/proof_v2.rs` (or extend `proof.rs` with
`IterationProofV2` and `AggregateProofV2`) so the v1 types stay intact.

```rust
pub struct IterationProofV2 {
    pub u1: Vec<RingElem>,                  // length κ₁
    pub p: Vec<i128>,                        // length 2λ = 256
    pub b_double_prime: Vec<RingElem>,       // length K''
    pub u2: Vec<RingElem>,                   // length κ₁
    pub z0: Vec<RingElem>,                   // length n
    pub z1: Vec<RingElem>,                   // length n
    pub v_chunks: Vec<Vec<Vec<RingElem>>>,   // [r][t₁][κ]
    pub g_chunks: Vec<Vec<Vec<RingElem>>>,   // [r][r][t₂] upper-triangular
    pub h_chunks: Vec<Vec<Vec<RingElem>>>,   // [r][r][t₁] upper-triangular
}
```

## Tests to add

In `crates/labrador/tests/single_iteration_v2.rs`:
- `honest_proof_v2_round_trips_for_n4` — analog of v1's
  `honest_proof_round_trips_for_n4`, using real Falcon sigs.
- `tampering_u1_rejects` — flipping any byte of `u_1` makes the verifier
  reject (outer-commit opening check fails).
- `tampering_p_rejects` — flipping a byte of `p` makes verify fail (JL
  bound or downstream check).
- `tampering_v_chunk_rejects` — flipping a `v_chunks[i][k][c]` byte breaks
  one of: outer-commit check, norm check, or A·z = Σc·v.
- All 5 v1 single-iteration tests must still pass (we're not touching v1).

## Critical gotchas (read these before coding)

1. **Centered arithmetic for decomposition.** `garbage::decompose` works in
   the centered representation. `recompose` reverses it. Don't centre twice.

2. **Norm of chunks vs ring-elements.** After decomposition, each chunk's
   coefficients are in `(-b/2, b/2]`. Use `RingElem::norm_sq(modulus)`
   (already centered) for the norm-bound check. Don't add `b/2` everywhere.

3. **Transcript order.** The new order is documented in `transcript.rs:10-27`
   already; the absorb/squeeze calls in `prove_v2`/`verify_v2` MUST match
   it exactly. Soundness depends on it. Round-trip tests do NOT catch
   ordering bugs.

4. **κ vs κ₁.** `κ` is the rank of the inner-commitment matrix `A`. `κ₁` is
   the rank of the outer-commitment matrices `B, C, D`. Don't conflate.

5. **`Statement::beta_sq`** is `i128` and currently set to `1<<60` at the
   top level. Session B's verifier should ALSO compute the next-iteration
   norm bound `β'²` (it's `params.iterations[0].next_beta_list` after
   conversion to `i128`). Read the param table carefully — `next_beta_list`
   has two entries (`β'₀, β'₁`); the combined bound is
   `‖next_beta_list[0]‖² + ‖next_beta_list[1]‖²`.

6. **JL projection columns.** For each `w_i`, `Π_i` has `n · D` columns
   (one per centered ℤ-coefficient). Session B can reuse
   `jl::sample_projection(t, label_per_i, n * D)` per witness vector i.

7. **Symmetric storage.** `g_chunks` and `h_chunks` are stored
   upper-triangular: index by `(i, j, k)` with `i ≤ j`. The lower triangle
   should NOT be populated (matches `outer_commit_sym`'s expectation).

8. **Don't refactor v1.** Add v2 in parallel; do not touch v1's
   `prover.rs`/`verifier.rs`/`proof.rs::IterationProof`. The existing
   tests must continue to pass without modification.

9. **Iteration 0's `r_list` has one entry.** For Session B's single-iteration
   v2, `r = r_list[0]`. Multi-entry `r_list` appears at higher iteration
   depths (Mid/SecLast) — irrelevant to Session B.

## Verification

End-of-session checklist:
- [ ] `cargo build --workspace` clean.
- [ ] `cargo test --workspace --release` — all 67 prior tests + new v2 tests pass.
- [ ] `cargo clippy --workspace -- -D warnings` clean (or matches existing
      warning level).
- [ ] A new test `cross_validate_v2_with_falcon_rust` is **NOT** required
      for Session B (that's Session D's e2e validation).
- [ ] Commit + push with a message naming "Phase 6 Session B".

## What comes after Session B

Don't attempt Sessions C/D/E in the same turn. Each is its own multi-hour
piece:
- **Session C**: `fold(stmt, proof_v2) → (next_stmt, next_witness)`
  — translates Protocol 3 checks 3-9 into a new `Statement`. Mathematical
  care needed: the verifier checks become dot-product constraints on
  `(z^{(0)}, z^{(1)}, v_chunks ‖ g_chunks ‖ h_chunks)`.
- **Session D**: recursion driver. Multi-iteration loop with `First → Mid
  → SecLast` stage handling. Stop sending chunks at non-final iterations.
  This is where proof size finally shrinks below naive concatenation.
- **Session E**: JL constraints in `F'` + tighten norm bound to match the
  paper estimator's predictions.

The plan file lives at `/home/galois/.claude/plans/do-an-audit-of-zazzy-owl.md`
— scroll to the bottom for the session table.
