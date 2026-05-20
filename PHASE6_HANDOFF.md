# Phase 6 handoff — recursive folding for LaBRADOR Falcon-512 aggregation

> **Status**: Session A + B done and pushed (commits `92f1b01`, `784eec4`).
> You're picking up **Session C** (fold function).

## What you're picking up

The goal of Phase 6 is to make the aggregate proof smaller than naive sig
concatenation at large N. The paper estimator predicts ≈ 80 KB at N=1024 (vs
666 KB of raw sigs — an 8× win). Today's v1 single-iteration proof is
**2.86 MB at N=64 and grows linearly**, so it loses to concatenation at every
tested N. Phase 6 is the fix.

**Sessions A + B already shipped:**
- Session A (`92f1b01`): outer-commitment helpers in `commit.rs`
  (`outer_commit_v`, `outer_commit_sym`, `expand_b_mats`, `expand_sym_mats`) + 3 unit tests.
- Session B (`784eec4`): paper-correct single-iteration `prove_v2`/`verify_v2`
  in `crates/labrador/src/{prover_v2,verifier_v2}.rs` + `IterationProofV2`
  in `proof.rs` + 5 tests in `tests/single_iteration_v2.rs`. All 72 tests pass.
  v1 prover/verifier is **unchanged** and v1 tests are **untouched** — v2
  lives in parallel.

**Sessions C, D, E still ahead:**
- Session C (this one): write `fold(stmt, proof_v2, it_params, transcript)
  → (new_stmt, new_witness)` that translates verifier checks 3-9 of
  Protocol 3 into LaBRADOR dot-product constraints over a new witness
  `(z^{(0)}, z^{(1)}, e = v ‖ g_chunks ‖ h_chunks)` with `(ν, μ)` splitting
  from `Params::for_n(N).iterations[k+1].{prev_nu, prev_mu}`.
- Session D: recursion driver — top-level loop that runs `prove_v2` →
  `fold` → `prove_v2` → ... → SecLast. Drop chunks/openings from intermediate
  iterations. THIS is where proof size finally shrinks below concatenation.
- Session E: JL projection constraints in `F'` + tighten norm bound to
  match paper estimator predictions.

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

## Session C scope (THIS SESSION)

Build `fold(stmt, proof_v2, it_params, transcript) → (new_stmt, new_witness)`.
The fold is the heart of recursion: it converts the just-completed iteration's
verifier checks 3-9 into a new LaBRADOR `Statement` whose witness is the data
the prover would otherwise send in the open. Re-running `prove_v2` on the
folded `(new_stmt, new_witness)` produces the next iteration's proof.

### Inputs to `fold`

- `stmt`: the statement that `prove_v2` was just run on (so we know `r, n, κ, κ₁, β²`).
- `proof_v2`: the prover messages `(u_1, p, b'', u_2, z^{(0)}, z^{(1)}, v, g, h)`.
- `it_params`: `Params::for_n(N).iterations[k]` (current iter).
- `transcript`: must be at the state AFTER `prove_v2`/`verify_v2` would have
  finished — so that re-deriving `A, B, C, D, Π, ψ, α, β, c_i` matches both
  sides exactly. The fold itself only DERIVES these (does not absorb new
  prover messages).
- `(ν, μ)`: take from `Params::for_n(N).iterations[k+1].prev_nu / prev_mu`
  (the params object for iteration `k+1` records the fold parameters that
  produced its witness from iteration `k`).
- For the prover side, also pass in the v_chunks / g_chunks / h_chunks (which
  `prove_v2` discards today — you'll want to plumb them through, or
  re-compute them deterministically via `garbage::decompose`).

### Logical witness layout `e`

`e ∈ R^m` where `m = r·t₁·κ + t₂·(r(r+1)/2) + t₁·(r(r+1)/2)`:
- positions `[0, r·t₁·κ)`: `v_chunks[i][k][idx]` at offset
  `i·t₁·κ + k·κ + idx`.
- positions `[r·t₁·κ, r·t₁·κ + t₂·r(r+1)/2)`: `g_chunks[(i,j)][k]` for `i ≤ j`,
  flattened upper-triangle.
- positions tail: `h_chunks[(i,j)][k]` for `i ≤ j`.

Write a helper `enum EIdx { V(i,k,idx), G(i,j,k), H(i,j,k) } → usize` that
maps each logical chunk to its position in `e`. Use it everywhere — bare
arithmetic is too easy to off-by-one.

### `(ν, μ)` splitting

New witness has `r' = 2ν + μ` vectors of common rank `n' = max(⌈n/ν⌉, ⌈m/μ⌉)`.
- `w'_1, …, w'_ν` ← `z^{(0)}` chopped into ν pieces of length `⌈n/ν⌉`,
  padded with zeros to length `n'`.
- `w'_{ν+1}, …, w'_{2ν}` ← `z^{(1)}` similarly.
- `w'_{2ν+1}, …, w'_{2ν+μ}` ← `e` chopped into μ pieces of length `⌈m/μ⌉`,
  padded.

For each global position `p` in the original `z^{(0)} / z^{(1)} / e`, you'll
need a `(witness_idx, local_pos)` lookup. The constraint encoding will use
this mapping pervasively.

### Constraints to emit (translated from Protocol 3 checks)

For each of these, build either a `DotConstraint` (full equality in S) or a
`ConstTermConstraint` (only constant term must equal). All of them go into
`new_stmt.full` or `new_stmt.const_term`. The `b_agg` value that the verifier
computes from the previous iteration becomes one of the constraint RHS values.

1. **Check 3 (κ constraints, linear)**: per row `k` of `A`,
   `Σ_j A[k][j]·z^{(0)}[j] + b·Σ_j A[k][j]·z^{(1)}[j] - Σ_i c_i Σ_l b₁^l v_chunks[i][l][k] = 0`.
   Encoded as `DotConstraint` with only `phi` populated (no quadratic part).

2. **Check 4 (1 constraint, quadratic)**:
   `⟨z, z⟩ - Σ c_i c_j g_ij = 0` where `z = z^{(0)} + b·z^{(1)}` and
   `g_ij = Σ_l b₂^l g_chunks[i][j][l]`. Expand:
   `⟨z^{(0)}, z^{(0)}⟩ + 2b·⟨z^{(0)}, z^{(1)}⟩ + b²·⟨z^{(1)}, z^{(1)}⟩ - Σ_{i,j} c_i c_j Σ_l b₂^l g_chunks[i][j][l] = 0`.
   The `⟨w_i, w_j⟩` inner products span MULTIPLE folded vectors (since z^{(0)}
   is now ν vectors); you need cross-terms between every pair `(w_a, w_b)`
   with `a ∈ z^{(0)} family, b ∈ z^{(0)} family`, etc. This is the most
   intricate constraint.

3. **Check 5 (1 constraint, bilinear)**:
   `Σ_i ⟨φ_i, z⟩·c_i - Σ_{i,j} c_i c_j h_ij = 0` where `φ_i` is the aggregated
   `phi_agg[i]` from iter's verifier. Encoded similarly to check 4 but with
   linear-in-z part instead of quadratic.

4. **Check 6 (1 constraint, linear)**:
   `Σ a_{ij}·g_ij + Σ h_ii - b_agg = 0`. Pure linear in `e`.

5. **Check 8 — outer commit `u_1` (κ₁ constraints, linear)**: per row `r₁` of `u_1`,
   `u_1[r₁] - Σ_i Σ_k B_{i,k}[r₁,:] · v_chunks[i][k] - Σ_{i ≤ j} Σ_k C_{i,j,k}[r₁] · g_chunks[i][j][k] = 0`.
   Note: the `Σ_i Σ_k B · v_chunks` term has a κ-length dot product inside;
   it's `Σ_{i,k,idx} B_{i,k}[r₁, idx] · v_chunks[i][k][idx]`. All entries are in
   `e` so it's purely linear.

6. **Check 9 — outer commit `u_2` (κ₁ constraints, linear)**:
   `u_2[r₁] - Σ_{i ≤ j} Σ_k D_{i,j,k}[r₁] · h_chunks[i][j][k] = 0`.

The norm bound `β'²` for the new statement is the previously-computed
`next_beta_sq = it_params.next_beta_list[0]² + it_params.next_beta_list[1]²`.

### New statement size

For N=8 iter[0]→iter[1] (from JSON): `ν=1, μ=7`, `n'=397`, `r'=9`. Constraint
count: `κ + 1 + 1 + 1 + κ₁ + κ₁ = 19 + 3 + 12 = 34` full constraints (plus
constant-term constraints from JL deferred to Session E).

### Tests for Session C

Add `tests/fold_v2.rs`:
- `fold_satisfies_iff_v2_verifies` — generate a real Falcon statement at N=4,
  run `prove_v2`, run `fold`, check `satisfies(new_stmt, new_witness)`.
- `fold_witness_norm_within_bound` — `new_witness.norm_sq() ≤ β'²`.
- `tampering_v2_proof_breaks_fold` — flip a byte of `proof_v2.z0`, verify
  `satisfies(new_stmt, new_witness_built_from_tampered_proof)` returns false.

### Critical gotchas for Session C

1. **Transcript is consumed twice**: `prove_v2`/`verify_v2` advance the
   transcript through all challenges. `fold` MUST advance the transcript to
   the same state — easiest: have `fold` accept the same transcript handle
   AFTER `prove_v2` runs, then `fold` re-derives the challenges by sampling
   without absorbing new prover messages.

2. **Symmetric storage of `g, h`**: only `i ≤ j` entries are populated. When
   building Check 4's `Σ_{i,j} c_i c_j g_ij`, double the `i < j` terms.

3. **Padding**: when `n%ν ≠ 0` or `m%μ ≠ 0`, the last chopped piece is padded
   with zeros. The constraint encoding must use the SAME mapping for both
   the witness (where padding zeros are stored) and the constraint φ
   (where the corresponding positions are not referenced).

4. **`b_agg` is iteration-specific**: it's computed inside `verify_v2` as
   `Σ α_k b^{(k)} + Σ β_k b''^{(k)}` AND uses the iter's `α, β` challenges.
   `fold` must rebuild it the same way.

## Original Session B scope (now done — leave for reference)

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
