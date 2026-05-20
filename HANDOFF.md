# Handoff prompt — continue building LaBRADOR Falcon-512 aggregation in Rust

Copy everything between the `---` markers as the prompt for the new agent.

---

You are continuing a from-scratch Rust implementation of the protocol from
the paper *"Aggregating Falcon Signatures with LaBRADOR"* (Aardal, Aranha,
Boudgoust, Kolby, Takahashi — CRYPTO 2024). The end-to-end goal is concrete:

```rust
let pairs: Vec<(FalconPublicKey, Vec<u8>)> = /* N pk + msg */;
let sigs:  Vec<FalconInstance>             = /* N full instances */;
let proof = aggregate_falcon::aggregate(&sigs).unwrap();
assert!(aggregate_falcon::verify(&pairs, &proof).is_ok());
// Mutating any pk, msg, sig, or proof byte must make verify return Err.
```

This must work for `N ∈ {2, 8, 64, 500, 1024}`. The serialized proof should
trend with the Python estimator's predictions (`tools/phase0_params.json`);
an exact match isn't expected since v1 uses `bincode`, not entropy coding.

## Read these first, in order

1. The approved plan **including its "Continuation guide" section** — the
   canonical source of truth for module layout and per-phase order:
   `/Users/fermat/.claude/plans/make-a-plan-to-cozy-yeti.md`
2. The paper PDF: `/Users/fermat/Downloads/2024-311 (1).pdf`
   (text-extracted page-by-page at `/tmp/falcon-paper/p{01..70}.txt`).
   The sections you need:
   - **§B.6 (pages 36–38) + Protocols 2 & 3 (pages 39–40)** — full LaBRADOR
     prover/verifier pseudocode. **This is the spec for Phase 5.**
   - **§2.5 (pages 10–11) + Lemma 2.2** — JL projection (Phase 4).
   - **§3 (pages 13–15)** — challenge-set definitions (Phase 4).
   - **§6 (pages 22–25)** — the Falcon-specific modifications already
     implemented in Phase 3 — read for context on `q'`, `v_i`, four-square.
   - **§F.1–F.2 (pages 47–50)** — the constraint set, already encoded
     (read for context if you need to extend the constraint builders).
3. The golden parameter table:
   `aggregate-falcon-rs/tools/phase0_params.json` — per-`N` values of
   `q_bitlen`, `depth`, and every iteration's
   `n, r, b, b1, b2, t, t1, t2, κ, κ₁, ν, μ, σz, σh, m, parallel_reps`.
   **Your `params.rs` must reproduce these exactly.**

## What's already built and locked (37 tests passing — do not touch unless extending)

Run `cd aggregate-falcon-rs && cargo test --workspace` to confirm. The
GitHub mirror is `partylikeits1983/aggregate-falcon-rs` (private).

| Crate | What's in it |
|---|---|
| `crates/modring` | `F_Q` field, `Z_Q[X]/(X^64+1)` ring, two-splitting CRT. `Ring`, `Modulus`, `RingElem`, `find_prime_5mod8`, `SplitMix64`. **Invariant tested:** CRT-mul ≡ schoolbook-mul. |
| `crates/falcon-relation` | `falcon_ring.rs`, `parse.rs` (pubkey/sig decode + HashToPoint, validated vs `pqcrypto-falcon`), `embed.rs` (`embed_signed`, `embed_fpoly_centered`, `extract_centered`, `mul_subring`, **proven equal to a direct degree-512 oracle**), `relation.rs` (the full Phase 3c assembly — `WitnessLayout`, `sigma_minus_one`, `build_honest_witness`, `add_falcon_eq_constraints`, `add_four_square_constraints`, all four §F.2 form-constraint families, `build_falcon_statement`). **Decisive tests:** `tests/modulus_lift.rs` proves `s1 + h·s2 + q·v − c = 0` in `S^8` on real Falcon sigs; `tests/honest_statement.rs` proves the assembled statement is satisfied by the honest witness on real `pqcrypto-falcon` sigs and rejects every distinct form-constraint tamper. |
| `crates/labrador` | `statement.rs` — `Statement`, `Witness`, `DotConstraint`, `ConstTermConstraint`, `ring_inner_product`, `eval_full`, `eval_const_term`, `satisfies`. **Note: `phi` is sparse** — `Vec<(usize, Vec<(usize, RingElem)>)>` — for memory/compute reasons (~2000·N form constraints with 1–2 non-zero entries each). |

Workspace `Cargo.toml` includes all three crates. The crate
`aggregate-falcon` (top-level public API) doesn't exist yet — you'll create
it in Phase 7.

## Your job

Build Phases 4 → 7 from the Continuation guide. Build one phase at a time.
After each phase, `cargo test --workspace` must be green before you move
on. Commit and push after each phase, with a one-paragraph description of
what was added and what is now testable.

### Phase 4 — proof-system primitives (~1500 LOC), `crates/labrador/`

Build these modules in order. Each must have unit tests against either the
paper formulas or `phase0_params.json`.

- **`params.rs`** — port the `Iteration` class from
  `aggregate-falcon-rs/tools/phase0_params.py` (which mirrors the paper
  estimator's `proof_size_estimate.py`) line-by-line. Reproduce
  `b, b1, b2, t, t1, t2, κ, κ₁, σz, σh, m, parallel_reps, nextbeta_list`.
  **Gate test:** for every `N` in the JSON, every iteration field of your
  `params(N).iterations[k]` equals the JSON value exactly. **Two-splitting
  has `parallel_reps = 1` always; the JSON confirms this.**

- **`transcript.rs`** — `Transcript` wrapping `Shake256`, with
  `absorb(label: &[u8], bytes: &[u8])` and
  `challenge_bytes(label: &[u8], n: usize) -> Vec<u8>`. **Absorb-before-
  squeeze ordering is the soundness invariant.** Document the order in a
  module-level doc comment that lists every absorb/squeeze call site in
  protocol-execution order. Round-trip tests do *not* catch ordering bugs.

- **`commit.rs`** — `expand_matrix(seed: &[u8], rows, cols, ring) -> Vec<Vec<RingElem>>`
  deterministically from a transcript seed (squeeze `rows·cols·D` u64s,
  reduce mod `q`). Inner commitment `t_i = A · w_i`, outer
  `u_1 = Σ B v_chunks + Σ C g_chunks`. Matrices are *regenerated* on the
  verifier side — never serialized. **Gate test:** `expand_matrix` is
  deterministic on the same seed; commitment homomorphism
  `commit(a + b) = commit(a) + commit(b)`.

- **`challenge.rs`** — sample `c ∈ S` with `w = 43` non-zeros and
  infinity-bound `γ = 2`, so `‖c‖₂² ≤ τ = 86` and `‖c‖_op ≤ T = 43`. The
  operator norm `‖c‖_op` is the max over the two CRT slots
  (`Ring::split`) of the slot's coefficient ℓ₂-norm. Reject-sample until
  both bounds hold. **Gate test:** every sampled challenge satisfies both
  bounds; over 10000 samples no challenge exceeds `T`.

- **`jl.rs`** — `project(witness, Π) -> [i128; 256]`: convert each witness
  ring element to its *centered* `i128` coefficient vector, do a
  `2λ = 256 × (nd)` `{-1, 0, +1}` matrix multiply. Sample `Π` from a
  transcript with `Pr[0] = 1/2, Pr[±1] = 1/4` per Lemma 2.2. **Gate
  tests:** matches a hand-computed `Π·w` on tiny inputs; the JL
  inequality holds for honest Falcon witnesses across many trials.

- **`garbage.rs`** — given the aggregated `φ_i`, compute
  `h_{ij} = (⟨φ_i, w_j⟩ + ⟨φ_j, w_i⟩) / 2`. Decompose `g_{ij}` and
  `h_{ij}` into base-`b₁` / `b₂` chunks. **Gate test:** decompose +
  recompose is the identity; `‖chunk‖_∞ < base/2`.

### Phase 5 — single-iteration prover/verifier (~800 LOC), `crates/labrador/`

Transcribe Protocol 2 (page 39, prover) and Protocol 3 (page 40, verifier)
literally. Steps:

1. **Commit.** `v_i = A w_i`, decompose; `g_{ij} = ⟨w_i, w_j⟩`,
   decompose; outer `u_1`.
2. **Project.** Sample `Π_i` via transcript; send `p_j = Σ ⟨π^(j)_i, w_i⟩`.
3. **Aggregate F'.** Sample `ψ, ω`; compute `b''`.
4. **Aggregate F.** Sample `α, β`; commit `h_{ij}`; send `u_2`.
5. **Amortize.** Sample `c_i`; send `z = Σ c_i w_i = z^(0) + b·z^(1)`,
   `v_i, g_{ij}, h_{ij}`.

Verifier checks: `A·z = Σ c_i v_i`; `⟨z, z⟩ = Σ g_{ij} c_i c_j`;
`Σ ⟨φ_i, z⟩ c_i = Σ h_{ij} c_i c_j`; `Σ a_{ij} g_{ij} + Σ h_{ii} = b`;
norm bound on the decomposed pieces; outer-commitment openings;
`b''_0` constant-term check.

**Gate test:** honest proof round-trips on the Phase-3c statement for
`N = 4`. Tampering each field of the proof in turn → verifier rejects.

### Phase 6 — recursive folding (~400 LOC), `crates/labrador/`

Define the new statement from the last-message check equations (§B.6
Step 5): `e = v ‖ g ‖ h`; new witness `(z^(0), z^(1), e)`; fold by
`(ν, μ)` into `r' = 2ν + μ` new vectors of rank `max(n/ν, m/μ)`.
Reformulate every verifier check as a new `Statement` so the recursion
re-enters Phase 5's prover/verifier on the folded statement. `ν, μ` per
depth come from `params.rs` / `phase0_params.json`.

**Gate test:** full proof at the depth selected for `N` round-trips for
`N ∈ {2, 8}`. Negative tests at every depth.

### Phase 7 — top-level API + serialization + end-to-end (~300 LOC)

Create `crates/aggregate-falcon/` with:

- `pub fn aggregate(sigs: &[FalconInstance]) -> Result<AggregateProof, AggregateError>`
- `pub fn verify(pairs: &[(FalconPublicKey, Vec<u8>)], proof: &AggregateProof) -> Result<(), VerifyError>`
- `serde` + `bincode` for the proof bytes.
- `examples/roundtrip.rs` exercising the success path.

**Gate test:** end-to-end aggregate→verify accepts for
`N ∈ {2, 8, 64, 500, 1024}`. Mutation tests (wrong pubkey, wrong message,
wrong signature, swapped sigs, flipped proof bytes) → `verify` returns
`Err`. Serialized size vs `phase0_params.json` reported but not asserted.

## Critical gotchas (each one silently breaks the protocol if mishandled)

1. **Sign of `v`.** Paper eq. (6) is `s1 + h·s2 + q·v − c = 0` ⇒
   `v = (c − s1 − h·s2)/q`. This was a real bug fixed in Phase 3 — don't
   reintroduce it.

2. **Centered coefficients.** Every conversion from a `u64` residue to a
   signed integer MUST go through `Modulus::centered`. Never cast
   `u64 → i64` directly. Norms always use centered coefficients.

3. **`τ` vs `T` naming.** In code: `tau = 86 = T_2` (ℓ₂² bound on
   challenges); `T = 43 = T_op` (operator-norm bound). The paper symbols
   are `T_2` and `T_op`. The estimator's `CHAL_2_SPLIT_64_128` halving
   (`/2`) is already baked into `phase0_params.json`.

4. **The LaBRADOR ring in code is `S = Z_{q'}[X]/(X^64+1)` — not Falcon's
   ring.** `modring`'s `Ring` is `S`. `FALCON_Q = 12289` is Falcon's
   modulus, distinct from `q'`. `Ring::new(Modulus::new(q'))` builds `S`
   with `r = √(-1) mod q'`.

5. **Witness rank is `8N` post-subring.** Every R-witness-vector of length
   `N` becomes an `S`-vector of length `8N`. R-position-`i` (1-indexed)
   occupies S-positions `8(i-1)..8i-1`.

6. **Operator norm uses the CRT split.** `‖c‖_op` for `c ∈ S` is the max
   over the two CRT slots (`Ring::split`) of the slot's coefficient
   ℓ₂-norm. Challenge sampling rejects until both norms hold.

7. **Fiat-Shamir absorb-before-squeeze is the soundness invariant.** Every
   prover message must be absorbed into the transcript before any
   challenge derived from it is squeezed. Match the paper's transcript
   order *exactly*. Document the order in `transcript.rs`. Round-trip
   tests do NOT detect ordering bugs — they detect *correctness* but not
   *soundness*.

8. **`q' < 2^50`** for `N` up to 4096 (verified in Phase 0). The
   `u64`+`u128` backend is exact — do not add `crypto-bigint`.
   `Modulus::new` panics if `q ≥ 2^62`.

9. **Parameters are dynamic, not hard-coded.** Recompute `q'` (a real
   prime `≡ 5 mod 8` of `q_bitlen` bits) at aggregate-time from `N`; the
   verifier re-derives it from `public_inputs.len()`. Stash `q'` and
   `r = √(-1)` in the proof header for robustness.

10. **`phi` is sparse.** `DotConstraint.phi` and `ConstTermConstraint.phi`
    are `Vec<(usize, Vec<(usize, RingElem)>)>` — per-witness-vector
    sparse outer, per-position sparse inner. Construct phis sparsely;
    never materialize a dense `Vec<RingElem>` of length `8N`.

11. **Phase 3c uses S-level `σ_{-1}` per slot — not paper's R-level
    `σ_{-1}`.** The paper's `s' = σ_{-1}^R(s)` would not make the natural
    `S`-inner product on length-`8N` witness vectors equal `‖s‖²_R`. I
    chose `σ_{-1}^S` applied independently to each of the 8 S-slots, so
    `ct(⟨ϕ(s), s'⟩_S) = Σ_k ‖ϕ(s)_k‖²₂ = ‖s‖²_R` directly. This is
    mathematically equivalent for the four-square's purpose, but the
    witness contents differ from the paper's formula. The form
    constraints (`add_form_constraints_yp`, `add_form_constraints_ep`)
    enforce the S-level relation. **If you re-derive any
    constraint or check from the paper, remember the encoding is
    S-level, not R-level.**

12. **Don't write decorative comments.** Per the project style, only
    comment the non-obvious *why*. Never explain what well-named code
    already says. Never add file/section banners that just restate the
    filename. Match the rustdoc style of `embed.rs` and `relation.rs`.

## Style and discipline

- Match existing style: terse top-of-file rustdoc that explains intent
  and cites the paper section; method docstrings only where non-obvious.
- Tests live alongside the code they test; integration tests in `tests/`.
- Use `pqcrypto-falcon` (already a `dev-dependency` of `falcon-relation`)
  for generating test signatures. Never use it in non-test code.
- Prefer deterministic `SplitMix64` over `proptest` for reproducibility.
- Run `cargo test --workspace` after every meaningful change. Don't
  proceed past a failing test; diagnose root cause.
- Don't refactor the existing crates unless you find an actual bug or
  hit a structural blocker (like the Phase 3c sparse-phi refactor). When
  you do refactor, update the dependent code in the same commit.
- After each phase: commit, push to `partylikeits1983/aggregate-falcon-rs`,
  and write a one-paragraph description of what was added and what is now
  testable.

## Bounds on what you decide alone

Make and document any reasonable engineering call needed to keep moving.
But **stop and ask the user before**:

- Adding any new external crate that isn't in the Continuation guide
  (`sha3`, `crypto-bigint` (already ruled out), `rand`/`getrandom`,
  `serde`, `bincode`, `proptest`, `pqcrypto-falcon`).
- Changing the workspace structure or the public API of an existing
  crate.
- Diverging from the paper's protocol order or constraint shapes.
- Skipping any constraint or protocol step on grounds it "seems redundant."
- Force-pushing or rewriting published commits on `main`.

## How to verify the final result

When all phases are done, this end-to-end flow must work and is the
shipping criterion:

```rust
let pairs: Vec<(FalconPublicKey, Vec<u8>)> = /* pk + msg for N signatures */;
let sigs: Vec<FalconInstance> = /* full instances incl. signatures */;
let proof = aggregate_falcon::aggregate(&sigs).unwrap();
assert!(aggregate_falcon::verify(&pairs, &proof).is_ok());
// Mutating any pk, msg, or proof byte → verify returns Err.
```

Test it for `N ∈ {2, 8, 64, 500, 1024}`. The serialized proof size
should trend with the Python estimator's predictions; exact match isn't
expected since v1 uses `bincode`.

Start at Phase 4 (`params.rs` against `tools/phase0_params.json`).
Reach `cargo test --workspace` green before moving on. Report progress
after each phase.

---
