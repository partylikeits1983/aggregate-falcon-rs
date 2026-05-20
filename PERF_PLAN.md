# Performance plan: cut Falcon aggregation wall-clock

## The question

Why does aggregating Falcon-512 signatures take so long, and what can we do about it on this machine?

## Where the time goes

The LaBRADOR prover is **inherently O(N²)** in the number of aggregated signatures — paper §F.1 admits this directly ("the recursion does not start from a balanced state"; first iteration has `r₀ = O(√N)`, `n₀ = O(√N)`, so the dominant `r²·n·D` work is `O(N²·D)`). That asymptotic is not a bug we can fix in software.

What *is* a software problem: the wall-clock cost of one operation step. Measured on this machine (Ryzen 7 7840HS, 8 cores / 16 threads, AVX-512 capable) at commit `c7c5374`:

- N=8 release-mode aggregate+verify: ~50 s
- Extrapolation under O(N²): N=128 ≈ 3.5 h, N=512 ≈ 57 h, N=1024 ≈ 230 h

That extrapolation is the constant factor the implementation is leaving on the table. The five reasons it is this bad on this hardware:

1. **No parallelism whatsoever.** `grep -r 'rayon\|par_iter\|parallel'` across `crates/` returns zero matches in production code. The prover runs on one of 16 hardware threads.
2. **Schoolbook ring multiplication.** `crates/modring/src/poly.rs:132-149` and `crates/modring/src/crt.rs:80-104` are naive O(D²) negacyclic multiplications — 4096 modular mults per ring multiplication at D = 64. No NTT, no Karatsuba, no FFT.
3. **No SIMD.** The base loops use scalar `u64`/`u128` arithmetic; the CPU exposes AVX2 *and* AVX-512 (including `avx512ifma`, the 52-bit integer-multiply-add extension that is essentially purpose-built for lattice ring multiplications).
4. **Hot-loop clones in `fold.rs`.** Lines 512, 514, 521, 530, 535 each `clone()` a `RingElem` (a 64-element `u64` array, ~512 B) inside `O(r·(t₁·κ + t₂·r))` loops.
5. **Matrix re-expansion.** Every iteration `expand_matrix` for A, B, C, D is rebuilt from the transcript fresh (`crates/labrador/src/commit.rs:18-47`, called from `prover_v2.rs:70,96-97,221` and `fold.rs:217-219`). The expansion is deterministic but uncached, so the prover and the fold each do the same SHAKE-out work twice.

These are all implementation concerns, not protocol concerns. Below is the order I would attack them.

## Optimization roadmap

### Tier 1 — easy, large wins (target: 5–10× speedup; weekend's work)

#### 1.1 Add `rayon` to independent prover loops

`Cargo.toml` workspace dep + `par_iter()` at the call sites where iterations are independent:

| Site | Loop | Estimated speedup |
|---|---|---|
| `crates/labrador/src/commit.rs:50-65` (`matmul`) | rows of `A` are independent | linear in cores per matmul |
| `crates/labrador/src/commit.rs:68-74` (`commit_inner`) | each `w_i → A·w_i` is independent | linear in cores, this is called `r` times per iteration |
| `crates/labrador/src/jl.rs:53-74` (`project_vector`) | the 256 rows are independent | linear in min(256, cores) |
| `crates/labrador/src/prover_v2.rs:88` (`compute_g`) — assuming it's `r²` independent inner products | the `r(r+1)/2` pairs are independent | linear in cores |
| `crates/labrador/src/prover_v2.rs:76-93` (decomposition of `v_chunks`, `g_chunks`) | per-element | linear in cores |

Cost: a few hundred LOC of changes, one new dependency. Risk: low — `rayon::par_iter` is a drop-in for `iter` when there are no captured-mut closures. The constraint-aggregation loop at `prover_v2.rs:145-161` is sparse-accumulator-shaped, so it needs a per-thread reducer rather than naive `par_iter` — slightly more careful but standard.

**Expected on this machine**: 8× on perfectly-parallel sections, ~6× overall after Amdahl. N=128 would drop from ~3.5 h to ~30 min.

#### 1.2 Cache the expanded commitment matrices per iteration

`expand_matrix` is deterministic from the transcript label + seed. Each iteration's `A, B, C, D` are referenced three times (prove → fold-prover, verify, fold-verifier — though in the v2 driver it's prove + fold-prover). Stash the matrices in a `OncePerIteration` or thread the expansion through the iteration once and pass by reference into `fold`.

Cost: small refactor of `aggregate_v2.rs` and `fold.rs` signatures. Risk: low — verifier *must* re-derive from the transcript anyway for soundness, but inside one prover/verifier run we can cache.

**Expected**: ~1.5×.

#### 1.3 Replace `clone()`s in fold hot loops with references / `mem::take`

`crates/labrador/src/fold.rs` lines 512, 514, 521, 530, 535 clone `RingElem`s in tight loops. Most of these can be `mem::take(&mut src)` since the source is consumed, or `&src` if the consumer accepts a reference. Profile-driven: a single `cargo flamegraph` run on `roundtrip_n8` will pinpoint the worst offenders.

Cost: micro-edits. Risk: low.

**Expected**: ~1.2× on the fold-heavy phase.

### Tier 2 — NTT-based ring multiplication (target: 30–80× on `Ring::mul`; week's work)

`crates/modring/src/poly.rs::mul` is the dominant arithmetic primitive: every commit, every fold, every constraint-aggregation pass calls it `O(r² · n · D)` times. Replacing the schoolbook O(D²) inner with an NTT/INTT round-trip cuts each call from 4096 modular mults to ~6·D·log₂D ≈ 2300 mults *and* makes the inner loop SIMD-friendly.

The prerequisite: the chosen LaBRADOR modulus `q'` must admit roots of unity of high enough order. The repo picks `q' ≡ 5 (mod 8)` (`crates/labrador/src/params.rs:140-142`) and already 2-splits the ring (`crates/modring/src/crt.rs`). With `q' ≡ 5 (mod 8)`, `q'` admits a primitive 8th root of unity but not necessarily a primitive 2D-th root (which would be needed for a full negacyclic NTT). Two paths:

- **Path A: deeper CRT splitting.** The current code splits `Z_{q'}[X]/(X^64+1)` into two factors of degree 32. Iterating the splitting recursively yields factors of degree 16, 8, 4, 2 — each step needs one more root of unity in `q'`. We can split as far as `q'` lets us, and run schoolbook on the resulting smaller factors. At splitting depth 4 each factor is degree 4 — schoolbook becomes 16 mults each, total ~16·16 = 256 mults instead of 4096. ~16×.
- **Path B: pick a different `q'`.** The paper's estimator allows `q'` to be larger than strictly needed; choosing one with order-2D roots of unity unlocks full NTT. This would mean tracking the *closest NTT-friendly prime above* `2^q_bitlen` in `select_modulus`. Cost: cross-checking that the paper's soundness margins still hold at the new (slightly larger) `q'`. The paper's §F.1 estimator outputs target bit-lengths, not exact primes; the slack is usually a few bits.

Path A is safer (no protocol-parameter change). Path B is faster but needs §F.1 re-verification.

**Expected**: 16–50×, depending on which path lands. Combined with Tier 1: N=128 falls into the minute range, N=1024 into the hour range.

### Tier 3 — AVX2 / AVX-512 hand-vectorization (target: 4–8× on the kernels; researcher's week)

The base modular multiplication in `crates/modring/src/modulus.rs:62` does one `u64×u64 → u128 → mod q` per call. AVX-512's `avx512ifma` (which this CPU has — confirmed in `/proc/cpuinfo`) does eight 52-bit `u64×u64 → u104` multiply-accumulates per cycle. Wrapping eight modular multiplications per SIMD instruction is the standard lattice-crypto trick. Same for `m.add` / `m.sub` / `m.neg` — eight-wide.

Implementation: a `Modulus::mul_x8(&[u64; 8], &[u64; 8]) -> [u64; 8]` intrinsic-gated to `target_feature = "avx512ifma"` with a scalar fallback. Lift to `RingElem::add`, `RingElem::sub`, `RingElem::scale`, the `mul_half` body in `crt.rs:80-104`. Coordinate-wise operations vectorize trivially (D = 64 = eight AVX-512 lanes); the multiplication needs careful lane interleaving.

This is the kind of work that benefits from a profile-guided sequence: only the kernels that flamegraph confirms dominate. Defer until Tier 1 + 2 expose what the new bottleneck is.

**Expected**: 4–8× on the kernels that actually run on hot data; less overall after Amdahl. Combined with Tier 1 + 2: another ~3× wall-clock improvement.

### Tier 4 — algorithmic / protocol-level (target: removes the O(N²) cliff; research project)

The paper's §F.1 paragraph that lists O(N²) also hints at the cause: the recursion does not start balanced. A different padding scheme that brings the first iteration into a balanced shape would in principle drop the asymptotic. This requires re-deriving the paper's parameter estimator and would likely change the soundness analysis. Out of scope unless someone takes it as a research project; mention here only to be clear the asymptotic ceiling isn't a wall of physics, it's a property of the specific reduction the paper chose.

## Suggested order of attack

1. Land Tier 1.1 (rayon) — gates the next benchmark and surfaces the real new bottleneck.
2. Re-measure N=8, N=64, N=128.
3. Decide between Tier 1.2 / 1.3 (more easy wins) vs. Tier 2 (NTT) based on where the new flamegraph points.
4. Tier 2 NTT (Path A, since it doesn't touch protocol params).
5. Re-measure. If still bottlenecked on `Modulus::mul`, do Tier 3 SIMD.
6. Tier 4 stays as a research note unless the project's goals shift.

Each tier is independently shippable and each one improves the realistic test ceiling: after Tier 1, the `roundtrip_n128` test from `large_n_e2e.rs` becomes a routine 30-minute CI gate; after Tier 2, `roundtrip_n512`; after Tier 3, `roundtrip_n1024`.

## How to measure as we go

```bash
# Flamegraph the prover at N=8 (already takes ~50 s, so the sample window is rich).
cargo install flamegraph
sudo cargo flamegraph -p aggregate-falcon --test roundtrip --release -- roundtrip_n8
```

Capture the resulting SVG before and after each tier lands. The shape of the flamegraph is the source of truth — pick the next tier based on what dominates.

## Files touched / created (when implemented)

- `Cargo.toml` (workspace deps: `rayon`)
- `crates/labrador/Cargo.toml`, `crates/modring/Cargo.toml` (add rayon)
- `crates/labrador/src/commit.rs` — par_iter in `matmul`, `commit_inner`
- `crates/labrador/src/jl.rs` — par_iter in `project_vector`, `project_combined`
- `crates/labrador/src/prover_v2.rs` — par_iter in `compute_g`, chunk decompositions; per-thread reducer in constraint aggregation
- `crates/labrador/src/fold.rs` — strip clones, par_iter constraint-rebuild loops
- `crates/labrador/src/aggregate_v2.rs` — thread cached matrices through `prove_aggregate`
- `crates/modring/src/poly.rs` and `crates/modring/src/crt.rs` — NTT path under a new module + selector based on `q'` admitting the needed roots of unity
- `crates/modring/src/modulus.rs` — `mul_x8` AVX-512 intrinsic under `target_feature` gate
- `crates/aggregate-falcon/benches/aggregate.rs` (new) — Criterion micro-bench so we have a stable wall-clock number across PRs

## Out of scope

- Tier 4 (protocol-level rebalancing).
- GPU offload — the working set is tiny per kernel but the kernels are deep; the round-trip cost of CPU↔GPU dominates at these sizes.
- Switching to a different aggregation scheme (e.g. SLAP / Greyhound). That is a research-level decision, not a perf change.
