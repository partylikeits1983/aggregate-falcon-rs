# aggregate-falcon-rs

LaBRADOR-based aggregation of Falcon-512 signatures.

## Run aggregation

Aggregate `N` real Falcon-512 signatures, verify, and report sizes:

```bash
cargo run --release --example roundtrip -p aggregate-falcon -- <N>
```

`<N>` is the number of signatures to aggregate. Example: `<N>` = 256.

```bash
cargo run --release --example roundtrip -p aggregate-falcon -- 8
cargo run --release --example roundtrip -p aggregate-falcon -- 32
cargo run --release --example roundtrip -p aggregate-falcon -- 256
```

### Optional knobs

| Flag / env | Effect |
|---|---|
| `--threads K` | Cap Rayon's global pool to `K` threads. Takes precedence over `RAYON_NUM_THREADS`. |
| `RAYON_NUM_THREADS=K` | Standard Rayon override. Honored when `--threads` isn't passed. |
| `STAGE_TIMING=1` | Print a per-stage timing table (matrix expansion, JL projection, constraint aggregation, etc.) for every iteration. |

Combined example:

```bash
STAGE_TIMING=1 cargo run --release --example roundtrip -p aggregate-falcon -- 32 --threads 8
```

The example always prints the resulting Rayon thread count up front so low CPU utilization is easy to diagnose.

## Benchmarks

Run the full sweep and emit JSON (defaults to `N = 8 32 64 128 256 512`,
override by passing N values as args):

```bash
./bench.sh                 # default sweep -> bench-results/results.json
./bench.sh 8 32 64         # custom N list
BENCH_OUT=out.json ./bench.sh
```

The script builds once in `--release`, runs each N, parses the roundtrip
output, and writes a JSON file with machine metadata and per-N results. A run
that is killed (e.g. OOM at large N) is recorded as `"status": "failed"` and
the sweep continues.

**Results** (AMD Ryzen 7 7840HS, 16 threads — values pending a clean run):

| N (sigs) | Naive concat factor¹ | Proof size | Proof gen time | Verification time | Verify slowdown² |
|---:|---:|---:|---:|---:|---:|
| 8   | — | — | — | — | — |
| 32  | — | — | — | — | — |
| 64  | — | — | — | — | — |
| 128 | — | — | — | — | — |
| 256 | — | — | — | — | — |
| 512 | — | — | — | — | — |

¹ Proof size ÷ Σ|sigᵢ| — the size of naively concatenating the raw signatures.
  Values **< 1.0×** mean the aggregate proof is *smaller* than shipping the raw
  signatures; **> 1.0×** means concatenation still wins at that N. The crossover
  sits around N ≈ 1024 analytically.

² Proof verification time ÷ time to naively verify all N Falcon signatures
  one-by-one (audited C reference impl). This is the *speed* cost of
  aggregation: e.g. a value of 13,000× means verifying the proof is ~13,000×
  slower than just checking the raw signatures. Aggregation trades verification
  speed for proof size — see the notes below.

Notes:

- **Proof size is ~constant in N** — the recursive LaBRADOR proof barely grows —
  so the concat factor falls as N rises. Aggregation is a *bandwidth/storage*
  win at scale, not a speed win.
- **Prove and verify time both scale ~linearly in N.** The first fold iteration
  carries all N signatures' constraints and dominates. The verifier is **not
  succinct**: it replays the full statement reconstruction, so verify ≈ prove.

### Future performance improvements

The hot ring-multiply paths currently use schoolbook O(D²) multiplication, even
though an NTT-friendly modulus (`q' ≡ 1 mod 2D`) and an `mul_ntt` implementation
already exist in the `modring` crate — they just aren't wired into the
prover/verifier yet.

| Improvement | Helps | Expected gain | Risk |
|---|---|---|---|
| Wire up the existing NTT (`mul_ntt`) into ring multiplies | prove a lot, verify some | ~3–5× prove, ~1.5× verify | safe, math-identical |

Larger-scope levers, in rough order of impact: parallelize the Fiat-Shamir
challenge derivation (helps verify, but soundness-sensitive), reduce the
~1.5k constraints emitted per signature (helps both), and — for verification
that is genuinely succinct and independent of N — wrap the proof in an outer
STARK/SNARK so end-verification is milliseconds at any N.

## Tests

```bash
cargo test --workspace --release
```
