# aggregate-falcon-rs

LaBRADOR-based aggregation of Falcon-512 signatures.

## Research status and limitations

This repository is a research implementation of LaBRADOR-based aggregation for
Falcon-512 signatures. Its purpose is to study the construction, implement its
main protocol components, and measure the resulting size and verification-cost
tradeoff on real signatures.

This implementation is **not suitable for production signature verification**.
The aggregate proof size is sublinear in the batch size over the measured range:
it grows much more slowly than a concatenation of `N` signatures. Verification
is **not succinct**. The verifier reconstructs the public Falcon relation,
Johnson-Lindenstrauss projection constraints, and recursive folds, and is
therefore thousands of times slower than direct Falcon verification.

The implementation also retains unresolved protocol limitations, including
loose recursive norm-bound handling. It is not an audited cryptographic
implementation.

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

**Results** (Apple M4 Pro, 12 threads, `--release`):

| N (sigs) | Naive concat factor¹ | Proof size | Proof gen time | Verification time | Verify slowdown² |
|---:|---:|---:|---:|---:|---:|
| 8   | 14.30× | 73.2 KB | 0.81 s  | 0.66 s  | 3,615× |
| 32  | 3.36×  | 68.9 KB | 2.60 s  | 2.03 s  | 4,077× |
| 64  | 2.01×  | 82.3 KB | 4.79 s  | 3.55 s  | 3,480× |
| 128 | 1.07×  | 87.7 KB | 14.07 s | 13.10 s | 6,150× |
| 256 | 0.55×  | 90.9 KB | 50.90 s | 42.86 s | 9,697× |
| 512 | —      | —       | —³      | —³      | —      |

¹ Proof size ÷ Σ|sigᵢ| — the size of naively concatenating the raw signatures.
  Values **< 1.0×** mean the aggregate proof is *smaller* than shipping the raw
  signatures; **> 1.0×** means concatenation still wins at that N. Observed
  crossover is **≈ N = 140** (near break-even at N = 128, clear win by N = 256) — far
  earlier than the loose analytical bound, because the proof is nearly
  constant-size while Σ|sigᵢ| grows ~655 B per signature.

² Proof verification time ÷ time to naively verify all N Falcon signatures
  one-by-one (audited C reference impl). This is the *speed* cost of
  aggregation: e.g. a value of 13,000× means verifying the proof is ~13,000×
  slower than just checking the raw signatures. Aggregation trades verification
  speed for proof size — see the notes below.

³ N = 512 **did not complete** — the process was terminated during aggregation
  (memory pressure). Current peak memory and runtime make large N impractical.

Notes:

- **Communication is sublinear over the measured range; verification is not.**
  The recursive proof grows slowly with `N`, so the concat factor falls as `N`
  rises. This is a bandwidth/storage tradeoff, not a verification-speed win.
- **The verifier is not succinct.** It replays the public relation and recursive
  statement reconstruction, including roughly 1.5k initial relation constraints
  per Falcon signature. At `N = 256`, the proof is `90.9 KB` rather than
  `167.7 KB` of concatenated signatures, but verification takes `42.86 s`
  instead of `4.42 ms` for native Falcon verification.
- **Local optimization cannot close that gap.** Faster arithmetic and additional
  parallelism can improve constant factors, but the verifier must still
  reconstruct and check the LaBRADOR relation.

### Alternative architecture

The current implementation already selects an NTT-friendly modulus and uses its
NTT-capable ring multiplication path. Further optimization is useful for
experiments, but does not make this construction a low-latency verifier.

For a production-oriented system that must compress the verification of many
Falcon signatures, a better direction is likely to verify the signatures inside
a zkVM or another succinct proving system and expose one succinct outer proof
to the final verifier. That is a different architecture, not a refactor of this
LaBRADOR verifier.

## Tests

```bash
cargo test --workspace --release
```
