# Aggregate-Falcon Benchmarks

Performance of the LaBRADOR-based Falcon-512 signature aggregator, measured with:

```
cargo run --release --example roundtrip -p aggregate-falcon -- <N>
```

**Test machine:** AMD Ryzen 7 7840HS (8C/16T), 24 GB RAM, Linux — `--release`
build, 16 rayon threads. Times are wall-clock for a single run; expect some
run-to-run variance.

## Results

| N (sigs) | Naive concat factor¹ | Proof size | Proof gen time | Verification time |
|---:|---:|---:|---:|---:|
| 8   | — | — | — | — |
| 32  | — | — | — | — |
| 64  | — | — | — | — |
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |

¹ Proof size ÷ Σ|sigᵢ| — the size of naively concatenating the raw signatures.
  Values **< 1.0×** mean the aggregate proof is *smaller* than shipping the raw
  signatures; **> 1.0×** means concatenation still wins at that N. The crossover
  (aggregation starts beating concatenation) sits around N ≈ 1024 analytically.

## How to read the numbers

- **Proof size is ~constant in N** — the recursive LaBRADOR proof barely grows —
  so the concat factor falls as N rises. That is the entire point of
  aggregation: it is a *bandwidth/storage* win at scale.
- **Prove and verify time both scale ~linearly in N.** The first fold iteration
  carries all N signatures' constraints and dominates. The verifier is **not
  succinct**: it replays the full statement reconstruction, so verify ≈ prove.

## Future performance improvements

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
