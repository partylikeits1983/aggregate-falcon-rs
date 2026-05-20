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

## Tests

```bash
cargo test --workspace --release
```
