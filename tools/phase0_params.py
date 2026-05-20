"""Phase 0: dump the golden LaBRADOR parameter table for the
two-splitting / 128-bit Falcon-512 set (FALCON_64_128 + CHAL_2_SPLIT_64_128).

Reuses the paper's estimator definitions verbatim (_estimator_defs.py is the
first 519 lines of proof_size_estimate.py, i.e. all defs, no import-time runs).

Run:  python3 tools/phase0_params.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _estimator_defs as E  # noqa: E402

# Two-splitting 128-bit set, matching proof_size_estimate.py's f512_2S search.
SCAL = 8
MAX_DEPTH = 15
TARGET_N = [2, 8, 64, 100, 500, 1024, 2000, 4096]

IT_FIELDS = ["stage", "n", "r_list", "beta", "logq", "b", "t",
             "t1", "b1", "t2", "b2", "kappa", "kappa1", "m",
             "sigz", "sigh", "prevnu", "prevmu"]


def iter_snapshot(it):
    snap = {}
    for f in IT_FIELDS:
        v = getattr(it, f, None)
        if isinstance(v, E.Stage):
            v = v.name
        snap[f] = v
    return snap


def best_recursion(num_sigs):
    """Mirror proof_size_estimate.search() but keep the winning iteration list."""
    falcon, chal = E.FALCON_64_128(), E.CHAL_2_SPLIT_64_128()
    q_, n, r_list, beta_list = E.get_initial_params(num_sigs, falcon, chal, SCAL)
    it0 = E.Iteration(q_, falcon.d, falcon.JL_slack, n, r_list, beta_list,
                      chal, falcon.SECPARAM, falcon.kappa_lim, E.Stage.FIRST)

    best = None
    for depth in range(1, MAX_DEPTH + 1):
        iters, size = E.recursion_to_depth_no_last_opt(it0, depth)
        if best is None or size < best[1]:
            best = (iters, size, depth)
    iters, size, depth = best
    return {
        "num_sigs": num_sigs,
        "q_bitlen": math.ceil(math.log2(q_)),
        "q_value_estimator": q_,            # 2^q_bitlen - 1 (composite placeholder)
        "parallel_reps": it0.parallel_reps(),
        "depth": depth,
        "total_proof_bits": size,
        "total_proof_bytes": math.ceil(size / 8),
        "iterations": [iter_snapshot(it) for it in iters],
    }


def main():
    table = []
    for n in TARGET_N:
        try:
            table.append(best_recursion(n))
        except Exception as exc:  # tiny N may break get_kappa / q condition
            table.append({"num_sigs": n, "error": repr(exc)})

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "phase0_params.json")
    with open(out, "w") as fh:
        json.dump(table, fh, indent=2)

    # Human-readable summary.
    print(f"{'N':>6} {'q_bits':>7} {'depth':>6} {'reps':>5} "
          f"{'proof':>12}  q_<2^63?")
    for row in table:
        if "error" in row:
            print(f"{row['num_sigs']:>6}  ERROR: {row['error']}")
            continue
        print(f"{row['num_sigs']:>6} {row['q_bitlen']:>7} {row['depth']:>6} "
              f"{row['parallel_reps']:>5} {E.format_size(row['total_proof_bits']).strip():>12}"
              f"  {'yes' if row['q_bitlen'] < 63 else 'NO -> wide'}")
    print(f"\nWrote {out}")

    qmax = max((r["q_bitlen"] for r in table if "q_bitlen" in r), default=0)
    print(f"max q_bitlen across targets: {qmax}  "
          f"=> backend: {'u64/u128 ok' if qmax < 63 else 'need crypto-bigint'}")


if __name__ == "__main__":
    main()
