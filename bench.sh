#!/usr/bin/env bash
#
# bench.sh — sweep the aggregate-falcon roundtrip over several N values and
# write the results to JSON (machine metadata + per-N prove/verify/size).
#
# Usage:
#   ./bench.sh                 # default sweep: 8 32 64 128 256 512
#   ./bench.sh 8 32 64         # custom N list
#   BENCH_OUT=out.json ./bench.sh
#
# A run that crashes or is OOM-killed is recorded as {"status":"failed"} and
# the sweep continues with the next N.

set -uo pipefail

NS=("$@")
if [ "${#NS[@]}" -eq 0 ]; then
  NS=(8 32 64 128 256 512)
fi

OUT="${BENCH_OUT:-bench-results/results.json}"
mkdir -p "$(dirname "$OUT")"

BIN="target/release/examples/roundtrip"

echo "==> building release binary (this may take a minute)..." >&2
if ! cargo build --release --example roundtrip -p aggregate-falcon >&2; then
  echo "build failed" >&2
  exit 1
fi

# --- helpers ---------------------------------------------------------------

# Convert a time token (e.g. "1.08s", "951.53ms", "1m2.3s") to seconds (float).
to_secs() {
  awk -v t="$1" 'BEGIN{
    if (t ~ /ms$/)     { sub(/ms$/,"",t); printf "%.6f", t/1000.0 }
    else if (t ~ /s$/) { sub(/s$/,"",t);  printf "%.6f", t }
    else if (t == "")  { printf "null" }
    else               { printf "%s", t }
  }'
}

cpu_model() {
  if [ -r /proc/cpuinfo ]; then
    grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^[[:space:]]*//'
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown"
  else
    echo "unknown"
  fi
}

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# --- machine metadata ------------------------------------------------------

MODEL="$(cpu_model)"
OS="$(uname -s) $(uname -r)"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- sweep -----------------------------------------------------------------

tmp="$(mktemp)"
entries=()
THREADS="null"

for N in "${NS[@]}"; do
  echo "==> running N=$N ..." >&2
  status="ok"
  if ! "$BIN" "$N" >"$tmp" 2>/dev/null; then
    echo "    N=$N FAILED (exit $?) — recording as failed" >&2
    status="failed"
  fi

  # thread count (cross-platform: parsed from the program's own banner)
  th="$(grep -m1 'rayon threads' "$tmp" | grep -oE '[0-9]+' | head -1)"
  [ -n "$th" ] && THREADS="$th"

  prove_raw="$(grep -E '^aggregate:' "$tmp" | awk '{print $2}' | head -1)"
  verify_raw="$(grep -E '^verify:'   "$tmp" | awk '{print $2}' | head -1)"
  proof_bytes="$(grep 'aggregate proof:' "$tmp" | grep -oE '[0-9]+ B' | head -1 | grep -oE '[0-9]+')"
  sig_bytes="$(grep 'what aggregation replaces' "$tmp" | grep -oE '[0-9]+ B' | head -1 | grep -oE '[0-9]+')"
  concat="$(grep 'proof /' "$tmp" | grep -oE '[0-9]+\.[0-9]+' | head -1)"
  # naive Falcon verification baseline (audited C ref impl) — total time to
  # verify all N signatures one-by-one, i.e. what a verifier of concatenated
  # raw signatures would pay.
  naive_raw="$(grep 'pqcrypto-falcon' "$tmp" | grep -oE '[0-9]+\.[0-9]+(ms|s)' | head -1)"

  prove_s="$(to_secs "$prove_raw")"
  verify_s="$(to_secs "$verify_raw")"
  naive_s="$(to_secs "$naive_raw")"
  # how many times slower proof verification is than naive sig verification
  slowdown="$(awk -v v="$verify_s" -v n="$naive_s" 'BEGIN{
    if (v=="null"||v==""||n=="null"||n==""||n+0==0) printf "null";
    else printf "%.1f", v/n
  }')"

  # fall back to JSON null for any field we couldn't parse
  : "${proof_bytes:=null}"
  : "${sig_bytes:=null}"
  : "${concat:=null}"
  : "${prove_s:=null}"
  : "${verify_s:=null}"
  : "${naive_s:=null}"
  : "${slowdown:=null}"
  prove_raw_j="$( [ -n "$prove_raw" ] && printf '"%s"' "$(json_escape "$prove_raw")" || printf 'null')"
  verify_raw_j="$( [ -n "$verify_raw" ] && printf '"%s"' "$(json_escape "$verify_raw")" || printf 'null')"
  naive_raw_j="$( [ -n "$naive_raw" ] && printf '"%s"' "$(json_escape "$naive_raw")" || printf 'null')"

  entries+=("$(cat <<JSON
    {
      "n": $N,
      "status": "$status",
      "concat_factor": $concat,
      "proof_bytes": $proof_bytes,
      "sig_concat_bytes": $sig_bytes,
      "prove_seconds": $prove_s,
      "verify_seconds": $verify_s,
      "naive_sig_verify_seconds": $naive_s,
      "verify_slowdown_vs_naive": $slowdown,
      "prove_raw": $prove_raw_j,
      "verify_raw": $verify_raw_j,
      "naive_sig_verify_raw": $naive_raw_j
    }
JSON
)")
  echo "    N=$N  prove=${prove_raw:-?}  verify=${verify_raw:-?}  proof=${proof_bytes:-?}B  concat=${concat:-?}x  naive=${naive_raw:-?}  slowdown=${slowdown:-?}x" >&2
done

rm -f "$tmp"

# --- emit JSON -------------------------------------------------------------

{
  printf '{\n'
  printf '  "machine": { "cpu": "%s", "os": "%s", "threads": %s },\n' \
    "$(json_escape "$MODEL")" "$(json_escape "$OS")" "$THREADS"
  printf '  "generated_at": "%s",\n' "$NOW"
  printf '  "results": [\n'
  for i in "${!entries[@]}"; do
    printf '%s' "${entries[$i]}"
    if [ "$i" -lt $(( ${#entries[@]} - 1 )) ]; then printf ','; fi
    printf '\n'
  done
  printf '  ]\n'
  printf '}\n'
} > "$OUT"

echo "==> wrote $OUT" >&2
cat "$OUT" >&2
