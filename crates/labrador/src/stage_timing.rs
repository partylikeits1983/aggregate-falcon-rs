//! Env-gated, thread-local stage timing for prover/verifier instrumentation.
//!
//! When `STAGE_TIMING=1` is set in the process environment at the time the
//! aggregator runs, every `record(name, dur)` call appends to a `thread_local!`
//! buffer. The caller (e.g. the roundtrip example) drains the buffer after
//! aggregation finishes via `drain()` and prints it.
//!
//! When the env var is unset, every public function below is a fast no-op:
//! one atomic load + branch, no allocation, no syscall. The check is cached in
//! a `OnceLock<bool>` so the cost is amortized across all calls in a process.
//!
//! Records from worker threads are dropped unless the worker calls `drain()`
//! itself. That's fine for our use: the heavy stages we time
//! (`expand_matrix`, `commit_inner`, `compute_g`, JL, constraint aggregation,
//! `compute_h`) run on the main prover thread, with Rayon parallelism living
//! *inside* the timed region. The wall-clock measured on the main thread is
//! what we care about.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Duration;

static ENABLED: OnceLock<bool> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("STAGE_TIMING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

thread_local! {
    static BUFFER: RefCell<Vec<(&'static str, Duration)>> = const { RefCell::new(Vec::new()) };
}

/// Append a `(stage, duration)` pair to the main thread's buffer.
pub fn record(stage: &'static str, dur: Duration) {
    if !enabled() {
        return;
    }
    BUFFER.with(|buf| buf.borrow_mut().push((stage, dur)));
}

/// Insert a sentinel marker so consumers can group timings by iteration. The
/// roundtrip example treats every `("iter-end", _)` record as the boundary
/// between successive `prove_v2` invocations.
pub fn mark(stage: &'static str) {
    record(stage, Duration::ZERO);
}

/// Drain the main thread's buffer and return its contents. Subsequent calls
/// return empty until more records are appended.
pub fn drain() -> Vec<(&'static str, Duration)> {
    BUFFER.with(|buf| std::mem::take(&mut *buf.borrow_mut()))
}

/// Whether `STAGE_TIMING=1` was set at process start. Caller-facing — the
/// roundtrip example uses this to decide whether to print the table at all.
pub fn is_enabled() -> bool {
    enabled()
}
