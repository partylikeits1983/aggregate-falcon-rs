//! LaBRADOR single-iteration prover.
//!
//! Engineering simplifications relative to Protocol 2 of the paper:
//!
//! - **No outer commitment.** `v_i = A · w_i` is transcript-absorbed
//!   directly, providing Fiat-Shamir binding without the `B, C, D` matrices.
//! - **No norm-bound decomposition.** `v, g, h` are sent in full (no
//!   base-`b₁/b₂` chunks). This makes the proof larger but the verification
//!   identities easier to write.
//! - **No JL projection / `ω` aggregation.** The 2λ projection constraints
//!   are dropped from `F'` for v1. The const-term constraint count stays the
//!   same; only the per-projection σ₋₁-shift in `b''_0^{(k)}` is omitted.
//! - **Single iteration.** No recursive folding. The verifier checks the
//!   last-message identities directly.
//!
//! The `b''` aggregation (Step 3) and `α, β` aggregation (Step 4) follow the
//! paper line-by-line, modulo the omissions above.

use crate::challenge::sample_challenge;
use crate::commit::{commit_inner, expand_matrix};
use crate::garbage::{compute_g, compute_h};
use crate::proof::IterationProof;
use crate::statement::{
    ring_inner_product, sparse_phi_inner_product, ConstTermConstraint, Statement, Witness,
};
use crate::transcript::Transcript;
use modring::{Modulus, RingElem, D};
use rayon::prelude::*;

const LABEL_A: &[u8] = b"labrador.A";
const LABEL_V: &[u8] = b"labrador.v";
const LABEL_PSI: &[u8] = b"labrador.psi";
const LABEL_BPP: &[u8] = b"labrador.b''";
const LABEL_ALPHA: &[u8] = b"labrador.alpha";
const LABEL_BETA: &[u8] = b"labrador.beta";
const LABEL_G: &[u8] = b"labrador.g";
const LABEL_H: &[u8] = b"labrador.h";
const LABEL_C: &[u8] = b"labrador.c";

/// Number of full constraints to extend `F'` into during the const-term
/// aggregation. The paper sets `K'' = ⌈λ/log₂(q')⌉`.
pub fn k_double_prime(stmt: &Statement, lambda: u32) -> usize {
    let logq = ceil_log2(stmt.ring.m.q);
    ((lambda as usize) + (logq as usize) - 1) / (logq as usize)
}

/// Ajtai rank `κ`. We hard-code a conservative value here because the v1
/// prover doesn't decompose `v` and thus doesn't need `κ` to match the
/// estimator's optimum. A modest value (8) keeps `A` small while staying
/// within the security bound for the tested signature counts.
pub fn kappa() -> usize {
    8
}

/// Run the LaBRADOR prover for one iteration and return its messages.
pub fn prove(
    stmt: &Statement,
    witness: &Witness,
    transcript: &mut Transcript,
) -> IterationProof {
    bind_statement(transcript, stmt);

    let ring = &stmt.ring;
    let m = &ring.m;
    let n = stmt.n;
    let r = stmt.r;
    let kap = kappa();

    // --- Step 1: expand A and compute v_i = A · w_i ---
    let a = expand_matrix(transcript, LABEL_A, kap, n, ring);
    let v = commit_inner(ring, &a, &witness.w);
    absorb_ring_matrix(transcript, LABEL_V, &v);

    // --- Step 3: aggregate F' const-term constraints into K'' full ones ---
    let lambda: u32 = 128;
    let k_pp = k_double_prime(stmt, lambda);
    let q = m.q;

    // For each k in [K''], sample ψ^{(k)} ∈ Z_{q'}^{|F'|}. Compute the
    // aggregated quadratic form (a_pp, phi_pp), then evaluate it on the
    // witness to obtain the full polynomial b''^{(k)}.
    let n_fp = stmt.const_term.len();
    let mut b_double_prime: Vec<RingElem> = Vec::with_capacity(k_pp);

    // We precompute, for each (k, witness-vector pair (i, j)), the aggregated
    // a''^{(k)}_{i,j} as a single RingElem (sum of ψ_l · a'^{(l)}_{i,j} for
    // each F' constraint l). Memory: O(K'' · r²) RingElems; manageable.
    let mut a_pp: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![RingElem::zero(); r]; r]; k_pp];
    let mut phi_pp: Vec<Vec<Vec<(usize, RingElem)>>> =
        vec![vec![vec![]; r]; k_pp];

    let psis: Vec<Vec<u64>> = (0..k_pp)
        .map(|k| {
            (0..n_fp)
                .map(|l| {
                    let label =
                        [LABEL_PSI, &(k as u64).to_le_bytes(), &(l as u64).to_le_bytes()].concat();
                    transcript.derive_field(&label, q)
                })
                .collect()
        })
        .collect();

    for k in 0..k_pp {
        // Σ_l ψ_l^{(k)} · a'^{(l)}_{i,j}
        for (l, c) in stmt.const_term.iter().enumerate() {
            let psi = psis[k][l];
            if psi == 0 {
                continue;
            }
            for &(i, j, ref aij) in &c.a {
                let scaled = aij.scale(m, psi);
                a_pp[k][i][j] = a_pp[k][i][j].add(m, &scaled);
            }
            for (wi, phi_i) in &c.phi {
                let bucket = &mut phi_pp[k][*wi];
                for (pos, coef) in phi_i {
                    let scaled = coef.scale(m, psi);
                    // Naive: append; we'll fold duplicate positions later.
                    bucket.push((*pos, scaled));
                }
            }
        }
        // Fold duplicate (pos) entries within each phi bucket.
        for bucket in phi_pp[k].iter_mut() {
            bucket.sort_by_key(|(p, _)| *p);
            let mut merged: Vec<(usize, RingElem)> = Vec::with_capacity(bucket.len());
            for (pos, coef) in bucket.drain(..) {
                if let Some(last) = merged.last_mut() {
                    if last.0 == pos {
                        last.1 = last.1.add(m, &coef);
                        continue;
                    }
                }
                merged.push((pos, coef));
            }
            *bucket = merged;
        }

        // Evaluate b''^{(k)} = Σ a''_{i,j} ⟨w_i, w_j⟩ + Σ ⟨φ''_i, w_i⟩
        // — a full RingElem.
        let mut b_full = RingElem::zero();
        for i in 0..r {
            for j in i..r {
                let coef = &a_pp[k][i][j];
                if coef.is_zero() {
                    continue;
                }
                let ip = ring_inner_product(ring, &witness.w[i], &witness.w[j]);
                let mut term = ring.mul(coef, &ip);
                if i != j {
                    term = term.add(m, &term.clone());
                }
                b_full = b_full.add(m, &term);
            }
        }
        for (i, phi_i) in phi_pp[k].iter().enumerate() {
            if phi_i.is_empty() {
                continue;
            }
            let ip = sparse_phi_inner_product(ring, phi_i, &witness.w[i]);
            b_full = b_full.add(m, &ip);
        }
        b_double_prime.push(b_full);
    }
    absorb_ring_vec(transcript, LABEL_BPP, &b_double_prime);

    // --- Step 4: aggregate F + F'' via (α, β) into a single full constraint ---
    let n_f = stmt.full.len();
    let alphas: Vec<RingElem> = (0..n_f)
        .map(|k| {
            sample_ring_element(transcript, LABEL_ALPHA, k as u64, ring)
        })
        .collect();
    let betas: Vec<RingElem> = (0..k_pp)
        .map(|k| sample_ring_element(transcript, LABEL_BETA, k as u64, ring))
        .collect();

    let (a_agg, phi_agg) = aggregate_full(stmt, &alphas, &betas, &a_pp, &phi_pp);

    // Compute g_{ij} = ⟨w_i, w_j⟩ and h_{ij} = (⟨φ_i, w_j⟩ + ⟨φ_j, w_i⟩)/2.
    let g = compute_g(ring, &witness.w);
    let h = compute_h(ring, &phi_agg, &witness.w);
    absorb_ring_matrix(transcript, LABEL_G, &g);
    absorb_ring_matrix(transcript, LABEL_H, &h);

    // --- Step 5: sample c_i, compute z = Σ c_i w_i ---
    let cs: Vec<RingElem> = (0..r)
        .map(|i| {
            let label = [LABEL_C, &(i as u64).to_le_bytes()].concat();
            sample_challenge(transcript, &label, ring)
        })
        .collect();

    let mut z = vec![RingElem::zero(); n];
    for i in 0..r {
        for k in 0..n {
            let p = ring.mul(&cs[i], &witness.w[i][k]);
            z[k] = z[k].add(m, &p);
        }
    }

    // We don't need a_agg in the proof — the verifier rebuilds it. But the
    // verifier also rebuilds the aggregated b. We package only the messages
    // the verifier can't compute itself.
    let _ = a_agg;

    IterationProof {
        v,
        b_double_prime,
        z,
        g,
        h,
    }
}

/// Bind the statement to the transcript (so `q'`, the constraint set, and
/// `β²` cannot be confused for another instance).
pub fn bind_statement(t: &mut Transcript, stmt: &Statement) {
    t.absorb(b"q'", &stmt.ring.m.q.to_le_bytes());
    t.absorb(b"n", &(stmt.n as u64).to_le_bytes());
    t.absorb(b"r", &(stmt.r as u64).to_le_bytes());
    t.absorb(b"beta_sq", &stmt.beta_sq.to_le_bytes());
    // Hash the constraint set via a SHAKE accumulator. We canonicalise full
    // and const-term constraints separately.
    t.absorb(b"|F|", &(stmt.full.len() as u64).to_le_bytes());
    for (k, c) in stmt.full.iter().enumerate() {
        t.absorb(b"F[k]", &(k as u64).to_le_bytes());
        absorb_a(t, &c.a);
        absorb_phi(t, &c.phi);
        absorb_ring_elem(t, b"b", &c.b);
    }
    t.absorb(b"|F'|", &(stmt.const_term.len() as u64).to_le_bytes());
    for (k, c) in stmt.const_term.iter().enumerate() {
        t.absorb(b"F'[k]", &(k as u64).to_le_bytes());
        absorb_a(t, &c.a);
        absorb_phi(t, &c.phi);
        t.absorb(b"b0", &c.b0.to_le_bytes());
    }
}

fn absorb_a(t: &mut Transcript, a: &[(usize, usize, RingElem)]) {
    t.absorb(b"|a|", &(a.len() as u64).to_le_bytes());
    for (i, j, coef) in a {
        let mut buf = Vec::with_capacity(16 + D * 8);
        buf.extend_from_slice(&(*i as u64).to_le_bytes());
        buf.extend_from_slice(&(*j as u64).to_le_bytes());
        for k in 0..D {
            buf.extend_from_slice(&coef.c[k].to_le_bytes());
        }
        t.absorb(b"a_ij", &buf);
    }
}

fn absorb_phi(t: &mut Transcript, phi: &[(usize, Vec<(usize, RingElem)>)]) {
    t.absorb(b"|phi|", &(phi.len() as u64).to_le_bytes());
    for (wi, entries) in phi {
        let mut buf = Vec::with_capacity(16 + entries.len() * (8 + D * 8));
        buf.extend_from_slice(&(*wi as u64).to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (pos, coef) in entries {
            buf.extend_from_slice(&(*pos as u64).to_le_bytes());
            for k in 0..D {
                buf.extend_from_slice(&coef.c[k].to_le_bytes());
            }
        }
        t.absorb(b"phi_i", &buf);
    }
}

fn absorb_ring_elem(t: &mut Transcript, label: &[u8], r: &RingElem) {
    let mut buf = Vec::with_capacity(D * 8);
    for k in 0..D {
        buf.extend_from_slice(&r.c[k].to_le_bytes());
    }
    t.absorb(label, &buf);
}

fn absorb_ring_vec(t: &mut Transcript, label: &[u8], v: &[RingElem]) {
    let mut buf = Vec::with_capacity(8 + v.len() * D * 8);
    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
    for r in v {
        for k in 0..D {
            buf.extend_from_slice(&r.c[k].to_le_bytes());
        }
    }
    t.absorb(label, &buf);
}

fn absorb_ring_matrix(t: &mut Transcript, label: &[u8], v: &[Vec<RingElem>]) {
    let mut buf = Vec::with_capacity(16 + v.iter().map(|r| r.len() * D * 8).sum::<usize>());
    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
    for row in v {
        buf.extend_from_slice(&(row.len() as u64).to_le_bytes());
        for r in row {
            for k in 0..D {
                buf.extend_from_slice(&r.c[k].to_le_bytes());
            }
        }
    }
    t.absorb(label, &buf);
}

fn sample_ring_element(
    t: &mut Transcript,
    label: &[u8],
    idx: u64,
    ring: &modring::Ring,
) -> RingElem {
    let mut e = RingElem::zero();
    for k in 0..D {
        let sub = [label, &idx.to_le_bytes(), &(k as u64).to_le_bytes()].concat();
        e.c[k] = t.derive_below(&sub, ring.m.q);
    }
    e
}

/// Combine `F` and `F''` into a single quadratic form `(a_agg, phi_agg)`.
pub fn aggregate_full(
    stmt: &Statement,
    alphas: &[RingElem],
    betas: &[RingElem],
    a_pp: &[Vec<Vec<RingElem>>],
    phi_pp: &[Vec<Vec<(usize, RingElem)>>],
) -> (Vec<Vec<RingElem>>, Vec<Vec<(usize, RingElem)>>) {
    let r = stmt.r;
    let m = &stmt.ring.m;
    let ring = &stmt.ring;

    // Each F constraint (indexed by k_f) and each F'' aggregator (indexed by
    // k_pp) contributes independently to (a_agg, phi_agg). Produce two streams
    // of sparse contributions in parallel, then merge serially. Sparse
    // representation keeps memory bounded by the contribution size (not r²).
    type AContrib = Vec<(usize, usize, RingElem)>;
    type PhiContrib = Vec<(usize, usize, RingElem)>; // (witness_idx, pos, scaled)

    let f_contribs: Vec<(AContrib, PhiContrib)> = stmt
        .full
        .par_iter()
        .enumerate()
        .filter_map(|(k, c)| {
            let alpha = &alphas[k];
            if alpha.is_zero() {
                return None;
            }
            let mut a_local: AContrib = Vec::with_capacity(c.a.len());
            for &(i, j, ref aij) in &c.a {
                a_local.push((i, j, ring.mul(alpha, aij)));
            }
            let mut phi_local: PhiContrib = Vec::new();
            for (wi, phi_i) in &c.phi {
                for (pos, coef) in phi_i {
                    phi_local.push((*wi, *pos, ring.mul(alpha, coef)));
                }
            }
            Some((a_local, phi_local))
        })
        .collect();

    let fpp_contribs: Vec<(AContrib, PhiContrib)> = betas
        .par_iter()
        .enumerate()
        .filter_map(|(k, beta)| {
            if beta.is_zero() {
                return None;
            }
            let mut a_local: AContrib = Vec::new();
            for i in 0..r {
                for j in 0..r {
                    let aij = &a_pp[k][i][j];
                    if aij.is_zero() {
                        continue;
                    }
                    a_local.push((i, j, ring.mul(beta, aij)));
                }
            }
            let mut phi_local: PhiContrib = Vec::new();
            for (wi, phi_i) in phi_pp[k].iter().enumerate() {
                for (pos, coef) in phi_i {
                    phi_local.push((wi, *pos, ring.mul(beta, coef)));
                }
            }
            Some((a_local, phi_local))
        })
        .collect();

    // Sequential merge: addition into the dense a_agg, push into the
    // per-witness phi buckets. Both are cheap relative to the parallel work.
    let mut a_agg: Vec<Vec<RingElem>> = vec![vec![RingElem::zero(); r]; r];
    let mut phi_agg: Vec<Vec<(usize, RingElem)>> = vec![vec![]; r];
    for (a_local, phi_local) in f_contribs.into_iter().chain(fpp_contribs.into_iter()) {
        for (i, j, scaled) in a_local {
            a_agg[i][j] = a_agg[i][j].add(m, &scaled);
        }
        for (wi, pos, scaled) in phi_local {
            phi_agg[wi].push((pos, scaled));
        }
    }

    // Fold duplicate (pos) entries.
    for bucket in phi_agg.iter_mut() {
        bucket.sort_by_key(|(p, _)| *p);
        let mut merged: Vec<(usize, RingElem)> = Vec::with_capacity(bucket.len());
        for (pos, coef) in bucket.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.0 == pos {
                    last.1 = last.1.add(m, &coef);
                    continue;
                }
            }
            merged.push((pos, coef));
        }
        *bucket = merged;
    }

    (a_agg, phi_agg)
}

/// Per-`k` ψ-aggregation of the constant-term constraints `F'`.
///
/// For each `k ∈ [k_pp]` this builds `(a_pp[k], phi_pp[k]) = Σ_l ψ_{k,l} · F'_l`,
/// the accumulated constraint that the prover, the verifier, and the fold
/// replay all rebuild identically.
///
/// The previous implementation parallelised **only** over `k_pp` (typically 3),
/// leaving most cores idle while each thread walked all ~2×10⁵ constraints
/// serially with a full `r × n` dense accumulator. Each `F'_l` is extremely
/// sparse — `a` is empty and `phi` touches just one or two (witness, position)
/// pairs — so instead we:
///
///   1. index the constraints once by witness id (`by_wi[wi]`), and
///   2. parallelise over the `k_pp × r` independent (k, witness) cells, each
///      accumulating only its own length-`n` row.
///
/// This gives ~`k_pp·r`-way parallelism with tiny per-task buffers (no 42 MB
/// dense scratch, no expensive dense merge). The `a` part — non-empty only for
/// the handful of quadratic constant-term constraints — is summed in a separate
/// `k`-parallel pass.
///
/// Re-association is exact: every accumulation is modular addition (commutative
/// + associative), so the result is byte-identical to the serial baseline. The
/// transcript-derived `psis` are computed by the caller and unchanged, so
/// soundness (challenge derivation order) is untouched.
pub(crate) fn aggregate_const_term_per_k(
    const_term: &[ConstTermConstraint],
    psis: &[Vec<u64>],
    k_pp: usize,
    r: usize,
    n: usize,
    m: &Modulus,
) -> (Vec<Vec<Vec<RingElem>>>, Vec<Vec<Vec<(usize, RingElem)>>>) {
    // Index constraint phi-rows by witness id, once (shared across all k).
    // `by_wi[wi]` holds (constraint index l, &phi-positions) for every F'_l
    // that touches witness wi.
    let mut by_wi: Vec<Vec<(usize, &Vec<(usize, RingElem)>)>> = vec![Vec::new(); r];
    for (l, c) in const_term.iter().enumerate() {
        for (wi, phi_i) in &c.phi {
            by_wi[*wi].push((l, phi_i));
        }
    }

    // phi_pp[k][wi]: parallelise over the flattened (k, wi) grid. Each cell
    // accumulates into a length-n dense row, then emits a sorted sparse row.
    let cells: Vec<(usize, usize, Vec<(usize, RingElem)>)> = (0..k_pp * r)
        .into_par_iter()
        .map(|idx| {
            let k = idx / r;
            let wi = idx % r;
            let psis_k = &psis[k];
            let mut row: Vec<Option<RingElem>> = vec![None; n];
            for &(l, phi_i) in &by_wi[wi] {
                let psi = psis_k[l];
                if psi == 0 {
                    continue;
                }
                for (pos, coef) in phi_i {
                    match &mut row[*pos] {
                        Some(existing) => existing.add_scaled_assign(m, coef, psi),
                        slot @ None => *slot = Some(coef.scale(m, psi)),
                    }
                }
            }
            let sparse: Vec<(usize, RingElem)> = row
                .into_iter()
                .enumerate()
                .filter_map(|(p, opt)| opt.map(|coef| (p, coef)))
                .collect();
            (k, wi, sparse)
        })
        .collect();
    let mut phi_pp: Vec<Vec<Vec<(usize, RingElem)>>> = vec![vec![Vec::new(); r]; k_pp];
    for (k, wi, sparse) in cells {
        phi_pp[k][wi] = sparse;
    }

    // a_pp[k]: only the rare quadratic constant-term constraints contribute.
    // Parallelise over k; the empty-`a` majority is skipped in O(1).
    let a_pp: Vec<Vec<Vec<RingElem>>> = (0..k_pp)
        .into_par_iter()
        .map(|k| {
            let psis_k = &psis[k];
            let mut a_k: Vec<Vec<RingElem>> = vec![vec![RingElem::zero(); r]; r];
            for (l, c) in const_term.iter().enumerate() {
                if c.a.is_empty() {
                    continue;
                }
                let psi = psis_k[l];
                if psi == 0 {
                    continue;
                }
                for &(i, j, ref aij) in &c.a {
                    a_k[i][j].add_scaled_assign(m, aij, psi);
                }
            }
            a_k
        })
        .collect();

    (a_pp, phi_pp)
}

fn ceil_log2(x: u64) -> u32 {
    if x <= 1 {
        return 0;
    }
    (x as f64).log2().ceil() as u32
}
