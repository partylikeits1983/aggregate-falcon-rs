//! Paper-correct LaBRADOR single-iteration prover (Protocol 2, pp. 39).
//!
//! Differences from v1 (`prover.rs`):
//! - **Outer commitment.** `v_i` (decomposed into `t₁` chunks base `b₁`) and
//!   `g_{ij}` (decomposed into `t₂` chunks base `b₂`) are committed via
//!   `u_1 = Σ B v + Σ C g`. `h_{ij}` (decomposed into `t₁` chunks base `b₁`)
//!   is committed via `u_2 = Σ D h`.
//! - **JL projection.** Step 2 emits `p = Σ Π_i · τ(w_i) ∈ Z^{2λ}` for the
//!   verifier to bound. (JL *constraints* added to F' are deferred to
//!   Session E; Session B only emits the projection vector.)
//! - **Witness decomposition.** The amortized opening `z = Σ c_i w_i` is
//!   decomposed `z = z^{(0)} + b · z^{(1)}` (base `b` from params).
//! - **Final message.** Sends `(u_1, p, b'', u_2, z^{(0)}, z^{(1)}, v_chunks,
//!   g_chunks, h_chunks)`. Still single-iteration; folding is Sessions C+.
//!
//! The transcript order is fixed in `transcript.rs:10-27` — soundness
//! depends on absorbing every prover message before any verifier challenge
//! derived from it is squeezed.

use crate::challenge::sample_challenge;
use crate::commit::{commit_inner, expand_b_mats, expand_matrix, expand_sym_mats, outer_commit_sym, outer_commit_v};
use crate::fold::IterationReplay;
use crate::garbage::{compute_g, compute_h, decompose};
use crate::jl::{build_jl_constraints, project_combined, sample_projection, PROJECTION_ROWS};
use crate::params::Iteration;
use crate::proof::{IterationLastMsg, IterationProofV2};
use crate::prover::{aggregate_full, bind_statement, k_double_prime};
use crate::stage_timing;
use crate::statement::{sparse_phi_inner_product, Statement, Witness};
use crate::transcript::Transcript;
use modring::{RingElem, D};
use rayon::prelude::*;
use std::time::Instant;

const LABEL_A: &[u8] = b"labrador.A";
const LABEL_B: &[u8] = b"labrador.B";
const LABEL_C: &[u8] = b"labrador.C";
const LABEL_D: &[u8] = b"labrador.D";
const LABEL_U1: &[u8] = b"labrador.u1";
const LABEL_PI: &[u8] = b"labrador.Pi";
const LABEL_P: &[u8] = b"labrador.p";
const LABEL_PSI: &[u8] = b"labrador.psi";
const LABEL_BPP: &[u8] = b"labrador.b''";
const LABEL_ALPHA: &[u8] = b"labrador.alpha";
const LABEL_BETA: &[u8] = b"labrador.beta";
const LABEL_U2: &[u8] = b"labrador.u2";
const LABEL_CHAL: &[u8] = b"labrador.c";

/// Run the v2 single-iteration prover with outer commitments + JL projection.
///
/// `it_params` carries the per-iteration parameters
/// (`b, t, b1, t1, b2, t2, kappa, kappa1`) from `Params::for_n(N).iterations[0]`.
///
/// Thin wrapper around [`prove_v2_with_replay`] that drops the
/// [`IterationReplay`]. Callers that want to feed the prover's matrix
/// expansions and aggregated relation straight into [`crate::fold::fold_with_replay`]
/// (saving one full transcript walk per iteration in the multi-iteration
/// driver) should use [`prove_v2_with_replay`] directly.
pub fn prove_v2(
    stmt: &Statement,
    witness: &Witness,
    it_params: &Iteration,
    transcript: &mut Transcript,
) -> IterationProofV2 {
    prove_v2_with_replay(stmt, witness, it_params, transcript).0
}

/// Like [`prove_v2`] but also returns the [`IterationReplay`] holding every
/// transcript-derived matrix and aggregated quantity the prover produced. The
/// fold step can consume this directly via [`crate::fold::fold_with_replay`]
/// to skip its own redundant transcript walk.
pub fn prove_v2_with_replay(
    stmt: &Statement,
    witness: &Witness,
    it_params: &Iteration,
    transcript: &mut Transcript,
) -> (IterationProofV2, IterationReplay) {
    bind_statement(transcript, stmt);

    let ring = &stmt.ring;
    let m = &ring.m;
    let n = stmt.n;
    let r = stmt.r;
    let kappa = it_params.kappa as usize;
    let kappa1 = it_params.kappa1 as usize;
    let b = it_params.b;
    let b1 = it_params.b1;
    let b2 = it_params.b2;
    let t1 = it_params.t1 as usize;
    let t2 = it_params.t2 as usize;

    // --- Step 1: inner commitments v_i = A·w_i, decompose, outer-commit u_1 ---
    let t = Instant::now();
    let a_mat = expand_matrix(transcript, LABEL_A, kappa, n, ring);
    stage_timing::record("expand_A", t.elapsed());

    let t = Instant::now();
    let v = commit_inner(ring, &a_mat, &witness.w);
    stage_timing::record("commit_inner", t.elapsed());

    // Decompose each v_i (length κ) into t1 chunks base b1.
    // v_chunks[i][k] = the k-th chunk of v_i, shape [κ]. Per-`i` independent.
    let t = Instant::now();
    let v_chunks: Vec<Vec<Vec<RingElem>>> = v
        .par_iter()
        .map(|vi| {
            let mut per_i: Vec<Vec<RingElem>> = vec![vec![RingElem::zero(); kappa]; t1];
            for (idx, e) in vi.iter().enumerate() {
                let chunks = decompose(e, m, b1, t1);
                for k in 0..t1 {
                    per_i[k][idx] = chunks[k].clone();
                }
            }
            per_i
        })
        .collect();
    stage_timing::record("decompose_v", t.elapsed());

    // g_{ij} = ⟨w_i, w_j⟩ upper-triangular, decompose into t2 chunks base b2.
    let t = Instant::now();
    let g = compute_g(ring, &witness.w);
    stage_timing::record("compute_g", t.elapsed());
    let g_pairs: Vec<(usize, usize)> = (0..r)
        .flat_map(|i| (i..r).map(move |j| (i, j)))
        .collect();
    let g_chunk_entries: Vec<(usize, usize, Vec<RingElem>)> = g_pairs
        .par_iter()
        .map(|&(i, j)| (i, j, decompose(&g[i][j], m, b2, t2)))
        .collect();
    let mut g_chunks: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![]; r]; r];
    for (i, j, chunks) in g_chunk_entries {
        g_chunks[i][j] = chunks;
    }

    let t = Instant::now();
    let b_mats = expand_b_mats(transcript, LABEL_B, r, t1, kappa1, kappa, ring);
    let c_mats = expand_sym_mats(transcript, LABEL_C, r, t2, kappa1, ring);
    let u1_v = outer_commit_v(ring, &b_mats, &v_chunks);
    let u1_g = outer_commit_sym(ring, &c_mats, &g_chunks);
    let u1: Vec<RingElem> = (0..kappa1).map(|k| u1_v[k].add(m, &u1_g[k])).collect();
    absorb_ring_vec(transcript, LABEL_U1, &u1);
    stage_timing::record("expand_BC + outer_commit_u1", t.elapsed());

    // --- Step 2: JL projection ---
    // Per witness vector i, sample Π_i ∈ {-1, 0, +1}^{2λ × (n·D)}.
    let t = Instant::now();
    let pis: Vec<Vec<Vec<(usize, i8)>>> = (0..r)
        .map(|i| {
            let label = [LABEL_PI, &(i as u64).to_le_bytes()].concat();
            sample_projection(transcript, &label, n * D)
        })
        .collect();
    let p_arr = project_combined(&pis, &witness.w, m);
    let p: Vec<i128> = p_arr.iter().copied().collect();
    absorb_p_vec(transcript, LABEL_P, &p);

    // Build the 2λ JL-projection constraints (Protocol 2 §B.6 Step 2) and
    // splice them into F' so the ψ-aggregation downstream ties `p` to the
    // witness. The original `stmt` is not mutated.
    let jl_extra = build_jl_constraints(&pis, &p, n, m);
    let const_term_extended: Vec<_> = stmt
        .const_term
        .iter()
        .cloned()
        .chain(jl_extra.into_iter())
        .collect();
    stage_timing::record("jl_project", t.elapsed());

    // --- Step 3: aggregate F' const-term constraints ---
    let t = Instant::now();
    let lambda: u32 = 128;
    let k_pp = k_double_prime(stmt, lambda);
    let q = m.q;
    let n_fp = const_term_extended.len();
    let psis: Vec<Vec<u64>> = (0..k_pp)
        .map(|k| {
            (0..n_fp)
                .map(|l| {
                    let label = [LABEL_PSI, &(k as u64).to_le_bytes(), &(l as u64).to_le_bytes()].concat();
                    transcript.derive_field(&label, q)
                })
                .collect()
        })
        .collect();

    // Each k builds an independent (a_pp[k], phi_pp[k], b_double_prime[k]).
    // Parallelise over k and collect at the end so insertion order matches the
    // serial baseline.
    let per_k: Vec<(Vec<Vec<RingElem>>, Vec<Vec<(usize, RingElem)>>, RingElem)> = (0..k_pp)
        .into_par_iter()
        .map(|k| {
            let mut a_pp_k: Vec<Vec<RingElem>> = vec![vec![RingElem::zero(); r]; r];
            // Dense accumulators indexed by (wi, pos). One Option per slot lets
            // us tell "uninitialized" from "zero" cheaply; the dense layout
            // replaces the previous push+sort+merge that dominated wall-clock.
            let mut phi_dense: Vec<Vec<Option<RingElem>>> = (0..r)
                .map(|_| (0..n).map(|_| None).collect())
                .collect();
            for (l, c) in const_term_extended.iter().enumerate() {
                let psi = psis[k][l];
                if psi == 0 {
                    continue;
                }
                for &(i, j, ref aij) in &c.a {
                    a_pp_k[i][j].add_scaled_assign(m, aij, psi);
                }
                for (wi, phi_i) in &c.phi {
                    let bucket = &mut phi_dense[*wi];
                    for (pos, coef) in phi_i {
                        match &mut bucket[*pos] {
                            Some(existing) => existing.add_scaled_assign(m, coef, psi),
                            slot @ None => *slot = Some(coef.scale(m, psi)),
                        }
                    }
                }
            }
            // Convert dense -> sparse. Already sorted by `pos` because we walk
            // the dense Vec in order.
            let phi_pp_k: Vec<Vec<(usize, RingElem)>> = phi_dense
                .into_iter()
                .map(|bucket| {
                    bucket
                        .into_iter()
                        .enumerate()
                        .filter_map(|(p, opt)| opt.map(|coef| (p, coef)))
                        .collect()
                })
                .collect();

            let mut b_full = RingElem::zero();
            for i in 0..r {
                for j in i..r {
                    let coef = &a_pp_k[i][j];
                    if coef.is_zero() {
                        continue;
                    }
                    // g[i][j] = ⟨w_i, w_j⟩ was already computed above for the
                    // garbage commitment; reuse it instead of recomputing the
                    // D=64 ring multiplications per k_pp.
                    let mut term = ring.mul(coef, &g[i][j]);
                    if i != j {
                        term = term.add(m, &term.clone());
                    }
                    b_full = b_full.add(m, &term);
                }
            }
            for (i, phi_i) in phi_pp_k.iter().enumerate() {
                if phi_i.is_empty() {
                    continue;
                }
                let ip = sparse_phi_inner_product(ring, phi_i, &witness.w[i]);
                b_full = b_full.add(m, &ip);
            }
            (a_pp_k, phi_pp_k, b_full)
        })
        .collect();

    let mut a_pp: Vec<Vec<Vec<RingElem>>> = Vec::with_capacity(k_pp);
    let mut phi_pp: Vec<Vec<Vec<(usize, RingElem)>>> = Vec::with_capacity(k_pp);
    let mut b_double_prime: Vec<RingElem> = Vec::with_capacity(k_pp);
    for (a_k, phi_k, b_k) in per_k {
        a_pp.push(a_k);
        phi_pp.push(phi_k);
        b_double_prime.push(b_k);
    }
    absorb_ring_vec(transcript, LABEL_BPP, &b_double_prime);
    stage_timing::record("constraint_agg", t.elapsed());

    // --- Step 4: aggregate F + F'' → (a_agg, phi_agg); commit h via u_2 ---
    let t = Instant::now();
    let n_f = stmt.full.len();
    let alphas: Vec<RingElem> = (0..n_f)
        .map(|k| sample_ring_element(transcript, LABEL_ALPHA, k as u64, ring))
        .collect();
    let betas: Vec<RingElem> = (0..k_pp)
        .map(|k| sample_ring_element(transcript, LABEL_BETA, k as u64, ring))
        .collect();
    let (a_agg, phi_agg) = aggregate_full(stmt, &alphas, &betas, &a_pp, &phi_pp);
    // Compute b_agg = Σ α_k · b_k + Σ β_k · b''_k so the replay carries the
    // RHS of the aggregated F + F'' identity that fold_statement (Check 6)
    // needs. The verifier and the old replay_iteration both compute this
    // exact sum from the same alphas/betas; doing it here lets the fold's
    // build_check6 use the precomputed value.
    let mut b_agg = RingElem::zero();
    for (k, c) in stmt.full.iter().enumerate() {
        let term = ring.mul(&alphas[k], &c.b);
        b_agg = b_agg.add(m, &term);
    }
    for k in 0..k_pp {
        let term = ring.mul(&betas[k], &b_double_prime[k]);
        b_agg = b_agg.add(m, &term);
    }
    stage_timing::record("aggregate_full", t.elapsed());

    let t = Instant::now();
    let h = compute_h(ring, &phi_agg, &witness.w);
    stage_timing::record("compute_h", t.elapsed());
    let t = Instant::now();
    let h_pairs: Vec<(usize, usize)> = (0..r)
        .flat_map(|i| (i..r).map(move |j| (i, j)))
        .collect();
    let h_chunk_entries: Vec<(usize, usize, Vec<RingElem>)> = h_pairs
        .par_iter()
        .map(|&(i, j)| (i, j, decompose(&h[i][j], m, b1, t1)))
        .collect();
    let mut h_chunks: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![]; r]; r];
    for (i, j, chunks) in h_chunk_entries {
        h_chunks[i][j] = chunks;
    }
    let d_mats = expand_sym_mats(transcript, LABEL_D, r, t1, kappa1, ring);
    let u2 = outer_commit_sym(ring, &d_mats, &h_chunks);
    absorb_ring_vec(transcript, LABEL_U2, &u2);
    stage_timing::record("expand_D + outer_commit_u2", t.elapsed());

    // --- Step 5: amortize z = Σ c_i w_i; decompose z = z^(0) + b·z^(1) ---
    let t = Instant::now();
    let cs: Vec<RingElem> = (0..r)
        .map(|i| {
            let label = [LABEL_CHAL, &(i as u64).to_le_bytes()].concat();
            sample_challenge(transcript, &label, ring)
        })
        .collect();
    // z[k] = Σ_i c_i · w_i[k]. Parallelise over the n output positions; each
    // position reads (c_i, w_i[k]) independently.
    let z: Vec<RingElem> = (0..n)
        .into_par_iter()
        .map(|k| {
            let mut acc = RingElem::zero();
            for i in 0..r {
                let p = ring.mul(&cs[i], &witness.w[i][k]);
                acc = acc.add(m, &p);
            }
            acc
        })
        .collect();
    // Decompose each z[k] into 2 chunks base b. The lossless `decompose`
    // puts any overflow into the high chunk so that
    // recompose([z0, z1], b) = z holds — fold's Check 4 depends on this.
    let split: Vec<(RingElem, RingElem)> = (0..n)
        .into_par_iter()
        .map(|k| {
            let chunks = decompose(&z[k], m, b, 2);
            (chunks[0].clone(), chunks[1].clone())
        })
        .collect();
    let mut z0 = Vec::with_capacity(n);
    let mut z1 = Vec::with_capacity(n);
    for (a, bb) in split {
        z0.push(a);
        z1.push(bb);
    }
    stage_timing::record("z_amortize + decompose_z", t.elapsed());

    // Upper-triangular g matrix for the wire (lower triangle = zero).
    let mut g_wire: Vec<Vec<RingElem>> = vec![vec![RingElem::zero(); r]; r];
    for i in 0..r {
        for j in i..r {
            g_wire[i][j] = g[i][j].clone();
        }
    }
    // Same for h: keep upper triangle only.
    let mut h_wire: Vec<Vec<RingElem>> = vec![vec![RingElem::zero(); r]; r];
    for i in 0..r {
        for j in i..r {
            h_wire[i][j] = h[i][j].clone();
        }
    }

    let _ = (v_chunks, g_chunks, h_chunks); // chunks are transient; verifier re-derives.

    let proof = IterationProofV2 {
        u1,
        p,
        b_double_prime,
        u2,
        last_msg: Some(IterationLastMsg {
            z0,
            z1,
            v,
            g: g_wire,
            h: h_wire,
        }),
    };
    let replay = IterationReplay {
        a_mat,
        b_mats,
        c_mats,
        d_mats,
        cs,
        a_agg,
        phi_agg,
        b_agg,
    };
    (proof, replay)
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

fn absorb_p_vec(t: &mut Transcript, label: &[u8], v: &[i128]) {
    assert_eq!(v.len(), PROJECTION_ROWS);
    let mut buf = Vec::with_capacity(v.len() * 16);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    t.absorb(label, &buf);
}

fn sample_ring_element(t: &mut Transcript, label: &[u8], idx: u64, ring: &modring::Ring) -> RingElem {
    // One SHAKE squeeze per RingElem instead of D=64 separate `derive_below`
    // calls. Soundness is preserved because the label binds (call-site,
    // idx) into the transcript before the squeeze, and our protocol never
    // reuses the same `(label, idx)` pair within one transcript.
    let sub = [label, &idx.to_le_bytes()].concat();
    let coeffs = t.derive_field_array(&sub, D, ring.m.q);
    let mut e = RingElem::zero();
    for (k, v) in coeffs.into_iter().enumerate() {
        e.c[k] = v;
    }
    e
}
