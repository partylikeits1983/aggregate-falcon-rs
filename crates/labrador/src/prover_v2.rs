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
use crate::garbage::{compute_g, compute_h, decompose};
use crate::jl::{build_jl_constraints, project_combined, sample_projection, PROJECTION_ROWS};
use crate::params::Iteration;
use crate::proof::{IterationLastMsg, IterationProofV2};
use crate::prover::{aggregate_full, bind_statement, k_double_prime};
use crate::statement::{ring_inner_product, sparse_phi_inner_product, Statement, Witness};
use crate::transcript::Transcript;
use modring::{RingElem, D};

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
pub fn prove_v2(
    stmt: &Statement,
    witness: &Witness,
    it_params: &Iteration,
    transcript: &mut Transcript,
) -> IterationProofV2 {
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
    let a_mat = expand_matrix(transcript, LABEL_A, kappa, n, ring);
    let v = commit_inner(ring, &a_mat, &witness.w);

    // Decompose each v_i (length κ) into t1 chunks base b1.
    // v_chunks[i][k] = the k-th chunk of v_i, shape [κ].
    let mut v_chunks: Vec<Vec<Vec<RingElem>>> = Vec::with_capacity(r);
    for vi in v.iter() {
        let mut per_i: Vec<Vec<RingElem>> = vec![vec![RingElem::zero(); kappa]; t1];
        for (idx, e) in vi.iter().enumerate() {
            let chunks = decompose(e, m, b1, t1);
            for k in 0..t1 {
                per_i[k][idx] = chunks[k].clone();
            }
        }
        v_chunks.push(per_i);
    }

    // g_{ij} = ⟨w_i, w_j⟩ upper-triangular, decompose into t2 chunks base b2.
    let g = compute_g(ring, &witness.w);
    let mut g_chunks: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![]; r]; r];
    for i in 0..r {
        for j in i..r {
            g_chunks[i][j] = decompose(&g[i][j], m, b2, t2);
        }
    }

    let b_mats = expand_b_mats(transcript, LABEL_B, r, t1, kappa1, kappa, ring);
    let c_mats = expand_sym_mats(transcript, LABEL_C, r, t2, kappa1, ring);
    let u1_v = outer_commit_v(ring, &b_mats, &v_chunks);
    let u1_g = outer_commit_sym(ring, &c_mats, &g_chunks);
    let u1: Vec<RingElem> = (0..kappa1).map(|k| u1_v[k].add(m, &u1_g[k])).collect();
    absorb_ring_vec(transcript, LABEL_U1, &u1);

    // --- Step 2: JL projection ---
    // Per witness vector i, sample Π_i ∈ {-1, 0, +1}^{2λ × (n·D)}.
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

    // --- Step 3: aggregate F' const-term constraints ---
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

    let mut a_pp: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![RingElem::zero(); r]; r]; k_pp];
    let mut phi_pp: Vec<Vec<Vec<(usize, RingElem)>>> = vec![vec![vec![]; r]; k_pp];
    let mut b_double_prime: Vec<RingElem> = Vec::with_capacity(k_pp);
    for k in 0..k_pp {
        for (l, c) in const_term_extended.iter().enumerate() {
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
                    bucket.push((*pos, scaled));
                }
            }
        }
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

    // --- Step 4: aggregate F + F'' → (a_agg, phi_agg); commit h via u_2 ---
    let n_f = stmt.full.len();
    let alphas: Vec<RingElem> = (0..n_f)
        .map(|k| sample_ring_element(transcript, LABEL_ALPHA, k as u64, ring))
        .collect();
    let betas: Vec<RingElem> = (0..k_pp)
        .map(|k| sample_ring_element(transcript, LABEL_BETA, k as u64, ring))
        .collect();
    let (_a_agg, phi_agg) = aggregate_full(stmt, &alphas, &betas, &a_pp, &phi_pp);

    let h = compute_h(ring, &phi_agg, &witness.w);
    let mut h_chunks: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![]; r]; r];
    for i in 0..r {
        for j in i..r {
            h_chunks[i][j] = decompose(&h[i][j], m, b1, t1);
        }
    }
    let d_mats = expand_sym_mats(transcript, LABEL_D, r, t1, kappa1, ring);
    let u2 = outer_commit_sym(ring, &d_mats, &h_chunks);
    absorb_ring_vec(transcript, LABEL_U2, &u2);

    // --- Step 5: amortize z = Σ c_i w_i; decompose z = z^(0) + b·z^(1) ---
    let cs: Vec<RingElem> = (0..r)
        .map(|i| {
            let label = [LABEL_CHAL, &(i as u64).to_le_bytes()].concat();
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
    // Decompose each z[k] into 2 chunks base b. The lossless `decompose`
    // puts any overflow into the high chunk so that
    // recompose([z0, z1], b) = z holds — fold's Check 4 depends on this.
    let mut z0 = vec![RingElem::zero(); n];
    let mut z1 = vec![RingElem::zero(); n];
    for k in 0..n {
        let chunks = decompose(&z[k], m, b, 2);
        z0[k] = chunks[0].clone();
        z1[k] = chunks[1].clone();
    }

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

    IterationProofV2 {
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
    }
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
    let mut e = RingElem::zero();
    for k in 0..D {
        let sub = [label, &idx.to_le_bytes(), &(k as u64).to_le_bytes()].concat();
        e.c[k] = t.derive_below(&sub, ring.m.q);
    }
    e
}
