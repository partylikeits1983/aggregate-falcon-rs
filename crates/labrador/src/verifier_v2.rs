//! Paper-correct LaBRADOR single-iteration verifier (Protocol 3, p. 40).
//!
//! Mirrors `prover_v2.rs`: re-derives every transcript squeeze
//! (`Π, ψ, α, β, c_i`), recomposes `v, g, h` from their chunks, checks the
//! four amortized identities, the outer-commitment openings, the JL bound,
//! and the norm bound on decomposed pieces.
//!
//! v2 omissions (Session B scope):
//! - JL projection constraints in `F'` are NOT yet emitted; only the
//!   `‖p‖ ≤ √λ·β` bound on the supplied `p` is checked. Session E adds the
//!   per-coordinate constraints that prove `p` is consistent with `w`.

use crate::challenge::sample_challenge;
use crate::commit::{expand_b_mats, expand_matrix, expand_sym_mats, matmul, outer_commit_sym, outer_commit_v};
use crate::garbage::{decompose, recompose};
use crate::jl::{sample_projection, PROJECTION_ROWS};
use crate::params::Iteration;
use crate::proof::{IterationProofV2, VerifyError};
use crate::prover::{aggregate_full, bind_statement, k_double_prime};
use crate::statement::{ring_inner_product, sparse_phi_inner_product, Statement};
use crate::transcript::Transcript;
use modring::{Ring, RingElem, D};

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

pub fn verify_v2(
    stmt: &Statement,
    proof: &IterationProofV2,
    it_params: &Iteration,
    transcript: &mut Transcript,
) -> Result<(), VerifyError> {
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

    // Shape checks.
    if proof.u1.len() != kappa1 {
        return Err(VerifyError::ProofShape("u1 length"));
    }
    if proof.u2.len() != kappa1 {
        return Err(VerifyError::ProofShape("u2 length"));
    }
    if proof.p.len() != PROJECTION_ROWS {
        return Err(VerifyError::ProofShape("p length"));
    }
    if proof.z0.len() != n || proof.z1.len() != n {
        return Err(VerifyError::ProofShape("z0/z1 length"));
    }
    if proof.v.len() != r {
        return Err(VerifyError::ProofShape("v outer"));
    }
    for i in 0..r {
        if proof.v[i].len() != kappa {
            return Err(VerifyError::ProofShape("v κ"));
        }
    }
    if proof.g.len() != r || proof.h.len() != r {
        return Err(VerifyError::ProofShape("g/h outer"));
    }
    for i in 0..r {
        if proof.g[i].len() != r || proof.h[i].len() != r {
            return Err(VerifyError::ProofShape("g/h inner"));
        }
    }

    // --- Step 1 mirror: expand A, B, C; decompose v, g; check u_1 ---
    let a_mat = expand_matrix(transcript, LABEL_A, kappa, n, ring);
    let b_mats = expand_b_mats(transcript, LABEL_B, r, t1, kappa1, kappa, ring);
    let c_mats = expand_sym_mats(transcript, LABEL_C, r, t2, kappa1, ring);

    // Re-derive v_chunks via deterministic centered-base-b1 decomposition.
    let mut v_chunks: Vec<Vec<Vec<RingElem>>> = Vec::with_capacity(r);
    for vi in proof.v.iter() {
        let mut per_i: Vec<Vec<RingElem>> = vec![vec![RingElem::zero(); kappa]; t1];
        for (idx, e) in vi.iter().enumerate() {
            let chunks = decompose(e, m, b1, t1);
            for k in 0..t1 {
                per_i[k][idx] = chunks[k].clone();
            }
        }
        v_chunks.push(per_i);
    }
    let mut g_chunks: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![]; r]; r];
    for i in 0..r {
        for j in i..r {
            g_chunks[i][j] = decompose(&proof.g[i][j], m, b2, t2);
        }
    }
    let u1_v = outer_commit_v(ring, &b_mats, &v_chunks);
    let u1_g = outer_commit_sym(ring, &c_mats, &g_chunks);
    for k in 0..kappa1 {
        let expected = u1_v[k].add(m, &u1_g[k]);
        if expected != proof.u1[k] {
            return Err(VerifyError::OuterCommitment("u_1"));
        }
    }
    absorb_ring_vec(transcript, LABEL_U1, &proof.u1);

    // Verifier-side aliases for v, g (we use proof.v, proof.g directly).
    let v: &Vec<Vec<RingElem>> = &proof.v;
    let g_mat: &Vec<Vec<RingElem>> = &proof.g;

    // --- Step 2 mirror: re-derive Π_i, check JL bound on p ---
    let _pis: Vec<_> = (0..r)
        .map(|i| {
            let label = [LABEL_PI, &(i as u64).to_le_bytes()].concat();
            sample_projection(transcript, &label, n * D)
        })
        .collect();
    absorb_p_vec(transcript, LABEL_P, &proof.p);
    // ‖p‖² check: ≤ 128 · β² (jl_const is 120 in our params; we use 128
    // here as the standard λ from the Lemma 2.2 bound).
    let p_norm_sq: i128 = proof.p.iter().map(|x| (*x) * (*x)).sum();
    let p_bound: i128 = 128i128 * stmt.beta_sq;
    if p_norm_sq > p_bound {
        return Err(VerifyError::JlBound);
    }

    // --- Step 3 mirror: re-derive ψ, check const-term aggregation ---
    let lambda: u32 = 128;
    let k_pp = k_double_prime(stmt, lambda);
    if proof.b_double_prime.len() != k_pp {
        return Err(VerifyError::ProofShape("b'' length"));
    }
    let n_fp = stmt.const_term.len();
    let q = m.q;
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
    for k in 0..k_pp {
        let mut expected: u64 = 0;
        for (l, c) in stmt.const_term.iter().enumerate() {
            let prod = m.mul(psis[k][l], c.b0);
            expected = m.add(expected, prod);
        }
        if proof.b_double_prime[k].c[0] != expected {
            return Err(VerifyError::ConstTermMismatch(k));
        }
    }
    absorb_ring_vec(transcript, LABEL_BPP, &proof.b_double_prime);

    // --- Step 4 mirror: rebuild (a_pp, phi_pp), aggregate F + F'', commit u_2 ---
    let mut a_pp: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![RingElem::zero(); r]; r]; k_pp];
    let mut phi_pp: Vec<Vec<Vec<(usize, RingElem)>>> = vec![vec![vec![]; r]; k_pp];
    for k in 0..k_pp {
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
    }

    let alphas: Vec<RingElem> = (0..stmt.full.len())
        .map(|k| sample_ring_element(transcript, LABEL_ALPHA, k as u64, ring))
        .collect();
    let betas: Vec<RingElem> = (0..k_pp)
        .map(|k| sample_ring_element(transcript, LABEL_BETA, k as u64, ring))
        .collect();
    let (a_agg, phi_agg) = aggregate_full(stmt, &alphas, &betas, &a_pp, &phi_pp);

    // Aggregated b for the F + F'' identity.
    let mut b_agg = RingElem::zero();
    for (k, c) in stmt.full.iter().enumerate() {
        let term = ring.mul(&alphas[k], &c.b);
        b_agg = b_agg.add(m, &term);
    }
    for k in 0..k_pp {
        let term = ring.mul(&betas[k], &proof.b_double_prime[k]);
        b_agg = b_agg.add(m, &term);
    }

    // Decompose h_{ij} and check u_2 = Σ D h_chunks.
    let mut h_chunks: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![]; r]; r];
    for i in 0..r {
        for j in i..r {
            h_chunks[i][j] = decompose(&proof.h[i][j], m, b1, t1);
        }
    }
    let d_mats = expand_sym_mats(transcript, LABEL_D, r, t1, kappa1, ring);
    let u2_check = outer_commit_sym(ring, &d_mats, &h_chunks);
    for k in 0..kappa1 {
        if u2_check[k] != proof.u2[k] {
            return Err(VerifyError::OuterCommitment("u_2"));
        }
    }
    absorb_ring_vec(transcript, LABEL_U2, &proof.u2);

    let h_mat: &Vec<Vec<RingElem>> = &proof.h;

    // --- Step 5 mirror: sample c_i, recompose z, run four identity checks ---
    let cs: Vec<RingElem> = (0..r)
        .map(|i| {
            let label = [LABEL_CHAL, &(i as u64).to_le_bytes()].concat();
            sample_challenge(transcript, &label, ring)
        })
        .collect();

    // Recompose z[k] = z^(0)[k] + b · z^(1)[k] via the canonical centered-
    // base-b expansion. We use `recompose` so that any centering convention
    // baked into the prover's `decompose` is matched here exactly.
    let mut z: Vec<RingElem> = vec![RingElem::zero(); n];
    for k in 0..n {
        let chunks = vec![proof.z0[k].clone(), proof.z1[k].clone()];
        z[k] = recompose(&chunks, m, b);
    }

    // Check 1: A · z = Σ c_i v_i.
    let az = matmul(ring, &a_mat, &z);
    let mut expected = vec![RingElem::zero(); kappa];
    for i in 0..r {
        for k in 0..kappa {
            let p = ring.mul(&cs[i], &v[i][k]);
            expected[k] = expected[k].add(m, &p);
        }
    }
    for k in 0..kappa {
        if az[k] != expected[k] {
            return Err(VerifyError::InnerCommitmentOpening);
        }
    }

    // Check 2: ⟨z, z⟩ = Σ c_i c_j g_{ij}.
    let zz = ring_inner_product(ring, &z, &z);
    let mut expected_zz = RingElem::zero();
    for i in 0..r {
        for j in 0..r {
            let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
            let gij = &g_mat[lo][hi];
            let cicj = ring.mul(&cs[i], &cs[j]);
            let term = ring.mul(&cicj, gij);
            expected_zz = expected_zz.add(m, &term);
        }
    }
    if zz != expected_zz {
        return Err(VerifyError::QuadraticOpening);
    }

    // Check 3: Σ ⟨φ_i^{agg}, z⟩ c_i = Σ c_i c_j h_{ij}.
    let mut lhs = RingElem::zero();
    for (i, phi_i) in phi_agg.iter().enumerate() {
        if phi_i.is_empty() {
            continue;
        }
        let inner = sparse_phi_inner_product(ring, phi_i, &z);
        let term = ring.mul(&cs[i], &inner);
        lhs = lhs.add(m, &term);
    }
    let mut rhs = RingElem::zero();
    for i in 0..r {
        for j in 0..r {
            let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
            let hij = &h_mat[lo][hi];
            let cicj = ring.mul(&cs[i], &cs[j]);
            let term = ring.mul(&cicj, hij);
            rhs = rhs.add(m, &term);
        }
    }
    if lhs != rhs {
        return Err(VerifyError::LinearOpening);
    }

    // Check 4: Σ a_{ij}^{agg} g_{ij} + Σ h_{ii} = b_agg.
    let mut lhs4 = RingElem::zero();
    for i in 0..r {
        for j in i..r {
            let aij = &a_agg[i][j];
            if aij.is_zero() {
                continue;
            }
            let term = ring.mul(aij, &g_mat[i][j]);
            let mut contrib = term.clone();
            if i != j {
                contrib = contrib.add(m, &term);
            }
            lhs4 = lhs4.add(m, &contrib);
        }
    }
    for i in 0..r {
        lhs4 = lhs4.add(m, &h_mat[i][i]);
    }
    if lhs4 != b_agg {
        return Err(VerifyError::AggregatedRelation);
    }

    // Check 5: norm of decomposed pieces ≤ β'².
    // We use the chunks computed above (v_chunks, g_chunks, h_chunks) — these
    // are the centered base-b decompositions that the prover hashed into u_1, u_2.
    let next_b0 = it_params.next_beta_list[0];
    let next_b1 = it_params.next_beta_list[1];
    let next_beta_sq = (next_b0 * next_b0 + next_b1 * next_b1) as i128;
    let mut norm_sq: i128 = 0;
    for k in 0..n {
        norm_sq += proof.z0[k].norm_sq(m);
        norm_sq += proof.z1[k].norm_sq(m);
    }
    for i in 0..r {
        for k in 0..t1 {
            for idx in 0..kappa {
                norm_sq += v_chunks[i][k][idx].norm_sq(m);
            }
        }
    }
    for i in 0..r {
        for j in i..r {
            for k in 0..t2 {
                norm_sq += g_chunks[i][j][k].norm_sq(m);
            }
            for k in 0..t1 {
                norm_sq += h_chunks[i][j][k].norm_sq(m);
            }
        }
    }
    if norm_sq > next_beta_sq {
        return Err(VerifyError::NormBound);
    }

    Ok(())
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

fn sample_ring_element(t: &mut Transcript, label: &[u8], idx: u64, ring: &Ring) -> RingElem {
    let mut e = RingElem::zero();
    for k in 0..D {
        let sub = [label, &idx.to_le_bytes(), &(k as u64).to_le_bytes()].concat();
        e.c[k] = t.derive_below(&sub, ring.m.q);
    }
    e
}
