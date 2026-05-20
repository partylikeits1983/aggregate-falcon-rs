//! LaBRADOR single-iteration verifier.
//!
//! Mirrors `prover.rs`'s engineering simplifications: no outer commitment,
//! no decomposition, single iteration. We re-derive every transcript squeeze
//! (`ψ, α, β, c_i`) and check the four amortized identities plus the
//! constant-term agreement of `b''^{(k)}` and the witness ℓ₂-bound on `z`.

use crate::challenge::sample_challenge;
use crate::commit::{expand_matrix, matmul};
use crate::proof::{IterationProof, VerifyError};
use crate::prover::{aggregate_full, bind_statement, k_double_prime, kappa};
use crate::statement::{ring_inner_product, sparse_phi_inner_product, Statement};
use crate::transcript::Transcript;
use modring::{Ring, RingElem, D};

const LABEL_A: &[u8] = b"labrador.A";
const LABEL_V: &[u8] = b"labrador.v";
const LABEL_PSI: &[u8] = b"labrador.psi";
const LABEL_BPP: &[u8] = b"labrador.b''";
const LABEL_ALPHA: &[u8] = b"labrador.alpha";
const LABEL_BETA: &[u8] = b"labrador.beta";
const LABEL_G: &[u8] = b"labrador.g";
const LABEL_H: &[u8] = b"labrador.h";
const LABEL_C: &[u8] = b"labrador.c";

pub fn verify(
    stmt: &Statement,
    proof: &IterationProof,
    transcript: &mut Transcript,
) -> Result<(), VerifyError> {
    bind_statement(transcript, stmt);

    let ring = &stmt.ring;
    let m = &ring.m;
    let n = stmt.n;
    let r = stmt.r;
    let kap = kappa();

    // Shape checks before anything else.
    if proof.v.len() != r || proof.v.iter().any(|vi| vi.len() != kap) {
        return Err(VerifyError::ProofShape("v shape"));
    }
    if proof.z.len() != n {
        return Err(VerifyError::ProofShape("z length"));
    }
    if proof.g.len() != r || proof.g.iter().any(|row| row.len() != r) {
        return Err(VerifyError::ProofShape("g shape"));
    }
    if proof.h.len() != r || proof.h.iter().any(|row| row.len() != r) {
        return Err(VerifyError::ProofShape("h shape"));
    }

    // --- Re-derive A and absorb v ---
    let a = expand_matrix(transcript, LABEL_A, kap, n, ring);
    absorb_ring_matrix(transcript, LABEL_V, &proof.v);

    // --- Re-derive ψ^{(k)} and check ct(b''^{(k)}) = b''_0^{(k)} ---
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
                    let label =
                        [LABEL_PSI, &(k as u64).to_le_bytes(), &(l as u64).to_le_bytes()].concat();
                    transcript.derive_field(&label, q)
                })
                .collect()
        })
        .collect();

    // Check ct(b''^{(k)}) = Σ ψ_l^{(k)} b'_0^{(l)} for each k.
    for k in 0..k_pp {
        let mut expected: u64 = 0;
        for (l, c) in stmt.const_term.iter().enumerate() {
            let prod = m.mul(psis[k][l], c.b0);
            expected = m.add(expected, prod);
        }
        let got = proof.b_double_prime[k].c[0];
        if got != expected {
            return Err(VerifyError::ConstTermMismatch(k));
        }
    }
    absorb_ring_vec(transcript, LABEL_BPP, &proof.b_double_prime);

    // --- Reconstruct (a''^{(k)}, phi''^{(k)}) and aggregate F + F'' ---
    let mut a_pp: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![RingElem::zero(); r]; r]; k_pp];
    let mut phi_pp: Vec<Vec<Vec<(usize, RingElem)>>> =
        vec![vec![vec![]; r]; k_pp];
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
        // Fold duplicates (same as prover for byte-equal aggregation).
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

    // --- α, β ---
    let alphas: Vec<RingElem> = (0..stmt.full.len())
        .map(|k| sample_ring_element(transcript, LABEL_ALPHA, k as u64, ring))
        .collect();
    let betas: Vec<RingElem> = (0..k_pp)
        .map(|k| sample_ring_element(transcript, LABEL_BETA, k as u64, ring))
        .collect();

    let (a_agg, phi_agg) = aggregate_full(stmt, &alphas, &betas, &a_pp, &phi_pp);

    // Aggregated b = Σ alpha_k · b^{(k)} + Σ beta_k · b''^{(k)}
    let mut b_agg = RingElem::zero();
    for (k, c) in stmt.full.iter().enumerate() {
        let term = ring.mul(&alphas[k], &c.b);
        b_agg = b_agg.add(m, &term);
    }
    for k in 0..k_pp {
        let term = ring.mul(&betas[k], &proof.b_double_prime[k]);
        b_agg = b_agg.add(m, &term);
    }

    absorb_ring_matrix(transcript, LABEL_G, &proof.g);
    absorb_ring_matrix(transcript, LABEL_H, &proof.h);

    // --- c_i ---
    let cs: Vec<RingElem> = (0..r)
        .map(|i| {
            let label = [LABEL_C, &(i as u64).to_le_bytes()].concat();
            sample_challenge(transcript, &label, ring)
        })
        .collect();

    // --- Check 1: A · z = Σ c_i v_i ---
    let az = matmul(ring, &a, &proof.z);
    let mut expected = vec![RingElem::zero(); kap];
    for i in 0..r {
        for k in 0..kap {
            let p = ring.mul(&cs[i], &proof.v[i][k]);
            expected[k] = expected[k].add(m, &p);
        }
    }
    for k in 0..kap {
        if az[k] != expected[k] {
            return Err(VerifyError::InnerCommitmentOpening);
        }
    }

    // --- Check 2: ⟨z, z⟩ = Σ c_i c_j g_{ij} ---
    let zz = ring_inner_product(ring, &proof.z, &proof.z);
    let mut expected_zz = RingElem::zero();
    for i in 0..r {
        for j in 0..r {
            let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
            let gij = &proof.g[lo][hi];
            let cicj = ring.mul(&cs[i], &cs[j]);
            let term = ring.mul(&cicj, gij);
            expected_zz = expected_zz.add(m, &term);
        }
    }
    if zz != expected_zz {
        return Err(VerifyError::QuadraticOpening);
    }

    // --- Check 3: Σ ⟨φ_i^{agg}, z⟩ c_i = Σ c_i c_j h_{ij} ---
    let mut lhs = RingElem::zero();
    for (i, phi_i) in phi_agg.iter().enumerate() {
        if phi_i.is_empty() {
            continue;
        }
        let inner = sparse_phi_inner_product(ring, phi_i, &proof.z);
        let term = ring.mul(&cs[i], &inner);
        lhs = lhs.add(m, &term);
    }
    let mut rhs = RingElem::zero();
    for i in 0..r {
        for j in 0..r {
            let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
            let hij = &proof.h[lo][hi];
            let cicj = ring.mul(&cs[i], &cs[j]);
            let term = ring.mul(&cicj, hij);
            rhs = rhs.add(m, &term);
        }
    }
    if lhs != rhs {
        return Err(VerifyError::LinearOpening);
    }

    // --- Check 4: Σ a_{ij}^{agg} g_{ij} + Σ h_{ii} = b ---
    let mut lhs4 = RingElem::zero();
    for i in 0..r {
        for j in i..r {
            let aij = &a_agg[i][j];
            if aij.is_zero() {
                continue;
            }
            let term = ring.mul(aij, &proof.g[i][j]);
            let mut contrib = term.clone();
            if i != j {
                contrib = contrib.add(m, &term);
            }
            lhs4 = lhs4.add(m, &contrib);
        }
    }
    for i in 0..r {
        lhs4 = lhs4.add(m, &proof.h[i][i]);
    }
    if lhs4 != b_agg {
        return Err(VerifyError::AggregatedRelation);
    }

    // --- Norm bound on z: a coarse `r · τ · β²` bound from the paper.
    // We use the witness norm bound directly since v1 doesn't decompose z.
    let z_norm_sq: i128 = proof.z.iter().map(|e| e.norm_sq(m)).sum();
    let bound = (r as i128) * (crate::challenge::TAU_SQ as i128) * stmt.beta_sq;
    if z_norm_sq > bound {
        return Err(VerifyError::NormBound);
    }

    Ok(())
}

fn sample_ring_element(
    t: &mut Transcript,
    label: &[u8],
    idx: u64,
    ring: &Ring,
) -> RingElem {
    let mut e = RingElem::zero();
    for k in 0..D {
        let sub = [label, &idx.to_le_bytes(), &(k as u64).to_le_bytes()].concat();
        e.c[k] = t.derive_below(&sub, ring.m.q);
    }
    e
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
