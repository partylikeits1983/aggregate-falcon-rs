//! Recursive fold: convert a finished v2 iteration into the next iteration's
//! statement + witness.
//!
//! Per paper §B.6 Step 5 (Recurse), once the prover has sent the last message
//! `(u_1, p, b'', u_2, z^(0), z^(1), v, g, h)`, the verifier's checks 3–9 of
//! Protocol 3 are reformulated as dot-product constraints over a new
//! witness `(z^(0), z^(1), ê)` where `ê = v ‖ g_chunks ‖ h_chunks`. The folded
//! witness vectors are obtained by chopping `z^(0)`, `z^(1)` into ν pieces and
//! `ê` into μ pieces (each padded to common rank `n' = max(⌈n/ν⌉, ⌈m/μ⌉)`).
//!
//! `fold` is called with a transcript at the SAME state as the one prove_v2 /
//! verify_v2 were entered with. It re-derives the iteration's challenges
//! (`A, B, C, D, Π, ψ, α, β, c_i`) by replaying the absorb / squeeze schedule.
//! The recursion driver in `aggregate_v2.rs` clones the transcript before
//! running prove_v2, then passes the clone here.

use crate::challenge::sample_challenge;
use crate::commit::{expand_b_mats, expand_matrix, expand_sym_mats};
use crate::garbage::decompose;
use crate::jl::{sample_projection, PROJECTION_ROWS};
use crate::params::{Iteration, Stage};
use crate::proof::IterationProofV2;
use crate::prover::{aggregate_full, bind_statement, k_double_prime};
use crate::statement::{DotConstraint, Statement, Witness};
use crate::transcript::Transcript;
use modring::{Modulus, Ring, RingElem, D};

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

/// Position of each chunk inside the flat `ê = v ‖ g_chunks ‖ h_chunks` layout.
#[derive(Copy, Clone, Debug)]
pub enum EIdx {
    /// v_chunks[i][k][idx] — the `k`-th decomposition chunk of `v_i`, position
    /// `idx ∈ [0, κ)` within that chunk.
    V(usize, usize, usize),
    /// g_chunks[i][j][k] for `i ≤ j` — the `k`-th decomposition chunk of `g_{ij}`.
    G(usize, usize, usize),
    /// h_chunks[i][j][k] for `i ≤ j`.
    H(usize, usize, usize),
}

/// Logical layout of the new witness BEFORE folding into ν/μ groups.
///
/// `e = v ‖ g_chunks ‖ h_chunks ∈ S^m` with
/// `m = r·t₁·κ + (t₁ + t₂) · r(r+1)/2`.
pub struct ELayout {
    pub r: usize,
    pub t1: usize,
    pub t2: usize,
    pub kappa: usize,
    pub m: usize,
    /// Offset of g_chunks block within e.
    pub g_off: usize,
    /// Offset of h_chunks block within e.
    pub h_off: usize,
    /// Number of upper-triangular (i, j) pairs with `i ≤ j`.
    pub n_upper: usize,
}

impl ELayout {
    pub fn new(r: usize, t1: usize, t2: usize, kappa: usize) -> Self {
        let n_upper = r * (r + 1) / 2;
        let g_off = r * t1 * kappa;
        let h_off = g_off + t2 * n_upper;
        let m = h_off + t1 * n_upper;
        Self { r, t1, t2, kappa, m, g_off, h_off, n_upper }
    }

    /// Upper-triangular index for the `(i, j)` pair (`i ≤ j`).
    fn upper(&self, i: usize, j: usize) -> usize {
        debug_assert!(i <= j);
        // Row-major upper triangle: rows `i ∈ [0, r)`, cols `j ∈ [i, r)`.
        // Count of pairs with row < i: i*r − i(i−1)/2.
        let row_offset = i * self.r - i * (i.saturating_sub(1)) / 2;
        row_offset + (j - i)
    }

    pub fn pos(&self, idx: EIdx) -> usize {
        match idx {
            EIdx::V(i, k, p) => {
                debug_assert!(i < self.r && k < self.t1 && p < self.kappa);
                i * self.t1 * self.kappa + k * self.kappa + p
            }
            EIdx::G(i, j, k) => {
                debug_assert!(i <= j && j < self.r && k < self.t2);
                self.g_off + k * self.n_upper + self.upper(i, j)
            }
            EIdx::H(i, j, k) => {
                debug_assert!(i <= j && j < self.r && k < self.t1);
                self.h_off + k * self.n_upper + self.upper(i, j)
            }
        }
    }
}

/// Maps an unfolded position to its folded (witness_idx, local_pos) location.
#[derive(Copy, Clone, Debug)]
pub struct ChopLayout {
    /// Number of pieces.
    pub pieces: usize,
    /// Length of each unpadded piece.
    pub piece_len: usize,
}

impl ChopLayout {
    /// Chop a length-`total` vector into `pieces` pieces of length
    /// `⌈total / pieces⌉`.
    pub fn new(total: usize, pieces: usize) -> Self {
        let piece_len = total.div_ceil(pieces);
        Self { pieces, piece_len }
    }

    pub fn locate(&self, p: usize) -> (usize, usize) {
        (p / self.piece_len, p % self.piece_len)
    }
}

pub struct FoldedLayout {
    /// Length of each new witness vector after padding.
    pub n_prime: usize,
    /// Number of new witness vectors, `r' = 2ν + μ`.
    pub r_prime: usize,
    pub nu: usize,
    pub mu: usize,
    /// Layout describing `z^(0)`, `z^(1)`.
    pub z: ChopLayout,
    /// Layout describing `e`.
    pub e: ChopLayout,
}

impl FoldedLayout {
    pub fn new(n: usize, m: usize, nu: usize, mu: usize) -> Self {
        let z = ChopLayout::new(n, nu);
        let e = ChopLayout::new(m, mu);
        let n_prime = z.piece_len.max(e.piece_len);
        let r_prime = 2 * nu + mu;
        Self { n_prime, r_prime, nu, mu, z, e }
    }

    /// Witness-index + local position for an entry of `z^(0)`.
    pub fn z0(&self, p: usize) -> (usize, usize) {
        let (piece, off) = self.z.locate(p);
        debug_assert!(piece < self.nu);
        (piece, off)
    }

    /// Witness-index + local position for an entry of `z^(1)`.
    pub fn z1(&self, p: usize) -> (usize, usize) {
        let (piece, off) = self.z.locate(p);
        debug_assert!(piece < self.nu);
        (self.nu + piece, off)
    }

    /// Witness-index + local position for an entry of `e`.
    pub fn e(&self, p: usize) -> (usize, usize) {
        let (piece, off) = self.e.locate(p);
        debug_assert!(piece < self.mu);
        (2 * self.nu + piece, off)
    }
}

/// Output of the fold: the next-iteration statement and its honest witness.
pub struct FoldOutput {
    pub statement: Statement,
    pub witness: Witness,
}

/// All challenges and aggregated quantities derived from an iteration's
/// transcript, in the order produced by Protocols 2 & 3.
pub struct IterationReplay {
    pub a_mat: Vec<Vec<RingElem>>,
    pub b_mats: Vec<Vec<Vec<Vec<RingElem>>>>,
    pub c_mats: Vec<Vec<Vec<Vec<RingElem>>>>,
    pub d_mats: Vec<Vec<Vec<Vec<RingElem>>>>,
    pub cs: Vec<RingElem>,
    /// Aggregated (a_{ij}, φ_i) of the combined F + F'' relation.
    pub a_agg: Vec<Vec<RingElem>>,
    pub phi_agg: Vec<Vec<(usize, RingElem)>>,
    pub b_agg: RingElem,
}

/// Replay every challenge of an iteration on the given transcript, returning
/// the derived quantities used by both `verify_v2` and `fold`. The transcript
/// must be at the state where `prove_v2` would have started.
///
/// After this call the transcript is at the state IMMEDIATELY AFTER `prove_v2`
/// finished — i.e. all prover messages absorbed, all challenges squeezed.
pub fn replay_iteration(
    stmt: &Statement,
    proof: &IterationProofV2,
    it_params: &Iteration,
    transcript: &mut Transcript,
) -> IterationReplay {
    bind_statement(transcript, stmt);

    let ring = &stmt.ring;
    let m = &ring.m;
    let n = stmt.n;
    let r = stmt.r;
    let kappa = it_params.kappa as usize;
    let kappa1 = it_params.kappa1 as usize;
    let t1 = it_params.t1 as usize;
    let t2 = it_params.t2 as usize;

    let a_mat = expand_matrix(transcript, LABEL_A, kappa, n, ring);
    let b_mats = expand_b_mats(transcript, LABEL_B, r, t1, kappa1, kappa, ring);
    let c_mats = expand_sym_mats(transcript, LABEL_C, r, t2, kappa1, ring);
    absorb_ring_vec(transcript, LABEL_U1, &proof.u1);

    for i in 0..r {
        let label = [LABEL_PI, &(i as u64).to_le_bytes()].concat();
        let _ = sample_projection(transcript, &label, n * D);
    }
    absorb_p_vec(transcript, LABEL_P, &proof.p);

    let lambda: u32 = 128;
    let k_pp = k_double_prime(stmt, lambda);
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
    absorb_ring_vec(transcript, LABEL_BPP, &proof.b_double_prime);

    let alphas: Vec<RingElem> = (0..stmt.full.len())
        .map(|k| sample_ring_element(transcript, LABEL_ALPHA, k as u64, ring))
        .collect();
    let betas: Vec<RingElem> = (0..k_pp)
        .map(|k| sample_ring_element(transcript, LABEL_BETA, k as u64, ring))
        .collect();

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
            merge_phi_bucket(bucket, m);
        }
    }
    let (a_agg, phi_agg) = aggregate_full(stmt, &alphas, &betas, &a_pp, &phi_pp);

    let mut b_agg = RingElem::zero();
    for (k, c) in stmt.full.iter().enumerate() {
        let term = ring.mul(&alphas[k], &c.b);
        b_agg = b_agg.add(m, &term);
    }
    for k in 0..k_pp {
        let term = ring.mul(&betas[k], &proof.b_double_prime[k]);
        b_agg = b_agg.add(m, &term);
    }

    let d_mats = expand_sym_mats(transcript, LABEL_D, r, t1, kappa1, ring);
    absorb_ring_vec(transcript, LABEL_U2, &proof.u2);

    let cs: Vec<RingElem> = (0..r)
        .map(|i| {
            let label = [LABEL_CHAL, &(i as u64).to_le_bytes()].concat();
            sample_challenge(transcript, &label, ring)
        })
        .collect();

    IterationReplay { a_mat, b_mats, c_mats, d_mats, cs, a_agg, phi_agg, b_agg }
}

/// Translate an iteration's verifier checks 3-9 into a new
/// `(statement, witness)`.
///
/// `transcript` must be at the same state as when `prove_v2` was entered.
/// After this call the transcript is at the post-iteration state, suitable for
/// chaining into the next iteration.
pub fn fold(
    stmt: &Statement,
    proof: &IterationProofV2,
    it_params: &Iteration,
    nu: usize,
    mu: usize,
    transcript: &mut Transcript,
) -> FoldOutput {
    assert!(
        matches!(it_params.stage, Stage::First | Stage::Mid),
        "fold called on SecLast iteration — there is no next iteration to fold into"
    );
    assert_eq!(it_params.t, 2, "fold expects iteration t=2 (z decomposed into 2 chunks)");

    let ring = stmt.ring;
    let m = ring.m;
    let n = stmt.n;
    let r = stmt.r;
    let kappa = it_params.kappa as usize;
    let kappa1 = it_params.kappa1 as usize;
    let b = it_params.b;
    let b1 = it_params.b1;
    let b2 = it_params.b2;
    let t1 = it_params.t1 as usize;
    let t2 = it_params.t2 as usize;

    let replay = replay_iteration(stmt, proof, it_params, transcript);

    // Re-derive chunk decompositions (same as verifier_v2 does).
    let v_chunks = decompose_v(&proof.v, &m, b1, t1, r, kappa);
    let g_chunks = decompose_sym(&proof.g, &m, b2, t2, r);
    let h_chunks = decompose_sym(&proof.h, &m, b1, t1, r);

    let e_layout = ELayout::new(r, t1, t2, kappa);
    let fold_layout = FoldedLayout::new(n, e_layout.m, nu, mu);

    // --- Build the folded witness ---
    let witness = build_folded_witness(
        &fold_layout,
        &e_layout,
        n,
        &proof.z0,
        &proof.z1,
        &v_chunks,
        &g_chunks,
        &h_chunks,
        r,
        t1,
        t2,
        kappa,
    );

    // --- Build the new statement's constraints ---
    let b_re = RingElem::constant(&m, b);
    let b_sq = RingElem::constant(&m, m.mul(b, b));
    let mut full: Vec<DotConstraint> = Vec::new();

    // Check 3: κ linear constraints — A·z = Σ c_i v_i.
    for k in 0..kappa {
        full.push(build_check3_row(
            k, &replay.a_mat, &replay.cs, &fold_layout, &e_layout,
            &b_re, b1, t1, &ring,
        ));
    }
    let _ = v_chunks; let _ = kappa1;
    // Check 4: ⟨z, z⟩ = Σ c_i c_j g_ij (quadratic + linear).
    full.push(build_check4(
        &replay.cs, &fold_layout, &e_layout, &b_re, &b_sq, b2, t2, r, &ring,
    ));
    // Check 5: Σ ⟨φ_i, z⟩ c_i = Σ c_i c_j h_ij (linear).
    full.push(build_check5(
        &replay.cs, &replay.phi_agg, &fold_layout, &e_layout, &b_re, b1, t1, r, &ring,
    ));
    // Check 6: Σ a_ij g_ij + Σ h_ii − b_agg = 0 (linear).
    full.push(build_check6(
        &replay.a_agg, &fold_layout, &e_layout, b1, b2, t1, t2, r, &replay.b_agg, &ring,
    ));
    // Check 8: u_1 openings (κ₁ linear, one per row).
    for r1 in 0..kappa1 {
        full.push(build_check8_row(
            r1, &proof.u1, &replay.b_mats, &replay.c_mats, &fold_layout, &e_layout, r, t1, t2, kappa, &ring,
        ));
    }
    // Check 9: u_2 openings (κ₁ linear, one per row).
    for r1 in 0..kappa1 {
        full.push(build_check9_row(
            r1, &proof.u2, &replay.d_mats, &fold_layout, &e_layout, r, t1, &ring,
        ));
    }

    let beta_sq = beta_prime_sq(it_params);
    let statement = Statement {
        ring,
        n: fold_layout.n_prime,
        r: fold_layout.r_prime,
        full,
        const_term: Vec::new(),
        beta_sq,
    };

    FoldOutput { statement, witness }
}

/// β'² = `next_beta_list[0]² + next_beta_list[1]²` as an i128.
pub fn beta_prime_sq(it_params: &Iteration) -> i128 {
    let nb0 = it_params.next_beta_list[0];
    let nb1 = it_params.next_beta_list[1];
    (nb0 * nb0 + nb1 * nb1).ceil() as i128
}

fn decompose_v(
    v: &[Vec<RingElem>],
    m: &Modulus,
    b1: u64,
    t1: usize,
    r: usize,
    kappa: usize,
) -> Vec<Vec<Vec<RingElem>>> {
    let mut out: Vec<Vec<Vec<RingElem>>> = Vec::with_capacity(r);
    for vi in v.iter() {
        let mut per_i: Vec<Vec<RingElem>> = vec![vec![RingElem::zero(); kappa]; t1];
        for (idx, e) in vi.iter().enumerate() {
            let chunks = decompose(e, m, b1, t1);
            for k in 0..t1 {
                per_i[k][idx] = chunks[k].clone();
            }
        }
        out.push(per_i);
    }
    out
}

fn decompose_sym(
    mat: &[Vec<RingElem>],
    m: &Modulus,
    base: u64,
    parts: usize,
    r: usize,
) -> Vec<Vec<Vec<RingElem>>> {
    let mut out: Vec<Vec<Vec<RingElem>>> = vec![vec![vec![]; r]; r];
    for i in 0..r {
        for j in i..r {
            out[i][j] = decompose(&mat[i][j], m, base, parts);
        }
    }
    out
}

fn build_folded_witness(
    fold_layout: &FoldedLayout,
    e_layout: &ELayout,
    n: usize,
    z0: &[RingElem],
    z1: &[RingElem],
    v_chunks: &[Vec<Vec<RingElem>>],
    g_chunks: &[Vec<Vec<RingElem>>],
    h_chunks: &[Vec<Vec<RingElem>>],
    r: usize,
    t1: usize,
    t2: usize,
    kappa: usize,
) -> Witness {
    let n_prime = fold_layout.n_prime;
    let r_prime = fold_layout.r_prime;
    let mut w: Vec<Vec<RingElem>> = (0..r_prime).map(|_| vec![RingElem::zero(); n_prime]).collect();
    for p in 0..n {
        let (wi, off) = fold_layout.z0(p);
        w[wi][off] = z0[p].clone();
        let (wi, off) = fold_layout.z1(p);
        w[wi][off] = z1[p].clone();
    }
    for i in 0..r {
        for k in 0..t1 {
            for idx in 0..kappa {
                let pos = e_layout.pos(EIdx::V(i, k, idx));
                let (wi, off) = fold_layout.e(pos);
                w[wi][off] = v_chunks[i][k][idx].clone();
            }
        }
    }
    for i in 0..r {
        for j in i..r {
            for k in 0..t2 {
                let pos = e_layout.pos(EIdx::G(i, j, k));
                let (wi, off) = fold_layout.e(pos);
                w[wi][off] = g_chunks[i][j][k].clone();
            }
            for k in 0..t1 {
                let pos = e_layout.pos(EIdx::H(i, j, k));
                let (wi, off) = fold_layout.e(pos);
                w[wi][off] = h_chunks[i][j][k].clone();
            }
        }
    }
    Witness { w }
}

/// Accumulator for sparse phi entries. Wraps a `Vec<Vec<(usize, RingElem)>>`
/// per witness index, with a finalisation step that sorts + merges duplicates.
struct PhiBuilder {
    buckets: Vec<Vec<(usize, RingElem)>>,
}

impl PhiBuilder {
    fn new(r: usize) -> Self {
        Self { buckets: vec![vec![]; r] }
    }

    fn add(&mut self, witness_idx: usize, pos: usize, coef: RingElem) {
        self.buckets[witness_idx].push((pos, coef));
    }

    fn add_scaled(&mut self, witness_idx: usize, pos: usize, coef: &RingElem, scalar: &RingElem, ring: &Ring) {
        let scaled = ring.mul(coef, scalar);
        self.add(witness_idx, pos, scaled);
    }

    fn finish(mut self, m: &Modulus) -> Vec<(usize, Vec<(usize, RingElem)>)> {
        let mut out = Vec::new();
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            merge_phi_bucket(bucket, m);
            // Drop merged zeros (rare but possible after add_scaled with cancelling pieces).
            bucket.retain(|(_, coef)| !coef.is_zero());
            if !bucket.is_empty() {
                out.push((i, std::mem::take(bucket)));
            }
        }
        out
    }
}

fn merge_phi_bucket(bucket: &mut Vec<(usize, RingElem)>, m: &Modulus) {
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

/// Check 3 (one row): `Σ_j A[k,j]·z[j] - Σ_i c_i Σ_l b₁^l v_chunks[i][l][k] = 0`,
/// where `z = z^(0) + b · z^(1)`.
#[allow(clippy::too_many_arguments)]
fn build_check3_row(
    k: usize,
    a_mat: &[Vec<RingElem>],
    cs: &[RingElem],
    fold_layout: &FoldedLayout,
    e_layout: &ELayout,
    b_re: &RingElem,
    b1: u64,
    t1: usize,
    ring: &Ring,
) -> DotConstraint {
    let m = &ring.m;
    let n = a_mat[k].len();
    let r = e_layout.r;
    let mut phi = PhiBuilder::new(fold_layout.r_prime);

    // Σ A[k][j] · z^(0)[j]  and  Σ b·A[k][j] · z^(1)[j].
    for j in 0..n {
        let coef = &a_mat[k][j];
        if coef.is_zero() {
            continue;
        }
        let (wi0, off0) = fold_layout.z0(j);
        phi.add(wi0, off0, coef.clone());
        let (wi1, off1) = fold_layout.z1(j);
        let coef_b = ring.mul(coef, b_re);
        phi.add(wi1, off1, coef_b);
    }

    // -Σ_i c_i · Σ_l b₁^l · v_chunks[i][l][k].
    let mut pow_b1 = 1u64;
    for l in 0..t1 {
        let pow_re = RingElem::constant(m, pow_b1);
        for i in 0..r {
            let scalar = ring.mul(&cs[i], &pow_re);
            let neg_scalar = scalar.neg(m);
            let pos = e_layout.pos(EIdx::V(i, l, k));
            let (wi, off) = fold_layout.e(pos);
            // φ coefficient is `-c_i · b₁^l · 1`. We store the negated scalar directly.
            // Use add_scaled with RingElem::constant(1).
            let one = RingElem::constant(m, 1);
            phi.add_scaled(wi, off, &one, &neg_scalar, ring);
        }
        pow_b1 = m.mul(pow_b1, b1);
    }

    DotConstraint { a: Vec::new(), phi: phi.finish(m), b: RingElem::zero() }
}

/// Check 4: `⟨z, z⟩ - Σ_{i,j} c_i c_j · Σ_l b₂^l g_chunks[i][j][l] = 0`.
#[allow(clippy::too_many_arguments)]
fn build_check4(
    cs: &[RingElem],
    fold_layout: &FoldedLayout,
    e_layout: &ELayout,
    b_re: &RingElem,
    b_sq: &RingElem,
    b2: u64,
    t2: usize,
    r: usize,
    ring: &Ring,
) -> DotConstraint {
    let m = &ring.m;
    let nu = fold_layout.nu;

    // a entries — quadratic part on the ν paired z-vectors.
    let one = RingElem::constant(m, 1);
    let mut a: Vec<(usize, usize, RingElem)> = Vec::with_capacity(3 * nu);
    for ai in 0..nu {
        // ⟨w'_a, w'_a⟩ (z^(0) self): coefficient 1.
        a.push((ai, ai, one.clone()));
        // ⟨w'_a, w'_{ν+a}⟩ cross: coefficient b (doubled by evaluator to 2b).
        a.push((ai, nu + ai, b_re.clone()));
        // ⟨w'_{ν+a}, w'_{ν+a}⟩ (z^(1) self): coefficient b².
        a.push((nu + ai, nu + ai, b_sq.clone()));
    }

    // -Σ_{i,j} c_i c_j · Σ_l b₂^l · g_chunks[i][j][l]. We iterate i ≤ j and
    // emit the diagonal once with c_i^2 and the off-diagonal once with 2·c_i·c_j
    // (the constraint is f(w) − b, with our φ being a flat sum: the symmetric
    // pair (i, j) and (j, i) both appear, contributing 2c_ic_j·g_{ij} when i<j).
    let mut phi = PhiBuilder::new(fold_layout.r_prime);
    for i in 0..r {
        let ci = &cs[i];
        for j in i..r {
            let cj = &cs[j];
            let mut cicj = ring.mul(ci, cj);
            if i != j {
                cicj = cicj.add(m, &cicj.clone()); // doubled by symmetry
            }
            let neg_cicj = cicj.neg(m);
            let mut pow_b2 = 1u64;
            for l in 0..t2 {
                let pow_re = RingElem::constant(m, pow_b2);
                let coef = ring.mul(&neg_cicj, &pow_re);
                let pos = e_layout.pos(EIdx::G(i, j, l));
                let (wi, off) = fold_layout.e(pos);
                phi.add(wi, off, coef);
                pow_b2 = m.mul(pow_b2, b2);
            }
        }
    }

    DotConstraint { a, phi: phi.finish(m), b: RingElem::zero() }
}

/// Check 5: `Σ_i ⟨φ_i^{agg}, z⟩ c_i - Σ_{i,j} c_i c_j Σ_l b₁^l h_chunks[i][j][l] = 0`.
#[allow(clippy::too_many_arguments)]
fn build_check5(
    cs: &[RingElem],
    phi_agg: &[Vec<(usize, RingElem)>],
    fold_layout: &FoldedLayout,
    e_layout: &ELayout,
    b_re: &RingElem,
    b1: u64,
    t1: usize,
    r: usize,
    ring: &Ring,
) -> DotConstraint {
    let m = &ring.m;
    let mut phi = PhiBuilder::new(fold_layout.r_prime);

    // Σ_i c_i · ⟨φ_i^{agg}, z⟩  with z = z^(0) + b · z^(1).
    for (i, phi_i) in phi_agg.iter().enumerate() {
        let ci = &cs[i];
        for (pos, coef) in phi_i.iter() {
            let ci_coef = ring.mul(ci, coef);
            let (wi0, off0) = fold_layout.z0(*pos);
            phi.add(wi0, off0, ci_coef.clone());
            let (wi1, off1) = fold_layout.z1(*pos);
            let ci_coef_b = ring.mul(&ci_coef, b_re);
            phi.add(wi1, off1, ci_coef_b);
        }
    }

    // -Σ_{i,j} c_i c_j · Σ_l b₁^l · h_chunks[i][j][l].
    for i in 0..r {
        let ci = &cs[i];
        for j in i..r {
            let cj = &cs[j];
            let mut cicj = ring.mul(ci, cj);
            if i != j {
                cicj = cicj.add(m, &cicj.clone());
            }
            let neg_cicj = cicj.neg(m);
            let mut pow_b1 = 1u64;
            for l in 0..t1 {
                let pow_re = RingElem::constant(m, pow_b1);
                let coef = ring.mul(&neg_cicj, &pow_re);
                let pos = e_layout.pos(EIdx::H(i, j, l));
                let (wi, off) = fold_layout.e(pos);
                phi.add(wi, off, coef);
                pow_b1 = m.mul(pow_b1, b1);
            }
        }
    }

    DotConstraint { a: Vec::new(), phi: phi.finish(m), b: RingElem::zero() }
}

/// Check 6: `Σ_{i ≤ j} a_{ij} g_{ij} + Σ_i h_{ii} − b_agg = 0`.
/// In LaBRADOR's `Σ a_{ij} ⟨w_i, w_j⟩` symmetric form, an off-diagonal `i<j`
/// pair counts twice; we mirror that here so it matches `verifier_v2::Check 4`.
#[allow(clippy::too_many_arguments)]
fn build_check6(
    a_agg: &[Vec<RingElem>],
    fold_layout: &FoldedLayout,
    e_layout: &ELayout,
    b1: u64,
    b2: u64,
    t1: usize,
    t2: usize,
    r: usize,
    b_agg: &RingElem,
    ring: &Ring,
) -> DotConstraint {
    let m = &ring.m;
    let mut phi = PhiBuilder::new(fold_layout.r_prime);

    for i in 0..r {
        for j in i..r {
            let aij = &a_agg[i][j];
            if !aij.is_zero() {
                let mut coef = aij.clone();
                if i != j {
                    coef = coef.add(m, &aij.clone());
                }
                let mut pow_b2 = 1u64;
                for l in 0..t2 {
                    let pow_re = RingElem::constant(m, pow_b2);
                    let final_coef = ring.mul(&coef, &pow_re);
                    let pos = e_layout.pos(EIdx::G(i, j, l));
                    let (wi, off) = fold_layout.e(pos);
                    phi.add(wi, off, final_coef);
                    pow_b2 = m.mul(pow_b2, b2);
                }
            }
        }
    }
    // Σ_i h_{ii} via Σ_l b₁^l · h_chunks[i][i][l].
    for i in 0..r {
        let mut pow_b1 = 1u64;
        for l in 0..t1 {
            let pow_re = RingElem::constant(m, pow_b1);
            let pos = e_layout.pos(EIdx::H(i, i, l));
            let (wi, off) = fold_layout.e(pos);
            phi.add(wi, off, pow_re);
            pow_b1 = m.mul(pow_b1, b1);
        }
    }

    DotConstraint { a: Vec::new(), phi: phi.finish(m), b: b_agg.clone() }
}

/// Check 8 (one row of u_1): pure linear in v_chunks and g_chunks.
#[allow(clippy::too_many_arguments)]
fn build_check8_row(
    r1: usize,
    u1: &[RingElem],
    b_mats: &[Vec<Vec<Vec<RingElem>>>],
    c_mats: &[Vec<Vec<Vec<RingElem>>>],
    fold_layout: &FoldedLayout,
    e_layout: &ELayout,
    r: usize,
    t1: usize,
    t2: usize,
    kappa: usize,
    ring: &Ring,
) -> DotConstraint {
    let m = &ring.m;
    let mut phi = PhiBuilder::new(fold_layout.r_prime);

    for i in 0..r {
        for k in 0..t1 {
            for idx in 0..kappa {
                let coef = &b_mats[i][k][r1][idx];
                if coef.is_zero() {
                    continue;
                }
                let pos = e_layout.pos(EIdx::V(i, k, idx));
                let (wi, off) = fold_layout.e(pos);
                phi.add(wi, off, coef.clone());
            }
        }
    }
    for i in 0..r {
        for j in i..r {
            for k in 0..t2 {
                let coef = &c_mats[i][j][k][r1];
                if coef.is_zero() {
                    continue;
                }
                let pos = e_layout.pos(EIdx::G(i, j, k));
                let (wi, off) = fold_layout.e(pos);
                phi.add(wi, off, coef.clone());
            }
        }
    }
    DotConstraint { a: Vec::new(), phi: phi.finish(m), b: u1[r1].clone() }
}

/// Check 9 (one row of u_2): pure linear in h_chunks.
fn build_check9_row(
    r1: usize,
    u2: &[RingElem],
    d_mats: &[Vec<Vec<Vec<RingElem>>>],
    fold_layout: &FoldedLayout,
    e_layout: &ELayout,
    r: usize,
    t1: usize,
    ring: &Ring,
) -> DotConstraint {
    let m = &ring.m;
    let mut phi = PhiBuilder::new(fold_layout.r_prime);

    for i in 0..r {
        for j in i..r {
            for k in 0..t1 {
                let coef = &d_mats[i][j][k][r1];
                if coef.is_zero() {
                    continue;
                }
                let pos = e_layout.pos(EIdx::H(i, j, k));
                let (wi, off) = fold_layout.e(pos);
                phi.add(wi, off, coef.clone());
            }
        }
    }
    DotConstraint { a: Vec::new(), phi: phi.finish(m), b: u2[r1].clone() }
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
