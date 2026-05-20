//! Building blocks for the LaBRADOR statement that asserts
//! "the N supplied Falcon-512 signatures verify."
//!
//! Two foundational operations live here:
//!
//! - [`compute_v_signed`] — the §6.1 *modulus-lifting* quotient `v_i`. The
//!   Falcon verification equation is mod `q = 12289`; LaBRADOR runs over a
//!   larger prime `q'`. We lift the equation to an exact integer identity
//!   `s1 + h·s2 + q·v_i − c = 0 ∈ R` by computing `v_i` so the residue
//!   cancels. The norm bound on `v_i` then guarantees no wrap-around mod `q'`.
//!
//! - [`four_square`] — Lagrange's four-square decomposition of a non-negative
//!   integer, used (§6.2) to express `β² − ‖s_{i,1}‖² − ‖s_{i,2}‖²` as
//!   `ε²_{i,0} + ε²_{i,1} + ε²_{i,2} + ε²_{i,3}`, encoding an *exact* norm
//!   proof via a quadratic ring constraint.
//!
//! Both are tested against real `pqcrypto-falcon` signatures: the lift
//! verifies `s1 + h·s2 + q·v − c ≡ 0 in S^8` after subring embedding, and
//! the decomposition's squares sum back to the input.

use crate::embed::{embed_fpoly_centered, embed_signed, C};
use crate::falcon_ring::{FPoly, FALCON_BETA_SQ, FALCON_N, FALCON_Q};
use crate::parse::FalconSig;
use labrador::statement::{ConstTermConstraint, DotConstraint, Statement, Witness};
use modring::{Modulus, Ring, RingElem, D};

/// Centered (signed) coefficients of a Falcon polynomial.
fn centered_vec(f: &FPoly) -> [i64; FALCON_N] {
    let mut out = [0i64; FALCON_N];
    for i in 0..FALCON_N {
        out[i] = f.centered(i) as i64;
    }
    out
}

/// Negacyclic product of two integer polynomials of degree < 512, modulo `X^512 + 1`.
///
/// No modular reduction on coefficients. With Falcon-512 magnitudes
/// (`|h|, |c| < q/2`, `|s| < ~2¹²`) all intermediate sums fit comfortably
/// in `i64`.
fn neg_mul_i64(a: &[i64; FALCON_N], b: &[i64; FALCON_N]) -> [i64; FALCON_N] {
    let mut out = [0i64; FALCON_N];
    for i in 0..FALCON_N {
        if a[i] == 0 {
            continue;
        }
        for j in 0..FALCON_N {
            let p = a[i] * b[j];
            let k = i + j;
            if k < FALCON_N {
                out[k] += p;
            } else {
                out[k - FALCON_N] -= p;
            }
        }
    }
    out
}

/// Compute the modulus-lifting quotient `v_i` for a single Falcon signature.
///
/// Returns the integer polynomial `v` (centered coefficients) satisfying
///
/// ```text
/// s1 + h·s2 − c = q · v        in R = Z[X]/(X^512 + 1)
/// ```
///
/// where `q = 12289`. **Precondition:** the input must satisfy the Falcon
/// verification equation over the integers — i.e. every coefficient of
/// `c − s1 − h·s2` must be divisible by `q`. A genuine signature decoded
/// through [`crate::parse::decode_instance`] always meets this, since the
/// decoder rebuilds `s1 = c − h·s2 (mod q)`. We enforce it with a real
/// release-active assertion so any future caller that fabricates `(s1, s2,
/// h, c)` cannot silently produce a truncated `v` with the wrong norm.
///
/// `c` is the `HashToPoint` output; we use its *centered* coefficients so
/// `v` ends up as small as possible.
pub fn compute_v_signed(
    s1: &FPoly,
    s2: &FPoly,
    h: &FPoly,
    c: &FPoly,
) -> [i64; FALCON_N] {
    let s1_i = centered_vec(s1);
    let s2_i = centered_vec(s2);
    let h_i = centered_vec(h);
    let c_i = centered_vec(c);
    let hs2 = neg_mul_i64(&h_i, &s2_i);
    let q = FALCON_Q as i64;
    let mut out = [0i64; FALCON_N];
    // Paper eq. (6) is s1 + h·s2 + q·v − c = 0, so v = (c − s1 − h·s2) / q.
    for i in 0..FALCON_N {
        let diff = c_i[i] - s1_i[i] - hs2[i];
        assert!(
            diff % q == 0,
            "compute_v_signed: Falcon verification equation must hold over the \
             integers; coeff {i} of (c - s1 - h·s2) = {diff} is not divisible by q={q}"
        );
        out[i] = diff / q;
    }
    out
}

/// Add a constant scalar (multiplied into each slot) to a subring-embedded
/// element: scales every slot of `x` by `k` mod `m.q`.
pub fn scale_slots(x: &[RingElem; C], k: u64, m: &Modulus) -> [RingElem; C] {
    let mut out = [
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
    ];
    for i in 0..C {
        out[i] = x[i].scale(m, k);
    }
    out
}

/// Coefficient-wise sum of two subring-embedded elements.
pub fn add_slots(a: &[RingElem; C], b: &[RingElem; C], m: &Modulus) -> [RingElem; C] {
    let mut out = [
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
    ];
    for i in 0..C {
        out[i] = a[i].add(m, &b[i]);
    }
    out
}

/// Coefficient-wise difference of two subring-embedded elements.
pub fn sub_slots(a: &[RingElem; C], b: &[RingElem; C], m: &Modulus) -> [RingElem; C] {
    let mut out = [
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
    ];
    for i in 0..C {
        out[i] = a[i].sub(m, &b[i]);
    }
    out
}

/// True if every slot is the zero polynomial in `S`.
pub fn all_slots_zero(x: &[RingElem; C]) -> bool {
    x.iter().all(|s| s.is_zero())
}

/// Lagrange's four-square decomposition: returns `(a, b, c, d)` with
/// `a² + b² + c² + d² = n`. Panics if `n < 0`.
///
/// This is correctness-first — a direct deterministic search over the
/// outermost two terms. For the magnitudes involved in §6.2
/// (`β² ≈ 5834² ≈ 3.4·10⁷`, so each square ≤ that) this is fast enough.
pub fn four_square(n: i64) -> (i64, i64, i64, i64) {
    assert!(n >= 0, "four_square requires a non-negative input");
    if n == 0 {
        return (0, 0, 0, 0);
    }
    // Try a from large to small; for each a, try b from large to small;
    // then test whether n - a² - b² is a sum of two squares.
    let isqrt = |x: i64| -> i64 { (x as f64).sqrt() as i64 };
    for a in (0..=isqrt(n)).rev() {
        let r = n - a * a;
        for b in (0..=isqrt(r)).rev() {
            let s = r - b * b;
            // Is s a sum of two squares?
            for c in 0..=isqrt(s) {
                let t = s - c * c;
                let d = isqrt(t);
                if d * d == t {
                    return (a, b, c, d);
                }
            }
        }
    }
    unreachable!("by Lagrange's theorem every non-negative integer is a sum of four squares");
}

/// Conjugation automorphism `σ_{-1}` on an `S = Z_{q'}[X]/(X^d+1)` element.
///
/// `σ_{-1}(a) = a(X^{-1}) = a_0 − a_{d-1}X − a_{d-2}X² − ⋯ − a_1 X^{d-1}` in
/// `S`. The same formula shape is used in §6.2(B) on `R`; we apply it slot-wise
/// in the subring picture so that the natural `S`-inner product on the
/// padded witness vectors recovers `R`-coefficient ℓ²-norms — i.e.
///
/// ```text
/// ct(⟨ϕ(s)_k, σ_{-1}(ϕ(s)_k)⟩_S) = ‖ϕ(s)_k‖²₂,
/// Σ_k ‖ϕ(s)_k‖²₂ = ‖s‖²_R.
/// ```
pub fn sigma_minus_one(s: &RingElem, m: &Modulus) -> RingElem {
    let mut r = RingElem::zero();
    r.c[0] = s.c[0];
    for j in 1..D {
        r.c[j] = m.neg(s.c[D - j]);
    }
    r
}

fn slot_array_sigma_minus_one(slots: &[RingElem; C], m: &Modulus) -> [RingElem; C] {
    let mut out = [
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
        RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
    ];
    for k in 0..C {
        out[k] = sigma_minus_one(&slots[k], m);
    }
    out
}

fn place_slots(dst: &mut [RingElem], pos: usize, src: &[RingElem; C]) {
    for k in 0..C {
        dst[pos + k] = src[k].clone();
    }
}

/// Indexing of the padded LaBRADOR witness for `N` Falcon signatures, per §F.1.
///
/// `ρ = round(√N)`, `num_y = ⌈N/ρ⌉`, `num_yp = ρ`, and `r = 3·num_y + 3·num_yp + 1`
/// witness vectors of `R`-rank `N` — which become S-rank `8·N` after the §6.4
/// subring decomposition. Each `R`-position `i ∈ [1, N]` lives at S-positions
/// `8(i-1)..8i-1` of the relevant witness vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WitnessLayout {
    /// Number of aggregated signatures `N`.
    pub n_sigs: usize,
    /// The √N rounding `ρ`.
    pub rho: usize,
    /// Number of `y_{·,j}` (and `e_·`) vectors per side: `⌈N/ρ⌉`.
    pub num_y: usize,
    /// Number of `y'_{·,j}` (and `e'_·`) vectors per side: `ρ`.
    pub num_yp: usize,
}

impl WitnessLayout {
    /// Build the layout for `N` signatures (`N ≥ 1`).
    pub fn new(n_sigs: usize) -> Self {
        assert!(n_sigs >= 1, "need at least one signature");
        let rho = ((n_sigs as f64).sqrt().round() as usize).max(1);
        let num_y = (n_sigs + rho - 1) / rho;
        let num_yp = rho;
        Self { n_sigs, rho, num_y, num_yp }
    }

    /// Number of witness vectors: `r = 3⌈N/ρ⌉ + 3ρ + 1`.
    pub fn r(&self) -> usize {
        3 * self.num_y + 3 * self.num_yp + 1
    }

    /// Length of each witness vector, in `S`: `8·N` (post-subring).
    pub fn n_s(&self) -> usize {
        C * self.n_sigs
    }

    /// Paper `index(i) = ⌈i/ρ⌉` (1-indexed).
    pub fn index(&self, i: usize) -> usize {
        debug_assert!(1 <= i && i <= self.n_sigs);
        (i + self.rho - 1) / self.rho
    }

    /// Paper `index'(i) = ((i-1) mod ρ) + 1` (1-indexed).
    pub fn index_prime(&self, i: usize) -> usize {
        debug_assert!(1 <= i && i <= self.n_sigs);
        ((i - 1) % self.rho) + 1
    }

    /// 0-indexed slot for `y_{i_y, j}` with `j ∈ {1, 2}` and `i_y ∈ [1, num_y]`.
    pub fn y_idx(&self, i_y: usize, j: usize) -> usize {
        debug_assert!(j == 1 || j == 2);
        debug_assert!(1 <= i_y && i_y <= self.num_y);
        (j - 1) * self.num_y + (i_y - 1)
    }

    /// 0-indexed slot for `y'_{i_yp, j}` with `j ∈ {1, 2}` and `i_yp ∈ [1, num_yp]`.
    pub fn yp_idx(&self, i_yp: usize, j: usize) -> usize {
        debug_assert!(j == 1 || j == 2);
        debug_assert!(1 <= i_yp && i_yp <= self.num_yp);
        2 * self.num_y + (j - 1) * self.num_yp + (i_yp - 1)
    }

    /// 0-indexed slot for `e_{i_y}`.
    pub fn e_idx(&self, i_y: usize) -> usize {
        debug_assert!(1 <= i_y && i_y <= self.num_y);
        2 * self.num_y + 2 * self.num_yp + (i_y - 1)
    }

    /// 0-indexed slot for `e'_{i_yp}`.
    pub fn ep_idx(&self, i_yp: usize) -> usize {
        debug_assert!(1 <= i_yp && i_yp <= self.num_yp);
        3 * self.num_y + 2 * self.num_yp + (i_yp - 1)
    }

    /// 0-indexed slot for the single `v` vector.
    pub fn v_idx(&self) -> usize {
        3 * self.num_y + 3 * self.num_yp
    }
}

/// Assemble the honest LaBRADOR witness from `N` decoded Falcon-512 signatures.
///
/// Each witness vector is a length-`8N` `S`-vector. Signature `i` (1-indexed)
/// places its `(s_{i,1}, s_{i,2}, s'_{i,1}, s'_{i,2}, ε_i, ε'_i, v_i)` content
/// at S-positions `8(i-1)..8i-1` of the vectors selected by [`WitnessLayout`].
/// The primed elements use the per-slot conjugation `σ_{-1}^S`, so that the
/// natural S-inner product `⟨y, y'⟩_S` recovers `‖s‖²_R`.
pub fn build_honest_witness(sigs: &[FalconSig], ring: &Ring) -> (Witness, WitnessLayout) {
    let layout = WitnessLayout::new(sigs.len());
    let m = &ring.m;

    let mut w: Vec<Vec<RingElem>> = (0..layout.r())
        .map(|_| vec![RingElem::zero(); layout.n_s()])
        .collect();

    for (sig_idx, sig) in sigs.iter().enumerate() {
        let i = sig_idx + 1;
        let i_y = layout.index(i);
        let i_yp = layout.index_prime(i);
        let pos = C * (i - 1);

        let s1_slots = embed_fpoly_centered(&sig.s1, m);
        let s2_slots = embed_fpoly_centered(&sig.s2, m);
        let s1p_slots = slot_array_sigma_minus_one(&s1_slots, m);
        let s2p_slots = slot_array_sigma_minus_one(&s2_slots, m);

        let v_signed = compute_v_signed(&sig.s1, &sig.s2, &sig.h, &sig.c);
        let v_slots = embed_signed(&v_signed, m);

        let n1 = sig.s1.norm_sq();
        let n2 = sig.s2.norm_sq();
        let remainder = FALCON_BETA_SQ - n1 - n2;
        assert!(remainder >= 0, "Falcon signature exceeds the published ℓ² bound");
        let (e0, e1, e2, e3) = four_square(remainder);
        let mut eps = [0i64; FALCON_N];
        eps[0] = e0;
        eps[1] = e1;
        eps[2] = e2;
        eps[3] = e3;
        let e_slots = embed_signed(&eps, m);
        let ep_slots = slot_array_sigma_minus_one(&e_slots, m);

        place_slots(&mut w[layout.y_idx(i_y, 1)], pos, &s1_slots);
        place_slots(&mut w[layout.y_idx(i_y, 2)], pos, &s2_slots);
        place_slots(&mut w[layout.yp_idx(i_yp, 1)], pos, &s1p_slots);
        place_slots(&mut w[layout.yp_idx(i_yp, 2)], pos, &s2p_slots);
        place_slots(&mut w[layout.e_idx(i_y)], pos, &e_slots);
        place_slots(&mut w[layout.ep_idx(i_yp)], pos, &ep_slots);
        place_slots(&mut w[layout.v_idx()], pos, &v_slots);
    }

    (Witness { w }, layout)
}

/// Append the §F.2 Falcon-verification constraints — eight `S`-full constraints
/// per signature, one per slot of the subring decomposition.
///
/// Each slot-`k` constraint encodes
/// `slot_k(s_{i,1} + h_i·s_{i,2} + q·v_i − c_i) = 0 ∈ S` as a pure linear
/// (φ-only) `DotConstraint` over the length-`8N` witness vectors. The `h·s_{i,2}`
/// contribution is the slot-`k` row of the explicit `mul_subring` bilinear form:
/// for each `b ∈ [0, 8)`, the coefficient of `s_{i,2}`'s slot `b` is
/// `h[k-b]` when `b ≤ k`, else `Y·h[k+8-b]`.
pub fn add_falcon_eq_constraints(
    stmt: &mut Statement,
    layout: &WitnessLayout,
    sigs: &[FalconSig],
    ring: &Ring,
) {
    let m = &ring.m;
    let q = FALCON_Q as u64;
    let y_mono = RingElem::monomial(m, 1);

    for (sig_idx, sig) in sigs.iter().enumerate() {
        let i = sig_idx + 1;
        let i_y = layout.index(i);
        let pos = C * (i - 1);

        let h_slots = embed_fpoly_centered(&sig.h, m);
        let c_slots = embed_fpoly_centered(&sig.c, m);

        let y1 = layout.y_idx(i_y, 1);
        let y2 = layout.y_idx(i_y, 2);
        let v = layout.v_idx();

        for k in 0..C {
            let phi_y1 = vec![(pos + k, RingElem::constant(m, 1))];
            let phi_v = vec![(pos + k, RingElem::constant(m, q))];
            let mut phi_y2: Vec<(usize, RingElem)> = Vec::with_capacity(C);
            for b in 0..C {
                let coef = if b <= k {
                    h_slots[k - b].clone()
                } else {
                    ring.mul(&y_mono, &h_slots[k + C - b])
                };
                if !coef.is_zero() {
                    phi_y2.push((pos + b, coef));
                }
            }

            stmt.full.push(DotConstraint {
                a: vec![],
                phi: vec![(y1, phi_y1), (y2, phi_y2), (v, phi_v)],
                b: c_slots[k].clone(),
            });
        }
    }
}

/// Append the §F.2 four-square constraints — one constant-term `S`-constraint
/// per signature checking `‖s_{i,1}‖² + ‖s_{i,2}‖² + ‖ε_i‖² = β²` in `Z_{q'}`.
///
/// Mechanism: with the witness primed-vectors set to `σ_{-1}^S(·)` slot-wise,
/// `ct(⟨y_{idx(i),j}, y'_{idx'(i),j}⟩_S) = ‖s_{i,j}‖²_R` (the padding makes a
/// single R-position contribute, and per-slot conjugation turns the diagonal
/// S-inner product into a coefficient ℓ²-norm). Each `(i, j)` pair is stored
/// once with `a_{i,j} = ½`; the evaluator's symmetric doubling then yields a
/// coefficient of `1` on each `⟨w_i, w_j⟩_S`.
pub fn add_four_square_constraints(
    stmt: &mut Statement,
    layout: &WitnessLayout,
    beta_sq: i64,
    ring: &Ring,
) {
    let m = &ring.m;
    let inv2 = RingElem::constant(m, m.inv(2));
    let beta_sq_residue = m.from_i64(beta_sq);

    for sig_idx in 0..layout.n_sigs {
        let i = sig_idx + 1;
        let i_y = layout.index(i);
        let i_yp = layout.index_prime(i);

        let pair = |a: usize, b: usize| -> (usize, usize, RingElem) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            (lo, hi, inv2.clone())
        };

        stmt.const_term.push(ConstTermConstraint {
            a: vec![
                pair(layout.y_idx(i_y, 1), layout.yp_idx(i_yp, 1)),
                pair(layout.y_idx(i_y, 2), layout.yp_idx(i_yp, 2)),
                pair(layout.e_idx(i_y), layout.ep_idx(i_yp)),
            ],
            phi: vec![],
            b0: beta_sq_residue,
        });
    }
}

/// Append §F.2 *padding* form constraints for the `y_{iy, j}` vectors.
///
/// `ȳ_{iy, j}` is non-zero only at R-positions `[(iy−1)·ρ + 1, iy·ρ] ∩ [1, N]`.
/// In the S-encoding each R-position spans 8 S-positions, so we pin every
/// S-position outside the active range to the zero S-element with a full
/// `DotConstraint` of the shape `w_{y_idx}[p] = 0`.
pub fn add_form_constraints_y_padding(
    stmt: &mut Statement,
    layout: &WitnessLayout,
    ring: &Ring,
) {
    let m = &ring.m;
    let one = RingElem::constant(m, 1);
    let zero_b = RingElem::zero();
    for i_y in 1..=layout.num_y {
        let active_lo = (i_y - 1) * layout.rho + 1;
        let active_hi = (i_y * layout.rho).min(layout.n_sigs);
        for j in 1..=2 {
            let yvec = layout.y_idx(i_y, j);
            for r_pos in 1..=layout.n_sigs {
                if r_pos >= active_lo && r_pos <= active_hi {
                    continue;
                }
                let base = C * (r_pos - 1);
                for s in 0..C {
                    stmt.full.push(DotConstraint {
                        a: vec![],
                        phi: vec![(yvec, vec![(base + s, one.clone())])],
                        b: zero_b.clone(),
                    });
                }
            }
        }
    }
}

/// Append §F.2 form constraints for the `y'_{iyp, j}` vectors:
/// (i) pin every padding S-position to zero, and (ii) tie `y'[p] = σ₋₁ˢ(y[p])`
/// at every active S-position `p`.
///
/// `ȳ'_{iyp, j}` is non-zero only at R-positions `k` with `k ≡ iyp (mod ρ)`.
/// At each such R-position `k`, the *honest* witness has
/// `y'[8(k−1) + s] = σ₋₁ˢ(y_{index(k), j}[8(k−1) + s])` for every slot `s`.
/// We enforce this with `d_S = 64` constant-term constraints per S-position:
///
/// ```text
/// ct(y[p] − y'[p]) = 0
/// ct(X^l · y[p] + X^{d_S − l} · y'[p]) = 0,   l ∈ [1, d_S − 1]
/// ```
///
/// (See the derivation in module-level rustdoc: `ct(X^l · a)` exposes
/// coefficient `−a[d_S − l]`, so the second family pins
/// `y'[p][l] = −y[p][d_S − l]`.)
pub fn add_form_constraints_yp(
    stmt: &mut Statement,
    layout: &WitnessLayout,
    ring: &Ring,
) {
    let m = &ring.m;
    let one = RingElem::constant(m, 1);
    let neg_one = RingElem::constant(m, m.neg(1));

    for i_yp in 1..=layout.num_yp {
        for j in 1..=2 {
            let ypvec = layout.yp_idx(i_yp, j);

            // Padding pins: r_pos NOT ≡ i_yp (mod ρ).
            for r_pos in 1..=layout.n_sigs {
                if layout.index_prime(r_pos) == i_yp {
                    continue;
                }
                let base = C * (r_pos - 1);
                for s in 0..C {
                    stmt.full.push(DotConstraint {
                        a: vec![],
                        phi: vec![(ypvec, vec![(base + s, one.clone())])],
                        b: RingElem::zero(),
                    });
                }
            }

            // σ₋₁ˢ ties at active S-positions: r_pos ≡ i_yp (mod ρ).
            for r_pos in 1..=layout.n_sigs {
                if layout.index_prime(r_pos) != i_yp {
                    continue;
                }
                let i_y_partner = layout.index(r_pos);
                let yvec = layout.y_idx(i_y_partner, j);
                let base = C * (r_pos - 1);
                for s in 0..C {
                    let p = base + s;
                    // ct(y[p] - y'[p]) = 0
                    stmt.const_term.push(ConstTermConstraint {
                        a: vec![],
                        phi: vec![
                            (yvec, vec![(p, one.clone())]),
                            (ypvec, vec![(p, neg_one.clone())]),
                        ],
                        b0: 0,
                    });
                    // ct(X^l · y[p] + X^{d_S - l} · y'[p]) = 0  for l in [1, d_S − 1]
                    for l in 1..D {
                        let xl = RingElem::monomial(m, l);
                        let xdl = RingElem::monomial(m, D - l);
                        stmt.const_term.push(ConstTermConstraint {
                            a: vec![],
                            phi: vec![
                                (yvec, vec![(p, xl)]),
                                (ypvec, vec![(p, xdl)]),
                            ],
                            b0: 0,
                        });
                    }
                }
            }
        }
    }
}

/// Append §F.2 form constraints for the `eᵢ` vectors.
///
/// The honest `eᵢ` carries ε₀..ε₃ from Lagrange's four-square at *slots 0..3*
/// (each slot's *constant term*, since `embed_signed` puts R-coefficient `r`
/// into slot `r mod 8`, inner-coefficient `r / 8`, and ε is supported on
/// R-coefficients 0..3). Slots 4..7 are entirely zero. We enforce:
///
/// - Padding R-positions: pin every S-position to zero.
/// - Active R-position `r_pos`: pin slots 4..7 to zero (full constraints),
///   and pin inner coefficients 1..d_S − 1 of slots 0..3 to zero
///   (constant-term constraints via `ct(X^{d_S − k} · e[p]) = −e[p][k]`).
pub fn add_form_constraints_e(
    stmt: &mut Statement,
    layout: &WitnessLayout,
    ring: &Ring,
) {
    let m = &ring.m;
    let one = RingElem::constant(m, 1);

    for i_y in 1..=layout.num_y {
        let active_lo = (i_y - 1) * layout.rho + 1;
        let active_hi = (i_y * layout.rho).min(layout.n_sigs);
        let evec = layout.e_idx(i_y);

        for r_pos in 1..=layout.n_sigs {
            let base = C * (r_pos - 1);
            if r_pos < active_lo || r_pos > active_hi {
                // padding: pin all 8 slots
                for s in 0..C {
                    stmt.full.push(DotConstraint {
                        a: vec![],
                        phi: vec![(evec, vec![(base + s, one.clone())])],
                        b: RingElem::zero(),
                    });
                }
            } else {
                // active R-position: pin slots 4..7 entirely
                for s in 4..C {
                    stmt.full.push(DotConstraint {
                        a: vec![],
                        phi: vec![(evec, vec![(base + s, one.clone())])],
                        b: RingElem::zero(),
                    });
                }
                // active slots 0..3: pin inner coefficients 1..d_S − 1.
                for s in 0..4 {
                    let p = base + s;
                    for k in 1..D {
                        let monomial = RingElem::monomial(m, D - k);
                        stmt.const_term.push(ConstTermConstraint {
                            a: vec![],
                            phi: vec![(evec, vec![(p, monomial)])],
                            b0: 0,
                        });
                    }
                }
            }
        }
    }
}

/// Append §F.2 form constraints for the `e'ᵢ` vectors.
///
/// The honest `e'ᵢ` is `σ₋₁ˢ(eᵢ)` slot-wise. Since `eᵢ` is constrained to be
/// the constant `ε_s` at slot `s ∈ [0, 3]` (and zero elsewhere), and
/// `σ₋₁ˢ(constant) = constant`, the honest `e'` mirrors `e` at slots 0..3
/// inner coefficient 0. We enforce only the four coef-0 ties: this is enough
/// for the four-square because `e` is already pinned to zero at every other
/// position, so any extra content in `e'` cannot enter the constant-term
/// inner product `ct(⟨e, e'⟩)` (paper §F.2: "it is enough to check that they
/// are 0 in ē_i").
///
/// - Padding R-positions: pin all 8 S-slots of `e'` to zero.
/// - Active R-position, slots 0..3: tie `ct(e[p] − e'[p]) = 0`. That's a
///   single constant-term constraint per slot.
pub fn add_form_constraints_ep(
    stmt: &mut Statement,
    layout: &WitnessLayout,
    ring: &Ring,
) {
    let m = &ring.m;
    let one = RingElem::constant(m, 1);
    let neg_one = RingElem::constant(m, m.neg(1));

    for i_yp in 1..=layout.num_yp {
        let epvec = layout.ep_idx(i_yp);

        // Padding pins.
        for r_pos in 1..=layout.n_sigs {
            if layout.index_prime(r_pos) == i_yp {
                continue;
            }
            let base = C * (r_pos - 1);
            for s in 0..C {
                stmt.full.push(DotConstraint {
                    a: vec![],
                    phi: vec![(epvec, vec![(base + s, one.clone())])],
                    b: RingElem::zero(),
                });
            }
        }

        // Full σ₋₁ˢ tie of e' to e at active R-positions, plus pinning of e'
        // outside the support of e. This makes the witness structurally rigid
        // (e' = σ₋₁ˢ(e) coefficient-by-coefficient) rather than only at the
        // four-square's required positions.
        for r_pos in 1..=layout.n_sigs {
            if layout.index_prime(r_pos) != i_yp {
                continue;
            }
            let i_y_partner = layout.index(r_pos);
            let evec = layout.e_idx(i_y_partner);
            let base = C * (r_pos - 1);

            for s in 0..4 {
                let p = base + s;
                // ct(e[p] − e'[p]) = 0 — c[0] tied.
                stmt.const_term.push(ConstTermConstraint {
                    a: vec![],
                    phi: vec![
                        (evec, vec![(p, one.clone())]),
                        (epvec, vec![(p, neg_one.clone())]),
                    ],
                    b0: 0,
                });
                // c[l] = -e[p][d_S - l] for l ∈ [1, d_S − 1]. Since e[p][k] = 0
                // for k ≥ 1 at active slots 0..3, this collapses to
                // ct(X^{d_S − l} · e'[p]) = -e'[p][l] = 0.
                for l in 1..D {
                    let xdl = RingElem::monomial(m, D - l);
                    stmt.const_term.push(ConstTermConstraint {
                        a: vec![],
                        phi: vec![(epvec, vec![(p, xdl)])],
                        b0: 0,
                    });
                }
            }
            // Pin slots 4..7 of e' entirely to zero (matches e at these slots).
            for s in 4..C {
                stmt.full.push(DotConstraint {
                    a: vec![],
                    phi: vec![(epvec, vec![(base + s, one.clone())])],
                    b: RingElem::zero(),
                });
            }
        }
    }
}

/// Build the LaBRADOR statement, honest witness, and layout for an aggregation
/// of `N` decoded Falcon-512 signatures.
///
/// `beta_sq` is the global ℓ²-norm bound on the entire padded witness.
pub fn build_falcon_statement(
    sigs: &[FalconSig],
    ring: &Ring,
    beta_sq: i128,
) -> (Statement, Witness, WitnessLayout) {
    let (witness, layout) = build_honest_witness(sigs, ring);
    let mut stmt = Statement {
        ring: *ring,
        n: layout.n_s(),
        r: layout.r(),
        full: vec![],
        const_term: vec![],
        beta_sq,
    };
    add_falcon_eq_constraints(&mut stmt, &layout, sigs, ring);
    add_four_square_constraints(&mut stmt, &layout, FALCON_BETA_SQ, ring);
    add_form_constraints_y_padding(&mut stmt, &layout, ring);
    add_form_constraints_yp(&mut stmt, &layout, ring);
    add_form_constraints_e(&mut stmt, &layout, ring);
    add_form_constraints_ep(&mut stmt, &layout, ring);
    (stmt, witness, layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use modring::{find_prime_5mod8, Modulus, Ring};

    fn ring() -> Ring {
        // A LaBRADOR modulus of realistic size (~2^44, matching N≈100 from Phase 0).
        Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
    }

    #[test]
    #[should_panic(expected = "Falcon verification equation must hold over the integers")]
    fn compute_v_signed_panics_on_non_divisible_input() {
        // Fabricate (s1, s2, h, c) where c - s1 - h·s2 is not divisible by q.
        // Easiest: take s1=s2=h=zero and c = polynomial whose constant coeff is 1.
        // Then diff[0] = 1 - 0 - 0 = 1, not divisible by 12289.
        let s1 = FPoly::zero();
        let s2 = FPoly::zero();
        let h = FPoly::zero();
        let mut c = FPoly::zero();
        c.c[0] = 1;
        let _ = compute_v_signed(&s1, &s2, &h, &c);
    }

    #[test]
    fn four_square_round_trips() {
        for n in [0i64, 1, 2, 7, 23, 100, 5_000, 34_035_556] {
            let (a, b, c, d) = four_square(n);
            assert_eq!(a * a + b * b + c * c + d * d, n, "n = {n}");
        }
    }

    #[test]
    fn slot_helpers_are_self_consistent() {
        let rg = ring();
        let mut rng = modring::rng::SplitMix64::new(7);
        let make = |seed: u64| {
            let mut r = modring::rng::SplitMix64::new(seed);
            let mut s = [
                RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
                RingElem::zero(), RingElem::zero(), RingElem::zero(), RingElem::zero(),
            ];
            for k in 0..C {
                for j in 0..modring::D {
                    s[k].c[j] = r.below(rg.m.q);
                }
            }
            s
        };
        let a = make(rng.next());
        let b = make(rng.next());
        let sum = add_slots(&a, &b, &rg.m);
        assert!(all_slots_zero(&sub_slots(&sub_slots(&sum, &a, &rg.m), &b, &rg.m)));
        let scaled = scale_slots(&a, 2, &rg.m);
        let twice = add_slots(&a, &a, &rg.m);
        for k in 0..C {
            assert_eq!(scaled[k], twice[k]);
        }
    }
}
