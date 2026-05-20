//! LaBRADOR's principal relation: a statement is a set of dot-product
//! constraints over the LaBRADOR ring `S_{q'} = Z_{q'}[X] / (X^64 + 1)`,
//! plus a global ℓ₂-norm bound on the witness.
//!
//! From "Aggregating Falcon Signatures with LaBRADOR" §2.5 / Protocol 2:
//!
//! Witness: vectors `w_1, …, w_r ∈ S_{q'}^n` (the multiplicity-`r`,
//! rank-`n` witness). A *full* constraint `f^(k)` is
//!
//! ```text
//! f^(k)(w) = Σ_{i,j} a^(k)_{i,j} ⟨w_i, w_j⟩
//!          + Σ_i ⟨φ^(k)_i, w_i⟩ - b^(k)   ∈ S_{q'}
//! ```
//!
//! with `a^(k)_{i,j} = a^(k)_{j,i}`, and the constraint demands `f^(k) = 0`
//! in `S_{q'}`. A *constant-term* constraint `f'^(l)` has the same shape but
//! the verifier only checks `ct(f'^(l)) = 0 (mod q')` — the constant term.
//!
//! The witness must additionally satisfy `Σ ‖w_i‖₂² ≤ β²`.
//!
//! This module models the relation abstractly and provides a *direct*
//! evaluator. The proof system (commitments, JL, recursion) is built on top
//! in later modules.

use modring::{Ring, RingElem};

/// Sparse symmetric quadratic form: only `(i, j)` pairs with non-zero `a_{i,j}`
/// are stored. By convention `i ≤ j`; the pair `(i, j)` represents both
/// `a_{i,j}` and `a_{j,i}` (which are equal), counted once in `i = j` and
/// twice in `i ≠ j` when evaluating `Σ a_{i,j} ⟨w_i, w_j⟩`.
///
/// `phi` is doubly sparse: outer entries `(i, sparse_φ)` skip witness vectors
/// with `φ_i = 0`, and `sparse_φ` itself lists only `(position, S_element)`
/// pairs with the rest of `φ_i` implicit zero. The §F.2 form constraints
/// produce many phis with one or two non-zero positions in a length-`8N`
/// vector; a dense representation would balloon both compute and memory.
#[derive(Clone, Debug)]
pub struct DotConstraint {
    /// `(i, j, a_{i,j})` with `i ≤ j`.
    pub a: Vec<(usize, usize, RingElem)>,
    /// `(i, [(pos, p)])` — `φ_i[pos] = p`, all other positions zero.
    pub phi: Vec<(usize, Vec<(usize, RingElem)>)>,
    /// Right-hand side `b ∈ S_{q'}`.
    pub b: RingElem,
}

/// A constant-term constraint: same shape as [`DotConstraint`] but only the
/// constant term of `f` must be zero (modulo `q'`).
#[derive(Clone, Debug)]
pub struct ConstTermConstraint {
    pub a: Vec<(usize, usize, RingElem)>,
    pub phi: Vec<(usize, Vec<(usize, RingElem)>)>,
    /// Right-hand side as a single coefficient of `Z_{q'}` — only the
    /// constant term is checked.
    pub b0: u64,
}

/// A LaBRADOR principal-relation statement.
#[derive(Clone, Debug)]
pub struct Statement {
    /// The LaBRADOR ring context.
    pub ring: Ring,
    /// Witness rank (length of each `w_i`).
    pub n: usize,
    /// Witness multiplicity (number of vectors).
    pub r: usize,
    /// Full dot-product constraints `f^(k)`.
    pub full: Vec<DotConstraint>,
    /// Constant-term constraints `f'^(l)`.
    pub const_term: Vec<ConstTermConstraint>,
    /// Norm bound `β²` (as `i128` to support large bounds and signed math).
    pub beta_sq: i128,
}

/// A LaBRADOR witness: `r` vectors of length `n` each.
#[derive(Clone, Debug)]
pub struct Witness {
    pub w: Vec<Vec<RingElem>>,
}

impl Witness {
    /// Number of vectors.
    pub fn r(&self) -> usize {
        self.w.len()
    }

    /// Length of each vector (assumes all equal).
    pub fn n(&self) -> usize {
        self.w.first().map_or(0, |v| v.len())
    }

    /// Total ℓ₂-norm squared of the witness, over the integers (centered coeffs).
    pub fn norm_sq(&self, ring: &Ring) -> i128 {
        let mut s: i128 = 0;
        for vi in &self.w {
            for p in vi {
                s += p.norm_sq(&ring.m);
            }
        }
        s
    }
}

/// `⟨a, b⟩ = Σ a_i · b_i` in `S_{q'}`.
pub fn ring_inner_product(ring: &Ring, a: &[RingElem], b: &[RingElem]) -> RingElem {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");
    let mut acc = RingElem::zero();
    for i in 0..a.len() {
        let p = ring.mul(&a[i], &b[i]);
        acc = acc.add(&ring.m, &p);
    }
    acc
}

fn sparse_phi_contribution(
    phi: &[(usize, Vec<(usize, RingElem)>)],
    ring: &Ring,
    w: &Witness,
) -> RingElem {
    let m = &ring.m;
    let mut acc = RingElem::zero();
    for (i, sparse) in phi {
        for (pos, p) in sparse {
            let term = ring.mul(p, &w.w[*i][*pos]);
            acc = acc.add(m, &term);
        }
    }
    acc
}

/// Evaluate a `DotConstraint` directly on a witness, returning
/// `Σ a_{i,j} ⟨w_i, w_j⟩ + Σ ⟨φ_i, w_i⟩ − b`.
pub fn eval_full(c: &DotConstraint, ring: &Ring, w: &Witness) -> RingElem {
    let m = &ring.m;
    let mut acc = RingElem::zero();

    for (i, j, aij) in &c.a {
        let ip = ring_inner_product(ring, &w.w[*i], &w.w[*j]);
        let mut term = ring.mul(aij, &ip);
        if i != j {
            // Symmetric pair contributes twice (a_{i,j} = a_{j,i}, both stored once).
            term = term.add(m, &term.clone());
        }
        acc = acc.add(m, &term);
    }
    let lin = sparse_phi_contribution(&c.phi, ring, w);
    acc = acc.add(m, &lin);
    acc.sub(m, &c.b)
}

/// Evaluate a `ConstTermConstraint`, returning its constant term in `[0, q')`.
pub fn eval_const_term(c: &ConstTermConstraint, ring: &Ring, w: &Witness) -> u64 {
    let m = &ring.m;
    let mut acc = RingElem::zero();
    for (i, j, aij) in &c.a {
        let ip = ring_inner_product(ring, &w.w[*i], &w.w[*j]);
        let mut term = ring.mul(aij, &ip);
        if i != j {
            term = term.add(m, &term.clone());
        }
        acc = acc.add(m, &term);
    }
    let lin = sparse_phi_contribution(&c.phi, ring, w);
    acc = acc.add(m, &lin);
    m.sub(acc.c[0], c.b0)
}

/// Check that `w` satisfies every constraint of `s` and the norm bound.
pub fn satisfies(s: &Statement, w: &Witness) -> bool {
    if w.r() != s.r || w.n() != s.n {
        return false;
    }
    for c in &s.full {
        if !eval_full(c, &s.ring, w).is_zero() {
            return false;
        }
    }
    for c in &s.const_term {
        if eval_const_term(c, &s.ring, w) != 0 {
            return false;
        }
    }
    w.norm_sq(&s.ring) <= s.beta_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use modring::{find_prime_5mod8, Modulus};

    fn ring() -> Ring {
        Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
    }

    fn random_vec(ring: &Ring, n: usize, seed: u64) -> Vec<RingElem> {
        let mut rng = modring::rng::SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                let mut p = RingElem::zero();
                for k in 0..modring::D {
                    p.c[k] = rng.below(ring.m.q);
                }
                p
            })
            .collect()
    }

    #[test]
    fn satisfies_trivially_zero_statement() {
        let ring = ring();
        let w = Witness {
            w: vec![random_vec(&ring, 3, 1), random_vec(&ring, 3, 2)],
        };
        let s = Statement {
            ring,
            n: 3,
            r: 2,
            full: vec![],
            const_term: vec![],
            beta_sq: i128::MAX,
        };
        assert!(satisfies(&s, &w));
    }

    fn into_sparse(v: Vec<RingElem>) -> Vec<(usize, RingElem)> {
        v.into_iter().enumerate().collect()
    }

    #[test]
    fn linear_constraint_built_from_witness_is_satisfied() {
        // Construct: ⟨φ_0, w_0⟩ - b = 0, where b is the actual inner product.
        let ring = ring();
        let w0 = random_vec(&ring, 5, 11);
        let phi0 = random_vec(&ring, 5, 12);
        let b = ring_inner_product(&ring, &phi0, &w0);
        let s = Statement {
            ring: ring.clone(),
            n: 5,
            r: 1,
            full: vec![DotConstraint {
                a: vec![],
                phi: vec![(0, into_sparse(phi0))],
                b,
            }],
            const_term: vec![],
            beta_sq: i128::MAX,
        };
        let w = Witness { w: vec![w0] };
        assert!(satisfies(&s, &w));
    }

    #[test]
    fn quadratic_constraint_built_from_witness_is_satisfied() {
        // Construct: 1·⟨w_0, w_0⟩ - ⟨w_0,w_0⟩ = 0.
        let ring = ring();
        let w0 = random_vec(&ring, 4, 21);
        let g = ring_inner_product(&ring, &w0, &w0);
        let one = RingElem::constant(&ring.m, 1);
        let s = Statement {
            ring: ring.clone(),
            n: 4,
            r: 1,
            full: vec![DotConstraint {
                a: vec![(0, 0, one)],
                phi: vec![],
                b: g,
            }],
            const_term: vec![],
            beta_sq: i128::MAX,
        };
        let w = Witness { w: vec![w0] };
        assert!(satisfies(&s, &w));
    }

    #[test]
    fn tampering_witness_breaks_satisfaction() {
        let ring = ring();
        let w0 = random_vec(&ring, 3, 31);
        let phi0 = random_vec(&ring, 3, 32);
        let b = ring_inner_product(&ring, &phi0, &w0);
        let mut tampered = w0.clone();
        tampered[0].c[0] = ring.m.add(tampered[0].c[0], 1);
        let s = Statement {
            ring: ring.clone(),
            n: 3,
            r: 1,
            full: vec![DotConstraint {
                a: vec![],
                phi: vec![(0, into_sparse(phi0))],
                b,
            }],
            const_term: vec![],
            beta_sq: i128::MAX,
        };
        assert!(!satisfies(&s, &Witness { w: vec![tampered] }));
    }

    #[test]
    fn norm_bound_is_enforced() {
        let ring = ring();
        let mut w0 = vec![RingElem::zero(); 2];
        w0[0].c[0] = 100;
        // norm² of single coeff 100 = 10000.
        let s = Statement {
            ring,
            n: 2,
            r: 1,
            full: vec![],
            const_term: vec![],
            beta_sq: 9999,
        };
        assert!(!satisfies(&s, &Witness { w: vec![w0.clone()] }));
        let s2 = Statement { beta_sq: 10000, ..s };
        assert!(satisfies(&s2, &Witness { w: vec![w0] }));
    }
}
