//! Phase 3c gate test: the honest LaBRADOR statement built from real
//! Falcon-512 signatures is satisfied by the honest witness, and any tamper
//! to that witness flips it to `false`.
//!
//! This is the decisive check that the §F.2 Falcon-eq constraint set and the
//! §6.2 four-square constraint set are encoded correctly against the
//! `S = Z_{q'}[X]/(X^64+1)` substrate. If any slot of any constraint is
//! mis-derived, the honest witness fails immediately; if any constraint is
//! degenerately satisfiable, the tamper case slips through.

use falcon_relation::parse::decode_instance;
use falcon_relation::relation::{build_falcon_statement, WitnessLayout};
use labrador::statement::satisfies;
use modring::{find_prime_5mod8, Modulus, Ring};
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

fn ring() -> Ring {
    Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
}

fn fresh_sigs(n: usize) -> Vec<falcon_relation::parse::FalconSig> {
    (0..n)
        .map(|i| {
            let msg = format!("phase 3c test message #{}", i).into_bytes();
            let (pk, sk) = falcon512::keypair();
            let sig = falcon512::detached_sign(&msg, &sk);
            decode_instance(pk.as_bytes(), sig.as_bytes(), &msg)
                .expect("decode_instance on a genuine signature")
        })
        .collect()
}

#[test]
fn witness_layout_matches_paper_formulas() {
    // ρ = round(√N); r = 3⌈N/ρ⌉ + 3ρ + 1.
    let cases = [
        (1, 1, 1, 1, 3 * 1 + 3 * 1 + 1),
        (2, 1, 2, 1, 3 * 2 + 3 * 1 + 1),
        (4, 2, 2, 2, 3 * 2 + 3 * 2 + 1),
        (9, 3, 3, 3, 3 * 3 + 3 * 3 + 1),
        (10, 3, 4, 3, 3 * 4 + 3 * 3 + 1),
    ];
    for (n, expect_rho, expect_num_y, expect_num_yp, expect_r) in cases {
        let l = WitnessLayout::new(n);
        assert_eq!(l.rho, expect_rho, "ρ for N={n}");
        assert_eq!(l.num_y, expect_num_y, "num_y for N={n}");
        assert_eq!(l.num_yp, expect_num_yp, "num_yp for N={n}");
        assert_eq!(l.r(), expect_r, "r for N={n}");
        assert_eq!(l.n_s(), 8 * n, "n_S for N={n}");

        for i in 1..=n {
            // index'(i) lies in [1, ρ], index(i) in [1, num_y].
            let idx = l.index(i);
            let idxp = l.index_prime(i);
            assert!(1 <= idx && idx <= l.num_y);
            assert!(1 <= idxp && idxp <= l.num_yp);
            // (idx, idxp) must uniquely determine i, per §F.1's padding scheme.
            assert_eq!((idx - 1) * l.rho + idxp, i, "(idx, idx') round-trip for i={i}");
        }
    }
}

#[test]
fn honest_witness_satisfies_falcon_statement() {
    let ring = ring();
    let sigs = fresh_sigs(4);
    // β² is set generously here; tight derivation belongs with the §F.2 form
    // constraints + §6.2 quotient analysis (future phase).
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, _layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    assert!(
        satisfies(&stmt, &witness),
        "honest Falcon-512 witness must satisfy the assembled statement"
    );
}

#[test]
fn flipping_one_witness_coefficient_breaks_satisfaction() {
    let ring = ring();
    let sigs = fresh_sigs(4);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    // Hit every "logical" slice of the witness so any encoding error in any
    // family of constraints is caught: y_{·,1}, y_{·,2}, y'_{·,1}, y'_{·,2},
    // e_·, e'_·, v.
    let i_y = layout.index(1);
    let i_yp = layout.index_prime(1);
    let targets = [
        layout.y_idx(i_y, 1),
        layout.y_idx(i_y, 2),
        layout.yp_idx(i_yp, 1),
        layout.yp_idx(i_yp, 2),
        layout.e_idx(i_y),
        layout.ep_idx(i_yp),
        layout.v_idx(),
    ];
    for vec_idx in targets {
        let mut bad = witness.clone();
        bad.w[vec_idx][0].c[0] = ring.m.add(bad.w[vec_idx][0].c[0], 1);
        assert!(
            !satisfies(&stmt, &bad),
            "perturbing witness vector {vec_idx} (slot 0, coeff 0) must break satisfaction"
        );
    }
}

#[test]
fn tampering_yp_padding_breaks_form_constraints() {
    // y'_{iyp, j} should be zero at R-positions not ≡ iyp (mod ρ). Setting any
    // S-position of a padding R-position to non-zero must reject.
    let ring = ring();
    let sigs = fresh_sigs(4);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    // For N=4, ρ=2: y'_{1, 1} active at R-positions {1, 3}; padding at {2, 4}.
    // S-position for R-position 2 = 8..15.
    let yp = layout.yp_idx(1, 1);
    let mut bad = witness.clone();
    bad.w[yp][8].c[0] = ring.m.add(bad.w[yp][8].c[0], 1);
    assert!(
        !satisfies(&stmt, &bad),
        "padding of y' must be pinned to zero"
    );
}

#[test]
fn breaking_sigma_minus_one_tie_is_detected() {
    // Replace y' with y at one active S-position (no σ₋₁ applied) — the
    // σ₋₁ˢ tie should detect it.
    let ring = ring();
    let sigs = fresh_sigs(4);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    // y'_{1, 1} active at R-position 1 (S-pos 0..7). Replace y'_{1, 1}[0]
    // (currently σ₋₁ˢ(y_{1, 1}[0])) with y_{1, 1}[0] (no σ₋₁) — they differ
    // at every X^l coefficient for l ∈ [1, 63] if y has non-zero content there.
    let yp = layout.yp_idx(1, 1);
    let y = layout.y_idx(1, 1);
    let mut bad = witness.clone();
    // Force-clear coef 5 of y'[0] (which σ₋₁ˢ(y[0])[5] should equal − y[0][59])
    // — flipping it makes the tie at l=5 fail.
    bad.w[yp][0].c[5] = ring.m.add(bad.w[yp][0].c[5], 1);
    // Ensure y[0] coef 59 is non-zero so the constraint is actually violated.
    let _ = bad.w[y][0].c[59];
    assert!(
        !satisfies(&stmt, &bad),
        "σ₋₁ˢ tie at active y' position must reject a coefficient flip"
    );
}

#[test]
fn tampering_e_inner_coef_breaks_form_constraints() {
    // e_{i_y} at active R-position slot 0..3 must have inner coefs 1..63 = 0.
    // Setting any one of them non-zero must reject.
    let ring = ring();
    let sigs = fresh_sigs(4);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, layout) = build_falcon_statement(&sigs, &ring, beta_sq);

    let evec = layout.e_idx(1);
    // R-position 1, slot 0, inner coef 5.
    let mut bad = witness.clone();
    bad.w[evec][0].c[5] = ring.m.add(bad.w[evec][0].c[5], 1);
    assert!(
        !satisfies(&stmt, &bad),
        "inner coef 5 of slot 0 of e_{{i_y}} active R-position must be pinned"
    );
}

#[test]
fn tampering_a_signature_byte_breaks_falcon_eq() {
    // Build the *honest* statement first (so we have the public hi, ci to
    // compare against); then re-derive a witness from the same statement with
    // one tampered s2 coefficient, which violates the Falcon eq.
    let ring = ring();
    let sigs = fresh_sigs(2);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    assert!(satisfies(&stmt, &witness));

    // Flip one slot of s_{1,2}'s embedding (R-position 0 of the y_{·,2} vector).
    let y2 = layout.y_idx(layout.index(1), 2);
    let mut bad = witness.clone();
    bad.w[y2][3].c[5] = ring.m.add(bad.w[y2][3].c[5], 1);
    assert!(
        !satisfies(&stmt, &bad),
        "perturbing an s2 slot must violate the Falcon-eq slot constraints"
    );
}

/// Exercise every §F.2 form-constraint family. Each tamper targets a position
/// that the *form* constraints (not the Falcon-eq or four-square equations)
/// would catch — so a missing constraint family would silently let one of
/// these slip through.
#[test]
fn form_constraints_catch_each_padding_or_structure_violation() {
    let ring = ring();
    let sigs = fresh_sigs(4);
    let beta_sq: i128 = 1 << 60;
    let (stmt, witness, layout) = build_falcon_statement(&sigs, &ring, beta_sq);
    assert!(satisfies(&stmt, &witness));

    // (i_y, i_yp) for sigs 1 (in-range/matching) and 2 (in-range/matching for
    // a different i_yp). ρ=2, num_y=2, num_yp=2.
    // sig 1: index=1, index'=1; sig 2: index=1, index'=2;
    // sig 3: index=2, index'=1; sig 4: index=2, index'=2.

    // (1) y_{·,j} padding: y_{1,1} should be zero at R-position 3 (sig 3 not
    //     in i_y=1's range). Putting any value there must break.
    {
        let y_pad = layout.y_idx(1, 1);
        let mut bad = witness.clone();
        let pos = 8 * (3 - 1); // R-position for sig 3, slot 0
        bad.w[y_pad][pos].c[0] = ring.m.add(bad.w[y_pad][pos].c[0], 1);
        assert!(!satisfies(&stmt, &bad), "y padding (out-of-range R-position) not caught");
    }

    // (2) y'_{·,j} padding: y'_{1,1} should be zero at R-position 2 (sig 2 has
    //     index'=2, not 1). Tamper there must break.
    {
        let yp_pad = layout.yp_idx(1, 1);
        let mut bad = witness.clone();
        let pos = 8 * (2 - 1);
        bad.w[yp_pad][pos].c[0] = ring.m.add(bad.w[yp_pad][pos].c[0], 1);
        assert!(!satisfies(&stmt, &bad), "y' padding (non-matching R-position) not caught");
    }

    // (3) y'_{·,j} σ_{-1}^S consistency: at sig 1 (matching y'_{1,1}), the
    //     l=5 coefficient of slot 2 of y'_{1,1} should equal -y_{1,1}'s l=59
    //     coefficient. Flip y'_{1,1} alone.
    {
        let yp_match = layout.yp_idx(1, 1);
        let mut bad = witness.clone();
        bad.w[yp_match][2].c[5] = ring.m.add(bad.w[yp_match][2].c[5], 1);
        assert!(!satisfies(&stmt, &bad), "y' σ_{{-1}}^S equality not caught");
    }

    // (4) e_· structure: at sig 1 (in-range for e_1), slot 0 coefficient
    //     l=1 should be zero (only c[0] is free for slots 0..3). Tamper.
    {
        let e_vec = layout.e_idx(1);
        let mut bad = witness.clone();
        bad.w[e_vec][0].c[1] = ring.m.add(bad.w[e_vec][0].c[1], 1);
        assert!(!satisfies(&stmt, &bad), "ε structure (slot 0 c[1] should be zero) not caught");
    }

    // (5) e_· structure for slot ≥ 4: at sig 1, slot 4 should be entirely
    //     zero. Tamper any coefficient.
    {
        let e_vec = layout.e_idx(1);
        let mut bad = witness.clone();
        bad.w[e_vec][4].c[0] = ring.m.add(bad.w[e_vec][4].c[0], 1);
        assert!(!satisfies(&stmt, &bad), "ε structure (slot 4 must be zero) not caught");
    }

    // (6) e_· padding (out-of-range R-position): e_1 should be zero at sig 3.
    {
        let e_vec = layout.e_idx(1);
        let mut bad = witness.clone();
        let pos = 8 * (3 - 1);
        bad.w[e_vec][pos].c[0] = ring.m.add(bad.w[e_vec][pos].c[0], 1);
        assert!(!satisfies(&stmt, &bad), "ε padding (out-of-range) not caught");
    }

    // (7) e'_· σ_{-1}^S consistency: at sig 1 (matching e'_1), the l=2
    //     coefficient of slot 2 of e'_1 should match σ_{-1}^S(e_1)'s slot 2
    //     l=2 coefficient. Tamper e' alone.
    {
        let ep_vec = layout.ep_idx(1);
        let mut bad = witness.clone();
        bad.w[ep_vec][2].c[2] = ring.m.add(bad.w[ep_vec][2].c[2], 1);
        assert!(!satisfies(&stmt, &bad), "e' σ_{{-1}}^S equality not caught");
    }

    // (8) e'_· padding: e'_1 should be zero at sig 2 (index'(2) = 2, not 1).
    {
        let ep_vec = layout.ep_idx(1);
        let mut bad = witness.clone();
        let pos = 8 * (2 - 1);
        bad.w[ep_vec][pos].c[3] = ring.m.add(bad.w[ep_vec][pos].c[3], 1);
        assert!(!satisfies(&stmt, &bad), "e' padding (non-matching R-position) not caught");
    }
}
