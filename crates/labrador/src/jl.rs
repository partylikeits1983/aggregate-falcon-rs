//! Johnson-Lindenstrauss projection (Lemma 2.2 / §2.5).
//!
//! `Π ∈ {−1, 0, +1}^{256 × (n · d)}` with `Pr[0] = 1/2`, `Pr[±1] = 1/4` each.
//! For a witness vector `w ∈ S^n` with centered ℤ-coefficient vector
//! `w̃ ∈ ℤ^{n·d}` (length-d integer slice per ring element), the projection
//! `p_j = Σ_i Π_{j, ·} · w̃_i` is an `i128` to avoid overflow.

use crate::transcript::Transcript;
use modring::{Modulus, RingElem, D};

pub const LAMBDA: usize = 128;
pub const PROJECTION_ROWS: usize = 2 * LAMBDA; // 256

/// Sample one row of Π (length `cols`) into a sparse `(position, ±1)` list.
/// The 50/25/25 distribution is implemented by drawing two bits and using
/// `(bit0, bit1)` as `(zero?, sign)`.
fn sample_pi_row(t: &mut Transcript, label: &[u8], cols: usize) -> Vec<(usize, i8)> {
    // Pack two bits per coordinate: zero/non-zero (bit0), sign if non-zero (bit1).
    let nbytes = (cols + 3) / 4; // 4 entries per byte
    let bytes = t.challenge_bytes(label, nbytes);
    let mut out = Vec::new();
    for k in 0..cols {
        let byte = bytes[k / 4];
        let pair = (byte >> ((k % 4) * 2)) & 0b11;
        match pair {
            0b00 | 0b01 => {} // zero with probability 2/4
            0b10 => out.push((k, 1)),
            0b11 => out.push((k, -1)),
            _ => unreachable!(),
        }
    }
    out
}

/// Sample the full projection matrix `Π` as `PROJECTION_ROWS` sparse rows.
pub fn sample_projection(
    t: &mut Transcript,
    label: &[u8],
    cols: usize,
) -> Vec<Vec<(usize, i8)>> {
    (0..PROJECTION_ROWS)
        .map(|j| {
            let row_label = [label, b"|row|", &(j as u64).to_le_bytes()].concat();
            sample_pi_row(t, &row_label, cols)
        })
        .collect()
}

/// Project a single witness vector `w ∈ S^n` using `pi` (sparse rows).
/// Returns `[i128; 256]` with each entry being a (possibly large) signed sum
/// over centered coefficients.
pub fn project_vector(
    pi: &[Vec<(usize, i8)>],
    w: &[RingElem],
    m: &Modulus,
) -> [i128; PROJECTION_ROWS] {
    let mut out = [0i128; PROJECTION_ROWS];
    // Flatten centered coefficients of w.
    let mut wflat: Vec<i64> = Vec::with_capacity(w.len() * D);
    for e in w {
        for k in 0..D {
            wflat.push(m.centered(e.c[k]));
        }
    }
    for (j, row) in pi.iter().enumerate() {
        let mut acc: i128 = 0;
        for &(pos, sign) in row {
            acc += (sign as i128) * (wflat[pos] as i128);
        }
        out[j] = acc;
    }
    out
}

/// Sum projections over many witness vectors: `p_j = Σ_i ⟨π_j^{(i)}, w_i⟩`.
/// `pis[i]` is the projection matrix for witness vector `i` (each over its
/// own `n_i · d` columns).
pub fn project_combined(
    pis: &[Vec<Vec<(usize, i8)>>],
    ws: &[Vec<RingElem>],
    m: &Modulus,
) -> [i128; PROJECTION_ROWS] {
    let mut out = [0i128; PROJECTION_ROWS];
    for (pi, w) in pis.iter().zip(ws.iter()) {
        let p = project_vector(pi, w, m);
        for j in 0..PROJECTION_ROWS {
            out[j] += p[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use modring::{find_prime_5mod8, Modulus, Ring};

    fn ring() -> Ring {
        Ring::new(Modulus::new(find_prime_5mod8(1 << 44)))
    }

    #[test]
    fn hand_computed_projection_matches() {
        // Use a tiny witness and a hand-built Π.
        let r = ring();
        let m = &r.m;
        let mut w = vec![RingElem::zero(); 2];
        w[0].c[0] = 5;
        w[0].c[1] = m.from_i64(-3);
        w[1].c[0] = 2;
        // Flatten: [5, -3, 0, 0, ..., 0, 2, 0, ...] of length 2*D.
        // Hand row: Π_0 = [+1 at pos 0, -1 at pos 1, +1 at pos D]
        let pi = vec![vec![(0usize, 1i8), (1, -1), (D, 1)]];
        let mut pi_padded: Vec<Vec<(usize, i8)>> = pi.clone();
        // pad up to PROJECTION_ROWS rows of zeros so the type matches:
        while pi_padded.len() < PROJECTION_ROWS {
            pi_padded.push(vec![]);
        }
        let p = project_vector(&pi_padded, &w, m);
        // 5 - (-3) + 2 = 10
        assert_eq!(p[0], 10);
        for j in 1..PROJECTION_ROWS {
            assert_eq!(p[j], 0);
        }
    }

    #[test]
    fn distribution_is_approximately_50_25_25() {
        let mut t = Transcript::new(b"jl-dist");
        let cols = 10_000;
        let row = sample_pi_row(&mut t, b"r0", cols);
        let total = row.len();
        let pos = row.iter().filter(|(_, s)| *s == 1).count();
        let neg = row.iter().filter(|(_, s)| *s == -1).count();
        // Expected: zeros ≈ cols/2, pos ≈ cols/4, neg ≈ cols/4.
        assert!((total as i64 - cols as i64 / 2).abs() < cols as i64 / 10);
        assert!((pos as i64 - cols as i64 / 4).abs() < cols as i64 / 10);
        assert!((neg as i64 - cols as i64 / 4).abs() < cols as i64 / 10);
    }
}
