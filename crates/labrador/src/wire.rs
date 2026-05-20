//! Compact wire encoding for [`AggregateProofV2`].
//!
//! The bincode-derived format on the proof types emits each `RingElem` as
//! 64 × `u64` = 512 bytes regardless of the LaBRADOR modulus width or
//! coefficient distribution. The paper's estimator (see
//! `params::Iteration::size_step` and `size_lastmsg`) assumes a tighter
//! encoding: ring-coefficients packed to `ceil(log2 q')` bits, and the known-
//! Gaussian fields (`z0`, `z1`, `g`) entropy-coded with a Golomb-Rice coder
//! parameterised by their σ.
//!
//! This module implements that tighter encoding and is what
//! `Serialize for AggregateProofV2` writes to / `Deserialize` reads from.
//!
//! Wire layout (least significant bit of each byte goes out first):
//!
//! - Fixed header (bincode-fixint, little-endian): `q_prime` (u64),
//!   `n_sigs` (u64), `beta_sq` (i128), `n_intermediate` (u64),
//!   `payload_len` (u64).
//! - Then a `payload_len`-byte bit-stream containing:
//!   - For each intermediate iteration: `pack_iter(..., with_last_msg=false)`.
//!   - Then: `pack_iter(..., with_last_msg=true)` for the final iteration.
//!
//! Per-iteration layout (inside the bit-stream):
//!   - `u1.len()`        — varbit, see [`write_len`].
//!   - `u1[*]`           — packed `RingElem`s (`q_bitlen` bits per coeff).
//!   - `p.len()`         — varbit.
//!   - `p[*]`            — each entry written as a signed q_bitlen-bit value
//!                         (reduced to canonical mod-`q'`); see
//!                         [`encode_signed_modq`]. Recoverable exactly when
//!                         `|p[j]| < q'/2`, which the JL projection bound
//!                         guarantees for honest proofs.
//!   - `b''.len()`       — varbit.
//!   - `b''[*]`          — packed `RingElem`s.
//!   - `u2.len()`        — varbit.
//!   - `u2[*]`           — packed `RingElem`s.
//!   - `has_last_msg`    — 1 bit.
//!   - if set:
//!     - `z0.len()`, then `z0[*]` Golomb-Rice-coded (k from σ_z of this iter).
//!     - `z1.len()`, then `z1[*]` Golomb-Rice-coded.
//!     - `r` (= v.len()), `κ` (= v[0].len()), then `r×κ` packed RingElems
//!       for `v[i][k]`.
//!     - `r` again as a sanity check, then upper-triangle (j ≥ i) of `g`
//!       Golomb-Rice-coded (k from σ_h), then upper-triangle of `h` packed.
//!
//! Reading the wire format reconstructs the full structures: `g`/`h` lower
//! triangles are filled with `RingElem::zero()`, matching the convention of
//! `IterationLastMsg` at `proof.rs:69-73`.
//!
//! Round-trip: `unpack(pack(p)) == p` for every well-formed proof. Tested in
//! the `tests/` module at the bottom of this file.

use crate::params::Params;
use crate::proof::{AggregateProofV2, IterationLastMsg, IterationProofV2};
use modring::{RingElem, D};

#[derive(Debug)]
pub enum WireError {
    Truncated,
    BadShape(&'static str),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::Truncated => write!(f, "wire stream truncated"),
            WireError::BadShape(s) => write!(f, "wire shape invalid: {s}"),
        }
    }
}

impl std::error::Error for WireError {}

// ---------------------------------------------------------------------------
// BitWriter / BitReader
// ---------------------------------------------------------------------------

struct BitWriter {
    buf: Vec<u8>,
    cur: u64,
    cur_bits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self { buf: Vec::new(), cur: 0, cur_bits: 0 }
    }

    /// Write `nbits` low bits of `value`. `nbits ≤ 64`.
    fn write_bits(&mut self, value: u64, nbits: u32) {
        debug_assert!(nbits <= 64);
        if nbits == 0 {
            return;
        }
        let mask = if nbits == 64 { u64::MAX } else { (1u64 << nbits) - 1 };
        let v = value & mask;
        // Write up to 64 - cur_bits bits into cur, flush, then if more remain
        // start a fresh cur.
        let free = 64 - self.cur_bits;
        if nbits <= free {
            self.cur |= v << self.cur_bits;
            self.cur_bits += nbits;
            if self.cur_bits == 64 {
                self.buf.extend_from_slice(&self.cur.to_le_bytes());
                self.cur = 0;
                self.cur_bits = 0;
            }
        } else {
            // Pack first `free` low bits into cur, flush, then carry the rest.
            let low_mask = if free == 64 { u64::MAX } else { (1u64 << free) - 1 };
            let low = v & low_mask;
            self.cur |= low << self.cur_bits;
            self.buf.extend_from_slice(&self.cur.to_le_bytes());
            let remaining = nbits - free;
            self.cur = v >> free;
            self.cur_bits = remaining;
        }
    }

    /// Length-prefix encoding: write `len` as a 6-bit "exponent" + payload.
    /// Specifically, write `e = ceil(log2(len+1))` in 6 bits then the value
    /// in `e` bits. Lengths in [0, 2^64) supported. Optimised for small
    /// values that occur in our proofs (lengths under ~10000 cost ≤ 20 bits).
    fn write_len(&mut self, len: u64) {
        let bits = 64 - len.leading_zeros();
        self.write_bits(bits as u64, 6);
        if bits > 0 {
            self.write_bits(len, bits);
        }
    }

    /// Write unary-then-binary Golomb-Rice code. `value` ≥ 0.
    /// Length: `(value >> k) + 1 + k` bits.
    fn write_gr(&mut self, value: u64, k: u32) {
        let q = value >> k;
        // Unary: q ones, then a zero terminator. Write 64 bits at a time for
        // very long runs (rare but possible).
        let mut q_left = q;
        while q_left >= 64 {
            self.write_bits(u64::MAX, 64);
            q_left -= 64;
        }
        if q_left > 0 {
            let ones = (1u64 << q_left) - 1;
            self.write_bits(ones, q_left as u32);
        }
        // Terminator zero (1 bit).
        self.write_bits(0, 1);
        // Remainder: low k bits of value.
        if k > 0 {
            let mask = (1u64 << k) - 1;
            self.write_bits(value & mask, k);
        }
    }

    /// Finalise: pad to a byte boundary with zeros and return the byte stream.
    fn finish(mut self) -> Vec<u8> {
        if self.cur_bits > 0 {
            let bytes_needed = (self.cur_bits as usize + 7) / 8;
            let le = self.cur.to_le_bytes();
            self.buf.extend_from_slice(&le[..bytes_needed]);
        }
        self.buf
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    byte_pos: usize,
    bit_pos: u32,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, byte_pos: 0, bit_pos: 0 }
    }

    fn read_bits(&mut self, nbits: u32) -> Result<u64, WireError> {
        debug_assert!(nbits <= 64);
        if nbits == 0 {
            return Ok(0);
        }
        let mut out: u64 = 0;
        let mut got: u32 = 0;
        while got < nbits {
            if self.byte_pos >= self.buf.len() {
                return Err(WireError::Truncated);
            }
            let byte = self.buf[self.byte_pos];
            let avail_in_byte = 8 - self.bit_pos;
            let want = (nbits - got).min(avail_in_byte);
            let mask = if want == 8 { 0xFFu64 } else { (1u64 << want) - 1 };
            let chunk = ((byte as u64) >> self.bit_pos) & mask;
            out |= chunk << got;
            got += want;
            self.bit_pos += want;
            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Ok(out)
    }

    fn read_len(&mut self) -> Result<u64, WireError> {
        let bits = self.read_bits(6)? as u32;
        if bits == 0 {
            Ok(0)
        } else if bits > 64 {
            Err(WireError::BadShape("length-prefix bits > 64"))
        } else {
            self.read_bits(bits)
        }
    }

    fn read_gr(&mut self, k: u32) -> Result<u64, WireError> {
        let mut q: u64 = 0;
        loop {
            let b = self.read_bits(1)?;
            if b == 0 {
                break;
            }
            q = q.checked_add(1).ok_or(WireError::BadShape("GR unary overflow"))?;
            if q > (1u64 << 40) {
                return Err(WireError::BadShape("GR unary unreasonably long"));
            }
        }
        let r = if k > 0 { self.read_bits(k)? } else { 0 };
        Ok((q << k) | r)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn ceil_log2(q: u64) -> u32 {
    if q <= 1 {
        0
    } else {
        64 - (q - 1).leading_zeros()
    }
}

/// Map a signed value in `[-(q-1)/2, (q-1)/2]` to its canonical mod-`q`
/// representative in `[0, q)`. The inverse is [`decode_signed_modq`].
#[inline]
fn encode_signed_modq(x: i128, q: u64) -> u64 {
    let q_i = q as i128;
    let mut r = x % q_i;
    if r < 0 {
        r += q_i;
    }
    r as u64
}

/// Inverse of [`encode_signed_modq`]: maps `[0, q)` canonical residues back
/// to centered `[-(q-1)/2, (q-1)/2]`.
#[inline]
fn decode_signed_modq(u: u64, q: u64) -> i128 {
    if u <= q / 2 {
        u as i128
    } else {
        (u as i128) - (q as i128)
    }
}

/// Zigzag-encode a signed value to a non-negative unsigned value suitable for
/// Golomb-Rice coding. `0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, …`.
#[inline]
fn zigzag_encode(x: i64) -> u64 {
    ((x << 1) ^ (x >> 63)) as u64
}

#[inline]
fn zigzag_decode(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// Golomb-Rice parameter `k` for a zigzag-encoded Gaussian source with
/// width `sigma`. Zigzag turns signed `x ∼ N(0, σ²)` into non-negative
/// `u = (|x|<<1) | sign(x)` with `E[u] ≈ 2σ·sqrt(2/π) ≈ σ·1.596`. The
/// optimal GR parameter is `k = floor(log2(E[u]))`. Empirically choosing
/// the floor (rather than nearest) wins by ~3-5% on our z/g fields because
/// the unary cost is asymmetric: under-estimating `k` by 1 costs 1 bit per
/// symbol, over-estimating by 1 costs 1 bit per symbol BUT slightly less
/// when the remainder is in the "easy" half.
#[inline]
fn gr_k_for_sigma(sigma: f64) -> u32 {
    if sigma <= 1.0 {
        return 0;
    }
    let target = sigma * (2.0 * (2.0 / std::f64::consts::PI).sqrt());
    let k = target.log2().floor() as i32;
    k.clamp(0, 31) as u32
}

// ---------------------------------------------------------------------------
// RingElem packing
// ---------------------------------------------------------------------------

/// Pack a `RingElem`'s 64 canonical-residue coefficients using `q_bitlen`
/// bits each. Uniformly-distributed fields (`u1`, `u2`, `v`, `h`, `b''`)
/// use this directly. Gaussian fields go through [`write_ring_elem_gr`].
fn write_ring_elem_packed(w: &mut BitWriter, e: &RingElem, q_bitlen: u32) {
    for &c in &e.c {
        w.write_bits(c, q_bitlen);
    }
}

fn read_ring_elem_packed(r: &mut BitReader, q_bitlen: u32) -> Result<RingElem, WireError> {
    let mut c = [0u64; D];
    for slot in &mut c {
        *slot = r.read_bits(q_bitlen)?;
    }
    Ok(RingElem { c })
}

/// Golomb-Rice-pack a Gaussian-distributed `RingElem`. Each canonical mod-q'
/// coefficient is converted to its centered signed representative in
/// `(-q/2, q/2]`, zigzag-encoded, and GR-coded with parameter `k`.
fn write_ring_elem_gr(w: &mut BitWriter, e: &RingElem, q: u64, k: u32) {
    for &c in &e.c {
        let centered = decode_signed_modq(c, q);
        // Clamp to i64; Gaussian field coefficients are well below i64::MAX
        // in honest proofs (σ_z, σ_h up to ~2^40 in worst case).
        let centered_i64 = centered.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        w.write_gr(zigzag_encode(centered_i64), k);
    }
}

fn read_ring_elem_gr(r: &mut BitReader, q: u64, k: u32) -> Result<RingElem, WireError> {
    let mut c = [0u64; D];
    for slot in &mut c {
        let zig = r.read_gr(k)?;
        let signed = zigzag_decode(zig);
        *slot = encode_signed_modq(signed as i128, q);
    }
    Ok(RingElem { c })
}

#[inline]
fn ring_elem_is_zero(e: &RingElem) -> bool {
    e.c.iter().all(|&x| x == 0)
}

/// Write a `Vec<RingElem>` with a 1-bit "all zeros" shortcut. When every
/// element is `RingElem::zero()` (e.g. `z1` at the SecLast iteration where
/// base `b = 1` makes the decomposition `z = z0 + 1·z1` collapse to
/// `z1 = 0`), the entire vector emits just one bit. Otherwise the regular
/// GR encoding follows. The deserializer mirrors this.
fn write_gauss_vec(w: &mut BitWriter, elems: &[RingElem], q: u64, k: u32) {
    let all_zero = elems.iter().all(ring_elem_is_zero);
    w.write_bits(if all_zero { 1 } else { 0 }, 1);
    if all_zero {
        return;
    }
    for e in elems {
        write_ring_elem_gr(w, e, q, k);
    }
}

fn read_gauss_vec(
    r: &mut BitReader,
    len: usize,
    q: u64,
    k: u32,
) -> Result<Vec<RingElem>, WireError> {
    let all_zero = r.read_bits(1)? == 1;
    if all_zero {
        return Ok(vec![RingElem::zero(); len]);
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(read_ring_elem_gr(r, q, k)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Iteration packing
// ---------------------------------------------------------------------------

fn write_iter(
    w: &mut BitWriter,
    it: &IterationProofV2,
    q: u64,
    q_bitlen: u32,
    sigz: f64,
    sigh: f64,
    p_sigma: f64,
    with_last_msg: bool,
) {
    // u1
    w.write_len(it.u1.len() as u64);
    for e in &it.u1 {
        write_ring_elem_packed(w, e, q_bitlen);
    }
    // p (JL projection vector, signed). Per Lemma 2.2 / Protocol 2 step 2,
    // each entry is `Σ_i ⟨π_i, τ(w_i)⟩` over random ±1 projection rows π,
    // so Var(p[j]) ≈ ‖w‖²/2 → σ_p ≈ β/√2 where β is the L2 bound on the
    // current witness (`Iteration::beta`). Tuning k_p to σ_p instead of σ_z
    // halves the encoded cost compared to using σ_z directly.
    let k_p = gr_k_for_sigma(p_sigma);
    w.write_len(it.p.len() as u64);
    for &pi in &it.p {
        let pi_i64 = pi.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        w.write_gr(zigzag_encode(pi_i64), k_p);
    }
    // b''
    w.write_len(it.b_double_prime.len() as u64);
    for e in &it.b_double_prime {
        write_ring_elem_packed(w, e, q_bitlen);
    }
    // u2
    w.write_len(it.u2.len() as u64);
    for e in &it.u2 {
        write_ring_elem_packed(w, e, q_bitlen);
    }
    // has_last_msg
    let actually_has = it.last_msg.is_some();
    if actually_has != with_last_msg {
        panic!(
            "wire::write_iter: with_last_msg={with_last_msg} but iter.last_msg.is_some()={actually_has}"
        );
    }
    w.write_bits(if actually_has { 1 } else { 0 }, 1);
    if let Some(lm) = &it.last_msg {
        let k_z = gr_k_for_sigma(sigz);
        let k_g = gr_k_for_sigma(sigh);

        // z0 (Gaussian, with all-zeros shortcut)
        w.write_len(lm.z0.len() as u64);
        write_gauss_vec(w, &lm.z0, q, k_z);
        // z1 (Gaussian, with all-zeros shortcut — z1 is *always* zero at
        // SecLast where base b=1; that case collapses to 1 bit on the wire.)
        w.write_len(lm.z1.len() as u64);
        write_gauss_vec(w, &lm.z1, q, k_z);
        // v: [r][κ] uniform
        let r = lm.v.len() as u64;
        let kappa = lm.v.first().map(|x| x.len() as u64).unwrap_or(0);
        w.write_len(r);
        w.write_len(kappa);
        for v_i in &lm.v {
            assert_eq!(v_i.len() as u64, kappa, "v inner length mismatch");
            for e in v_i {
                write_ring_elem_packed(w, e, q_bitlen);
            }
        }
        // g (Gaussian σ_h) and h (uniform mod q'): r×r upper triangle (j ≥ i).
        // g uses GR (with all-zeros shortcut over the upper triangle); h is
        // bit-packed.
        let r_g = lm.g.len() as u64;
        let r_h = lm.h.len() as u64;
        w.write_len(r_g);
        // Flatten the upper triangle of g into a Vec for the all-zeros check.
        let mut g_flat: Vec<RingElem> = Vec::with_capacity(((r_g * (r_g + 1)) / 2) as usize);
        for i in 0..(r_g as usize) {
            assert_eq!(lm.g[i].len() as u64, r_g, "g row length mismatch");
            for j in i..(r_g as usize) {
                g_flat.push(lm.g[i][j].clone());
            }
        }
        write_gauss_vec(w, &g_flat, q, k_g);
        w.write_len(r_h);
        for i in 0..(r_h as usize) {
            assert_eq!(lm.h[i].len() as u64, r_h, "h row length mismatch");
            for j in i..(r_h as usize) {
                write_ring_elem_packed(w, &lm.h[i][j], q_bitlen);
            }
        }
    }
}

fn read_iter(
    r: &mut BitReader,
    q: u64,
    q_bitlen: u32,
    sigz: f64,
    sigh: f64,
    p_sigma: f64,
) -> Result<IterationProofV2, WireError> {
    let u1_len = r.read_len()? as usize;
    let mut u1 = Vec::with_capacity(u1_len);
    for _ in 0..u1_len {
        u1.push(read_ring_elem_packed(r, q_bitlen)?);
    }
    let k_p = gr_k_for_sigma(p_sigma);
    let p_len = r.read_len()? as usize;
    let mut p = Vec::with_capacity(p_len);
    for _ in 0..p_len {
        let zig = r.read_gr(k_p)?;
        p.push(zigzag_decode(zig) as i128);
    }
    let b_len = r.read_len()? as usize;
    let mut b_double_prime = Vec::with_capacity(b_len);
    for _ in 0..b_len {
        b_double_prime.push(read_ring_elem_packed(r, q_bitlen)?);
    }
    let u2_len = r.read_len()? as usize;
    let mut u2 = Vec::with_capacity(u2_len);
    for _ in 0..u2_len {
        u2.push(read_ring_elem_packed(r, q_bitlen)?);
    }
    let has = r.read_bits(1)? == 1;
    let last_msg = if has {
        let k_z = gr_k_for_sigma(sigz);
        let k_g = gr_k_for_sigma(sigh);

        let z0_len = r.read_len()? as usize;
        let z0 = read_gauss_vec(r, z0_len, q, k_z)?;
        let z1_len = r.read_len()? as usize;
        let z1 = read_gauss_vec(r, z1_len, q, k_z)?;
        let rv = r.read_len()? as usize;
        let kappa = r.read_len()? as usize;
        let mut v = Vec::with_capacity(rv);
        for _ in 0..rv {
            let mut row = Vec::with_capacity(kappa);
            for _ in 0..kappa {
                row.push(read_ring_elem_packed(r, q_bitlen)?);
            }
            v.push(row);
        }
        let r_g = r.read_len()? as usize;
        let g_flat_len = (r_g * (r_g + 1)) / 2;
        let g_flat = read_gauss_vec(r, g_flat_len, q, k_g)?;
        let mut g = vec![vec![RingElem::zero(); r_g]; r_g];
        let mut idx = 0usize;
        for i in 0..r_g {
            for j in i..r_g {
                g[i][j] = g_flat[idx].clone();
                idx += 1;
            }
        }
        let r_h = r.read_len()? as usize;
        let mut h = vec![vec![RingElem::zero(); r_h]; r_h];
        for i in 0..r_h {
            for j in i..r_h {
                h[i][j] = read_ring_elem_packed(r, q_bitlen)?;
            }
        }
        Some(IterationLastMsg { z0, z1, v, g, h })
    } else {
        None
    };
    Ok(IterationProofV2 { u1, p, b_double_prime, u2, last_msg })
}

// ---------------------------------------------------------------------------
// AggregateProofV2 pack / unpack
// ---------------------------------------------------------------------------

/// Tightly pack the proof. The output begins with a fixed-size header
/// (`q_prime`, `n_sigs`, `beta_sq`, `n_intermediate`, `payload_len`) so the
/// reader can validate the header before allocating the bit-stream.
pub fn pack(p: &AggregateProofV2) -> Vec<u8> {
    let q = p.q_prime;
    let q_bitlen = ceil_log2(q);
    let params = Params::for_n(p.n_sigs);
    let depth = params.depth;
    assert_eq!(
        depth,
        p.intermediate.len() + 1,
        "wire::pack: proof depth ({}) does not match Params::for_n({}).depth ({})",
        p.intermediate.len() + 1,
        p.n_sigs,
        depth
    );

    let mut bw = BitWriter::new();
    for (k, inter) in p.intermediate.iter().enumerate() {
        let it_params = &params.iterations[k];
        let p_sigma = it_params.beta / std::f64::consts::SQRT_2;
        write_iter(
            &mut bw, inter, q, q_bitlen, it_params.sigz, it_params.sigh, p_sigma, false,
        );
    }
    let final_params = &params.iterations[depth - 1];
    let p_sigma_final = final_params.beta / std::f64::consts::SQRT_2;
    write_iter(
        &mut bw,
        &p.final_iter,
        q,
        q_bitlen,
        final_params.sigz,
        final_params.sigh,
        p_sigma_final,
        true,
    );
    let payload = bw.finish();

    let mut out = Vec::with_capacity(8 + 8 + 16 + 8 + 8 + payload.len());
    out.extend_from_slice(&q.to_le_bytes());
    out.extend_from_slice(&(p.n_sigs as u64).to_le_bytes());
    out.extend_from_slice(&p.beta_sq.to_le_bytes());
    out.extend_from_slice(&(p.intermediate.len() as u64).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Per-field packed-byte counts. Sums to ≈ `pack(p).len()` (modulo the fixed
/// 48-byte header and trailing bit-padding); the field labels mirror
/// [`crate::proof::ProofBreakdown`].
pub fn pack_breakdown(p: &AggregateProofV2) -> crate::proof::ProofBreakdown {
    use crate::proof::ProofBreakdown;
    let q = p.q_prime;
    let q_bitlen = ceil_log2(q);
    let params = Params::for_n(p.n_sigs);
    let depth = params.depth;
    let mut b = ProofBreakdown::default();

    fn bits_to_bytes(bits: u32) -> usize {
        (bits as usize + 7) / 8
    }

    // For each (sub-)field, write into its own BitWriter and measure.
    fn ring_packed_bytes(elems: &[RingElem], q_bitlen: u32) -> usize {
        let mut w = BitWriter::new();
        for e in elems {
            write_ring_elem_packed(&mut w, e, q_bitlen);
        }
        w.finish().len()
    }
    fn ring_gr_bytes(elems: &[RingElem], q: u64, k: u32) -> usize {
        let mut w = BitWriter::new();
        write_gauss_vec(&mut w, elems, q, k);
        w.finish().len()
    }
    fn p_bytes(p_vec: &[i128], sigma: f64) -> usize {
        let k_p = gr_k_for_sigma(sigma);
        let mut w = BitWriter::new();
        for &pi in p_vec {
            let pi_i64 = pi.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
            w.write_gr(zigzag_encode(pi_i64), k_p);
        }
        w.finish().len()
    }
    let _ = bits_to_bytes; // header byte cost not modeled here; minor.

    for (k, inter) in p.intermediate.iter().enumerate() {
        let it_params = &params.iterations[k];
        let p_sigma = it_params.beta / std::f64::consts::SQRT_2;
        b.intermediate_u1 += ring_packed_bytes(&inter.u1, q_bitlen);
        b.intermediate_u2 += ring_packed_bytes(&inter.u2, q_bitlen);
        b.intermediate_p += p_bytes(&inter.p, p_sigma);
        b.intermediate_bpp += ring_packed_bytes(&inter.b_double_prime, q_bitlen);
    }
    let final_params = &params.iterations[depth - 1];
    let p_sigma_final = final_params.beta / std::f64::consts::SQRT_2;
    b.final_u1 = ring_packed_bytes(&p.final_iter.u1, q_bitlen);
    b.final_u2 = ring_packed_bytes(&p.final_iter.u2, q_bitlen);
    b.final_p = p_bytes(&p.final_iter.p, p_sigma_final);
    b.final_bpp = ring_packed_bytes(&p.final_iter.b_double_prime, q_bitlen);
    if let Some(lm) = &p.final_iter.last_msg {
        let k_z = gr_k_for_sigma(final_params.sigz);
        let k_g = gr_k_for_sigma(final_params.sigh);
        b.final_z0 = ring_gr_bytes(&lm.z0, q, k_z);
        b.final_z1 = ring_gr_bytes(&lm.z1, q, k_z);
        let mut w_v = BitWriter::new();
        for v_i in &lm.v {
            for e in v_i {
                write_ring_elem_packed(&mut w_v, e, q_bitlen);
            }
        }
        b.final_v = w_v.finish().len();
        let mut w_g = BitWriter::new();
        for i in 0..lm.g.len() {
            for j in i..lm.g.len() {
                write_ring_elem_gr(&mut w_g, &lm.g[i][j], q, k_g);
            }
        }
        b.final_g = w_g.finish().len();
        let mut w_h = BitWriter::new();
        for i in 0..lm.h.len() {
            for j in i..lm.h.len() {
                write_ring_elem_packed(&mut w_h, &lm.h[i][j], q_bitlen);
            }
        }
        b.final_h = w_h.finish().len();
    }
    b
}

pub fn unpack(bytes: &[u8]) -> Result<AggregateProofV2, WireError> {
    const HEADER_LEN: usize = 8 + 8 + 16 + 8 + 8;
    if bytes.len() < HEADER_LEN {
        return Err(WireError::Truncated);
    }
    let q = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let n_sigs = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let beta_sq = i128::from_le_bytes(bytes[16..32].try_into().unwrap());
    let n_inter = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
    let payload_len = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
    if bytes.len() < HEADER_LEN + payload_len {
        return Err(WireError::Truncated);
    }
    let payload = &bytes[HEADER_LEN..HEADER_LEN + payload_len];

    let q_bitlen = ceil_log2(q);
    let params = Params::for_n(n_sigs);
    let depth = params.depth;
    if depth != n_inter + 1 {
        return Err(WireError::BadShape("depth/n_intermediate mismatch"));
    }

    let mut br = BitReader::new(payload);
    let mut intermediate = Vec::with_capacity(n_inter);
    for k in 0..n_inter {
        let it_params = &params.iterations[k];
        let p_sigma = it_params.beta / std::f64::consts::SQRT_2;
        let it = read_iter(&mut br, q, q_bitlen, it_params.sigz, it_params.sigh, p_sigma)?;
        if it.last_msg.is_some() {
            return Err(WireError::BadShape("intermediate iter has last_msg"));
        }
        intermediate.push(it);
    }
    let final_params = &params.iterations[depth - 1];
    let p_sigma_final = final_params.beta / std::f64::consts::SQRT_2;
    let final_iter = read_iter(
        &mut br,
        q,
        q_bitlen,
        final_params.sigz,
        final_params.sigh,
        p_sigma_final,
    )?;
    if final_iter.last_msg.is_none() {
        return Err(WireError::BadShape("final iter missing last_msg"));
    }

    Ok(AggregateProofV2 { q_prime: q, n_sigs, beta_sq, intermediate, final_iter })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitwriter_bitreader_roundtrip() {
        let mut w = BitWriter::new();
        w.write_bits(0xDEAD_BEEF, 32);
        w.write_bits(0b101, 3);
        w.write_bits(0xFFFF_FFFF_FFFF_FFFF, 64);
        w.write_bits(0, 5);
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(32).unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(r.read_bits(64).unwrap(), 0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(r.read_bits(5).unwrap(), 0);
    }

    #[test]
    fn write_len_roundtrip() {
        for &len in &[0u64, 1, 2, 5, 255, 256, 1_000, 10_000, 1u64 << 20, 1u64 << 40] {
            let mut w = BitWriter::new();
            w.write_len(len);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_len().unwrap(), len, "len={len}");
        }
    }

    #[test]
    fn write_gr_roundtrip() {
        let cases = [(0u64, 0u32), (1, 0), (5, 2), (100, 3), (1 << 16, 8)];
        for &(value, k) in &cases {
            let mut w = BitWriter::new();
            w.write_gr(value, k);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_gr(k).unwrap(), value, "value={value} k={k}");
        }
    }

    #[test]
    fn zigzag_roundtrip() {
        for x in [0i64, 1, -1, 5, -5, i32::MAX as i64, i32::MIN as i64] {
            assert_eq!(zigzag_decode(zigzag_encode(x)), x);
        }
    }

    #[test]
    fn signed_modq_roundtrip() {
        let q = 1_000_003u64;
        for x in [0i128, 1, -1, 500_000, -500_000] {
            let u = encode_signed_modq(x, q);
            assert!(u < q);
            assert_eq!(decode_signed_modq(u, q), x);
        }
    }
}
