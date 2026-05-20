//! Falcon-512 public-key / signature decoding and `HashToPoint`.
//!
//! The cryptography (keygen, signing, verification) is delegated to the
//! audited `pqcrypto-falcon` C implementation; this module only reverses the
//! published *byte formats* to recover the polynomial coefficients the
//! LaBRADOR relation operates on. Format decoding is not security-sensitive
//! and is fully self-checked: a correct decode must satisfy the Falcon
//! verification equation `s1 + s2·h = c (mod 12289)`.
//!
//! Formats (Falcon-512, as emitted by `pqcrypto-falcon`):
//! - public key:  `[0x09] ‖ 896 bytes` — `h`, 512 coeffs packed 14 bits each,
//!   most-significant-bit first.
//! - signature:   `[0x39] ‖ nonce[40] ‖ compressed(s2)` — `s2` in Falcon's
//!   sign/low-7/unary-high compression code, MSB-first, zero-padded.

use crate::falcon_ring::{FPoly, FALCON_N, FALCON_Q};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

/// Length of a Falcon-512 nonce.
pub const NONCE_LEN: usize = 40;

const PK_HEADER: u8 = 0x09; // 0x00 + logn(=9)
const SIG_HEADER: u8 = 0x39; // 0x30 (compressed) + logn(=9)
const PK_LEN: usize = 1 + (FALCON_N * 14 + 7) / 8; // 1 + 896

/// Errors from decoding Falcon byte blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Wrong length for the claimed object.
    BadLength,
    /// Unexpected format/header byte.
    BadHeader,
    /// A packed public-key coefficient was `>= q`.
    CoeffOutOfRange,
    /// The compressed signature stream is malformed (bad terminator,
    /// negative zero, overlong unary part, or truncated).
    BadCompression,
    /// Padding bits after the encoded data were not all zero.
    NonZeroPadding,
}

/// A fully decoded Falcon-512 signature instance, ready for relation encoding.
#[derive(Debug, Clone)]
pub struct FalconSig {
    /// Public-key polynomial `h`.
    pub h: FPoly,
    /// Hashed message point `c = HashToPoint(nonce ‖ message)`.
    pub c: FPoly,
    /// Recovered short polynomial `s1 = c - s2·h`.
    pub s1: FPoly,
    /// Signature short polynomial `s2`.
    pub s2: FPoly,
    /// The 40-byte nonce.
    pub nonce: [u8; NONCE_LEN],
}

/// Reads bits most-significant-first from a byte slice.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // bit index
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// Next bit, or `None` if the stream is exhausted.
    fn bit(&mut self) -> Option<u32> {
        let byte = self.pos / 8;
        if byte >= self.data.len() {
            return None;
        }
        let bit = 7 - (self.pos % 8);
        self.pos += 1;
        Some(((self.data[byte] >> bit) & 1) as u32)
    }

    /// Read `n` bits as an integer (MSB-first).
    fn bits(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Some(v)
    }

    /// True if every remaining bit is zero.
    fn rest_is_zero(&mut self) -> bool {
        while let Some(b) = self.bit() {
            if b != 0 {
                return false;
            }
        }
        true
    }
}

/// Decode a Falcon-512 public key into the polynomial `h`.
pub fn decode_public_key(bytes: &[u8]) -> Result<FPoly, ParseError> {
    if bytes.len() != PK_LEN {
        return Err(ParseError::BadLength);
    }
    if bytes[0] != PK_HEADER {
        return Err(ParseError::BadHeader);
    }
    let mut h = [0u32; FALCON_N];
    let mut acc = 0u32;
    let mut acc_len = 0u32;
    let mut idx = 0usize;
    for &b in &bytes[1..] {
        acc = (acc << 8) | b as u32;
        acc_len += 8;
        if acc_len >= 14 {
            acc_len -= 14;
            let w = (acc >> acc_len) & 0x3FFF;
            if w >= FALCON_Q {
                return Err(ParseError::CoeffOutOfRange);
            }
            h[idx] = w;
            idx += 1;
        }
    }
    debug_assert_eq!(idx, FALCON_N);
    // 896 bytes = 512*14 bits exactly, so acc_len must be 0.
    if acc & ((1 << acc_len) - 1) != 0 {
        return Err(ParseError::NonZeroPadding);
    }
    Ok(FPoly::from_residues(h))
}

/// Decode a Falcon-512 detached signature into `(nonce, s2)`.
pub fn decode_signature(bytes: &[u8]) -> Result<([u8; NONCE_LEN], FPoly), ParseError> {
    if bytes.len() < 1 + NONCE_LEN {
        return Err(ParseError::BadLength);
    }
    if bytes[0] != SIG_HEADER {
        return Err(ParseError::BadHeader);
    }
    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&bytes[1..1 + NONCE_LEN]);

    let mut r = BitReader::new(&bytes[1 + NONCE_LEN..]);
    let mut s2 = [0i32; FALCON_N];
    for slot in s2.iter_mut() {
        let sign = r.bit().ok_or(ParseError::BadCompression)?;
        let low = r.bits(7).ok_or(ParseError::BadCompression)? as i32;
        // High part: count zero bits up to a terminating 1.
        let mut high = 0i32;
        loop {
            let b = r.bit().ok_or(ParseError::BadCompression)?;
            if b == 1 {
                break;
            }
            high += 1;
            // Falcon-512 |coeff| stays well under 2^12; cap defensively.
            if high > 2047 {
                return Err(ParseError::BadCompression);
            }
        }
        let mag = (high << 7) | low;
        if sign == 1 && mag == 0 {
            // "Negative zero" is not a valid encoding.
            return Err(ParseError::BadCompression);
        }
        *slot = if sign == 1 { -mag } else { mag };
    }
    if !r.rest_is_zero() {
        return Err(ParseError::NonZeroPadding);
    }
    Ok((nonce, FPoly::from_i32(&s2)))
}

/// Falcon `HashToPoint`: `c = SHAKE256(nonce ‖ message)` reject-sampled into
/// `R_f`. Matches Falcon's constant-time variant (same accepted stream).
pub fn hash_to_point(nonce: &[u8; NONCE_LEN], message: &[u8]) -> FPoly {
    const ACCEPT_BOUND: u32 = 5 * FALCON_Q; // 61445 = 2^16 - (2^16 mod q)

    let mut shake = Shake256::default();
    shake.update(nonce);
    shake.update(message);
    let mut xof = shake.finalize_xof();

    let mut c = [0u32; FALCON_N];
    let mut i = 0usize;
    let mut buf = [0u8; 2];
    while i < FALCON_N {
        xof.read(&mut buf);
        let w = ((buf[0] as u32) << 8) | buf[1] as u32;
        if w < ACCEPT_BOUND {
            c[i] = w % FALCON_Q;
            i += 1;
        }
    }
    FPoly::from_residues(c)
}

/// Decode a complete Falcon-512 verification instance and recover `s1`.
///
/// This does *not* itself check the norm bound or the verification equation
/// — callers (and the relation encoder) do — but `s1` is recovered exactly
/// as Falcon verification would: `s1 = c - s2·h (mod q)`.
pub fn decode_instance(
    public_key: &[u8],
    signature: &[u8],
    message: &[u8],
) -> Result<FalconSig, ParseError> {
    let h = decode_public_key(public_key)?;
    let (nonce, s2) = decode_signature(signature)?;
    let c = hash_to_point(&nonce, message);
    let s1 = c.sub(&s2.mul(&h));
    Ok(FalconSig {
        h,
        c,
        s1,
        s2,
        nonce,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_reads_msb_first() {
        let mut r = BitReader::new(&[0b1011_0010]);
        assert_eq!(r.bits(4), Some(0b1011));
        assert_eq!(r.bits(4), Some(0b0010));
        assert_eq!(r.bit(), None);
    }
}
