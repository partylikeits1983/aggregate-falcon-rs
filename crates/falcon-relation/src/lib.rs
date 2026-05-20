//! `falcon-relation` — Falcon-512 ring arithmetic and (later) the encoding of
//! "N Falcon signatures are valid" as a LaBRADOR quadratic constraint system.
//!
//! Modules:
//! - [`falcon_ring`] — arithmetic in `Z_12289[X]/(X^512+1)` (done).
//! - `parse` — Falcon-512 public-key/signature decoding + HashToPoint (Phase 2,
//!   pending the Falcon C-FFI binding).
//! - `embed` / `relation` — Falcon→LaBRADOR embedding and constraint assembly
//!   (Phase 3).

pub mod embed;
pub mod falcon_ring;
pub mod parse;
pub mod relation;

pub use embed::{embed_fpoly_centered, embed_signed, extract_centered, mul_subring, C};
pub use falcon_ring::{FPoly, FALCON_BETA_SQ, FALCON_N, FALCON_Q};
pub use parse::{decode_instance, FalconSig, ParseError};
