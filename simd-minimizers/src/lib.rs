//! # `simd-minimizers` library
//!
//! Use the `minimizer_positions` and `canonical_minimizer_positions` functions here  to compute (canonical) minimizer positions.
//! Adjacent equal positions are deduplicated, but since the canonical minimizer is not _forward_, a position could appear more than once.
//! The `scalar` versions may be more efficient for short sequences
//! Submodules are exported for testing and benchmarking purposes, but should not be considered part of the stable API.
//!
//! For the SIMD versions, input must be a packed sequence: `packed_seq::PackedSeq`.
//! The scalar versions also accept plain ASCII `&[u8]` sequences that must only consist of `ACGTacgt` characters.
//! If you want to use the SIMD version on `ACGTacgt` input, pack the sequence using `PackedSeq::new(seq)`.
//!
//! General ASCII alphabet is not supported, since ntHash relies on the 2-bit encoding.
//!
//! ## `packed-seq` overview
//!
//! The `packed-seq` crate provides the `Seq` trait that models a non-owned sequence of bases, like a `&[u8]` with `ACTGactg` values.
//! It also has a `SeqVec` trait for owned variants.
//!
//! When dealing with ASCII input, use the `AsciiSeq` and `AsciiSeqVec` types.
//!
//! When dealing with packed sequences, use the `PackedSeq` and `PackedSeqVec` types.
//!
//!
//! ## Examples
//!
//! ```
//! // Scalar ASCII version.
//! use packed_seq::{Seq, AsciiSeq};
//! let seq = b"ACGTGCTCAGAGACTCAG";
//! let ascii_seq = AsciiSeq(seq);
//! let k = 5;
//! let w = 7;
//! let mut out_vec = Vec::new();
//! simd_minimizers::minimizer_positions_scalar(ascii_seq, k, w, &mut out_vec);
//! assert_eq!(out_vec, vec![0, 6, 8, 10, 12]);
//! ```
//!
//! ```
//! // Packed SIMD version.
//! use packed_seq::{complement_char, PackedSeqVec, SeqVec};
//! let seq = b"ACGTGCTCAGAGACTCAG";
//! let k = 5;
//! let w = 7;
//!
//! let packed_seq = PackedSeqVec::from_ascii(seq);
//! let mut fwd_pos = Vec::new();
//! // Unfortunately, `PackedSeqVec` can not `Deref` into a `PackedSeq`.
//! simd_minimizers::canonical_minimizer_positions(packed_seq.as_slice(), k, w, &mut fwd_pos);
//! assert_eq!(fwd_pos, vec![3, 5, 12]);
//!
//! // Check that reverse complement sequence has minimizers at 'reverse' positions.
//! let rc_seq = seq.iter().rev().map(|&b| complement_char(b)).collect::<Vec<_>>();
//! let rc_packed_seq = PackedSeqVec::from_ascii(&rc_seq);
//! let mut rc_pos = Vec::new();
//! simd_minimizers::canonical_minimizer_positions(rc_packed_seq.as_slice(), k, w, &mut rc_pos);
//! assert_eq!(rc_pos, vec![1, 8, 10]);
//! for (fwd, &rc) in std::iter::zip(fwd_pos, rc_pos.iter().rev()) {
//!     assert_eq!(fwd as usize, seq.len() - k - rc as usize);
//! }
//! ```
#![cfg_attr(
    not(any(
        doc,
        target_feature = "avx2",
        target_feature = "neon",
        feature = "hide-simd-warning"
    )),
    deprecated(
        note = "This implementation uses SIMD, make sure you are compiling using `-C target-cpu=native` to get the expected performance. You can hide this warning by enabling the `hide-simd-warning` feature."
    )
)]

// Private modules.
mod intrinsics;

// Re-exported modules.
mod anti_lex;
mod canonical;
mod collect;
mod minimizers;
mod nthash;
mod sliding_min;

#[cfg(test)]
mod test;

// TODO: Old and in-development modules.
// mod linearize;

/// Re-exported internals. Not part of the server-compatible stable API.
pub mod private {
    pub mod anti_lex {
        pub use crate::anti_lex::*;
    }
    pub mod canonical {
        pub use crate::canonical::*;
    }
    pub mod collect {
        pub use crate::collect::*;
    }
    pub mod minimizers {
        pub use crate::minimizers::*;
    }
    pub mod nthash {
        pub use crate::nthash::*;
    }
    pub mod sliding_min {
        pub use crate::sliding_min::*;
    }
    pub use packed_seq::u32x8 as S;
}

/// Re-export of the `packed-seq` crate.
pub use packed_seq;

use collect::collect_and_dedup_into;
use itertools::Itertools;
use minimizers::{
    canonical_minimizers_seq_scalar, canonical_minimizers_seq_simd, minimizers_seq_scalar,
    minimizers_seq_simd,
};
use nthash::{MulHasher, NtHasher};
use packed_seq::u32x8 as S;
use packed_seq::Seq;

/// Minimizer of a single window.
pub fn one_minimizer<'s, S: Seq<'s>>(seq: S, k: usize) -> usize {
    if S::BITS_PER_CHAR == 2 {
        minimizers::minimizer::<NtHasher>(seq, k)
    } else {
        minimizers::minimizer::<MulHasher>(seq, k)
    }
}
/// Minimizer of a single window.
pub fn one_canonical_minimizer<'s, S: Seq<'s>>(seq: S, k: usize) -> usize {
    if S::BITS_PER_CHAR == 2 {
        minimizers::minimizer::<NtHasher>(seq, k)
    } else {
        minimizers::minimizer::<MulHasher>(seq, k)
    }
}

/// Deduplicated positions of all minimizers in the sequence, using SIMD.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
pub fn minimizer_positions<'s, S: Seq<'s>>(seq: S, k: usize, w: usize, out_vec: &mut Vec<u32>) {
    if S::BITS_PER_CHAR == 2 {
        let head_tail = minimizers_seq_simd::<_, NtHasher>(seq, k, w);
        collect_and_dedup_into::<false>(head_tail, out_vec);
    } else {
        let head_tail = minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_into::<false>(head_tail, out_vec);
    }
}

/// Deduplicated positions of all canonical minimizers in the sequence, using SIMD.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
/// l=w+k-1 must be odd to determine the strand of each window.
pub fn canonical_minimizer_positions<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    if S::BITS_PER_CHAR == 2 {
        let head_tail = canonical_minimizers_seq_simd::<_, NtHasher>(seq, k, w);
        collect_and_dedup_into::<false>(head_tail, out_vec);
    } else {
        let head_tail = canonical_minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_into::<false>(head_tail, out_vec);
    }
}

/// Variants that always use `mulHash`, instead of defaulting to `ntHash` for DNA data.
pub mod mul_hash {
    use super::*;

    pub fn minimizer_positions<'s, S: Seq<'s>>(seq: S, k: usize, w: usize, out_vec: &mut Vec<u32>) {
        let head_tail = minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_into::<false>(head_tail, out_vec);
    }

    /// Deduplicated positions of all canonical minimizers in the sequence, using SIMD.
    ///
    /// Positions are appended to a reusable `out_vec` to avoid allocations.
    /// l=w+k-1 must be odd to determine the strand of each window.
    pub fn canonical_minimizer_positions<'s, S: Seq<'s>>(
        seq: S,
        k: usize,
        w: usize,
        out_vec: &mut Vec<u32>,
    ) {
        let head_tail = canonical_minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_into::<false>(head_tail, out_vec);
    }
}

/// Deduplicated positions of all minimizers in the sequence, not using SIMD.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
/// This scalar version can be faster for sequences known to be short.
pub fn minimizer_positions_scalar<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    if S::BITS_PER_CHAR == 2 {
        out_vec.extend(minimizers_seq_scalar::<NtHasher>(seq, k, w).dedup());
    } else {
        out_vec.extend(minimizers_seq_scalar::<MulHasher>(seq, k, w).dedup());
    }
}

/// Deduplicated positions of all canonical minimizers in the sequence, not using SIMD.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
/// l=w+k-1 must be odd to determine the strand of each window.
/// This scalar version can be faster for sequences known to be short.
pub fn canonical_minimizer_positions_scalar<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    if S::BITS_PER_CHAR == 2 {
        out_vec.extend(canonical_minimizers_seq_scalar::<NtHasher>(seq, k, w).dedup());
    } else {
        out_vec.extend(canonical_minimizers_seq_scalar::<MulHasher>(seq, k, w).dedup());
    }
}

// TODO: Make scalar methods return an iterator.
