//! A library to quickly compute (canonical) minimizers of DNA and text sequences.
//!
//! The main functions are:
//! - [`minimizer_positions`]: compute the positions of all minimizers of a sequence.
//! - [`canonical_minimizer_positions`]: compute the positions of all _canonical_ minimizers of a sequence.
//! Adjacent equal positions are deduplicated, but since the canonical minimizer is _not_ _forward_, a position could appear more than once.
//!
//! The implementation uses SIMD by splitting each sequence into 8 chunks and processing those in parallel.
//! The [`minimizer_positions_scalar`] and [`canonical_minimizer_positions_scalar`] versions can be more efficient on short sequences where the overhead of chunking is large.
//!
//! The minimizer of a single window can be found using [`one_minimizer`] and [`one_canonical_minimizer`].
//!
//! ## Minimizers
//!
//! The code is explained in detail in our [preprint](https://doi.org/10.1101/2025.01.27.634998):
//!
//! > SimdMinimizers: Computing random minimizers, fast.
//! > Ragnar Groot Koerkamp, Igor Martayan
//!
//! Briefly, minimizers are defined using two parameters `k` and `w`.
//! Given a sequence of characters, all k-mers (substrings of length `k`) are hashed,
//! and for each _window_ of `k` consecutive k-mers (of length `l = w + k - 1` characters),
//! (the position of) the smallest k-mer is sampled.
//!
//! Minimizers are found as follows:
//! 1. Split the input to 8 chunks that are processed in parallel using SIMD.
//! 2. Compute a 32-bit ntHash rolling hash of the k-mers.
//! 3. Use the 'two stacks' sliding window minimum on the top 16 bits of each hash.
//! 4. Break ties towards the leftmost position by storing the position in the bottom 16 bits.
//! 5. Compute 8 consecutive minimizer positions, and dedup them.
//! 6. Collect the deduplicated minimizer positions from all 8 chunks into a single vector.
//!
//! ## Canonical minimizers
//!
//! _Canonical_ minimizers have the property that the sampled k-mers of a DNA sequence are the same as those sampled from the _reverse complement_ sequence.
//!
//! This works as follows:
//! 1. ntHash is modified to use the canonical version that computes the xor of the hash of the forward and reverse complement k-mer.
//! 2. Compute the leftmost and rightmost minimal k-mer.
//! 3. Compute the 'preferred' strand of the current window as the one with more `TG` characters. This requires `l=w+k-1` to be odd for proper tie-breaking.
//! 4. Return either the leftmost or rightmost smallest k-mer, depending on the preferred strand.
//!
//! ## Input types
//!
//! This crate depends on [`packed-seq`] to handle generic types of input sequences.
//! Most commonly, one should use [`packed_seq::PackedSeqVec`] for packed DNA sequences, but one can also simply wrap a sequence of `ACTGactg` characters in [`packed_seq::AsciiSeqVec`].
//! Additionally, [`simd-minimizers`] works on general (ASCII) `&[u8]` text.
//!
//! The main function provided by [`packed_seq`] is [`packed_seq::Seq::iter_bp`], which splits the input into 8 chunks and iterates them in parallel using SIMD.
//!
//! When dealing with ASCII input, use the `AsciiSeq` and `AsciiSeqVec` types.
//!
//! ## Hash function
//!
//! By default, the library uses the `ntHash` hash function, which maps each DNA base `ACTG` to a pseudo-random value using a table lookup.
//! This hash function is specifically designed to be fast for hashing DNA sequences with input type [`packed_seq::PackedSeq`] and [`packed_seq::AsciiSeq`].
//!
//! For general ASCII sequences (`&[u8]`), `mulHash` is used instead, which instead multiplies each character value by a pseudo-random constant.
//! The `mul_hash` module provides functions that _always_ use mulHash, also for DNA sequences.
//!
//! ## Performance
//!
//! This library depends on AVX2 or NEON SIMD instructions to achieve good performance.
//! Make sure to compile with `-C target-cpu=native` to enable these instructions.
//!
//! All functions take a `out_vec: &mut Vec<u32>` parameter to which positions are _appended_.
//! For best performance, re-use the same `out_vec` between invocations, and [`Vec::clear`] it before or after each call.
//!
//! ## Features
//!
//! - `hide-simd-warning`: If your system does not support AVX2 or NEON, enable this feature to disable the compile warning that will be shown.
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
        note = "simd-minimizers uses AVX2 or NEON SIMD instructions. Compile using `-C target-cpu=native` to get the expected performance. Hide this warning using the `hide-simd-warning` feature."
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

/// Re-exported internals. Used for benchmarking, and not part of the semver-compatible stable API.
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

use collect::{collect_and_dedup_into, collect_and_dedup_with_index_into};
use itertools::Itertools;
use minimizers::{
    canonical_minimizers_seq_scalar, canonical_minimizers_seq_simd, minimizers_seq_scalar,
    minimizers_seq_simd,
};
use nthash::{MulHasher, NtHasher};
use packed_seq::u32x8 as S;
use packed_seq::Seq;

/// Minimizer position of a single window.
pub fn one_minimizer<'s, S: Seq<'s>>(seq: S, k: usize) -> usize {
    if S::BITS_PER_CHAR == 2 {
        minimizers::minimizer::<NtHasher>(seq, k)
    } else {
        minimizers::minimizer::<MulHasher>(seq, k)
    }
}
/// Canonical minimizer position of a single window.
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
        let head_padding = minimizers_seq_simd::<_, NtHasher>(seq, k, w);
        collect_and_dedup_into(head_padding, out_vec);
    } else {
        let head_padding = minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_into(head_padding, out_vec);
    }
}

/// Deduplicated positions of all canonical minimizers in the sequence, using SIMD.
///
/// `l=w+k-1` must be odd to determine the strand of each window.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
pub fn canonical_minimizer_positions<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    if S::BITS_PER_CHAR == 2 {
        let head_padding = canonical_minimizers_seq_simd::<_, NtHasher>(seq, k, w);
        collect_and_dedup_into(head_padding, out_vec);
    } else {
        let head_padding = canonical_minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_into(head_padding, out_vec);
    }
}

/// Deduplicated positions of all minimizers in the sequence with starting positions of the corresponding super-k-mers, using SIMD.
///
/// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
pub fn minimizer_and_superkmer_positions<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    w: usize,
    min_pos_vec: &mut Vec<u32>,
    sk_pos_vec: &mut Vec<u32>,
) {
    if S::BITS_PER_CHAR == 2 {
        let head_tail = minimizers_seq_simd::<_, NtHasher>(seq, k, w);
        collect_and_dedup_with_index_into(head_tail, min_pos_vec, sk_pos_vec);
    } else {
        let head_tail = minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_with_index_into(head_tail, min_pos_vec, sk_pos_vec);
    }
}

/// Deduplicated positions of all canonical minimizers in the sequence with starting positions of the corresponding super-k-mers, using SIMD.
///
/// `l=w+k-1` must be odd to determine the strand of each window.
///
/// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
pub fn canonical_minimizer_and_superkmer_positions<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    w: usize,
    min_pos_vec: &mut Vec<u32>,
    sk_pos_vec: &mut Vec<u32>,
) {
    if S::BITS_PER_CHAR == 2 {
        let head_tail = canonical_minimizers_seq_simd::<_, NtHasher>(seq, k, w);
        collect_and_dedup_with_index_into(head_tail, min_pos_vec, sk_pos_vec);
    } else {
        let head_tail = canonical_minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_with_index_into(head_tail, min_pos_vec, sk_pos_vec);
    }
}

/// Variants that always use mulHash, instead of the default ntHash for DNA and mulHash for text.
pub mod mul_hash {
    use super::*;

    /// Deduplicated positions of all minimizers in the sequence, using SIMD.
    ///
    /// Positions are appended to a reusable `out_vec` to avoid allocations.
    pub fn minimizer_positions<'s, S: Seq<'s>>(seq: S, k: usize, w: usize, out_vec: &mut Vec<u32>) {
        let head_padding = minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_into(head_padding, out_vec);
    }

    /// Deduplicated positions of all canonical minimizers in the sequence, using SIMD.
    ///
    /// `l=w+k-1` must be odd to determine the strand of each window.
    ///
    /// Positions are appended to a reusable `out_vec` to avoid allocations.
    pub fn canonical_minimizer_positions<'s, S: Seq<'s>>(
        seq: S,
        k: usize,
        w: usize,
        out_vec: &mut Vec<u32>,
    ) {
        let head_tail = canonical_minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_into(head_tail, out_vec);
    }

    /// Deduplicated positions of all minimizers in the sequence with starting positions of the corresponding super-k-mers, using SIMD.
    ///
    /// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
    pub fn minimizer_and_superkmer_positions<'s, S: Seq<'s>>(
        seq: S,
        k: usize,
        w: usize,
        min_pos_vec: &mut Vec<u32>,
        sk_pos_vec: &mut Vec<u32>,
    ) {
        let head_tail = minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_with_index_into(head_tail, min_pos_vec, sk_pos_vec);
    }

    /// Deduplicated positions of all canonical minimizers in the sequence with starting positions of the corresponding super-k-mers, using SIMD.
    ///
    /// `l=w+k-1` must be odd to determine the strand of each window.
    ///
    /// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
    pub fn canonical_minimizer_and_superkmer_positions<'s, S: Seq<'s>>(
        seq: S,
        k: usize,
        w: usize,
        min_pos_vec: &mut Vec<u32>,
        sk_pos_vec: &mut Vec<u32>,
    ) {
        let head_tail = canonical_minimizers_seq_simd::<_, MulHasher>(seq, k, w);
        collect_and_dedup_with_index_into(head_tail, min_pos_vec, sk_pos_vec);
    }
}

/// Deduplicated positions of all minimizers in the sequence.
/// This scalar version can be faster for short sequences.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
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

/// Deduplicated positions of all canonical minimizers in the sequence.
/// This scalar version can be faster for short sequences.
///
/// `l=w+k-1` must be odd to determine the strand of each window.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
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

/// Deduplicated positions of all minimizers in the sequence with starting positions of the corresponding super-k-mers.
/// This scalar version can be faster for short sequences.
///
/// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
pub fn minimizer_and_superkmer_positions_scalar<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    w: usize,
    min_pos_vec: &mut Vec<u32>,
    sk_pos_vec: &mut Vec<u32>,
) {
    if S::BITS_PER_CHAR == 2 {
        let (sk_pos, min_pos): (Vec<_>, Vec<_>) = minimizers_seq_scalar::<NtHasher>(seq, k, w)
            .enumerate()
            .dedup_by(|x, y| x.1 == y.1)
            .map(|(x, y)| (x as u32, y))
            .unzip();
        min_pos_vec.extend(min_pos);
        sk_pos_vec.extend(sk_pos);
    } else {
        let (sk_pos, min_pos): (Vec<_>, Vec<_>) = minimizers_seq_scalar::<MulHasher>(seq, k, w)
            .enumerate()
            .dedup_by(|x, y| x.1 == y.1)
            .map(|(x, y)| (x as u32, y))
            .unzip();
        min_pos_vec.extend(min_pos);
        sk_pos_vec.extend(sk_pos);
    }
}

/// Deduplicated positions of all canonical minimizers in the sequence with starting positions of the corresponding super-k-mers.
/// This scalar version can be faster for short sequences.
///
/// `l=w+k-1` must be odd to determine the strand of each window.
///
/// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
pub fn canonical_minimizer_and_superkmer_positions_scalar<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    w: usize,
    min_pos_vec: &mut Vec<u32>,
    sk_pos_vec: &mut Vec<u32>,
) {
    if S::BITS_PER_CHAR == 2 {
        let (sk_pos, min_pos): (Vec<_>, Vec<_>) =
            canonical_minimizers_seq_scalar::<NtHasher>(seq, k, w)
                .enumerate()
                .dedup_by(|x, y| x.1 == y.1)
                .map(|(x, y)| (x as u32, y))
                .unzip();
        min_pos_vec.extend(min_pos);
        sk_pos_vec.extend(sk_pos);
    } else {
        let (sk_pos, min_pos): (Vec<_>, Vec<_>) =
            canonical_minimizers_seq_scalar::<MulHasher>(seq, k, w)
                .enumerate()
                .dedup_by(|x, y| x.1 == y.1)
                .map(|(x, y)| (x as u32, y))
                .unzip();
        min_pos_vec.extend(min_pos);
        sk_pos_vec.extend(sk_pos);
    }
}

// TODO: Make scalar methods return an iterator.
