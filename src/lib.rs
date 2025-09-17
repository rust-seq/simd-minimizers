//! A library to quickly compute (canonical) minimizers of DNA and text sequences.
//!
//! The main functions are:
//! - [`minimizer_positions`]: compute the positions of all minimizers of a sequence.
//! - [`canonical_minimizer_positions`]: compute the positions of all _canonical_ minimizers of a sequence.
//! Adjacent equal positions are deduplicated, but since the canonical minimizer is _not_ _forward_, a position could appear more than once.
//!
//! The implementation uses SIMD by splitting each sequence into 8 chunks and processing those in parallel.
//!
//! When using super-k-mers, use the `_and_superkmer` variants to additionally return a vector containing the index of the first window the minimizer is minimal.
//!
//! The minimizer of a single window can be found using [`one_minimizer`] and [`one_canonical_minimizer`], but note that these functions are not nearly as efficient.
//!
//! The [`scalar`] versions are mostly for testing only, and basically always slower.
//! Only for short sequences with length up to 100 is [`scalar::minimizer_positions_scalar`] faster than the SIMD version.
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
//! simd_minimizers::scalar::minimizer_positions_scalar(ascii_seq, k, w, &mut out_vec);
//! assert_eq!(out_vec, vec![0, 6, 8, 10, 12]);
//! ```
//!
//! ```
//! // Packed SIMD version.
//! use packed_seq::{PackedSeqVec, SeqVec, Seq};
//! let seq = b"ACGTGCTCAGAGACTCAG";
//! let k = 5;
//! let w = 7;
//!
//! let packed_seq = PackedSeqVec::from_ascii(seq);
//! let mut fwd_pos = Vec::new();
//! // Unfortunately, `PackedSeqVec` can not `Deref` into a `PackedSeq`, so `as_slice` is needed.
//! simd_minimizers::canonical_minimizer_positions(packed_seq.as_slice(), k, w, &mut fwd_pos);
//! assert_eq!(fwd_pos, vec![3, 5, 12]);
//!
//! let fwd_vals: Vec<_> = simd_minimizers::iter_canonical_minimizer_values(packed_seq.as_slice(), k, &fwd_pos).collect();
//! assert_eq!(fwd_vals, vec![
//!     // A C G A G, GAGCA is rc of TGCTC at pos 3
//!     0b0001110011,
//!     // G A C T C, CTCAG is at pos 5
//!     0b1100011001,
//!     // A C T C A, ACTCA is at pos 12
//!     0b0001100100
//! ]);
//!
//! // Check that reverse complement sequence has minimizers at 'reverse' positions.
//! let rc_packed_seq = packed_seq.as_slice().to_revcomp();
//! let mut rc_pos = Vec::new();
//! simd_minimizers::canonical_minimizer_positions(rc_packed_seq.as_slice(), k, w, &mut rc_pos);
//! assert_eq!(rc_pos, vec![1, 8, 10]);
//! for (fwd, &rc) in std::iter::zip(fwd_pos, rc_pos.iter().rev()) {
//!     assert_eq!(fwd as usize, seq.len() - k - rc as usize);
//! }
//! let mut rc_vals: Vec<_> = simd_minimizers::iter_canonical_minimizer_values(rc_packed_seq.as_slice(), k, &rc_pos).collect();
//! rc_vals.reverse();
//! assert_eq!(rc_vals, fwd_vals);
//! ```
//!
//! ```
//! // Packed SIMD version with seeded hashes.
//! use packed_seq::{PackedSeqVec, SeqVec, Seq};
//! let seq = b"ACGTGCTCAGAGACTCAG";
//! let k = 5;
//! let w = 7;
//! let seed = 101010;
//!
//! let packed_seq = PackedSeqVec::from_ascii(seq);
//! let mut fwd_pos = Vec::new();
//! simd_minimizers::seeded::canonical_minimizer_positions(packed_seq.as_slice(), k, w, seed, &mut fwd_pos);
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

mod canonical;
pub mod collect;
mod minimizers;
mod sliding_min;
mod intrinsics {
    mod dedup;
    pub use dedup::{append_unique_vals, append_unique_vals_2};
}

#[cfg(test)]
mod test;

/// Re-exported internals. Used for benchmarking, and not part of the semver-compatible stable API.
pub mod private {
    pub mod canonical {
        pub use crate::canonical::*;
    }
    pub mod minimizers {
        pub use crate::minimizers::*;
    }
    pub mod sliding_min {
        pub use crate::sliding_min::*;
    }
    pub use packed_seq::u32x8 as S;
}

use collect::CollectAndDedup;
/// Re-export of the `packed-seq` crate.
pub use packed_seq;
/// Re-export of the `seq-hash` crate.
pub use seq_hash;

use itertools::Itertools;
use minimizers::{
    canonical_minimizers_seq_scalar, canonical_minimizers_seq_simd, minimizers_seq_scalar,
    minimizers_seq_simd,
};
use packed_seq::u32x8 as S;
use packed_seq::Seq;
use seq_hash::KmerHasher;

pub use minimizers::one_minimizer;
pub use sliding_min::Cache;

thread_local! {
    static CACHE: std::cell::RefCell<Cache> = std::cell::RefCell::new(Cache::default());
}

/// Deduplicated positions of all minimizers in the sequence, using SIMD.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
pub fn minimizer_positions<'s>(
    seq: impl Seq<'s>,
    hasher: &impl KmerHasher,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    CACHE.with_borrow_mut(|cache| {
        minimizers_seq_simd(seq, hasher, w, cache).collect_and_dedup_into(out_vec)
    })
}

/// Deduplicated positions of all canonical minimizers in the sequence, using SIMD.
///
/// `l=w+k-1` must be odd to determine the strand of each window.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
pub fn canonical_minimizer_positions<'s>(
    seq: impl Seq<'s>,
    hasher: &impl KmerHasher,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    CACHE.with_borrow_mut(|cache| {
        canonical_minimizers_seq_simd(seq, hasher, w, cache).collect_and_dedup_into(out_vec)
    })
}

/// Deduplicated positions of all minimizers in the sequence with starting positions of the corresponding super-k-mers, using SIMD.
///
/// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
pub fn minimizer_and_superkmer_positions<'s, S: Seq<'s>>(
    seq: S,
    hasher: &impl KmerHasher,
    w: usize,
    min_pos_vec: &mut Vec<u32>,
    sk_pos_vec: &mut Vec<u32>,
) {
    CACHE.with_borrow_mut(|cache| {
        minimizers_seq_simd(seq, hasher, w, cache)
            .collect_and_dedup_with_index_into(min_pos_vec, sk_pos_vec)
    })
}

/// Deduplicated positions of all canonical minimizers in the sequence with starting positions of the corresponding super-k-mers, using SIMD.
///
/// `l=w+k-1` must be odd to determine the strand of each window.
///
/// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
pub fn canonical_minimizer_and_superkmer_positions<'s, S: Seq<'s>>(
    seq: S,
    hasher: &impl KmerHasher,
    w: usize,
    min_pos_vec: &mut Vec<u32>,
    sk_pos_vec: &mut Vec<u32>,
) {
    CACHE.with_borrow_mut(|cache| {
        canonical_minimizers_seq_simd(seq, hasher, w, cache)
            .collect_and_dedup_with_index_into(min_pos_vec, sk_pos_vec);
    })
}

/// Given a sequence and a list of positions, iterate over the k-mer values at those positions.
#[inline(always)]
pub fn iter_minimizer_values<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    positions: &'s [u32],
) -> impl ExactSizeIterator<Item = u64> + Clone {
    positions
        .iter()
        .map(move |&pos| seq.read_kmer(k, pos as usize))
}

/// Given a sequence and a list of positions, iterate over the *canonical* k-mer values at those positions.
///
/// Canonical k-mers are defined as the *minimum* of the k-mer and its reverse complement.
/// Note that this also works for even `k`, but typically one would want `k` to be odd.
#[inline(always)]
pub fn iter_canonical_minimizer_values<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    positions: &'s [u32],
) -> impl ExactSizeIterator<Item = u64> + Clone {
    positions.iter().map(move |&pos| {
        let a = seq.read_kmer(k, pos as usize);
        let b = seq.read_revcomp_kmer(k, pos as usize);
        core::cmp::min(a, b)
    })
}

/// Given a sequence and a list of positions, iterate over the k-mer values at those positions.
#[inline(always)]
pub fn iter_minimizer_values_u128<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    positions: &'s [u32],
) -> impl ExactSizeIterator<Item = u128> + Clone {
    positions
        .iter()
        .map(move |&pos| seq.read_kmer_u128(k, pos as usize))
}

/// Given a sequence and a list of positions, iterate over the *canonical* k-mer values at those positions.
///
/// Canonical k-mers are defined as the *minimum* of the k-mer and its reverse complement.
/// Note that this also works for even `k`, but typically one would want `k` to be odd.
#[inline(always)]
pub fn iter_canonical_minimizer_values_u128<'s, S: Seq<'s>>(
    seq: S,
    k: usize,
    positions: &'s [u32],
) -> impl ExactSizeIterator<Item = u128> + Clone {
    positions.iter().map(move |&pos| {
        let a = seq.read_kmer_u128(k, pos as usize);
        let b = seq.read_revcomp_kmer_u128(k, pos as usize);
        core::cmp::min(a, b)
    })
}

/// Scalar variants that are nearly always slower.
///
/// Can be used for testing and debugging.
pub mod scalar {
    use crate::collect::collect_and_dedup_into_scalar;

    use super::*;

    /// Deduplicated positions of all minimizers in the sequence.
    /// This scalar version can be faster for short sequences.
    ///
    /// Positions are appended to a reusable `out_vec` to avoid allocations.
    pub fn minimizer_positions_scalar<'s, S: Seq<'s>>(
        seq: S,
        hasher: &impl KmerHasher,
        w: usize,
        out_vec: &mut Vec<u32>,
    ) {
        CACHE.with_borrow_mut(|cache| {
            collect_and_dedup_into_scalar(minimizers_seq_scalar(seq, hasher, w, cache), out_vec);
        })
    }

    /// Deduplicated positions of all canonical minimizers in the sequence.
    /// This scalar version can be faster for short sequences.
    ///
    /// `l=w+k-1` must be odd to determine the strand of each window.
    ///
    /// Positions are appended to a reusable `out_vec` to avoid allocations.
    pub fn canonical_minimizer_positions_scalar<'s, S: Seq<'s>>(
        seq: S,
        hasher: &impl KmerHasher,
        w: usize,
        out_vec: &mut Vec<u32>,
    ) {
        CACHE.with_borrow_mut(|cache| {
            collect_and_dedup_into_scalar(
                canonical_minimizers_seq_scalar(seq, hasher, w, cache),
                out_vec,
            );
        })
    }

    /// Deduplicated positions of all minimizers in the sequence with starting positions of the corresponding super-k-mers.
    /// This scalar version can be faster for short sequences.
    ///
    /// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
    pub fn minimizer_and_superkmer_positions_scalar<'s, S: Seq<'s>>(
        seq: S,
        hasher: &impl KmerHasher,
        w: usize,
        min_pos_vec: &mut Vec<u32>,
        sk_pos_vec: &mut Vec<u32>,
    ) {
        CACHE.with_borrow_mut(|cache| {
            let (sk_pos, min_pos): (Vec<_>, Vec<_>) = minimizers_seq_scalar(seq, hasher, w, cache)
                .enumerate()
                .dedup_by(|x, y| x.1 == y.1)
                .map(|(x, y)| (x as u32, y))
                .unzip();
            min_pos_vec.extend(min_pos);
            sk_pos_vec.extend(sk_pos);
        })
    }

    /// Deduplicated positions of all canonical minimizers in the sequence with starting positions of the corresponding super-k-mers.
    /// This scalar version can be faster for short sequences.
    ///
    /// `l=w+k-1` must be odd to determine the strand of each window.
    ///
    /// Positions are appended to reusable `min_pos_vec` and `sk_pos_vec` to avoid allocations.
    pub fn canonical_minimizer_and_superkmer_positions_scalar<'s, S: Seq<'s>>(
        seq: S,
        hasher: &impl KmerHasher,
        w: usize,
        min_pos_vec: &mut Vec<u32>,
        sk_pos_vec: &mut Vec<u32>,
    ) {
        CACHE.with_borrow_mut(|cache| {
            let (sk_pos, min_pos): (Vec<_>, Vec<_>) =
                canonical_minimizers_seq_scalar(seq, hasher, w, cache)
                    .enumerate()
                    .dedup_by(|x, y| x.1 == y.1)
                    .map(|(x, y)| (x as u32, y))
                    .unzip();
            min_pos_vec.extend(min_pos);
            sk_pos_vec.extend(sk_pos);
        })
    }
}
