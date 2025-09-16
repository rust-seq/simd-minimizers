//! Find the (canonical) minimizers of a sequence.
use std::iter::zip;

use crate::canonical;

use super::{
    canonical::canonical_mapper,
    sliding_min::{sliding_lr_min_mapper, sliding_min_mapper, sliding_min_scalar},
};
use itertools::{Either, Itertools};
use packed_seq::{ChunkIt, PaddedIt, Seq};
use seq_hash::SeqHasher;
use wide::u32x8;

/// Minimizer position of a single window.
pub fn one_minimizer<'s>(seq: impl Seq<'s>, hasher: &impl SeqHasher) -> usize {
    hasher
        .hash_kmers_scalar(seq)
        .map(|x| x & 0xffff_0000)
        .position_min()
        .unwrap()
}

// FIMXE: Add one_canonical_minimizer

/// Returns an iterator over the absolute positions of the minimizers of a sequence.
/// Returns one value for each window of size `w+k-1` in the input. Use
/// `Itertools::dedup()` to obtain the distinct positions of the minimizers.
///
/// Prefer `minimizer_simd_it` that internally used SIMD, or `minimizer_par_it` if it works for you.
pub fn minimizers_seq_scalar<'s>(
    seq: impl Seq<'s>,
    hasher: &impl SeqHasher,
    w: usize,
) -> impl ExactSizeIterator<Item = u32> {
    let it = hasher.hash_kmers_scalar(seq);
    sliding_min_scalar::<true>(it, w)
}

/// Split the windows of the sequence into 8 chunks of equal length ~len/8.
/// Then return the positions of the minimizers of each of them in parallel using SIMD,
/// and return the remaining few using the second iterator.
// TODO: Take a hash function as argument.
pub fn minimizers_seq_simd<'s, H: SeqHasher>(
    seq: impl Seq<'s>,
    hasher: &H,
    w: usize,
) -> PaddedIt<impl ChunkIt<u32x8>> {
    let k = hasher.k();
    let kmer_hashes = {
        if H::MAPPER_NEEDS_OUT {
            let delay = (&hasher).delay();
            assert!(delay.0 <= k - 1);
            let padded_it = seq.par_iter_bp_delayed(w + k - 1, delay);
            padded_it
                .map((&hasher).in_out_mapper_simd(seq))
                .map_it(Either::Left)
        } else {
            let padded_it = seq.par_iter_bp(w + k - 1);
            padded_it
                .map(|a| (a, u32x8::splat(0)))
                .map((&hasher).in_out_mapper_simd(seq))
                .map_it(Either::Right)
        }
    };
    let len = kmer_hashes.it.len();
    kmer_hashes
        .map(sliding_min_mapper::<true>(w, k, len))
        .dropping(w - 1)
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TRULY CANONICAL MINIMIZERS BELOW HERE
// The minimizers above can take a canonical hash, but do not correctly break ties.
// Below we fix that.

pub fn canonical_minimizers_seq_scalar<'s>(
    seq: impl Seq<'s>,
    hasher: &impl SeqHasher,
    w: usize,
) -> impl ExactSizeIterator<Item = u32> {
    // TODO: Change to compile-time check on `impl SeqHasher<RC=true>` once supported.
    assert!(hasher.is_canonical());

    let kmer_hashes = hasher.hash_kmers_scalar(seq);
    // FIXME: Instead of cloning the `kmer_hashes` iterator, use a `sliding_min_scalar_mapper` instead.
    let left = sliding_min_scalar::<true>(kmer_hashes.clone(), w);
    let right = sliding_min_scalar::<false>(kmer_hashes, w);
    // indicators whether each window is canonical
    let k = hasher.k();
    let canonical = canonical::canonical_windows_seq_scalar(seq, k + w - 1);
    zip(canonical, zip(left, right)).map(|(canonical, (left, right))| {
        // Select left or right based on canonical mask.
        if canonical {
            left
        } else {
            right
        }
    })
}

/// Use canonical NtHash, and keep both leftmost and rightmost minima.
pub fn canonical_minimizers_seq_simd<'s>(
    seq: impl Seq<'s>,
    hasher: &impl SeqHasher,
    w: usize,
) -> PaddedIt<impl ChunkIt<u32x8>> {
    assert!(hasher.is_canonical());

    let k = hasher.k();
    let l = k + w - 1;
    let mut hash_mapper = hasher.in_out_mapper_simd(seq);
    let (c_delay, mut canonical_mapper) = canonical_mapper(l);

    let padded_it = seq.par_iter_bp_delayed_2(l, hasher.delay(), c_delay);
    let mut sliding_min_mapper = sliding_lr_min_mapper(w, k, padded_it.it.len());

    padded_it
        .map(move |(a, rh, rc)| {
            let hash = hash_mapper((a, rh));
            let canonical = canonical_mapper((a, rc));
            let (lmin, rmin) = sliding_min_mapper(hash);
            unsafe { std::mem::transmute::<_, u32x8>(canonical) }.blend(lmin, rmin)
        })
        .dropping(l - 1)
}
