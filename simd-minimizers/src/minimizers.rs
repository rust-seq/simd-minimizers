//! Find the (canonical) minimizers of a sequence.
use std::iter::zip;

use crate::{
    canonical,
    nthash::{Captures, CharHasher},
};

use super::{
    canonical::canonical_mapper,
    nthash::{nthash_mapper, nthash_seq_scalar},
    sliding_min::{sliding_lr_min_mapper, sliding_min_mapper, sliding_min_scalar},
};
use itertools::Itertools;
use packed_seq::Seq;
use std::simd::u32x16 as u32x8;

/// Returns the minimizer of a window using a naive linear scan.
pub fn minimizer<'s, H: CharHasher>(seq: impl Seq<'s>, k: usize) -> usize {
    nthash_seq_scalar::<false, H>(seq, k)
        .map(|x| x & 0xffff_0000)
        .position_min()
        .unwrap()
}

/// Returns an iterator over the absolute positions of the minimizers of a sequence.
/// Returns one value for each window of size `w+k-1` in the input. Use
/// `Itertools::dedup()` to obtain the distinct positions of the minimizers.
///
/// Prefer `minimizer_simd_it` that internally used SIMD, or `minimizer_par_it` if it works for you.
pub fn minimizers_seq_scalar<'s, H: CharHasher>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    let it = nthash_seq_scalar::<false, H>(seq, k);
    sliding_min_scalar::<true>(it, w)
}

/// Split the windows of the sequence into 8 chunks of equal length ~len/8.
/// Then return the positions of the minimizers of each of them in parallel using SIMD,
/// and return the remaining few using the second iterator.
// TODO: Take a hash function as argument.
pub fn minimizers_seq_simd<'s, SEQ: Seq<'s>, H: CharHasher>(
    seq: SEQ,
    k: usize,
    w: usize,
) -> (
    impl ExactSizeIterator<Item = u32x8> + Captures<&'s ()>,
    usize,
) {
    let l = k + w - 1;

    let (add_remove, padding) = seq.par_iter_bp_delayed(k + w - 1, k - 1);

    let mut nthash = nthash_mapper::<false, SEQ, H>(k, w);
    let mut sliding_min = sliding_min_mapper::<true>(w, k, add_remove.len());

    let mut head = add_remove.map(move |(a, rk)| {
        let nthash = nthash((a, rk));
        sliding_min(nthash)
    });

    head.by_ref().take(l - 1).for_each(drop);
    (head, padding)
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TRULY CANONICAL MINIMIZERS BELOW HERE
// The minimizers above can take a canonical hash, but do not correctly break ties.
// Below we fix that.

pub fn canonical_minimizers_seq_scalar<'s, H: CharHasher>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    // true: canonical
    let kmer_hashes = nthash_seq_scalar::<true, H>(seq, k);
    // true: leftmost
    let left = sliding_min_scalar::<true>(kmer_hashes.clone(), w);
    // false: rightmost
    let right = sliding_min_scalar::<false>(kmer_hashes, w);
    // indicators whether each window is canonical
    let canonical = canonical::canonical_windows_seq_scalar(seq, k, w);
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
pub fn canonical_minimizers_seq_simd<'s, SEQ: Seq<'s>, H: CharHasher>(
    seq: SEQ,
    k: usize,
    w: usize,
) -> (
    impl ExactSizeIterator<Item = u32x8> + Captures<&'s ()>,
    usize,
) {
    let l = k + w - 1;

    // TODO: NtHash takes the return value *before* dropping the given character so has k-1,
    // while canonical first drops the character, so has l without -1.
    let (add_remove, padding) = seq.par_iter_bp_delayed_2(k + w - 1, k - 1, l);

    let mut nthash = nthash_mapper::<true, SEQ, H>(k, w);
    let mut canonical = canonical_mapper(k, w);
    let mut sliding_min = sliding_lr_min_mapper(w, k, add_remove.len());

    let mut head = add_remove.map(move |(a, rk, rl)| {
        let nthash = nthash((a, rk));
        let canonical = canonical((a, rl));
        let (lmin, rmin) = sliding_min(nthash);
        canonical.select(lmin, rmin)
    });

    head.by_ref().take(l - 1).for_each(drop);
    (head, padding)
}
