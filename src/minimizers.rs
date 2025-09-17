//! Find the (canonical) minimizers of a sequence.
use std::iter::zip;

use crate::{
    canonical,
    sliding_min::{sliding_lr_min_mapper_scalar, sliding_min_mapper_scalar},
};

use super::{
    canonical::canonical_mapper_simd,
    sliding_min::{sliding_lr_min_mapper_simd, sliding_min_mapper_simd},
};
use itertools::{izip, Itertools};
use packed_seq::{Advance, ChunkIt, Delay, PaddedIt, Seq};
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
#[inline(always)]
pub fn minimizers_seq_scalar<'s>(
    seq: impl Seq<'s>,
    hasher: &impl SeqHasher,
    w: usize,
) -> impl ExactSizeIterator<Item = u32> {
    let kmer_hashes = hasher.hash_kmers_scalar(seq);
    let len = kmer_hashes.len();
    kmer_hashes
        .map(sliding_min_mapper_scalar::<true>(w, len))
        .advance(w - 1)
}

/// Split the windows of the sequence into 8 chunks of equal length ~len/8.
/// Then return the positions of the minimizers of each of them in parallel using SIMD,
/// and return the remaining few using the second iterator.
// TODO: Take a hash function as argument.
#[inline(always)]
pub fn minimizers_seq_simd<'s>(
    seq: impl Seq<'s>,
    hasher: &impl SeqHasher,
    w: usize,
) -> PaddedIt<impl ChunkIt<u32x8>> {
    let kmer_hashes = hasher.hash_kmers_simd(seq, w);
    let len = kmer_hashes.it.len();
    kmer_hashes
        .map(sliding_min_mapper_simd::<true>(w, len))
        .advance(w - 1)
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TRULY CANONICAL MINIMIZERS BELOW HERE
// The minimizers above can take a canonical hash, but do not correctly break ties.
// Below we fix that.

#[inline(always)]
pub fn canonical_minimizers_seq_scalar<'s>(
    seq: impl Seq<'s>,
    hasher: &impl SeqHasher,
    w: usize,
) -> impl ExactSizeIterator<Item = u32> {
    // TODO: Change to compile-time check on `impl SeqHasher<RC=true>` once supported.
    assert!(hasher.is_canonical());

    let k = hasher.k();
    let delay1 = hasher.delay().0;
    let mut hash_mapper = hasher.in_out_mapper_scalar(seq);
    // TODO: Merge into a single mapper?
    let mut sliding_min_mapper = sliding_lr_min_mapper_scalar(w, seq.len());
    let (Delay(delay2), mut canonical_mapper) = canonical::canonical_mapper_scalar(k + w - 1);

    assert!(delay1 <= k - 1);
    assert!(k - 1 <= delay2);
    assert!(delay2 == k + w - 2);

    let mut a = seq.iter_bp();
    let mut rh = seq.iter_bp();
    let rc = seq.iter_bp();

    for a in a.by_ref().take(delay1) {
        hash_mapper((a, 0));
        canonical_mapper((a, 0));
    }

    for (a, rh) in zip(a.by_ref(), rh.by_ref()).take((k - 1) - delay1) {
        hash_mapper((a, rh));
        canonical_mapper((a, 0));
    }

    for (a, rh) in zip(a.by_ref(), rh.by_ref()).take(delay2 - (k - 1)) {
        let hash = hash_mapper((a, rh));
        canonical_mapper((a, 0));
        sliding_min_mapper(hash);
    }

    izip!(a, rh, rc).map(
        #[inline(always)]
        move |(a, rh, rc)| {
            let hash = hash_mapper((a, rh));
            let canonical = canonical_mapper((a, rc));
            let (left, right) = sliding_min_mapper(hash);
            // Assigning to x ensures we get a cmov here.
            let x = if canonical { left } else { right };
            x
        },
    )
}

/// Use canonical NtHash, and keep both leftmost and rightmost minima.
#[inline(always)]
pub fn canonical_minimizers_seq_simd<'s>(
    seq: impl Seq<'s>,
    hasher: &impl SeqHasher,
    w: usize,
) -> PaddedIt<impl ChunkIt<u32x8>> {
    assert!(hasher.is_canonical());

    let k = hasher.k();
    let l = k + w - 1;
    let mut hash_mapper = hasher.in_out_mapper_simd(seq);
    let (c_delay, mut canonical_mapper) = canonical_mapper_simd(l);

    let mut padded_it = seq.par_iter_bp_delayed_2(l, hasher.delay(), c_delay);

    // Process first k-1 characters separately, to initialize hash values.
    {
        let hash_mapper = &mut hash_mapper;
        let canonical_mapper = &mut canonical_mapper;
        padded_it
            .it
            .by_ref()
            .take(k - 1)
            .for_each(move |(a, rh, rc)| {
                hash_mapper((a, rh));
                canonical_mapper((a, rc));
            });
    }
    let mut sliding_min_mapper = sliding_lr_min_mapper_simd(w, padded_it.it.len());

    padded_it
        .map(move |(a, rh, rc)| {
            let hash = hash_mapper((a, rh));
            let canonical = canonical_mapper((a, rc));
            let (lmin, rmin) = sliding_min_mapper(hash);
            unsafe { std::mem::transmute::<_, u32x8>(canonical) }.blend(lmin, rmin)
        })
        .advance(w - 1)
}
