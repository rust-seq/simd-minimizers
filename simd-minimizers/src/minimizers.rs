use std::iter::zip;

use crate::{canonical, Captures};

use super::{
    canonical::canonical_mapper,
    nthash::{hash_mapper, hash_seq_scalar},
    sliding_min::{sliding_lr_min_mapper, sliding_min_mapper, sliding_min_scalar},
};
use itertools::Itertools;
use packed_seq::{Seq, S};

/// Returns the minimizer of a window using a naive linear scan.
pub fn minimizer<'s>(seq: impl Seq<'s>, k: usize) -> usize {
    hash_seq_scalar::<false>(seq, k)
        .map(|x| x & 0xffff_0000)
        .position_min()
        .unwrap()
}

/// Returns an iterator over the absolute positions of the minimizers of a sequence.
/// Returns one value for each window of size `w+k-1` in the input. Use
/// `Itertools::dedup()` to obtain the distinct positions of the minimizers.
///
/// Prefer `minimizer_simd_it` that internally used SIMD, or `minimizer_par_it` if it works for you.
pub fn minimizers_seq_scalar<'s>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    let it = hash_seq_scalar::<false>(seq, k);
    sliding_min_scalar::<true>(it, w)
}

/// Split the windows of the sequence into 8 chunks of equal length ~len/8.
/// Then return the positions of the minimizers of each of them in parallel using SIMD,
/// and return the remaining few using the second iterator.
// TODO: Take a hash function as argument.
pub fn minimizers_seq_simd<'s>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
) -> (
    impl ExactSizeIterator<Item = S> + Captures<&'s ()>,
    impl ExactSizeIterator<Item = u32> + Captures<&'s ()>,
) {
    let l = k + w - 1;

    let (add_remove, tail) = seq.par_iter_bp_delayed(k + w - 1, k - 1);

    let mut nthash = hash_mapper::<false>(k, w);
    // let mut alex = alex::alex_mapper(k, w);
    let mut sliding_min = sliding_min_mapper::<true>(w, k, add_remove.len());

    let mut head = add_remove.map(move |(a, rk)| {
        let nthash = nthash((a, rk));
        sliding_min(nthash)
        // let alex = alex(a);
        // sliding_min(alex)
    });

    head.by_ref().take(l - 1).for_each(drop);
    let head_len = head.len();

    let tail = minimizers_seq_scalar(tail, k, w).map(move |p| p + 8 * head_len as u32);
    (head, tail)
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TRULY CANONICAL MINIMIZERS BELOW HERE
// The minimizers above can take a canonical hash, but do not correctly break ties.
// Below we fix that.

pub fn canonical_minimizers_seq_scalar<'s>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    // true: canonical
    let kmer_hashes = hash_seq_scalar::<true>(seq, k);
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
pub fn canonical_minimizers_seq_simd<'s>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
) -> (
    impl ExactSizeIterator<Item = S> + Captures<&'s ()>,
    impl ExactSizeIterator<Item = u32> + Captures<&'s ()>,
) {
    let l = k + w - 1;

    // FIXME: NtHash takes the return value *before* dropping the given character,
    // while canonical first drops the character.
    let (add_remove, tail) = seq.par_iter_bp_delayed_2(k + w - 1, k - 1, l);

    let mut nthash = hash_mapper::<true>(k, w);
    let mut canonical = canonical_mapper(k, w);
    let mut sliding_min = sliding_lr_min_mapper(w, add_remove.len());

    let mut head = add_remove.map(move |(a, rk, rl)| {
        let nthash = nthash((a, rk));
        let canonical = canonical((a, rl));
        let (lmin, rmin) = sliding_min(nthash);
        unsafe { std::mem::transmute::<_, S>(canonical).blend(lmin, rmin) }
    });

    head.by_ref().take(l - 1).for_each(drop);

    let tail = canonical_minimizers_seq_scalar(tail, k, w);
    (head, tail)
}

#[cfg(test)]
mod test {
    use crate::{collect, minimizers_collect_and_dedup};

    use super::*;
    use packed_seq::{AsciiSeq, AsciiSeqVec, PackedSeqVec, SeqVec};
    use std::{cell::LazyCell, iter::once};

    const ASCII_SEQ: LazyCell<AsciiSeqVec> = LazyCell::new(|| AsciiSeqVec::random(1024 * 1024));
    const PACKED_SEQ: LazyCell<PackedSeqVec> = LazyCell::new(|| PackedSeqVec::random(1024 * 1024));

    #[test]
    fn scalar_ascii() {
        let seq = &*ASCII_SEQ;
        for k in [1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65] {
            for w in [1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65] {
                for len in (0..100).chain(once(1024 * 32)) {
                    let seq = seq.slice(0..len);
                    let single = seq.0[0..len]
                        .windows(w + k - 1)
                        .enumerate()
                        .map(|(pos, seq)| {
                            (pos + minimizer(AsciiSeq::new(seq, w + k - 1), k)) as u32
                        })
                        .collect::<Vec<_>>();
                    let scalar = minimizers_seq_scalar(seq, k, w).collect::<Vec<_>>();
                    assert_eq!(single, scalar, "k={k}, w={w}, len={len}");
                }
            }
        }
    }

    #[ignore = "delayed iteration not yet implemented for ASCII"]
    #[test]
    fn simd_ascii() {
        let seq = &*ASCII_SEQ;
        for k in [1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65] {
            for w in [1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65] {
                for len in (0..100).chain(once(1024 * 128)) {
                    let seq = seq.slice(0..len);
                    let scalar = minimizers_seq_scalar(seq, k, w).collect::<Vec<_>>();
                    let (par_head, tail) = minimizers_seq_simd(seq, k, w);
                    let par_head = par_head.collect::<Vec<_>>();
                    let parallel_iter = (0..8)
                        .flat_map(|l| par_head.iter().map(move |x| x.as_array_ref()[l]))
                        .chain(tail)
                        .collect::<Vec<_>>();

                    assert_eq!(scalar, parallel_iter, "k={k}, w={w}, len={len}");
                }
            }
        }
    }

    #[test]
    fn simd_packed() {
        let seq = &*PACKED_SEQ;
        for k in [1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65] {
            for w in [1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65] {
                for len in (0..100).chain(once(1024 * 128)) {
                    let seq = seq.slice(0..len);
                    let scalar = minimizers_seq_scalar(seq, k, w).collect::<Vec<_>>();
                    let (par_head, tail) = minimizers_seq_simd(seq, k, w);
                    let par_head = par_head.collect::<Vec<_>>();
                    let parallel_iter = (0..8)
                        .flat_map(|l| par_head.iter().map(move |x| x.as_array_ref()[l]))
                        .chain(tail)
                        .collect::<Vec<_>>();
                    if w == 1 && parallel_iter.len() > 0 {
                        assert_eq!(parallel_iter[0], 0);
                    }
                    assert_eq!(scalar, parallel_iter, "k={k}, w={w}, len={len}");
                }
            }
        }
    }

    #[test]
    fn linearized() {
        let seq = &*PACKED_SEQ;
        for k in [1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65] {
            for w in [1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65] {
                for len in (0..100).chain(once(1024 * 128 + 765)) {
                    let seq = seq.slice(0..len);
                    let scalar = minimizers_seq_scalar(seq, k, w).collect::<Vec<_>>();
                    let simd = collect(minimizers_seq_simd(seq, k, w));
                    assert_eq!(
                        scalar,
                        simd,
                        "k={}, len={} len left {} len right {}",
                        k,
                        len,
                        scalar.len(),
                        simd.len()
                    );
                }
            }
        }
    }

    #[test]
    fn test_sliding_min() {
        let n = 1000;
        let seq = PackedSeqVec::random(n);
        let mut poss = vec![];
        for k in 1..10 {
            for w in 1..10 {
                poss.clear();
                minimizers_collect_and_dedup::<false>(seq.as_slice(), k, w, &mut poss);
                for &x in &poss {
                    assert!(
                        x <= (n - k) as u32,
                        "Index {x} is not in range for n={n}, k={k}, w={w}. Should be in 0..{}\n{poss:?}",
                        n - k
                    );
                }
            }
        }
    }
}
