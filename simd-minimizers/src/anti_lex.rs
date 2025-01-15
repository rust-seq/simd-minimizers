//! A fast implementation of 'anti lexicographic' hashing:
//! A kmer's hash is simply its bit representation, but with the first character inverted.
//! When k > 16, only the last 16 characters are used.
//!
//! TODO: Figure out how this works with reverse complements.

use crate::{linearize, Captures};

use packed_seq::{Seq, S};

/// Naively compute the 32-bit anti-lex hash of a single k-mer.
pub fn anti_lex_kmer<'s>(seq: impl Seq<'s>) -> u32 {
    let k = seq.len();
    let mut hfw: u32 = 0;
    let anti = if k <= 16 { 3 << (2 * k - 2) } else { 3 << 30 };
    seq.iter_bp().for_each(|a| {
        hfw = (hfw << 2) ^ a as u32;
    });
    hfw ^ anti
}

/// Returns a scalar iterator over the 32-bit anti-lex hashes of all k-mers in the sequence.
/// Prefer `anti_lex_seq_simd`.
pub fn anti_lex_seq_scalar<'s>(
    seq: impl Seq<'s>,
    k: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    assert!(k > 0);
    let mut hfw: u32 = 0;
    let mut add = seq.iter_bp();
    let mask = if k < 16 { (1 << (2 * k)) - 1 } else { u32::MAX };
    let anti = if k <= 16 { 3 << (2 * k - 2) } else { 3 << 30 };
    add.by_ref().take(k - 1).for_each(|a| {
        hfw = (hfw << 2) ^ (a as u32);
    });
    add.map(move |a| {
        hfw = ((hfw << 2) ^ (a as u32)) & mask;
        hfw ^ anti
    })
}

/// Returns a simd-iterator over the 8 chunks 32-bit anti-lex hashes of all k-mers in the sequence.
/// The tail is returned separately.
/// Returned chunks overlap by w-1 hashes. Set w=1 for non-overlapping chunks.
pub fn anti_lex_seq_simd<'s>(
    seq: PackedSeq<'s>,
    k: usize,
    w: usize,
) -> (
    impl ExactSizeIterator<Item = S> + Captures<&'s ()>,
    impl ExactSizeIterator<Item = u32> + Captures<&'s ()>,
) {
    assert!(k > 0);
    assert!(w > 0);

    let mut h_fw = S::splat(0);
    let (mut add, tail) = seq.par_iter_bp(k + w - 1);

    let mask = S::splat(if k < 16 { (1 << (2 * k)) - 1 } else { u32::MAX });
    let anti = S::splat(if k <= 16 { 3 << (2 * k - 2) } else { 3 << 30 });

    add.by_ref().take(k - 1).for_each(|a| {
        h_fw = (h_fw << 2) ^ a;
    });

    let it = add.map(move |a| {
        h_fw = ((h_fw << 2) ^ a) & mask;
        h_fw ^ anti
    });

    let tail = anti_lex_seq_scalar(tail, k);

    (it, tail)
}

/// A function that 'eats' added and removed bases, and returns the updated hash.
/// The distance between them must be k-1, and the first k-1 removed bases must be 0.
/// The first k-1 returned values will be useless.
pub fn anti_lex_mapper(k: usize, w: usize) -> impl FnMut(S) -> S + Clone {
    assert!(k > 0);
    assert!(w > 0);

    let mask = S::splat(if k < 16 { (1 << (2 * k)) - 1 } else { u32::MAX });
    let anti = S::splat(if k <= 16 { 3 << (2 * k - 2) } else { 3 << 30 });

    let mut h_fw = S::splat(0);

    move |a| {
        h_fw = ((h_fw << 2) ^ a) & mask;
        h_fw ^ anti
    }
}

#[cfg(test)]
mod test {
    use crate::collect;

    use super::*;
    use itertools::Itertools;
    use packed_seq::{AsciiSeq, AsciiSeqVec, PackedSeqVec, SeqVec};
    use std::{iter::once, sync::LazyLock};

    static ASCII_SEQ: LazyLock<AsciiSeqVec> = LazyLock::new(|| AsciiSeqVec::random(1024));
    static PACKED_SEQ: LazyLock<PackedSeqVec> =
        LazyLock::new(|| PackedSeqVec::from_ascii(&ASCII_SEQ.seq));

    #[test]
    fn test_nthash() {
        let ascii_seq = &*ASCII_SEQ;
        let packed_seq = &*PACKED_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let ascii_seq = ascii_seq.slice(0..len);
                let packed_seq = packed_seq.slice(0..len);
                // naive
                let naive = ascii_seq
                    .0
                    .windows(k)
                    .map(|seq| anti_lex_kmer(AsciiSeq(seq)))
                    .collect::<Vec<_>>();
                // scalar ascii
                let scalar_ascii = anti_lex_seq_scalar(ascii_seq, k).collect::<Vec<_>>();
                // scalar packed
                let scalar_packed = anti_lex_seq_scalar(packed_seq, k).collect::<Vec<_>>();
                // simd packed
                let simd_packed = collect(anti_lex_seq_simd(packed_seq, k, 1));
                assert_eq!(scalar_ascii, naive, "k={}, len={}", k, len);
                assert_eq!(scalar_packed, naive, "k={}, len={}", k, len);
                assert_eq!(simd_packed, naive, "k={}, len={}", k, len);
            }
        }
    }
}
