#![allow(unused)]
//! A fast implementation of 'anti lexicographic' hashing:
//! A kmer's hash is simply its bit representation, but with the first character inverted.
//! When k > 16, only the last 16 characters are used.
//!
//! TODO: Figure out how this works with reverse complements.

use crate::Captures;

use super::linearize;
use packed_seq::{Seq, S};

pub fn alex_kmer_naive<'s>(seq: impl Seq<'s>) -> u32 {
    let k = seq.len();
    let mut hfw: u32 = 0;
    let anti = if k <= 16 { 3 << (2 * k - 2) } else { 3 << 30 };
    seq.iter_bp().for_each(|a| {
        hfw = (hfw << 2) ^ a as u32;
    });
    hfw ^ anti
}

pub fn alex_scalar_it<'s>(
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

pub fn alex_simd_it<'s>(
    seq: impl Seq<'s>,
    k: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    linearize::linearize(seq, k, move |seq| alex_par_it(seq, k, 1))
}

pub fn alex_par_it<'s>(
    seq: impl Seq<'s>,
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

    let tail = alex_scalar_it(tail, k);

    (it, tail)
}

/// NOTE: First k-1 values are bogus.
pub fn alex_mapper(k: usize, w: usize) -> impl FnMut(S) -> S + Clone {
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
    use super::*;
    use packed_seq::{AsciiSeq, AsciiSeqVec, PackedSeqVec, SeqVec, L};
    use std::{cell::LazyCell, iter::once};

    const ASCII_SEQ: LazyCell<AsciiSeqVec> = LazyCell::new(|| AsciiSeqVec::random(1024));
    const PACKED_SEQ: LazyCell<PackedSeqVec> = LazyCell::new(|| PackedSeqVec::random(1024));

    #[test]
    fn scalar_byte() {
        let seq = &*ASCII_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let single = seq
                    .0
                    .windows(k)
                    .map(|seq| alex_kmer_naive(AsciiSeq::new(seq, k)))
                    .collect::<Vec<_>>();
                let scalar = alex_scalar_it(seq, k).collect::<Vec<_>>();
                assert_eq!(single, scalar, "k={}, len={}", k, len);
            }
        }
    }

    #[test]
    fn parallel_byte() {
        let seq = &*ASCII_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let scalar = alex_scalar_it(seq, k).collect::<Vec<_>>();
                let parallel = alex_simd_it(seq, k).collect::<Vec<_>>();
                assert_eq!(scalar, parallel, "k={}, len={}", k, len);
            }
        }
    }

    #[test]
    fn parallel_packed() {
        let seq = &*PACKED_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let scalar = alex_scalar_it(seq, k).collect::<Vec<_>>();
                let parallel = alex_simd_it(seq, k).collect::<Vec<_>>();
                assert_eq!(scalar, parallel, "k={}, len={}", k, len);
            }
        }
    }

    #[test]
    fn parallel_iter_byte() {
        let seq = &*ASCII_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let scalar = alex_scalar_it(seq, k).collect::<Vec<_>>();
                let (par_head, tail) = alex_par_it(seq, k, 1);
                let par_head = par_head.collect::<Vec<_>>();
                let parallel_iter = (0..L)
                    .flat_map(|l| par_head.iter().map(move |x| x.as_array_ref()[l]))
                    .chain(tail)
                    .collect::<Vec<_>>();

                assert_eq!(scalar, parallel_iter, "k={}, len={}", k, len);
            }
        }
    }

    #[test]
    fn parallel_iter_packed() {
        let seq = &*PACKED_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let scalar = alex_scalar_it(seq, k).collect::<Vec<_>>();
                let (par_head, tail) = alex_par_it(seq, k, 1);
                let par_head = par_head.collect::<Vec<_>>();
                let parallel_iter = (0..L)
                    .flat_map(|l| par_head.iter().map(move |x| x.as_array_ref()[l]))
                    .chain(tail)
                    .collect::<Vec<_>>();
                assert_eq!(scalar, parallel_iter, "k={}, len={}", k, len);
            }
        }
    }

    #[test]
    fn linearized() {
        let seq = &*PACKED_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain([973, 1024]) {
                let seq = seq.slice(0..len);
                let scalar = alex_scalar_it(seq, k).collect::<Vec<_>>();
                let simd = alex_simd_it(seq, k).collect::<Vec<_>>();
                assert_eq!(scalar, simd, "k={}, len={}", k, len);
            }
        }
    }
}
