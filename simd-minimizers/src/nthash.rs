//! A fast implementation of ntHash.
//! Input:
//! - A 2-bit packed DNA sequence, only ACGT.
//! - It's length in bp.
//! - The k-mer length k.
//!
//! Output:
//! - An iterator over all 32-hashes of all k-mers in the sequence.

// TODO: Write about 2bit encoded packed represenatation.

use crate::Captures;

use super::intrinsics;
use super::linearize;
use packed_seq::{Seq, S};

// TODO: Update to guarantee unique hash values for k<=16?
const HASHES_F: [u32; 4] = [
    0x3c8b_fbb3_95c6_0474u64 as u32,
    0x3193_c185_62a0_2b4cu64 as u32,
    0x2032_3ed0_8257_2324u64 as u32,
    0x2955_49f5_4be2_4456u64 as u32,
];
const fn cmpl(base: u32) -> u32 {
    assert!(base < 4);
    base ^ 2
}
/// Complement hashes.
const HASHES_C: [u32; 4] = [
    HASHES_F[cmpl(0) as usize],
    HASHES_F[cmpl(1) as usize],
    HASHES_F[cmpl(2) as usize],
    HASHES_F[cmpl(3) as usize],
];

/// Naively compute the 32-bit NT hash of a single k-mer.
/// When `RC` is false, compute a forward hash.
/// When `RC` is true, compute a canonical hash.
pub fn nthash32_kmer_naive<'s, const RC: bool>(seq: impl Seq<'s>) -> u32 {
    let k = seq.len();
    let mut hfw: u32 = 0;
    let mut hrc: u32 = 0;
    seq.iter_bp().for_each(|a| {
        hfw = hfw.rotate_left(1) ^ HASHES_F[a as usize];
        if RC {
            hrc = hrc.rotate_right(1) ^ HASHES_C[a as usize];
        }
    });
    hfw.wrapping_add(hrc.rotate_left(k as u32 - 1))
}

/// Returns an iterator over the 32-bit NT hashes of all k-mers in the sequence.
/// Set `RC` to true for canonical hashes.
///
/// Prefer `nthash32_simd_it` which fills an internal buffer using the parallel SIMD version.
pub fn nthash32_scalar_it<'s, const RC: bool>(
    seq: impl Seq<'s>,
    k: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> + Clone {
    assert!(k > 0);
    let mut hfw: u32 = 0;
    let mut hrc: u32 = 0;
    let mut add = seq.iter_bp();
    let remove = seq.iter_bp();
    add.by_ref().take(k - 1).for_each(|a| {
        hfw = hfw.rotate_left(1) ^ HASHES_F[a as usize];
        if RC {
            hrc = hrc.rotate_right(1) ^ HASHES_C[a as usize].rotate_left(k as u32 - 1);
        }
    });
    add.zip(remove).map(move |(a, r)| {
        let hfw_out = hfw.rotate_left(1) ^ HASHES_F[a as usize];
        hfw = hfw_out ^ HASHES_F[r as usize].rotate_left(k as u32 - 1);
        if RC {
            let hrc_out = hrc.rotate_right(1) ^ HASHES_C[a as usize].rotate_left(k as u32 - 1);
            hrc = hrc_out ^ HASHES_C[r as usize];
            hfw_out.wrapping_add(hrc_out)
        } else {
            hfw_out
        }
    })
}

/// Returns an iterator over the 32-bit NT hashes of all k-mers in the sequence.
/// Set `RC` to true for canonical hashes.
///
/// This splits the sequence into chunks of size 2^13 bp.
/// Hashes for each chunk are eagerly computed using 8 parallel streams using SIMD using `nthash32_par_it`.
/// Then returns a linear iterator over the buffer.
/// Once the buffer runs out, the next chunk is computed.
pub fn nthash32_simd_it<'s, const RC: bool>(
    seq: impl Seq<'s>,
    k: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    linearize::linearize(seq, k, move |seq| nthash32_par_it::<RC>(seq, k, 1))
}

/// Split the kmers of the sequence into 8 chunks of equal length ~len/8.
/// Then return the hashes of each of them in parallel using SIMD,
/// and return the remaining few using the second iterator.
/// The tail end has up to 31*8 = 248 elements.
// TODO: SMALL_K + reusing gathers?
pub fn nthash32_par_it<'s, const RC: bool>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
) -> (
    impl ExactSizeIterator<Item = S> + Captures<&'s ()> + Clone,
    impl ExactSizeIterator<Item = u32> + Captures<&'s ()> + Clone,
) {
    let (add_remove, tail) = seq.par_iter_bp_delayed(k + w - 1, k - 1);

    let mut it = add_remove.map(nthash_mapper::<RC>(k, w));
    it.by_ref().take(k - 1).for_each(drop);

    let tail = nthash32_scalar_it::<RC>(tail, k);

    (it, tail)
}

/// NOTE: First k-1 values are bogus.
pub fn nthash_mapper<const RC: bool>(k: usize, w: usize) -> impl FnMut((S, S)) -> S + Clone {
    assert!(k > 0);
    assert!(w > 0);
    // Each 128-bit half has a copy of the 4 32-bit hashes.
    let table_fw: S = [0, 1, 2, 3, 0, 1, 2, 3]
        .map(|c| HASHES_F[c as usize])
        .into();
    let table_fw_rot: S = [0, 1, 2, 3, 0, 1, 2, 3]
        .map(|c| HASHES_F[c as usize].rotate_left(k as u32 - 1))
        .into();
    // TODO: Reuse tables above instead of making new ones?
    let table_rc: S = [0, 1, 2, 3, 0, 1, 2, 3]
        .map(|c| HASHES_C[c as usize])
        .into();
    let table_rc_rot: S = [0, 1, 2, 3, 0, 1, 2, 3]
        .map(|c| HASHES_C[c as usize].rotate_left(k as u32 - 1))
        .into();

    let mut fw = 0u32;
    let mut rc = 0u32;
    for _ in 0..k - 1 {
        fw = fw.rotate_left(1) ^ HASHES_F[0];
        rc = rc.rotate_right(1) ^ HASHES_C[0].rotate_left(k as u32 - 1);
    }

    let mut h_fw = S::splat(fw);
    let mut h_rc = S::splat(rc);

    move |(a, r)| {
        let hfw_out = ((h_fw << 1) | (h_fw >> 31)) ^ intrinsics::table_lookup(table_fw, a);
        h_fw = hfw_out ^ intrinsics::table_lookup(table_fw_rot, r);
        if RC {
            let hrc_out = ((h_rc >> 1) | (h_rc << 31)) ^ intrinsics::table_lookup(table_rc_rot, a);
            h_rc = hrc_out ^ intrinsics::table_lookup(table_rc, r);
            // Wrapping SIMD add
            hfw_out + hrc_out
        } else {
            hfw_out
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;
    use packed_seq::{AsciiSeq, AsciiSeqVec, PackedSeqVec, SeqVec, L};
    use std::{iter::once, sync::LazyLock};

    static ASCII_SEQ: LazyLock<AsciiSeqVec> = LazyLock::new(|| AsciiSeqVec::random(1024));
    static PACKED_SEQ: LazyLock<PackedSeqVec> =
        LazyLock::new(|| PackedSeqVec::from_ascii(&ASCII_SEQ.seq));

    #[test]
    fn scalar_ascii() {
        let seq = &*ASCII_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let single = seq
                    .0
                    .windows(k)
                    .map(|seq| nthash32_kmer_naive::<false>(AsciiSeq::new(seq, k)))
                    .collect::<Vec<_>>();
                let scalar = nthash32_scalar_it::<false>(seq, k).collect::<Vec<_>>();
                assert_eq!(single, scalar, "k={}, len={}", k, len);
            }
        }
    }

    #[test]
    fn scalar_packed() {
        let ascii_seq = &*ASCII_SEQ;
        let seq = &*PACKED_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let ascii_seq = ascii_seq.slice(0..len);
                let seq = seq.slice(0..len);

                let scalar_ascii = nthash32_scalar_it::<false>(ascii_seq, k).collect::<Vec<_>>();
                let scalar = nthash32_scalar_it::<false>(seq, k).collect::<Vec<_>>();
                assert_eq!(scalar_ascii, scalar, "k={}, len={}", k, len);
            }
        }
    }

    #[test]
    fn par_it_packed() {
        let seq = &*PACKED_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let scalar = nthash32_scalar_it::<false>(seq, k).collect::<Vec<_>>();
                let (par_head, tail) = nthash32_par_it::<false>(seq, k, 1);
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
    fn parallel_packed() {
        let seq = &*PACKED_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let scalar = nthash32_scalar_it::<false>(seq, k).collect::<Vec<_>>();
                let parallel = nthash32_simd_it::<false>(seq, k).collect::<Vec<_>>();
                assert_eq!(scalar, parallel, "k={}, len={}", k, len);
            }
        }
    }

    // ignore test
    #[ignore = "par_iter_bp_delayed is not implemented for ASCII"]
    #[test]
    fn parallel_byte() {
        let seq = &*ASCII_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let scalar = nthash32_scalar_it::<false>(seq, k).collect::<Vec<_>>();
                let parallel = nthash32_simd_it::<false>(seq, k).collect::<Vec<_>>();
                assert_eq!(scalar, parallel, "k={}, len={}", k, len);
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
                let scalar = nthash32_scalar_it::<false>(seq, k).collect::<Vec<_>>();
                let (par_head, tail) = nthash32_par_it::<false>(seq, k, 1);
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
    fn canonical_is_revcomp() {
        let seq = &*ASCII_SEQ;
        let seq_rc = AsciiSeqVec::from_vec(
            seq.seq
                .iter()
                .rev()
                .map(|c| packed_seq::complement_char(*c))
                .collect_vec(),
        );
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let seq_rc = seq_rc.slice(seq_rc.len() - len..seq_rc.len());
                let scalar = nthash32_scalar_it::<true>(seq, k).collect::<Vec<_>>();
                let scalar_rc = nthash32_scalar_it::<true>(seq_rc, k).collect::<Vec<_>>();
                let scalar_rc_rc = scalar_rc.iter().rev().copied().collect_vec();
                assert_eq!(
                    scalar_rc_rc,
                    scalar,
                    "k={}, len={} {:032b} {:032b}",
                    k,
                    len,
                    scalar.first().unwrap_or(&0),
                    scalar_rc_rc.first().unwrap_or(&0)
                );
            }
        }
    }

    #[test]
    fn scalar_byte_canonical() {
        let seq = &*ASCII_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let single = seq
                    .0
                    .windows(k)
                    .map(|seq| nthash32_kmer_naive::<true>(AsciiSeq::new(seq, k)))
                    .collect::<Vec<_>>();
                let scalar = nthash32_scalar_it::<true>(seq, k).collect::<Vec<_>>();
                assert_eq!(single, scalar, "k={}, len={}", k, len);
            }
        }
    }

    #[test]
    fn parallel_iter_packed_canonical() {
        let seq = &*PACKED_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let scalar = nthash32_scalar_it::<true>(seq, k).collect::<Vec<_>>();
                let (par_head, tail) = nthash32_par_it::<true>(seq, k, 1);
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
                let scalar = nthash32_scalar_it::<true>(seq, k).collect::<Vec<_>>();
                let simd = nthash32_simd_it::<true>(seq, k).collect::<Vec<_>>();
                assert_eq!(scalar, simd, "k={}, len={}", k, len);
            }
        }
    }
}
