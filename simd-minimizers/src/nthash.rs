//! NtHash the kmers in a sequence.
use super::intrinsics;
use packed_seq::{complement_base, PackedSeq};
use packed_seq::{Seq, S};

pub trait Captures<U> {}
impl<T: ?Sized, U> Captures<U> for T {}

/// Original ntHash seed values.
// TODO: Update to guarantee unique hash values for k<=16?
const HASHES_F: [u32; 4] = [
    0x3c8b_fbb3_95c6_0474u64 as u32,
    0x3193_c185_62a0_2b4cu64 as u32,
    0x2032_3ed0_8257_2324u64 as u32,
    0x2955_49f5_4be2_4456u64 as u32,
];
/// Hashes of complement bases.
const HASHES_C: [u32; 4] = [
    HASHES_F[complement_base(0) as usize],
    HASHES_F[complement_base(1) as usize],
    HASHES_F[complement_base(2) as usize],
    HASHES_F[complement_base(3) as usize],
];

/// Naively compute the 32-bit NT hash of a single k-mer.
/// When `RC` is false, compute a forward hash.
/// When `RC` is true, compute a canonical hash.
/// TODO: Investigate if we can use CLMUL instruction for speedup.
pub fn hash_kmer<'s, const RC: bool>(seq: impl Seq<'s>) -> u32 {
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

/// Returns a scalar iterator over the 32-bit NT hashes of all k-mers in the sequence.
/// Prefer `hash_seq_simd`.
///
/// Set `RC` to true for canonical ntHash.
pub fn hash_seq_scalar<'s, const RC: bool>(
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

/// Returns a simd-iterator over the 8 chunks 32-bit ntHashes of all k-mers in the sequence.
/// The tail is returned separately.
/// Returned chunks overlap by w-1 hashes. Set w=1 for non-overlapping chunks.
///
/// Set `RC` to true for canonical ntHash.
pub fn hash_seq_simd<'s, const RC: bool>(
    seq: PackedSeq<'s>,
    k: usize,
    w: usize,
) -> (
    impl ExactSizeIterator<Item = S> + Captures<&'s ()> + Clone,
    impl ExactSizeIterator<Item = u32> + Captures<&'s ()> + Clone,
) {
    let (add_remove, tail) = seq.par_iter_bp_delayed(k + w - 1, k - 1);

    let mut it = add_remove.map(hash_mapper::<RC>(k, w));
    it.by_ref().take(k - 1).for_each(drop);

    let tail = hash_seq_scalar::<RC>(tail, k);

    (it, tail)
}

/// A function that 'eats' added and removed bases, and returns the updated hash.
/// The distance between them must be k-1, and the first k-1 removed bases must be 0.
/// The first k-1 returned values will be useless.
///
/// Set `RC` to true for canonical ntHash.
pub fn hash_mapper<const RC: bool>(k: usize, w: usize) -> impl FnMut((S, S)) -> S + Clone {
    assert!(k > 0);
    assert!(w > 0);
    // Each 128-bit half has a copy of the 4 32-bit hashes.
    let table_fw: S = [0, 1, 2, 3, 0, 1, 2, 3]
        .map(|c| HASHES_F[c as usize])
        .into();
    let table_fw_rot: S = [0, 1, 2, 3, 0, 1, 2, 3]
        .map(|c| HASHES_F[c as usize].rotate_left(k as u32 - 1))
        .into();
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
    use crate::collect;

    use super::*;
    use itertools::Itertools;
    use packed_seq::{AsciiSeq, AsciiSeqVec, PackedSeqVec, SeqVec};
    use std::{iter::once, sync::LazyLock};

    static ASCII_SEQ: LazyLock<AsciiSeqVec> = LazyLock::new(|| AsciiSeqVec::random(1024));
    static PACKED_SEQ: LazyLock<PackedSeqVec> =
        LazyLock::new(|| PackedSeqVec::from_ascii(&ASCII_SEQ.seq));

    fn test_nthash<const RC: bool>() {
        let ascii_seq = &*ASCII_SEQ;
        let packed_seq = &*PACKED_SEQ;
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let ascii_seq = ascii_seq.slice(0..len);
                let packed_seq = packed_seq.slice(0..len);

                let naive = ascii_seq
                    .0
                    .windows(k)
                    .map(|seq| hash_kmer::<RC>(AsciiSeq(seq)))
                    .collect::<Vec<_>>();
                let scalar_ascii = hash_seq_scalar::<RC>(ascii_seq, k).collect::<Vec<_>>();
                let scalar_packed = hash_seq_scalar::<RC>(packed_seq, k).collect::<Vec<_>>();
                let simd_packed = collect(hash_seq_simd::<RC>(packed_seq, k, 1));

                assert_eq!(scalar_ascii, naive, "k={}, len={}", k, len);
                assert_eq!(scalar_packed, naive, "k={}, len={}", k, len);
                assert_eq!(simd_packed, naive, "k={}, len={}", k, len);
            }
        }
    }

    #[test]
    fn forward() {
        test_nthash::<false>();
    }

    #[test]
    fn canonical() {
        test_nthash::<true>();
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
                let scalar = hash_seq_scalar::<true>(seq, k).collect::<Vec<_>>();
                let scalar_rc = hash_seq_scalar::<true>(seq_rc, k).collect::<Vec<_>>();
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
}
