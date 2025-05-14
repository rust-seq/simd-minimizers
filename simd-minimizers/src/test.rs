use super::*;
use crate::{minimizers::*, nthash::*};
use collect::collect;
use itertools::Itertools;
use packed_seq::{AsciiSeq, AsciiSeqVec, PackedSeq, PackedSeqVec, SeqVec};
use rand::{random_range, Rng};
use std::sync::LazyLock;

/// Swap G and T, so that the lex order is the same as for the packed version.
fn swap_gt(c: u8) -> u8 {
    match c {
        b'G' => b'T',
        b'T' => b'G',
        c => c,
    }
}

static ASCII_SEQ: LazyLock<AsciiSeqVec> = LazyLock::new(|| AsciiSeqVec::random(1024 * 8));
static SLICE: LazyLock<Vec<u8>> =
    LazyLock::new(|| ASCII_SEQ.seq.iter().copied().map(swap_gt).collect_vec());
static PACKED_SEQ: LazyLock<PackedSeqVec> =
    LazyLock::new(|| PackedSeqVec::from_ascii(&ASCII_SEQ.seq));

fn test_on_inputs(f: impl Fn(usize, usize, &[u8], AsciiSeq, PackedSeq)) {
    let slice = &*SLICE;
    let ascii_seq = &*ASCII_SEQ;
    let packed_seq = &*PACKED_SEQ;
    let mut rng = rand::rng();
    let mut ks = vec![1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65];
    let mut ws = vec![1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65];
    let mut lens = (0..100).collect_vec();
    ks.extend((0..10).map(|_| rng.random_range(6..100)).collect_vec());
    ws.extend((0..10).map(|_| rng.random_range(6..100)).collect_vec());
    lens.extend(
        (0..10)
            .map(|_| rng.random_range(100..1024 * 8))
            .collect_vec(),
    );
    for &k in &ks {
        for &w in &ws {
            for &len in &lens {
                let offset = rng.random_range(0..=3.min(len));
                let slice = slice.slice(offset..len);
                let ascii_seq = ascii_seq.slice(offset..len);
                let packed_seq = packed_seq.slice(offset..len);

                f(k, w, slice, ascii_seq, packed_seq);
            }
        }
    }
}

fn test_nthash<const RC: bool, H: CharHasher>() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        if w > 1 {
            return;
        }

        let naive = ascii_seq
            .0
            .windows(k)
            .map(|seq| nthash_kmer::<RC, H>(AsciiSeq(seq)))
            .collect::<Vec<_>>();
        let scalar_ascii = nthash_seq_scalar::<RC, H>(ascii_seq, k).collect::<Vec<_>>();
        let scalar_packed = nthash_seq_scalar::<RC, H>(packed_seq, k).collect::<Vec<_>>();
        let simd_ascii = collect(nthash_seq_simd::<RC, AsciiSeq, H>(ascii_seq, k, 1));
        let simd_packed = collect(nthash_seq_simd::<RC, PackedSeq, H>(packed_seq, k, 1));

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, naive, "k={}, len={}", k, len);
        assert_eq!(scalar_packed, naive, "k={}, len={}", k, len);
        assert_eq!(simd_ascii, naive, "k={}, len={}", k, len);
        assert_eq!(simd_packed, naive, "k={}, len={}", k, len);
    });
}

#[test]
fn nthash_forward() {
    test_nthash::<false, NtHasher>();
}

#[test]
fn nthash_canonical() {
    test_nthash::<true, NtHasher>();
}

#[test]
fn nthash_forward_mul() {
    test_nthash::<false, MulHasher>();
}

#[test]
fn nthash_canonical_mul() {
    test_nthash::<true, MulHasher>();
}

#[test]
fn nthash_canonical_is_revcomp() {
    fn f<H: CharHasher>() {
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
            for len in (0..100).chain((0..10).map(|_| random_range(1024..8 * 1024))) {
                let seq = seq.slice(0..len);
                let seq_rc = seq_rc.slice(seq_rc.len() - len..seq_rc.len());
                let scalar = nthash_seq_scalar::<true, H>(seq, k).collect::<Vec<_>>();
                let scalar_rc = nthash_seq_scalar::<true, H>(seq_rc, k).collect::<Vec<_>>();
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
    f::<NtHasher>();
    f::<MulHasher>();
}

#[test]
fn test_anti_lex_hash() {
    use anti_lex::*;
    test_on_inputs(|k, w, slice, ascii_seq, packed_seq| {
        if w > 1 {
            return;
        }
        // naive
        let naive = ascii_seq
            .0
            .windows(k)
            .map(|seq| anti_lex_hash_kmer(AsciiSeq(seq)))
            .collect::<Vec<_>>();
        let scalar_ascii = anti_lex_hash_seq_scalar(ascii_seq, k).collect::<Vec<_>>();
        let scalar_packed = anti_lex_hash_seq_scalar(packed_seq, k).collect::<Vec<_>>();
        let simd_ascii = collect(anti_lex_hash_seq_simd(ascii_seq, k, 1));
        let simd_packed = collect(anti_lex_hash_seq_simd(packed_seq, k, 1));
        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, naive, "k={}, len={}", k, len);
        assert_eq!(scalar_packed, naive, "k={}, len={}", k, len);
        assert_eq!(simd_ascii, naive, "k={}, len={}", k, len);
        assert_eq!(simd_packed, naive, "k={}, len={}", k, len);

        let scalar_slice = anti_lex_hash_seq_scalar(slice, k).collect::<Vec<_>>();
        let simd_slice = collect(anti_lex_hash_seq_simd(slice, k, 1));
        assert_eq!(simd_slice, scalar_slice, "k={}, len={}", k, len);
    });
}

#[test]
fn minimizers_fwd() {
    fn f<H: CharHasher>() {
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            let naive = ascii_seq
                .0
                .windows(w + k - 1)
                .enumerate()
                .map(|(pos, seq)| (pos + minimizer::<H>(AsciiSeq(seq), k)) as u32)
                .collect::<Vec<_>>();

            let scalar_ascii = minimizers_seq_scalar::<H>(ascii_seq, k, w).collect::<Vec<_>>();
            let scalar_packed = minimizers_seq_scalar::<H>(packed_seq, k, w).collect::<Vec<_>>();
            let simd_ascii = collect(minimizers_seq_simd::<_, H>(ascii_seq, k, w));
            let simd_packed = collect(minimizers_seq_simd::<_, H>(packed_seq, k, w));

            let len = ascii_seq.len();
            assert_eq!(naive, scalar_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(naive, scalar_packed, "k={k}, w={w}, len={len}");
            assert_eq!(naive, simd_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(naive, simd_packed, "k={k}, w={w}, len={len}");
        });
    }
    f::<NtHasher>();
    f::<MulHasher>();
}

#[test]
fn minimizers_canonical() {
    fn f<H: CharHasher>() {
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            if (k + w - 1) % 2 == 0 {
                return;
            }
            let scalar_ascii =
                canonical_minimizers_seq_scalar::<H>(ascii_seq, k, w).collect::<Vec<_>>();
            let scalar_packed =
                canonical_minimizers_seq_scalar::<H>(packed_seq, k, w).collect::<Vec<_>>();
            let simd_ascii = collect(canonical_minimizers_seq_simd::<_, H>(ascii_seq, k, w));
            let simd_packed = collect(canonical_minimizers_seq_simd::<_, H>(packed_seq, k, w));

            let len = ascii_seq.len();
            assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
        });
    }
    f::<NtHasher>();
    f::<MulHasher>();
}

#[test]
fn minimizer_positions() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        let mut scalar_ascii = vec![];
        scalar::minimizer_positions_scalar(ascii_seq, k, w, &mut scalar_ascii);
        let mut scalar_packed = vec![];
        scalar::minimizer_positions_scalar(packed_seq, k, w, &mut scalar_packed);
        let mut simd_ascii = vec![];
        super::minimizer_positions(ascii_seq, k, w, &mut simd_ascii);
        let mut simd_packed = vec![];
        super::minimizer_positions(packed_seq, k, w, &mut simd_packed);

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
    });
}

#[test]
fn canonical_minimizer_positions() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        if (k + w - 1) % 2 == 0 {
            return;
        }
        let mut scalar_ascii = vec![];
        scalar::canonical_minimizer_positions_scalar(ascii_seq, k, w, &mut scalar_ascii);
        let mut scalar_packed = vec![];
        scalar::canonical_minimizer_positions_scalar(packed_seq, k, w, &mut scalar_packed);
        let mut simd_ascii = vec![];
        super::canonical_minimizer_positions(ascii_seq, k, w, &mut simd_ascii);
        let mut simd_packed = vec![];
        super::canonical_minimizer_positions(packed_seq, k, w, &mut simd_packed);

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
    });
}

#[test]
fn minimizer_and_superkmer_positions() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        let scalar_ascii = &mut vec![];
        let scalar_ascii_skmer = &mut vec![];
        scalar::minimizer_and_superkmer_positions_scalar(
            ascii_seq,
            k,
            w,
            scalar_ascii,
            scalar_ascii_skmer,
        );
        let scalar_packed = &mut vec![];
        let scalar_packed_skmer = &mut vec![];
        scalar::minimizer_and_superkmer_positions_scalar(
            packed_seq,
            k,
            w,
            scalar_packed,
            scalar_packed_skmer,
        );
        let simd_ascii = &mut vec![];
        let simd_ascii_skmer = &mut vec![];
        super::minimizer_and_superkmer_positions(ascii_seq, k, w, simd_ascii, simd_ascii_skmer);
        let simd_packed = &mut vec![];
        let simd_packed_skmer = &mut vec![];
        super::minimizer_and_superkmer_positions(packed_seq, k, w, simd_packed, simd_packed_skmer);

        let len = ascii_seq.len();
        assert_eq!(
            scalar_ascii.len(),
            scalar_ascii_skmer.len(),
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(
            scalar_packed.len(),
            scalar_packed_skmer.len(),
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(
            simd_ascii.len(),
            simd_ascii_skmer.len(),
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(
            simd_packed.len(),
            simd_packed_skmer.len(),
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
        assert_eq!(
            scalar_ascii_skmer, scalar_packed_skmer,
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
        assert_eq!(
            scalar_ascii_skmer, simd_ascii_skmer,
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
        assert_eq!(
            scalar_ascii_skmer, simd_packed_skmer,
            "k={k}, w={w}, len={len}"
        );
    });
}

#[test]
fn canonical_minimizer_and_superkmer_positions() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        if (k + w - 1) % 2 == 0 {
            return;
        }
        let mut scalar_ascii = vec![];
        let mut scalar_ascii_skmer = vec![];
        scalar::canonical_minimizer_and_superkmer_positions_scalar(
            ascii_seq,
            k,
            w,
            &mut scalar_ascii,
            &mut scalar_ascii_skmer,
        );
        let mut scalar_packed = vec![];
        let mut scalar_packed_skmer = vec![];
        scalar::canonical_minimizer_and_superkmer_positions_scalar(
            packed_seq,
            k,
            w,
            &mut scalar_packed,
            &mut scalar_packed_skmer,
        );
        let mut simd_ascii = vec![];
        let mut simd_ascii_skmer = vec![];
        super::canonical_minimizer_and_superkmer_positions(
            ascii_seq,
            k,
            w,
            &mut simd_ascii,
            &mut simd_ascii_skmer,
        );
        let mut simd_packed = vec![];
        let mut simd_packed_skmer = vec![];
        super::canonical_minimizer_and_superkmer_positions(
            packed_seq,
            k,
            w,
            &mut simd_packed,
            &mut simd_packed_skmer,
        );

        let len = ascii_seq.len();
        assert_eq!(
            scalar_ascii.len(),
            scalar_ascii_skmer.len(),
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(
            scalar_packed.len(),
            scalar_packed_skmer.len(),
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(
            simd_ascii.len(),
            simd_ascii_skmer.len(),
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(
            simd_packed.len(),
            simd_packed_skmer.len(),
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
        assert_eq!(
            scalar_ascii_skmer, scalar_packed_skmer,
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
        assert_eq!(
            scalar_ascii_skmer, simd_ascii_skmer,
            "k={k}, w={w}, len={len}"
        );
        assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
        assert_eq!(
            scalar_ascii_skmer, simd_packed_skmer,
            "k={k}, w={w}, len={len}"
        );
    });
}
