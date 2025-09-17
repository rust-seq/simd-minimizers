use super::*;
use crate::minimizers::*;
use itertools::Itertools;
use packed_seq::{AsciiSeq, AsciiSeqVec, PackedSeq, PackedSeqVec, SeqVec};
use rand::Rng;
use seq_hash::{AntiLexHasher, MulHasher, NtHasher};
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

fn test_on_inputs(mut f: impl FnMut(usize, usize, &[u8], AsciiSeq, PackedSeq)) {
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

#[test]
fn minimizers_fwd() {
    fn f<H: SeqHasher>(hasher: impl Fn(usize) -> H) {
        let mut cache = Cache::default();
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            let hasher = hasher(k);
            let naive = ascii_seq
                .0
                .windows(w + k - 1)
                .enumerate()
                .map(|(pos, seq)| (pos + one_minimizer(AsciiSeq(seq), &hasher)) as u32)
                .collect::<Vec<_>>();

            let scalar_ascii =
                minimizers_seq_scalar(ascii_seq, &hasher, w, &mut cache).collect::<Vec<_>>();
            let scalar_packed =
                minimizers_seq_scalar(packed_seq, &hasher, w, &mut cache).collect::<Vec<_>>();
            let simd_ascii = minimizers_seq_simd(ascii_seq, &hasher, w, &mut cache).collect();
            let simd_packed = minimizers_seq_simd(packed_seq, &hasher, w, &mut cache).collect();

            let len = ascii_seq.len();
            assert_eq!(naive, scalar_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(naive, scalar_packed, "k={k}, w={w}, len={len}");
            assert_eq!(naive, simd_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(naive, simd_packed, "k={k}, w={w}, len={len}");
        });
    }
    f(|k| NtHasher::<false>::new(k));
    f(|k| MulHasher::<false>::new(k));
    f(|k| AntiLexHasher::<false>::new(k));
}

#[test]
fn minimizers_canonical() {
    fn f<H: SeqHasher>(hasher: impl Fn(usize) -> H) {
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            if (k + w - 1) % 2 == 0 {
                return;
            }
            let hasher = hasher(k);
            let scalar_ascii =
                canonical_minimizers_seq_scalar(ascii_seq, &hasher, w, &mut Cache::default())
                    .collect::<Vec<_>>();
            let scalar_packed =
                canonical_minimizers_seq_scalar(packed_seq, &hasher, w, &mut Cache::default())
                    .collect::<Vec<_>>();
            let simd_ascii =
                canonical_minimizers_seq_simd(ascii_seq, &hasher, w, &mut Default::default())
                    .collect();
            let simd_packed =
                canonical_minimizers_seq_simd(packed_seq, &hasher, w, &mut Default::default())
                    .collect();

            let len = ascii_seq.len();

            assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
        });
    }
    f(|k| NtHasher::<true>::new(k));
    f(|k| MulHasher::<true>::new(k));
    f(|k| AntiLexHasher::<true>::new(k));
}

#[test]
fn minimizer_positions() {
    fn f<H: SeqHasher>(hasher: impl Fn(usize) -> H) {
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            let hasher = hasher(k);
            let mut scalar_ascii = vec![];
            scalar::minimizer_positions_scalar(ascii_seq, &hasher, w, &mut scalar_ascii);
            let mut scalar_packed = vec![];
            scalar::minimizer_positions_scalar(packed_seq, &hasher, w, &mut scalar_packed);
            let mut simd_ascii = vec![];
            super::minimizer_positions(ascii_seq, &hasher, w, &mut simd_ascii);
            let mut simd_packed = vec![];
            super::minimizer_positions(packed_seq, &hasher, w, &mut simd_packed);

            let len = ascii_seq.len();
            assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
        });
    }
    f(|k| NtHasher::<true>::new(k));
    f(|k| MulHasher::<true>::new(k));
    f(|k| AntiLexHasher::<true>::new(k));
}

#[test]
fn canonical_minimizer_positions() {
    fn f<H: SeqHasher>(hasher: impl Fn(usize) -> H) {
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            if (k + w - 1) % 2 == 0 {
                return;
            }
            let hasher = hasher(k);
            let mut scalar_ascii = vec![];
            scalar::canonical_minimizer_positions_scalar(ascii_seq, &hasher, w, &mut scalar_ascii);
            let mut scalar_packed = vec![];
            scalar::canonical_minimizer_positions_scalar(
                packed_seq,
                &hasher,
                w,
                &mut scalar_packed,
            );
            let mut simd_ascii = vec![];
            super::canonical_minimizer_positions(ascii_seq, &hasher, w, &mut simd_ascii);
            let mut simd_packed = vec![];
            super::canonical_minimizer_positions(packed_seq, &hasher, w, &mut simd_packed);

            let len = ascii_seq.len();
            assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
        });
    }
    f(|k| NtHasher::<true>::new(k));
    f(|k| MulHasher::<true>::new(k));
    f(|k| AntiLexHasher::<true>::new(k));
}

#[test]
fn canonical_minimizer_positions_and_values() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        if k > 32 {
            return;
        }
        if (k + w - 1) % 2 == 0 {
            return;
        }
        let hasher = NtHasher::<true>::new(k);

        let packed_seq_rc = packed_seq.to_revcomp();
        let packed_seq_rc = packed_seq_rc.as_slice();

        let mut fwd_positions = vec![];
        super::canonical_minimizer_positions(packed_seq, &hasher, w, &mut fwd_positions);
        let mut rc_positions = vec![];
        super::canonical_minimizer_positions(packed_seq_rc, &hasher, w, &mut rc_positions);

        // Check that positions are symmetric.
        let len = ascii_seq.len();
        for (&x, &y) in fwd_positions.iter().zip(rc_positions.iter().rev()) {
            assert_eq!((x + y) as usize, len - k, "k={k}, w={w}, fwd={x}, rc={y}");
        }

        // Extract canonical minimizer values.
        let fwd_values: Vec<_> =
            super::iter_canonical_minimizer_values(packed_seq, k, &fwd_positions).collect();

        let mut rc_values: Vec<_> =
            super::iter_canonical_minimizer_values(packed_seq_rc, k, &rc_positions).collect();

        // Check that values are the same.
        rc_values.reverse();
        assert_eq!(
            fwd_values,
            rc_values,
            "k={k}, w={w}, len={}",
            ascii_seq.len()
        );
    });
}

#[test]
fn minimizer_and_superkmer_positions() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        let hasher = NtHasher::<false>::new(k);
        let scalar_ascii = &mut vec![];
        let scalar_ascii_skmer = &mut vec![];
        scalar::minimizer_and_superkmer_positions_scalar(
            ascii_seq,
            &hasher,
            w,
            scalar_ascii,
            scalar_ascii_skmer,
        );
        let scalar_packed = &mut vec![];
        let scalar_packed_skmer = &mut vec![];
        scalar::minimizer_and_superkmer_positions_scalar(
            packed_seq,
            &hasher,
            w,
            scalar_packed,
            scalar_packed_skmer,
        );
        let simd_ascii = &mut vec![];
        let simd_ascii_skmer = &mut vec![];
        super::minimizer_and_superkmer_positions(
            ascii_seq,
            &hasher,
            w,
            simd_ascii,
            simd_ascii_skmer,
        );
        let simd_packed = &mut vec![];
        let simd_packed_skmer = &mut vec![];
        super::minimizer_and_superkmer_positions(
            packed_seq,
            &hasher,
            w,
            simd_packed,
            simd_packed_skmer,
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

#[test]
fn canonical_minimizer_and_superkmer_positions() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        if (k + w - 1) % 2 == 0 {
            return;
        }
        let hasher = NtHasher::<true>::new(k);
        let mut scalar_ascii = vec![];
        let mut scalar_ascii_skmer = vec![];
        scalar::canonical_minimizer_and_superkmer_positions_scalar(
            ascii_seq,
            &hasher,
            w,
            &mut scalar_ascii,
            &mut scalar_ascii_skmer,
        );
        let mut scalar_packed = vec![];
        let mut scalar_packed_skmer = vec![];
        scalar::canonical_minimizer_and_superkmer_positions_scalar(
            packed_seq,
            &hasher,
            w,
            &mut scalar_packed,
            &mut scalar_packed_skmer,
        );
        let mut simd_ascii = vec![];
        let mut simd_ascii_skmer = vec![];
        super::canonical_minimizer_and_superkmer_positions(
            ascii_seq,
            &hasher,
            w,
            &mut simd_ascii,
            &mut simd_ascii_skmer,
        );
        let mut simd_packed = vec![];
        let mut simd_packed_skmer = vec![];
        super::canonical_minimizer_and_superkmer_positions(
            packed_seq,
            &hasher,
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
