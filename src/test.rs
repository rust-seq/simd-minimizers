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
    fn f<H: KmerHasher>(hasher: impl Fn(usize) -> H) {
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            let hasher = hasher(k);
            let m = minimizers(k, w).hasher(&hasher);

            let naive = ascii_seq
                .0
                .windows(w + k - 1)
                .enumerate()
                .map(|(pos, seq)| (pos + one_minimizer(AsciiSeq(seq), &hasher)) as u32)
                .dedup()
                .collect::<Vec<_>>();

            let scalar_ascii = m.run_scalar_once(ascii_seq);
            let scalar_packed = m.run_scalar_once(packed_seq);
            let simd_ascii = m.run_once(ascii_seq);
            let simd_packed = m.run_once(packed_seq);

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
    fn f<H: KmerHasher>(hasher: impl Fn(usize) -> H) {
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            if (k + w - 1) % 2 == 0 {
                return;
            }
            let hasher = hasher(k);
            let m = canonical_minimizers(k, w).hasher(&hasher);

            let scalar_ascii = m.run_scalar_once(ascii_seq);
            let scalar_packed = m.run_scalar_once(packed_seq);
            let simd_ascii = m.run_once(ascii_seq);
            let simd_packed = m.run_once(packed_seq);

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
        let m = canonical_minimizers(k, w);

        let packed_seq_rc = packed_seq.to_revcomp();
        let packed_seq_rc = packed_seq_rc.as_slice();

        let mut fwd_positions = vec![];
        let mut rc_positions = vec![];
        let fwd_values = m
            .run(packed_seq, &mut fwd_positions)
            .values_u64()
            .collect_vec();
        let mut rc_values = m
            .run(packed_seq_rc, &mut rc_positions)
            .values_u64()
            .collect_vec();

        // Check that positions are symmetric.
        let len = ascii_seq.len();
        for (&x, &y) in fwd_positions.iter().zip(rc_positions.iter().rev()) {
            assert_eq!((x + y) as usize, len - k, "k={k}, w={w}, fwd={x}, rc={y}");
        }

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
        let m = minimizers(k, w);

        let mut scalar_ascii = vec![];
        let mut scalar_ascii_skmer = vec![];
        m.super_kmers(&mut scalar_ascii_skmer)
            .run_scalar(ascii_seq, &mut scalar_ascii);
        let mut scalar_packed = vec![];
        let mut scalar_packed_skmer = vec![];
        m.super_kmers(&mut scalar_packed_skmer)
            .run_scalar(packed_seq, &mut scalar_packed);
        let mut simd_ascii = vec![];
        let mut simd_ascii_skmer = vec![];
        m.super_kmers(&mut simd_ascii_skmer)
            .run(ascii_seq, &mut simd_ascii);
        let mut simd_packed = vec![];
        let mut simd_packed_skmer = vec![];
        m.super_kmers(&mut simd_packed_skmer)
            .run(packed_seq, &mut simd_packed);

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
        let m = canonical_minimizers(k, w);

        let mut scalar_ascii = vec![];
        let mut scalar_ascii_skmer = vec![];
        m.super_kmers(&mut scalar_ascii_skmer)
            .run_scalar(ascii_seq, &mut scalar_ascii);
        let mut scalar_packed = vec![];
        let mut scalar_packed_skmer = vec![];
        m.super_kmers(&mut scalar_packed_skmer)
            .run_scalar(packed_seq, &mut scalar_packed);
        let mut simd_ascii = vec![];
        let mut simd_ascii_skmer = vec![];
        m.super_kmers(&mut simd_ascii_skmer)
            .run(ascii_seq, &mut simd_ascii);
        let mut simd_packed = vec![];
        let mut simd_packed_skmer = vec![];
        m.super_kmers(&mut simd_packed_skmer)
            .run(packed_seq, &mut simd_packed);

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

/// Test to make sure that the builder compiles.
fn _builder<'s>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
    min_pos: &'s mut Vec<u32>,
    sk_pos: &'s mut Vec<u32>,
) {
    let hasher = &<MulHasher>::new_with_seed(k, 1234);

    // warning: unused
    // minimizers(k, w);
    // minimizers(k, w).hasher(hasher);

    minimizers(k, w).run(seq, min_pos);
    canonical_minimizers(k, w).run(seq, min_pos);
    // with super_kmers
    minimizers(k, w).super_kmers(sk_pos).run(seq, min_pos);
    canonical_minimizers(k, w)
        .super_kmers(sk_pos)
        .run(seq, min_pos);
    // with hasher
    canonical_minimizers(k, w).hasher(hasher).run(seq, min_pos);
    canonical_minimizers(k, w)
        .hasher(hasher)
        .super_kmers(sk_pos)
        .run(seq, min_pos);
    // with values
    let out = canonical_minimizers(k, w)
        .hasher(hasher)
        .super_kmers(sk_pos)
        .run(seq, min_pos);
    out.values_u64().sum::<u64>();
    out.values_u128().sum::<u128>();
    // reusing the minimizer
    let m = canonical_minimizers(k, w).hasher(hasher);
    for _ in 0..10 {
        m.super_kmers(sk_pos).run(seq, min_pos);
    }
}

#[test]
fn collect_and_dedup_scalar() {
    let mut out = vec![];
    collect_and_dedup_into_scalar([0, 1, 2, 3, 4, 5].into_iter(), &mut out);
    assert_eq!(out, [0, 1, 2, 3, 4, 5]);
    let mut out = vec![];
    collect_and_dedup_into_scalar([0, 0, 1, 1, 2, 2].into_iter(), &mut out);
    assert_eq!(out, [0, 1, 2]);
}

#[test]
fn collect_and_dedup_with_index_scalar() {
    let mut out = vec![];
    let mut pos = vec![];
    collect_and_dedup_with_index_into_scalar([0, 1, 2, 3, 4, 5].into_iter(), &mut out, &mut pos);
    assert_eq!(out, [0, 1, 2, 3, 4, 5]);
    assert_eq!(pos, [0, 1, 2, 3, 4, 5]);
    let mut out = vec![];
    let mut pos = vec![];
    collect_and_dedup_with_index_into_scalar([0, 0, 1, 1, 2, 2].into_iter(), &mut out, &mut pos);
    assert_eq!(out, [0, 1, 2]);
    assert_eq!(pos, [0, 2, 4]);
}
