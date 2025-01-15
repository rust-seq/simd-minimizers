use super::*;
use crate::{minimizers::*, nthash::*};
use collect::collect;
use itertools::Itertools;
use packed_seq::{AsciiSeq, AsciiSeqVec, PackedSeqVec, SeqVec};
use rand::Rng;
use std::{iter::once, sync::LazyLock};

static ASCII_SEQ: LazyLock<AsciiSeqVec> = LazyLock::new(|| AsciiSeqVec::random(1024 * 32));
static PACKED_SEQ: LazyLock<PackedSeqVec> =
    LazyLock::new(|| PackedSeqVec::from_ascii(&ASCII_SEQ.seq));

fn test_on_inputs(f: impl Fn(usize, usize, AsciiSeq, PackedSeq)) {
    let ascii_seq = &*ASCII_SEQ;
    let packed_seq = &*PACKED_SEQ;
    let mut rng = rand::thread_rng();
    let mut ks = vec![1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65];
    let mut ws = vec![1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65];
    let mut lens = (0..100).collect_vec();
    ks.extend((0..10).map(|_| rng.gen_range(6..100)).collect_vec());
    ws.extend((0..10).map(|_| rng.gen_range(6..100)).collect_vec());
    lens.extend((0..10).map(|_| rng.gen_range(100..1024 * 32)).collect_vec());
    for &k in &ks {
        for &w in &ws {
            for &len in &lens {
                let ascii_seq = ascii_seq.slice(0..len);
                let packed_seq = packed_seq.slice(0..len);

                f(k, w, ascii_seq, packed_seq);
            }
        }
    }
}

fn test_nthash<const RC: bool>() {
    test_on_inputs(|k, w, ascii_seq, packed_seq| {
        if w > 1 {
            return;
        }

        let naive = ascii_seq
            .0
            .windows(k)
            .map(|seq| hash_kmer::<RC>(AsciiSeq(seq)))
            .collect::<Vec<_>>();
        let scalar_ascii = hash_seq_scalar::<RC>(ascii_seq, k).collect::<Vec<_>>();
        let scalar_packed = hash_seq_scalar::<RC>(packed_seq, k).collect::<Vec<_>>();
        let simd_packed = collect(hash_seq_simd::<RC>(packed_seq, k, 1));

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, naive, "k={}, len={}", k, len);
        assert_eq!(scalar_packed, naive, "k={}, len={}", k, len);
        assert_eq!(simd_packed, naive, "k={}, len={}", k, len);
    });
}

#[test]
fn nthash_forward() {
    test_nthash::<false>();
}

#[test]
fn nthash_canonical() {
    test_nthash::<true>();
}

#[test]
fn nthash_canonical_is_revcomp() {
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
        for len in (0..100).chain(once(1024 * 32)) {
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

#[test]
fn minimizers_fwd() {
    test_on_inputs(|k, w, ascii_seq, packed_seq| {
        let naive = ascii_seq
            .0
            .windows(w + k - 1)
            .enumerate()
            .map(|(pos, seq)| (pos + minimizer(AsciiSeq(seq), k)) as u32)
            .collect::<Vec<_>>();

        let scalar_ascii = minimizers_seq_scalar(ascii_seq, k, w).collect::<Vec<_>>();
        let scalar_packed = minimizers_seq_scalar(packed_seq, k, w).collect::<Vec<_>>();
        let simd_packed = collect(minimizers_seq_simd(packed_seq, k, w));

        let len = ascii_seq.len();
        assert_eq!(naive, scalar_ascii, "k={k}, w={w}, len={len}");
        assert_eq!(naive, simd_packed, "k={k}, w={w}, len={len}");
        assert_eq!(naive, scalar_packed, "k={k}, w={w}, len={len}");
    });
}

#[test]
fn minimizers_canonical() {
    test_on_inputs(|k, w, ascii_seq, packed_seq| {
        if (k + w - 1) % 2 == 0 {
            return;
        }
        let scalar_ascii = canonical_minimizers_seq_scalar(ascii_seq, k, w).collect::<Vec<_>>();
        let scalar_packed = canonical_minimizers_seq_scalar(packed_seq, k, w).collect::<Vec<_>>();
        let simd_packed = collect(canonical_minimizers_seq_simd(packed_seq, k, w));

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
    });
}

#[test]
fn minimizer_positions() {
    test_on_inputs(|k, w, ascii_seq, packed_seq| {
        let mut scalar_ascii = vec![];
        minimizer_positions_scalar(ascii_seq, k, w, &mut scalar_ascii);
        let mut scalar_packed = vec![];
        minimizer_positions_scalar(packed_seq, k, w, &mut scalar_packed);
        let mut simd_packed = vec![];
        super::minimizer_positions(packed_seq, k, w, &mut simd_packed);

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
    });
}

#[test]
fn canonical_minimizer_positions() {
    test_on_inputs(|k, w, ascii_seq, packed_seq| {
        if (k + w - 1) % 2 == 0 {
            return;
        }
        let mut scalar_ascii = vec![];
        canonical_minimizer_positions_scalar(ascii_seq, k, w, &mut scalar_ascii);
        let mut scalar_packed = vec![];
        canonical_minimizer_positions_scalar(packed_seq, k, w, &mut scalar_packed);
        let mut simd_packed = vec![];
        super::canonical_minimizer_positions(packed_seq, k, w, &mut simd_packed);

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
    });
}
