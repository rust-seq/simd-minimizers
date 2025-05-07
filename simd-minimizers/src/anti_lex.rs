//! A fast implementation of 'anti lexicographic' hashing:
//! A kmer's hash is simply its bit representation, but with the first character inverted.
//! When k > 16, only the last 16 characters are used.
//!
//! TODO: Figure out how this works with reverse complements.

use crate::nthash::Captures;
use crate::S;
use packed_seq::Seq;

/// For k b-bit chars, the k*b-bit mask, and the mask for the most significant character.
fn anti_and_mask(k: usize, b: usize) -> (u32, u32) {
    let mask = if b * k < 32 {
        (1 << (b * k)) - 1
    } else {
        u32::MAX
    };
    let anti = if b * k <= 32 {
        ((1 << b) - 1) << (b * (k - 1))
    } else {
        ((1 << b) - 1) << (32 - b)
    };
    (anti, mask)
}

/// Naively compute the 32-bit anti-lex hash of a single k-mer.
pub fn anti_lex_hash_kmer<'s>(seq: impl Seq<'s>) -> u32 {
    let b = seq.bits_per_char();
    let k = seq.len();
    let mut hfw: u32 = 0;
    let (anti, _mask) = anti_and_mask(k, b);
    seq.iter_bp().for_each(|a| {
        hfw = (hfw << b) ^ a as u32;
    });
    hfw ^ anti
}

/// Returns a scalar iterator over the 32-bit anti-lex hashes of all k-mers in the sequence.
/// Prefer `anti_lex_seq_simd`.
pub fn anti_lex_hash_seq_scalar<'s>(
    seq: impl Seq<'s>,
    k: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    let b = seq.bits_per_char();
    assert!(k > 0);
    let mut hfw: u32 = 0;
    let mut add = seq.iter_bp();
    let (anti, mask) = anti_and_mask(k, b);
    add.by_ref().take(k - 1).for_each(|a| {
        hfw = (hfw << b) ^ (a as u32);
    });
    add.map(move |a| {
        hfw = ((hfw << b) ^ (a as u32)) & mask;
        hfw ^ anti
    })
}

/// Returns a simd-iterator over the 8 chunks 32-bit anti-lex hashes of all k-mers in the sequence.
/// The tail is returned separately.
/// Returned chunks overlap by w-1 hashes. Set w=1 for non-overlapping chunks.
pub fn anti_lex_hash_seq_simd<'s>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
) -> (impl ExactSizeIterator<Item = S> + Captures<&'s ()>, usize) {
    let b = seq.bits_per_char();
    assert!(k > 0);
    assert!(w > 0);

    let mut h_fw = S::splat(0);
    let (mut add, padding) = seq.par_iter_bp(k + w - 1);

    let (anti, mask) = anti_and_mask(k, b);
    let anti = S::splat(anti);
    let mask = S::splat(mask);

    add.by_ref().take(k - 1).for_each(|a| {
        h_fw = (h_fw << b as i32) ^ a;
    });

    let it = add.map(move |a| {
        h_fw = ((h_fw << b as i32) ^ a) & mask;
        h_fw ^ anti
    });

    (it, padding)
}

/// A function that 'eats' added and removed bases, and returns the updated hash.
/// The distance between them must be k-1, and the first k-1 removed bases must be 0.
/// The first k-1 returned values will be useless.
pub fn anti_lex_hash_mapper<'s, Sq: Seq<'s>>(k: usize, w: usize) -> impl FnMut(S) -> S + Clone {
    let b = Sq::BITS_PER_CHAR;
    assert!(k > 0);
    assert!(k > 0);
    assert!(w > 0);

    let (anti, mask) = anti_and_mask(k, b);
    let anti = S::splat(anti);
    let mask = S::splat(mask);

    let mut h_fw = S::splat(0);

    move |a| {
        h_fw = ((h_fw << b as i32) ^ a) & mask;
        h_fw ^ anti
    }
}
