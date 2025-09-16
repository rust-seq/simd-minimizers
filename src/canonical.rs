//! Determine whether each window is canonical, when `#GT > #AC`.
use std::mem::transmute;

use crate::S;
use packed_seq::{PackedSeq, Seq};
use wide::{i32x8, CmpGt};

use crate::nthash::Captures;

/// An iterator over windows that returns for each whether it's canonical or not.
/// Canonical windows have >half TG characters.
/// Window length l=k+w-1 must be odd for this to never tie.
pub fn canonical_windows_seq_scalar<'s>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
) -> impl ExactSizeIterator<Item = bool> + Captures<&'s ()> {
    let l = k + w - 1;
    assert!(
        l % 2 == 1,
        "Window length {l}={k}+{w}-1 must be odd to guarantee canonicality"
    );

    let mut add = seq.iter_bp();
    let remove = seq.iter_bp();

    // Cnt of odd characters, offset by -l/2 so >0 is canonical and <0 is not.
    let mut cnt = -(l as isize);

    add.by_ref().take(l - 1).for_each(|a| {
        cnt += a as isize & 2;
    });
    add.zip(remove).map(move |(a, r)| {
        cnt += a as isize & 2;
        let is_canonical = cnt > 0;
        cnt -= r as isize & 2;
        is_canonical
    })
}

/// An iterator over windows that returns for each whether it's canonical or not.
/// Canonical windows have >half odd characters.
/// Window length l=k+w-1 must be odd for this to never tie.
///
/// Split the kmers of the sequence into 8 chunks of equal length ~len/8.
/// Then compute of each of them in parallel using SIMD,
/// and return the remaining few using the second iterator.
pub fn canonical_windows_seq_simd<'s>(
    seq: PackedSeq<'s>,
    k: usize,
    w: usize,
) -> (
    impl ExactSizeIterator<Item = i32x8> + Captures<&'s ()>,
    usize,
) {
    let l = k + w - 1;
    let (add_remove, padding) = seq.par_iter_bp_delayed(k + w - 1, l - 1);

    let mut head = add_remove.map(canonical_mapper(k, w));
    head.by_ref().take(l - 1).for_each(drop);

    (head, padding)
}

/// NOTE: First l-1 values are bogus.
pub fn canonical_mapper(k: usize, w: usize) -> impl FnMut((S, S)) -> i32x8 {
    let l = k + w - 1;
    assert!(
        l % 2 == 1,
        "Window length {l}={k}+{w}-1 must be odd to guarantee canonicality"
    );

    // Cnt of odd characters, offset by -l/2 so >0 is canonical and <0 is not.
    // TODO: Verify that the delayed removed characters are indeed 0.
    let mut cnt = i32x8::splat(-(l as i32));
    let two = i32x8::splat(2);

    #[inline(always)]
    move |(a, r)| {
        cnt += unsafe { transmute::<_, i32x8>(a) } & two;
        cnt -= unsafe { transmute::<_, i32x8>(r) } & two;
        cnt.cmp_gt(i32x8::splat(0))
    }
}
