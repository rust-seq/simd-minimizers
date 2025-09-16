//! Determine whether each window is canonical, when `#GT > #AC`.
use std::mem::transmute;

use crate::S;
use packed_seq::{Delay, Seq};
use wide::{i32x8, CmpGt};

/// An iterator over windows that returns for each whether it's canonical or not.
/// Canonical windows have >half TG characters.
/// Window length l=k+w-1 must be odd for this to never tie.
pub fn canonical_windows_seq_scalar<'s>(
    seq: impl Seq<'s>,
    l: usize,
) -> impl ExactSizeIterator<Item = bool> {
    assert!(
        l % 2 == 1,
        "Window length l={l} must be odd to guarantee canonicality"
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
/// NOTE: First l-1 values are bogus.
pub fn canonical_mapper(l: usize) -> (Delay, impl FnMut((S, S)) -> i32x8) {
    assert!(
        l % 2 == 1,
        "Window length l={l} must be odd to guarantee canonicality"
    );

    // Cnt of odd characters, offset by -l/2 so >0 is canonical and <0 is not.
    // TODO: Verify that the delayed removed characters are indeed 0.
    let mut cnt = i32x8::splat(-(l as i32));
    let two = i32x8::splat(2);

    (
        Delay(l),
        #[inline(always)]
        move |(a, r)| {
            cnt += unsafe { transmute::<_, i32x8>(a) } & two;
            cnt -= unsafe { transmute::<_, i32x8>(r) } & two;
            cnt.cmp_gt(i32x8::splat(0))
        },
    )
}
