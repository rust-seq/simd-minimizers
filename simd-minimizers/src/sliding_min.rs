//! Sliding window minimum over windows of size `w`.
//!
//! For each window, the absolute position of the minimum is returned.
//!
//! Each method takes a `LEFT: bool` const generic. Set to `true` to break ties
//! towards the leftmost minimum, and false for the rightmost minimum.
//!
//! All these methods take 32 bit input values, **but they only use the upper 16 bits!**
//!
//! Positions ar
use crate::S;
use core::array::from_fn;
use std::hint::assert_unchecked;

/// A custom RingBuf implementation that has a fixed size `w` and wraps around.
struct RingBuf<V> {
    w: usize,
    idx: usize,
    data: Vec<V>,
}

impl<V: Clone> RingBuf<V> {
    #[inline(always)]
    fn new(w: usize, v: V) -> Self {
        assert!(w > 0);
        let data = vec![v; w];
        Self { w, idx: 0, data }
    }

    /// Returns the next index to be written.
    #[inline(always)]
    const fn idx(&self) -> usize {
        self.idx
    }

    #[inline(always)]
    fn push(&mut self, v: V) {
        *unsafe { self.data.get_unchecked_mut(self.idx) } = v;
        self.idx += 1;
        if self.idx == self.w {
            self.idx = 0;
        }
    }
}

/// A RingBuf can be used as a slice.
impl<V> std::ops::Deref for RingBuf<V> {
    type Target = [V];

    #[inline(always)]
    fn deref(&self) -> &[V] {
        &self.data
    }
}

/// A RingBuf can be used as a mutable slice.
impl<V> std::ops::DerefMut for RingBuf<V> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [V] {
        &mut self.data
    }
}

/// Scalar version. Takes an iterator over values and returns an iterator over positions.
#[inline(always)]
pub fn sliding_min_scalar<const LEFT: bool>(
    it: impl ExactSizeIterator<Item = u32>,
    w: usize,
) -> impl ExactSizeIterator<Item = u32> {
    assert!(w > 0);
    assert!(
        w < (1 << 15),
        "sliding_min is not tested for windows of length > 2^15."
    );
    assert!(
        it.len() < (1 << 32),
        "sliding_min returns 32bit indices. Try splitting the input into 4GB chunks first."
    );
    let mut prefix_min = u32::MAX;
    let mut ring_buf = RingBuf::new(w, prefix_min);
    // We only compare the upper 16 bits of each hash.
    // Ties are broken automatically in favour of lower pos.
    let val_mask = 0xffff_0000;
    let pos_mask = 0x0000_ffff;
    let mut pos = 0;
    let max_pos = (1 << 16) - 1;
    let mut pos_offset = 0;

    fn min<const LEFT: bool>(a: u32, b: u32) -> u32 {
        if LEFT {
            a.min(b)
        } else {
            a.max(b)
        }
    }

    let mut it = it.map(
        #[inline(always)]
        move |val| {
            // Make sure the position does not interfere with the hash value.
            if pos == max_pos {
                let delta = ((1 << 16) - 2 - w) as u32;
                pos -= delta;
                prefix_min -= delta;
                pos_offset += delta;
                for x in &mut *ring_buf {
                    *x -= delta;
                }
            }
            let elem = (if LEFT { val } else { !val } & val_mask) | pos;
            pos += 1;
            ring_buf.push(elem);
            prefix_min = min::<LEFT>(prefix_min, elem);
            // After a chunk has been filled, compute suffix minima.
            if ring_buf.idx() == 0 {
                let mut suffix_min = ring_buf[w - 1];
                for i in (0..w - 1).rev() {
                    suffix_min = min::<LEFT>(suffix_min, ring_buf[i]);
                    ring_buf[i] = suffix_min;
                }
                prefix_min = elem; // slightly faster than assigning S::splat(u32::MAX)
            }
            let suffix_min = unsafe { *ring_buf.get_unchecked(ring_buf.idx()) };
            (min::<LEFT>(prefix_min, suffix_min) & pos_mask) + pos_offset
        },
    );
    // This optimizes better than it.skip(w-1).
    it.by_ref().take(w - 1).for_each(drop);
    it
}

fn simd_min<const LEFT: bool>(a: S, b: S) -> S {
    if LEFT {
        a.min(b)
    } else {
        a.max(b)
    }
}

/// SIMD version. Takes a SIMD-iterator over 8 lanes of values and returns an iterator over 8 lanes of positions.
/// Requires an `ExactSizeIterator` to determine the length of the input.
#[inline(always)]
pub fn sliding_min_simd<const LEFT: bool>(
    it: impl ExactSizeIterator<Item = S>,
    w: usize,
    k: usize,
) -> impl ExactSizeIterator<Item = S> {
    let len = it.len();
    let mut it = it.map(sliding_min_mapper::<LEFT>(w, k, len));
    // This optimizes better than it.skip(w-1).
    it.by_ref().take(w - 1).for_each(drop);
    it
}

/// Mapper version, that returns a function that can be called with new inputs as needed.
/// Output values are offset by `-(k-1)`, so that the k'th returned value (the first kmer) is at position 0.
/// `len` is the number of values in each chunk. The SIMD lanes will be offset by `len-(k+w-2)`.
/// The first `k+w-2` returned values are bogus, since they correspond to incomplete windows.
pub fn sliding_min_mapper<const LEFT: bool>(w: usize, k: usize, len: usize) -> impl FnMut(S) -> S {
    assert!(w > 0);
    assert!(w < (1 << 15), "This method is not tested for large w.");
    assert!(len * 8 < (1 << 32));
    let mut prefix_min = S::splat(u32::MAX);
    let mut ring_buf = RingBuf::new(w, prefix_min);
    // We only compare the upper 16 bits of each hash.
    // Ties are broken automatically in favour of lower pos.
    let val_mask = S::splat(0xffff_0000);
    let pos_mask = S::splat(0x0000_ffff);
    let max_pos = S::splat((1 << 16) - 1);
    let mut pos = S::splat(0);
    // Sliding min is over w+k-1 characters, so chunks overlap w+k-2.
    // Thus, the true length of each lane is len-(k+w-2).
    //
    // The k-mer starting at position 0 is done after processing the char at
    // position k-1, so we compensate for that as well.
    let mut pos_offset: S =
        from_fn(|l| (l * len.saturating_sub(k + w - 2)).wrapping_sub(k - 1) as u32).into();

    #[inline(always)]
    move |val| {
        // Make sure the position does not interfere with the hash value.
        if pos == max_pos {
            // Slow case extracted to a function to have better inlining here.
            reset_positions_offsets(w, &mut pos, &mut prefix_min, &mut pos_offset, &mut ring_buf);
        }
        // slightly faster than assigning S::splat(u32::MAX)
        let elem = (if LEFT { val } else { !val } & val_mask) | pos;
        pos += S::splat(1);
        ring_buf.push(elem);
        prefix_min = simd_min::<LEFT>(prefix_min, elem);
        // After a chunk has been filled, compute suffix minima.
        if ring_buf.idx() == 0 {
            // Slow case extracted to a function to have better inlining here.
            suffix_minima::<LEFT>(&mut ring_buf, w, &mut prefix_min, elem);
        }

        let suffix_min = unsafe { *ring_buf.get_unchecked(ring_buf.idx()) };
        (simd_min::<LEFT>(prefix_min, suffix_min) & pos_mask) + pos_offset
    }
}

fn suffix_minima<const LEFT: bool>(
    ring_buf: &mut RingBuf<S>,
    w: usize,
    prefix_min: &mut S,
    elem: S,
) {
    // Avoid some bounds checks when this function is not inlined.
    unsafe { assert_unchecked(ring_buf.len() == w) };
    unsafe { assert_unchecked(w > 0) };
    let mut suffix_min = ring_buf[w - 1];
    for i in (0..w - 1).rev() {
        suffix_min = simd_min::<LEFT>(suffix_min, ring_buf[i]);
        ring_buf[i] = suffix_min;
    }
    *prefix_min = elem;
}

fn reset_positions_offsets(
    w: usize,
    pos: &mut S,
    prefix_min: &mut S,
    pos_offset: &mut S,
    ring_buf: &mut RingBuf<S>,
) {
    let delta = S::splat((1 << 16) - 2 - w as u32);
    *pos -= delta;
    *prefix_min -= delta;
    *pos_offset += delta;
    for x in &mut **ring_buf {
        *x -= delta;
    }
}

/// Like `sliding_min_simd`, but returns both the leftmost and the rightmost minimum.
#[inline(always)]
pub fn sliding_lr_min_simd(
    it: impl ExactSizeIterator<Item = S>,
    w: usize,
    k: usize,
) -> impl ExactSizeIterator<Item = (S, S)> {
    let len = it.len();
    let mut it = it.map(sliding_lr_min_mapper(w, k, len));
    // This optimizes better than it.skip(w-1).
    it.by_ref().take(w - 1).for_each(drop);
    it
}

/// Like `sliding_min_mapper`, but returns both the leftmost and the rightmost minimum.
pub fn sliding_lr_min_mapper(w: usize, k: usize, len: usize) -> impl FnMut(S) -> (S, S) {
    assert!(w > 0);
    assert!(w < (1 << 15), "This method is not tested for large w.");
    assert!(len * 8 < (1 << 32));
    let mut prefix_lr_min = (S::splat(u32::MAX), S::splat(u32::MAX));
    let mut ring_buf = RingBuf::new(w, prefix_lr_min);
    // We only compare the upper 16 bits of each hash.
    // Ties are broken automatically in favour of lower pos.
    let val_mask = S::splat(0xffff_0000);
    let pos_mask = S::splat(0x0000_ffff);
    let max_pos = S::splat((1 << 16) - 1);
    let mut pos = S::splat(0);
    let mut pos_offset: S =
        from_fn(|l| (l * len.saturating_sub(k + w - 2)).wrapping_sub(k - 1) as u32).into();

    #[inline(always)]
    move |val| {
        // Make sure the position does not interfere with the hash value.
        if pos == max_pos {
            // Slow case extracted to a function to have better inlining here.
            reset_positions_offsets_lr(
                w,
                &mut pos,
                &mut prefix_lr_min,
                &mut pos_offset,
                &mut ring_buf,
            );
        }
        // slightly faster than assigning S::splat(u32::MAX)
        let lelem = (val & val_mask) | pos;
        let relem = (!val & val_mask) | pos;
        let elem = (lelem, relem);
        pos += S::splat(1);
        ring_buf.push(elem);
        prefix_lr_min = simd_lr_min(prefix_lr_min, elem);
        // After a chunk has been filled, compute suffix minima.
        if ring_buf.idx() == 0 {
            // Slow case extracted to a function to have better inlining here.
            suffix_lr_minima(&mut ring_buf, w, &mut prefix_lr_min, elem);
        }

        let suffix_lr_min = unsafe { *ring_buf.get_unchecked(ring_buf.idx()) };
        let (lmin, rmin) = simd_lr_min(prefix_lr_min, suffix_lr_min);
        (
            (lmin & pos_mask) + pos_offset,
            (rmin & pos_mask) + pos_offset,
        )
    }
}

#[inline(always)]
fn simd_lr_min((al, ar): (S, S), (bl, br): (S, S)) -> (S, S) {
    (al.min(bl), ar.max(br))
}

#[inline(always)]
fn suffix_lr_minima(
    ring_buf: &mut RingBuf<(S, S)>,
    w: usize,
    prefix_min: &mut (S, S),
    elem: (S, S),
) {
    // Avoid some bounds checks when this function is not inlined.
    unsafe { assert_unchecked(ring_buf.len() == w) };
    unsafe { assert_unchecked(w > 0) };
    let mut suffix_min = ring_buf[w - 1];
    for i in (0..w - 1).rev() {
        suffix_min = simd_lr_min(suffix_min, ring_buf[i]);
        ring_buf[i] = suffix_min;
    }
    *prefix_min = elem;
}

#[inline(always)]
fn reset_positions_offsets_lr(
    w: usize,
    pos: &mut S,
    prefix_min: &mut (S, S),
    pos_offset: &mut S,
    ring_buf: &mut RingBuf<(S, S)>,
) {
    let delta = S::splat((1 << 16) - 2 - w as u32);
    *pos -= delta;
    *pos_offset += delta;
    prefix_min.0 -= delta;
    prefix_min.1 -= delta;
    for x in &mut **ring_buf {
        x.0 -= delta;
        x.1 -= delta;
    }
}
