//! Collect (and dedup) SIMD-iterator values into a flat `Vec<u32>`.
use std::{
    array::{self, from_fn},
    cell::RefCell,
    mem::transmute,
};

use crate::S;
use packed_seq::L;
use std::simd::u32x16 as u32x8;

use packed_seq::intrinsics::transpose;

/// Convenience wrapper around `collect_into`.
pub fn collect((par_head, padding): (impl ExactSizeIterator<Item = S>, usize)) -> Vec<u32> {
    let mut v = vec![];
    collect_into((par_head, padding), &mut v);
    v
}

/// Collect a SIMD-iterator into a single flat vector.
/// Works by taking 8 elements from each stream, and transposing this SIMD-matrix before writing out the results.
/// The `tail` is appended at the end.
#[inline(always)]
pub fn collect_into(
    (par_head, padding): (impl ExactSizeIterator<Item = S>, usize),
    out_vec: &mut Vec<u32>,
) {
    let len = par_head.len();
    out_vec.resize(len * L, 0);

    let mut m = [unsafe { transmute([0; L]) }; L];
    let mut i = 0;
    par_head.for_each(|x| {
        m[i % L] = x;
        if i % L == L - 1 {
            let t = transpose(m);
            for j in 0..L {
                unsafe {
                    *out_vec
                        .get_unchecked_mut(j * len + L * (i / L)..)
                        .split_first_chunk_mut::<L>()
                        .unwrap()
                        .0 = transmute(t[j]);
                }
            }
        }
        i += 1;
    });

    // Manually write the unfinished parts of length k=i%L.
    let t = transpose(m);
    let k = i % L;
    for j in 0..L {
        unsafe {
            out_vec[j * len + L * (i / L)..j * len + L * (i / L) + k]
                .copy_from_slice(&transmute::<_, [u32; L]>(t[j])[..k]);
        }
    }

    out_vec.resize(out_vec.len() - padding, 0);
}

thread_local! {
    static CACHE: RefCell<[Vec<u32>; L]> = RefCell::new(array::from_fn(|_| Vec::new()));
}

/// Convenience wrapper around `collect_and_dedup_into`.
pub fn collect_and_dedup<const SUPER: bool>(
    (par_head, padding): (impl ExactSizeIterator<Item = S>, usize),
) -> Vec<u32> {
    let mut v = vec![];
    collect_and_dedup_into::<SUPER>((par_head, padding), &mut v);
    v
}

/// Collect a SIMD-iterator into a single vector, and duplicate adjacent equal elements.
/// Works by taking 8 elements from each stream, and then transposing the SIMD-matrix before writing out the results.
///
/// By default (when `SUPER` is false), the output is simply the deduplicated input values.
/// When `SUPER` is true, each returned `u32` is a tuple of `(u16,16)` where the low bits are those of the input value,
/// and the high bits are the index of the stream it first appeared, i.e., the start of its super-k-mer.
/// These positions are mod 2^16. When the window length is <2^16, this is sufficient to recover full super-k-mers.
#[inline(always)]
pub fn collect_and_dedup_into<const SUPER: bool>(
    (par_head, padding): (impl ExactSizeIterator<Item = S>, usize),
    out_vec: &mut Vec<u32>,
) {
    CACHE.with(|v| {
        let mut v = v.borrow_mut();

        let mut write_idx = [0; L];
        // Vec of last pushed elements in each lane.
        let mut old = [unsafe { transmute([u32::MAX; 8]) }; L];

        let len = par_head.len();
        let lane_offsets: [u32x8; L] = from_fn(|i| u32x8::splat(((i * len) << 16) as u32));
        let offsets: [u32; L] = from_fn(|i| (i << 16) as u32);
        let mut offsets: u32x8 = unsafe { transmute(offsets) };

        let mut mask = u32x8::default();
        let mut padding_i = 0;
        let mut padding_idx = 0;
        assert!(padding <= L * len, "padding {padding} <= L {L} * len {len}");
        let mut remaining_padding = padding;
        for i in (0..L).rev() {
            if remaining_padding >= len {
                mask.as_mut_array()[i] = u32::MAX;
                remaining_padding -= len;
                continue;
            }
            padding_i = len - remaining_padding;
            padding_idx = i;
            break;
        }

        let mut m = [u32x8::default(); L];
        let mut i = 0;
        par_head.for_each(|x| {
            if i == padding_i {
                mask.as_mut_array()[padding_idx] = u32::MAX;
            }
            let x = x | mask;
            m[i % L] = x;
            if i % L == L - 1 {
                let t = transpose(m);
                offsets += u32x8::splat((L as u32) << 16);
                for j in 0..8 {
                    let lane = t[j];
                    let vals = if SUPER {
                        // High 16 bits are the index where the minimizer first becomes minimal.
                        // Low 16 bits are the position of the minimizer itself.
                        (offsets + lane_offsets[j]) | (lane & u32x8::splat(0xFFFF))
                    } else {
                        lane
                    };
                    if write_idx[j] + L > v[j].len() {
                        let new_len = v[j].len() + 1024;
                        v[j].resize(new_len, 0);
                    }
                    unsafe {
                        let lane0 = transmute(*lane.as_array().split_first_chunk::<8>().unwrap().0);
                        let vals0 = transmute(*vals.as_array().split_first_chunk::<8>().unwrap().0);
                        let lane1 = transmute(*lane.as_array().split_last_chunk::<8>().unwrap().1);
                        let vals1 = transmute(*vals.as_array().split_last_chunk::<8>().unwrap().1);
                        crate::intrinsics::append_unique_vals(
                            old[j],
                            lane0,
                            vals0,
                            &mut v[j],
                            &mut write_idx[j],
                        );
                        old[j] = lane0;
                        crate::intrinsics::append_unique_vals(
                            old[j],
                            lane1,
                            vals1,
                            &mut v[j],
                            &mut write_idx[j],
                        );
                        old[j] = lane1;
                    }
                }
            }
            i += 1;
        });

        for j in 0..8 {
            v[j].truncate(write_idx[j]);
        }

        // Manually write the unfinished parts of length k=i%8.
        let t = transpose(m);
        let k = i % 8;
        for j in 0..8 {
            let lane = &unsafe { transmute::<_, [u32; L]>(t[j]) }[..k];
            for x in lane {
                if v[j].last() != Some(x) {
                    v[j].push(*x);
                }
            }
        }

        // Flatten v.
        for lane in v.iter() {
            let mut lane = lane.as_slice();
            while !lane.is_empty() && Some(lane[0]) == out_vec.last().copied() {
                lane = &lane[1..];
            }
            out_vec.extend_from_slice(lane);
        }

        // If we had padding, pop the last element.
        if out_vec.last() == Some(&u32::MAX) {
            assert!(padding > 0);
            out_vec.pop();
        }
    })
}
