use std::{
    array::{self, from_fn},
    cell::RefCell,
    mem::transmute,
};

use packed_seq::S;
use wide::u32x8;

use super::dedup::write_unique_with_old_distinct_vals;

/// Collect a parallel stream into a single vector.
/// Works by taking 8 elements from each stream, and then transposing the SIMD-matrix before writing out the results.
#[inline(always)]
pub fn collect(
    par_head: impl ExactSizeIterator<Item = S>,
    tail: impl ExactSizeIterator<Item = u32>,
) -> Vec<u32> {
    let len = par_head.len();
    let mut v = vec![0; len * 8 + tail.len()];

    let mut m = [unsafe { transmute([0; 8]) }; 8];
    let mut i = 0;
    par_head.for_each(|x| {
        m[i % 8] = x;
        if i % 8 == 7 {
            let t = transpose(m);
            for j in 0..8 {
                unsafe {
                    *v.get_unchecked_mut(j * len + 8 * (i / 8)..j * len + 8 * (i / 8) + 8)
                        .split_first_chunk_mut::<8>()
                        .unwrap()
                        .0 = transmute(t[j]);
                }
            }
        }
        i += 1;
    });

    // Manually write the unfinished parts of length k=i%8.
    let t = transpose(m);
    let k = i % 8;
    for j in 0..8 {
        unsafe {
            v[j * len + 8 * (i / 8)..j * len + 8 * (i / 8) + k]
                .copy_from_slice(&transmute::<_, [u32; 8]>(t[j])[..k]);
        }
    }

    // Manually write the explicit tail.
    for (i, x) in tail.enumerate() {
        v[8 * len + i] = x;
    }

    v
}

thread_local! {
    static CACHE: RefCell<[Vec<u32>; 8]> = RefCell::new(array::from_fn(|_| Vec::new()));
}

/// Collect a parallel stream into a single vector.
/// Works by taking 8 elements from each stream, and then transposing the SIMD-matrix before writing out the results.
///
/// By default (when `SUPER` is false), the output is simply the 32 bit positions of all distinct minimizers.
/// When `SUPER` is true, each u32 is a tuple of `(u16,16)` where the low bits are the position of the minimizer,
/// and the high bits are the index of the stream it first appeared, i.e., the start of its super-k-mer.
/// These positions are mod 2^16. When the window length is <2^16, this is sufficient to recover full positions.
#[inline(always)]
pub fn collect_and_dedup<const SUPER: bool>(
    par_head: impl ExactSizeIterator<Item = S>,
    tail: impl ExactSizeIterator<Item = u32>,
    out_vec: &mut Vec<u32>,
) {
    let len = par_head.len();
    CACHE.with(|v| {
        let mut v = v.borrow_mut();
        for x in v.iter_mut() {
            x.resize(len, 0);
        }
        // let mut v: [Vec<u32>; 8] = array::from_fn(|_| vec![0; len]);

        let mut write_idx = [0; 8];
        // Vec of last pushed elements in each lane.
        let mut old = [unsafe { transmute([u32::MAX; 8]) }; 8];

        let len = par_head.len();
        let lane_offsets: [u32x8; 8] = from_fn(|i| u32x8::splat(((i * len) << 16) as u32));
        let offsets: [u32; 8] = from_fn(|i| (i << 16) as u32);
        let mut offsets: u32x8 = unsafe { transmute(offsets) };

        let mut m = [u32x8::ZERO; 8];
        let mut i = 0;
        par_head.for_each(|x| {
            m[i % 8] = x;
            if i % 8 == 7 {
                let t = transpose(m);
                offsets += u32x8::splat(8 << 16);
                for j in 0..8 {
                    let lane = t[j];
                    let vals = if SUPER {
                        // High 16 bits are the index where the minimizer first becomes minimal.
                        // Low 16 bits are the position of the minimizer itself.
                        (offsets + lane_offsets[j]) | (lane & u32x8::splat(0xFFFF))
                    } else {
                        lane
                    };
                    unsafe {
                        write_unique_with_old_distinct_vals(
                            old[j],
                            transmute(lane),
                            transmute(vals),
                            &mut v[j],
                            &mut write_idx[j],
                        );
                        old[j] = transmute(lane);
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
            let lane = &unsafe { transmute::<_, [u32; 8]>(t[j]) }[..k];
            for x in lane {
                if v[j].last() != Some(x) {
                    v[j].push(*x);
                }
            }
        }

        // Flatten v.
        for lane in v.iter() {
            let mut lane = lane.as_slice();
            while Some(lane[0]) == out_vec.last().copied() {
                lane = &lane[1..];
            }
            out_vec.extend_from_slice(lane);
        }

        // Manually write the dedup'ed explicit tail.
        for x in tail {
            if out_vec.last() != Some(&x) {
                out_vec.push(x);
            }
        }

        // v_flat
    })
}

/// A utility function for creating masks to use with Intel shuffle and
/// permute intrinsics.
///
/// Copied from the standard library, since it is unstable.
const fn _mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

/// Transpose a matrix of 8 SIMD vectors.
/// https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
///
/// TODO: Investigate other transpose functions mentioned there?
#[inline(always)]
fn transpose(m: [S; 8]) -> [S; 8] {
    unsafe {
        use std::arch::x86_64::*;
        let m: [__m256; 8] = transmute(m);
        let x0 = _mm256_unpacklo_ps(m[0], m[1]);
        let x1 = _mm256_unpackhi_ps(m[0], m[1]);
        let x2 = _mm256_unpacklo_ps(m[2], m[3]);
        let x3 = _mm256_unpackhi_ps(m[2], m[3]);
        let x4 = _mm256_unpacklo_ps(m[4], m[5]);
        let x5 = _mm256_unpackhi_ps(m[4], m[5]);
        let x6 = _mm256_unpacklo_ps(m[6], m[7]);
        let x7 = _mm256_unpackhi_ps(m[6], m[7]);
        let y0 = _mm256_shuffle_ps(x0, x2, _mm_shuffle(1, 0, 1, 0));
        let y1 = _mm256_shuffle_ps(x0, x2, _mm_shuffle(3, 2, 3, 2));
        let y2 = _mm256_shuffle_ps(x1, x3, _mm_shuffle(1, 0, 1, 0));
        let y3 = _mm256_shuffle_ps(x1, x3, _mm_shuffle(3, 2, 3, 2));
        let y4 = _mm256_shuffle_ps(x4, x6, _mm_shuffle(1, 0, 1, 0));
        let y5 = _mm256_shuffle_ps(x4, x6, _mm_shuffle(3, 2, 3, 2));
        let y6 = _mm256_shuffle_ps(x5, x7, _mm_shuffle(1, 0, 1, 0));
        let y7 = _mm256_shuffle_ps(x5, x7, _mm_shuffle(3, 2, 3, 2));
        let mut t: [__m256; 8] = [transmute([0; 8]); 8];
        t[0] = _mm256_permute2f128_ps(y0, y4, 0x20);
        t[1] = _mm256_permute2f128_ps(y1, y5, 0x20);
        t[2] = _mm256_permute2f128_ps(y2, y6, 0x20);
        t[3] = _mm256_permute2f128_ps(y3, y7, 0x20);
        t[4] = _mm256_permute2f128_ps(y0, y4, 0x31);
        t[5] = _mm256_permute2f128_ps(y1, y5, 0x31);
        t[6] = _mm256_permute2f128_ps(y2, y6, 0x31);
        t[7] = _mm256_permute2f128_ps(y3, y7, 0x31);
        transmute(t)
    }
}
