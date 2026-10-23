use crate::S;
#[allow(unused)]
use core::mem::transmute;
#[cfg(not(any(
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
use seq_hash::packed_seq::L;

/// Append the values of `x` where `mask` is *false* to `v`.
#[cfg(not(any(
    target_feature = "avx2",
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[inline(always)]
pub unsafe fn append_filtered_vals(vals: S, mask: S, v: &mut [u32], write_idx: &mut usize) {
    unsafe {
        let mask = mask.to_array();
        let vals = vals.to_array();
        for i in 0..L {
            if mask[i] == 0 {
                v.as_mut_ptr().add(*write_idx).write(vals[i]);
                *write_idx += 1;
            }
        }
    }
}

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]`.
#[cfg(not(any(
    target_feature = "avx2",
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[inline(always)]
pub unsafe fn append_unique_vals<const SKIP_MAX: bool>(
    old: S,
    new: S,
    vals: S,
    v: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        let old = old.to_array();
        let new = new.to_array();
        let vals = vals.to_array();
        let mut prec = old[7];
        for (i, &curr) in new.iter().enumerate() {
            if curr != prec && (!SKIP_MAX || curr != crate::minimizers::SKIPPED) {
                v.as_mut_ptr().add(*write_idx).write(vals[i]);
                *write_idx += 1;
                prec = curr;
            }
        }
    }
}

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]` and `vals2` to `v2[write_idx]`.
#[cfg(not(any(
    target_feature = "avx2",
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[inline(always)]
pub unsafe fn append_unique_vals_2(
    old: S,
    new: S,
    vals: S,
    vals2: S,
    v: &mut [u32],
    v2: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        let old = old.to_array();
        let new = new.to_array();
        let vals = vals.to_array();
        let vals2 = vals2.to_array();
        let mut prec = old[7];
        for (i, &curr) in new.iter().enumerate() {
            if curr != prec {
                v.as_mut_ptr().add(*write_idx).write(vals[i]);
                v2.as_mut_ptr().add(*write_idx).write(vals2[i]);
                *write_idx += 1;
                prec = curr;
            }
        }
    }
}

/// Append the values of `x` where `mask` is *false* to `v`.
#[cfg(target_feature = "avx2")]
#[inline(always)]
pub unsafe fn append_filtered_vals(vals: S, mask: S, v: &mut [u32], write_idx: &mut usize) {
    unsafe {
        use core::arch::x86_64::*;
        let mask = _mm256_movemask_ps(transmute(mask)) as usize;
        let numberofnewvalues = L - mask.count_ones() as usize;
        let key = UNIQSHUF[mask];
        append_filtered_vals_from_key(vals, key, v, write_idx);
        *write_idx += numberofnewvalues;
    }
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
pub unsafe fn append_filtered_vals_2(
    vals: S,
    vals2: S,
    mask: S,
    v: &mut [u32],
    v2: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        use core::arch::x86_64::*;
        let mask = _mm256_movemask_ps(transmute(mask)) as usize;
        let numberofnewvalues = L - mask.count_ones() as usize;
        let key = UNIQSHUF[mask];
        append_filtered_vals_from_key(vals, key, v, write_idx);
        append_filtered_vals_from_key(vals2, key, v2, write_idx);
        *write_idx += numberofnewvalues;
    }
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn append_filtered_vals_from_key(vals: S, key: S, v: &mut [u32], write_idx: &mut usize) {
    unsafe {
        use core::arch::x86_64::*;
        let val = _mm256_permutevar8x32_epi32(transmute(vals), transmute(key));
        _mm256_storeu_si256(v.as_mut_ptr().add(*write_idx) as *mut __m256i, val);
    }
}

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]`.
///
/// Based on Daniel Lemire's blog.
/// <https://lemire.me/blog/2017/04/10/removing-duplicates-from-lists-quickly/>
/// <https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/edfd0e8b809d9a57527a7990c4bb44b9d1d05a69/2017/04/10/removeduplicates.cpp>
#[cfg(target_feature = "avx2")]
#[inline(always)]
pub unsafe fn append_unique_vals<const SKIP_MAX: bool>(
    old: S,
    new: S,
    vals: S,
    v: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        use core::arch::x86_64::*;

        let old = transmute(old);

        let recon = _mm256_blend_epi32(old, transmute(new), 0b01111111);
        let movebyone_mask = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7); // rotate shuffle
        let vec_tmp: S = transmute(_mm256_permutevar8x32_epi32(recon, movebyone_mask));

        let mut mask = vec_tmp.simd_eq(new);
        if SKIP_MAX {
            // skip everything equal to prev, or equal to MAX.
            mask |= new.simd_eq(crate::minimizers::SIMD_SKIPPED);
        }

        append_filtered_vals(vals, mask, v, write_idx);
    }
}

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]` and `vals2` to `v2[write_idx]`.
///
/// Based on Daniel Lemire's blog.
/// <https://lemire.me/blog/2017/04/10/removing-duplicates-from-lists-quickly/>
/// <https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/edfd0e8b809d9a57527a7990c4bb44b9d1d05a69/2017/04/10/removeduplicates.cpp>
#[cfg(target_feature = "avx2")]
#[inline(always)]
pub unsafe fn append_unique_vals_2(
    old: S,
    new: S,
    vals: S,
    vals2: S,
    v: &mut [u32],
    v2: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        use core::arch::x86_64::*;

        let old = transmute(old);
        let new = transmute(new);

        let recon = _mm256_blend_epi32(old, new, 0b01111111);
        let movebyone_mask = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7); // rotate shuffle
        let vec_tmp = _mm256_permutevar8x32_epi32(recon, movebyone_mask);
        let mask = transmute(_mm256_cmpeq_epi32(vec_tmp, new));

        append_filtered_vals_2(vals, vals2, mask, v, v2, write_idx);
    }
}

/// 128-bit primitives used by the shared NEON / wasm simd128 backend below.
#[cfg(target_feature = "neon")]
mod v128 {
    use core::arch::aarch64::*;

    pub type V = uint32x4_t;

    /// One bit per lane, set where the lane is all ones.
    #[inline(always)]
    pub unsafe fn bitmask(m: V) -> usize {
        const POW: [u32; 4] = [1, 2, 4, 8];
        unsafe { vaddvq_u32(vandq_u32(m, vld1q_u32(POW.as_ptr()))) as usize }
    }

    /// `[a[3], b[0], b[1], b[2]]`.
    #[inline(always)]
    pub unsafe fn shift_in(a: V, b: V) -> V {
        unsafe { vextq_u32::<3>(a, b) }
    }

    /// Byte shuffle of the 32 bytes `[v[0], v[1]]` by `key`; out-of-range indices give 0.
    #[inline(always)]
    pub unsafe fn shuffle(v: [V; 2], key: &[u8; 32]) -> [V; 2] {
        unsafe {
            let t = uint8x16x2_t(vreinterpretq_u8_u32(v[0]), vreinterpretq_u8_u32(v[1]));
            let k: [uint8x16_t; 2] = core::mem::transmute(*key);
            [
                vreinterpretq_u32_u8(vqtbl2q_u8(t, k[0])),
                vreinterpretq_u32_u8(vqtbl2q_u8(t, k[1])),
            ]
        }
    }

    #[inline(always)]
    pub unsafe fn store(ptr: *mut u32, v: V) {
        unsafe { vst1q_u32(ptr, v) }
    }
}

/// 128-bit primitives used by the shared NEON / wasm simd128 backend below.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod v128 {
    use core::arch::wasm32::*;

    pub type V = v128;

    /// One bit per lane, set where the lane is all ones.
    #[inline(always)]
    pub unsafe fn bitmask(m: V) -> usize {
        i32x4_bitmask(m) as usize
    }

    /// `[a[3], b[0], b[1], b[2]]`.
    #[inline(always)]
    pub unsafe fn shift_in(a: V, b: V) -> V {
        i32x4_shuffle::<3, 4, 5, 6>(a, b)
    }

    /// Byte shuffle of the 32 bytes `[v[0], v[1]]` by `key`; out-of-range indices give 0.
    #[inline(always)]
    pub unsafe fn shuffle(v: [V; 2], key: &[u8; 32]) -> [V; 2] {
        unsafe {
            let k: [v128; 2] = core::mem::transmute(*key);
            let sixteen = u8x16_splat(16);
            k.map(|k| {
                v128_or(
                    i8x16_swizzle(v[0], k),
                    i8x16_swizzle(v[1], i8x16_sub(k, sixteen)),
                )
            })
        }
    }

    #[inline(always)]
    pub unsafe fn store(ptr: *mut u32, v: V) {
        unsafe { v128_store(ptr as *mut v128, v) }
    }
}

/// For each of 256 masks of duplicate lanes, a byte shuffle that sends the
/// other u32 lanes to the beginning.
#[cfg(any(
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const UNIQSHUF_256: [[u8; 32]; 256] = {
    let mut t = [[0x80; 32]; 256];
    let mut mask = 0;
    while mask < 256 {
        let mut out = 0;
        let mut lane = 0;
        while lane < 8 {
            if mask & (1 << lane) == 0 {
                let mut b = 0;
                while b < 4 {
                    t[mask][4 * out + b] = (4 * lane + b) as u8;
                    b += 1;
                }
                out += 1;
            }
            lane += 1;
        }
        mask += 1;
    }
    t
};

/// Append the values of `x` where `mask` is *false* to `v`.
#[cfg(any(
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
pub unsafe fn append_filtered_vals(vals: S, mask: S, v: &mut [u32], write_idx: &mut usize) {
    unsafe {
        let vals: [v128::V; 2] = transmute(vals);
        let mask: [v128::V; 2] = transmute(mask);
        let m = v128::bitmask(mask[0]) | v128::bitmask(mask[1]) << 4;
        let r = v128::shuffle(vals, &UNIQSHUF_256[m]);
        let p = v.as_mut_ptr().add(*write_idx);
        v128::store(p, r[0]);
        v128::store(p.add(4), r[1]);
        *write_idx += 8 - m.count_ones() as usize;
    }
}

#[cfg(any(
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
pub unsafe fn append_filtered_vals_2(
    vals: S,
    vals2: S,
    mask: S,
    v: &mut [u32],
    v2: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        let vals: [v128::V; 2] = transmute(vals);
        let vals2: [v128::V; 2] = transmute(vals2);
        let mask: [v128::V; 2] = transmute(mask);
        let m = v128::bitmask(mask[0]) | v128::bitmask(mask[1]) << 4;
        let key = &UNIQSHUF_256[m];
        let r = v128::shuffle(vals, key);
        let r2 = v128::shuffle(vals2, key);
        let p = v.as_mut_ptr().add(*write_idx);
        let p2 = v2.as_mut_ptr().add(*write_idx);
        v128::store(p, r[0]);
        v128::store(p.add(4), r[1]);
        v128::store(p2, r2[0]);
        v128::store(p2.add(4), r2[1]);
        *write_idx += 8 - m.count_ones() as usize;
    }
}

/// The element preceding each lane of `new`: `[old[7], new[0], ..., new[6]]`.
#[cfg(any(
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
unsafe fn preceding(old: S, new: S) -> S {
    unsafe {
        let [_, old_hi]: [v128::V; 2] = transmute(old);
        let [new_lo, new_hi]: [v128::V; 2] = transmute(new);
        transmute([
            v128::shift_in(old_hi, new_lo),
            v128::shift_in(new_lo, new_hi),
        ])
    }
}

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]`.
///
/// Somewhat based on Daniel Lemire's blog.
/// <https://lemire.me/blog/2017/04/10/removing-duplicates-from-lists-quickly/>
/// <https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/edfd0e8b809d9a57527a7990c4bb44b9d1d05a69/2017/04/10/removeduplicates.cpp>
#[cfg(any(
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
pub unsafe fn append_unique_vals<const SKIP_MAX: bool>(
    old: S,
    new: S,
    vals: S,
    v: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        let mut dup = preceding(old, new).simd_eq(new);
        if SKIP_MAX {
            dup |= new.simd_eq(crate::minimizers::SIMD_SKIPPED);
        }
        append_filtered_vals(vals, dup, v, write_idx);
    }
}

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]` and `vals2` to `v2[write_idx]`.
///
/// Somewhat based on Daniel Lemire's blog.
/// <https://lemire.me/blog/2017/04/10/removing-duplicates-from-lists-quickly/>
/// <https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/edfd0e8b809d9a57527a7990c4bb44b9d1d05a69/2017/04/10/removeduplicates.cpp>
#[cfg(any(
    target_feature = "neon",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
pub unsafe fn append_unique_vals_2(
    old: S,
    new: S,
    vals: S,
    vals2: S,
    v: &mut [u32],
    v2: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        let dup = preceding(old, new).simd_eq(new);
        append_filtered_vals_2(vals, vals2, dup, v, v2, write_idx);
    }
}

/// For each of 256 masks of which elements are different than their predecessor,
/// a shuffle that sends those new elements to the beginning.
#[cfg(target_feature = "avx2")]
#[rustfmt::skip]
const UNIQSHUF: [S; 256] = unsafe {transmute([
0,1,2,3,4,5,6,7,
1,2,3,4,5,6,7,0,
0,2,3,4,5,6,7,0,
2,3,4,5,6,7,0,0,
0,1,3,4,5,6,7,0,
1,3,4,5,6,7,0,0,
0,3,4,5,6,7,0,0,
3,4,5,6,7,0,0,0,
0,1,2,4,5,6,7,0,
1,2,4,5,6,7,0,0,
0,2,4,5,6,7,0,0,
2,4,5,6,7,0,0,0,
0,1,4,5,6,7,0,0,
1,4,5,6,7,0,0,0,
0,4,5,6,7,0,0,0,
4,5,6,7,0,0,0,0,
0,1,2,3,5,6,7,0,
1,2,3,5,6,7,0,0,
0,2,3,5,6,7,0,0,
2,3,5,6,7,0,0,0,
0,1,3,5,6,7,0,0,
1,3,5,6,7,0,0,0,
0,3,5,6,7,0,0,0,
3,5,6,7,0,0,0,0,
0,1,2,5,6,7,0,0,
1,2,5,6,7,0,0,0,
0,2,5,6,7,0,0,0,
2,5,6,7,0,0,0,0,
0,1,5,6,7,0,0,0,
1,5,6,7,0,0,0,0,
0,5,6,7,0,0,0,0,
5,6,7,0,0,0,0,0,
0,1,2,3,4,6,7,0,
1,2,3,4,6,7,0,0,
0,2,3,4,6,7,0,0,
2,3,4,6,7,0,0,0,
0,1,3,4,6,7,0,0,
1,3,4,6,7,0,0,0,
0,3,4,6,7,0,0,0,
3,4,6,7,0,0,0,0,
0,1,2,4,6,7,0,0,
1,2,4,6,7,0,0,0,
0,2,4,6,7,0,0,0,
2,4,6,7,0,0,0,0,
0,1,4,6,7,0,0,0,
1,4,6,7,0,0,0,0,
0,4,6,7,0,0,0,0,
4,6,7,0,0,0,0,0,
0,1,2,3,6,7,0,0,
1,2,3,6,7,0,0,0,
0,2,3,6,7,0,0,0,
2,3,6,7,0,0,0,0,
0,1,3,6,7,0,0,0,
1,3,6,7,0,0,0,0,
0,3,6,7,0,0,0,0,
3,6,7,0,0,0,0,0,
0,1,2,6,7,0,0,0,
1,2,6,7,0,0,0,0,
0,2,6,7,0,0,0,0,
2,6,7,0,0,0,0,0,
0,1,6,7,0,0,0,0,
1,6,7,0,0,0,0,0,
0,6,7,0,0,0,0,0,
6,7,0,0,0,0,0,0,
0,1,2,3,4,5,7,0,
1,2,3,4,5,7,0,0,
0,2,3,4,5,7,0,0,
2,3,4,5,7,0,0,0,
0,1,3,4,5,7,0,0,
1,3,4,5,7,0,0,0,
0,3,4,5,7,0,0,0,
3,4,5,7,0,0,0,0,
0,1,2,4,5,7,0,0,
1,2,4,5,7,0,0,0,
0,2,4,5,7,0,0,0,
2,4,5,7,0,0,0,0,
0,1,4,5,7,0,0,0,
1,4,5,7,0,0,0,0,
0,4,5,7,0,0,0,0,
4,5,7,0,0,0,0,0,
0,1,2,3,5,7,0,0,
1,2,3,5,7,0,0,0,
0,2,3,5,7,0,0,0,
2,3,5,7,0,0,0,0,
0,1,3,5,7,0,0,0,
1,3,5,7,0,0,0,0,
0,3,5,7,0,0,0,0,
3,5,7,0,0,0,0,0,
0,1,2,5,7,0,0,0,
1,2,5,7,0,0,0,0,
0,2,5,7,0,0,0,0,
2,5,7,0,0,0,0,0,
0,1,5,7,0,0,0,0,
1,5,7,0,0,0,0,0,
0,5,7,0,0,0,0,0,
5,7,0,0,0,0,0,0,
0,1,2,3,4,7,0,0,
1,2,3,4,7,0,0,0,
0,2,3,4,7,0,0,0,
2,3,4,7,0,0,0,0,
0,1,3,4,7,0,0,0,
1,3,4,7,0,0,0,0,
0,3,4,7,0,0,0,0,
3,4,7,0,0,0,0,0,
0,1,2,4,7,0,0,0,
1,2,4,7,0,0,0,0,
0,2,4,7,0,0,0,0,
2,4,7,0,0,0,0,0,
0,1,4,7,0,0,0,0,
1,4,7,0,0,0,0,0,
0,4,7,0,0,0,0,0,
4,7,0,0,0,0,0,0,
0,1,2,3,7,0,0,0,
1,2,3,7,0,0,0,0,
0,2,3,7,0,0,0,0,
2,3,7,0,0,0,0,0,
0,1,3,7,0,0,0,0,
1,3,7,0,0,0,0,0,
0,3,7,0,0,0,0,0,
3,7,0,0,0,0,0,0,
0,1,2,7,0,0,0,0,
1,2,7,0,0,0,0,0,
0,2,7,0,0,0,0,0,
2,7,0,0,0,0,0,0,
0,1,7,0,0,0,0,0,
1,7,0,0,0,0,0,0,
0,7,0,0,0,0,0,0,
7,0,0,0,0,0,0,0,
0,1,2,3,4,5,6,0,
1,2,3,4,5,6,0,0,
0,2,3,4,5,6,0,0,
2,3,4,5,6,0,0,0,
0,1,3,4,5,6,0,0,
1,3,4,5,6,0,0,0,
0,3,4,5,6,0,0,0,
3,4,5,6,0,0,0,0,
0,1,2,4,5,6,0,0,
1,2,4,5,6,0,0,0,
0,2,4,5,6,0,0,0,
2,4,5,6,0,0,0,0,
0,1,4,5,6,0,0,0,
1,4,5,6,0,0,0,0,
0,4,5,6,0,0,0,0,
4,5,6,0,0,0,0,0,
0,1,2,3,5,6,0,0,
1,2,3,5,6,0,0,0,
0,2,3,5,6,0,0,0,
2,3,5,6,0,0,0,0,
0,1,3,5,6,0,0,0,
1,3,5,6,0,0,0,0,
0,3,5,6,0,0,0,0,
3,5,6,0,0,0,0,0,
0,1,2,5,6,0,0,0,
1,2,5,6,0,0,0,0,
0,2,5,6,0,0,0,0,
2,5,6,0,0,0,0,0,
0,1,5,6,0,0,0,0,
1,5,6,0,0,0,0,0,
0,5,6,0,0,0,0,0,
5,6,0,0,0,0,0,0,
0,1,2,3,4,6,0,0,
1,2,3,4,6,0,0,0,
0,2,3,4,6,0,0,0,
2,3,4,6,0,0,0,0,
0,1,3,4,6,0,0,0,
1,3,4,6,0,0,0,0,
0,3,4,6,0,0,0,0,
3,4,6,0,0,0,0,0,
0,1,2,4,6,0,0,0,
1,2,4,6,0,0,0,0,
0,2,4,6,0,0,0,0,
2,4,6,0,0,0,0,0,
0,1,4,6,0,0,0,0,
1,4,6,0,0,0,0,0,
0,4,6,0,0,0,0,0,
4,6,0,0,0,0,0,0,
0,1,2,3,6,0,0,0,
1,2,3,6,0,0,0,0,
0,2,3,6,0,0,0,0,
2,3,6,0,0,0,0,0,
0,1,3,6,0,0,0,0,
1,3,6,0,0,0,0,0,
0,3,6,0,0,0,0,0,
3,6,0,0,0,0,0,0,
0,1,2,6,0,0,0,0,
1,2,6,0,0,0,0,0,
0,2,6,0,0,0,0,0,
2,6,0,0,0,0,0,0,
0,1,6,0,0,0,0,0,
1,6,0,0,0,0,0,0,
0,6,0,0,0,0,0,0,
6,0,0,0,0,0,0,0,
0,1,2,3,4,5,0,0,
1,2,3,4,5,0,0,0,
0,2,3,4,5,0,0,0,
2,3,4,5,0,0,0,0,
0,1,3,4,5,0,0,0,
1,3,4,5,0,0,0,0,
0,3,4,5,0,0,0,0,
3,4,5,0,0,0,0,0,
0,1,2,4,5,0,0,0,
1,2,4,5,0,0,0,0,
0,2,4,5,0,0,0,0,
2,4,5,0,0,0,0,0,
0,1,4,5,0,0,0,0,
1,4,5,0,0,0,0,0,
0,4,5,0,0,0,0,0,
4,5,0,0,0,0,0,0,
0,1,2,3,5,0,0,0,
1,2,3,5,0,0,0,0,
0,2,3,5,0,0,0,0,
2,3,5,0,0,0,0,0,
0,1,3,5,0,0,0,0,
1,3,5,0,0,0,0,0,
0,3,5,0,0,0,0,0,
3,5,0,0,0,0,0,0,
0,1,2,5,0,0,0,0,
1,2,5,0,0,0,0,0,
0,2,5,0,0,0,0,0,
2,5,0,0,0,0,0,0,
0,1,5,0,0,0,0,0,
1,5,0,0,0,0,0,0,
0,5,0,0,0,0,0,0,
5,0,0,0,0,0,0,0,
0,1,2,3,4,0,0,0,
1,2,3,4,0,0,0,0,
0,2,3,4,0,0,0,0,
2,3,4,0,0,0,0,0,
0,1,3,4,0,0,0,0,
1,3,4,0,0,0,0,0,
0,3,4,0,0,0,0,0,
3,4,0,0,0,0,0,0,
0,1,2,4,0,0,0,0,
1,2,4,0,0,0,0,0,
0,2,4,0,0,0,0,0,
2,4,0,0,0,0,0,0,
0,1,4,0,0,0,0,0,
1,4,0,0,0,0,0,0,
0,4,0,0,0,0,0,0,
4,0,0,0,0,0,0,0,
0,1,2,3,0,0,0,0,
1,2,3,0,0,0,0,0,
0,2,3,0,0,0,0,0,
2,3,0,0,0,0,0,0,
0,1,3,0,0,0,0,0,
1,3,0,0,0,0,0,0,
0,3,0,0,0,0,0,0,
3,0,0,0,0,0,0,0,
0,1,2,0,0,0,0,0,
1,2,0,0,0,0,0,0,
0,2,0,0,0,0,0,0,
2,0,0,0,0,0,0,0,
0,1,0,0,0,0,0,0,
1,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
])};

#[cfg(test)]
mod test {
    use super::*;
    use seq_hash::packed_seq::u32x8 as S;
    use std::time::Instant;
    const L: usize = 8;

    #[test]
    fn test_append_unique_vals() {
        let len = 1 << 20;
        for max in [len / 10, len / 3, len, len * 3, len * 10] {
            // 1M random numbers up to max.
            let mut v: Vec<u32> = (0..len).map(|_| rand::random::<u32>() % max).collect();
            v.sort();

            let mut v1 = v.clone();
            let start = Instant::now();
            v1.dedup();
            eprintln!("dedup_std {} in {:?}", max, start.elapsed());

            const IDX: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let len = len as usize;
            let chunks: Vec<S> = (0..(len / L))
                .map(|i| S::new(IDX.map(|j| v[i * L + j])))
                .collect();
            let mut old = S::MAX;
            let mut v2 = Vec::with_capacity(len);
            let mut write_idx = 0;
            let start = Instant::now();
            for new in chunks {
                unsafe {
                    append_unique_vals::<false>(old, new, new, &mut v2, &mut write_idx);
                }
                old = new;
            }
            eprintln!("dedup_new {} in {:?}", max, start.elapsed());
            unsafe {
                v2.set_len(write_idx);
            }
            assert_eq!(v1, v2, "Failure for\n      : {v:?}");
        }
    }
}
