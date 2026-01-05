use crate::S;
use crate::minimizers::SIMD_SKIPPED;
use core::mem::transmute;
use packed_seq::L;

/// Append the values of `x` selected by `mask` to `v`.
#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
#[inline(always)]
pub unsafe fn append_filtered_vals(vals: S, mask: S, v: &mut [u32], write_idx: &mut usize) {
    unsafe {
        for i in 0..L {
            if mask.as_array()[i] != 0 {
                v.as_mut_ptr().add(*write_idx).write(x.as_array()[i]);
                *write_idx += 1;
            }
        }
    }
}

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]`.
#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
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
            if curr != prec && cur != SKIPPED {
                v.as_mut_ptr().add(*write_idx).write(vals[i]);
                *write_idx += 1;
                prec = curr;
            }
        }
    }
}

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]` and `vals2` to `v2[write_idx]`.
#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
#[inline(always)]
pub unsafe fn append_unique_vals_2<const SKIP_MAX: bool>(
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
        let key = transmute(UNIQSHUF[mask]);
        let val = _mm256_permutevar8x32_epi32(transmute(vals), key);
        _mm256_storeu_si256(v.as_mut_ptr().add(*write_idx) as *mut __m256i, val);
        *write_idx += numberofnewvalues;
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
        let vals = transmute(vals);

        let recon = _mm256_blend_epi32(old, transmute(new), 0b01111111);
        let movebyone_mask = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7); // rotate shuffle
        let vec_tmp: S = transmute(_mm256_permutevar8x32_epi32(recon, movebyone_mask));

        let mut mask = vec_tmp.cmp_eq(new);
        if SKIP_MAX {
            // skip everything equal to prev, or equal to MAX.
            mask |= new.cmp_eq(SIMD_SKIPPED);
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
        let vals = transmute(vals);
        let vals2 = transmute(vals2);

        let recon = _mm256_blend_epi32(old, new, 0b01111111);
        let movebyone_mask = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7); // rotate shuffle
        let vec_tmp = _mm256_permutevar8x32_epi32(recon, movebyone_mask);

        let mask = _mm256_movemask_ps(transmute(_mm256_cmpeq_epi32(vec_tmp, new))) as usize;
        let numberofnewvalues = L - mask.count_ones() as usize;
        let key = transmute(UNIQSHUF[mask]);
        let val = _mm256_permutevar8x32_epi32(vals, key);
        _mm256_storeu_si256(v.as_mut_ptr().add(*write_idx) as *mut __m256i, val);
        let val2 = _mm256_permutevar8x32_epi32(vals2, key);
        _mm256_storeu_si256(v2.as_mut_ptr().add(*write_idx) as *mut __m256i, val2);
        *write_idx += numberofnewvalues;
    }
}

/// Append the values of `x` selected by `mask` to `v`.
#[cfg(target_feature = "neon")]
#[inline(always)]
pub unsafe fn append_filtered_vals(vals: S, mask: S, v: &mut [u32], write_idx: &mut usize) {
    unsafe {
        use core::arch::aarch64::{vaddvq_u32, vqtbl2q_u8, vst1_u32_x4};
        use wide::u32x4;

        const POW1: u32x4 = u32x4::new([1, 2, 4, 8]);
        const POW2: u32x4 = u32x4::new([16, 32, 64, 128]);

        let (m1, m2): (u32x4, u32x4) = transmute(mask);
        let m1 = vaddvq_u32(transmute(m1 & POW1));
        let m2 = vaddvq_u32(transmute(m2 & POW2));
        let mask = (m1 | m2) as usize;
        let numberofnewvalues = L - mask.count_ones() as usize;
        let idx = UNIQSHUF_NEON[mask];
        let (i1, i2) = transmute(idx);
        let t = transmute(vals);
        let r1 = vqtbl2q_u8(t, i1);
        let r2 = vqtbl2q_u8(t, i2);
        let val: S = transmute((r1, r2));
        vst1_u32_x4(v.as_mut_ptr().add(*write_idx), transmute(val));
        *write_idx += numberofnewvalues;
    }
}

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]`.
///
/// Somewhat based on Daniel Lemire's blog.
/// <https://lemire.me/blog/2017/04/10/removing-duplicates-from-lists-quickly/>
/// <https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/edfd0e8b809d9a57527a7990c4bb44b9d1d05a69/2017/04/10/removeduplicates.cpp>
#[inline(always)]
#[cfg(target_feature = "neon")]
pub unsafe fn append_unique_vals<const SKIP_MAX: bool>(
    old: S,
    new: S,
    vals: S,
    v: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        use core::arch::aarch64::vqtbl2q_u8;

        const NEW_OLD_MASK: S = S::new([
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            0,
        ]);
        let recon = NEW_OLD_MASK.blend(new, old);
        let (i1, i2) = transmute([
            0x1F1E1D1Cu32,
            0x03020100,
            0x07060504,
            0x0B0A0908,
            0x0F0E0D0C,
            0x13121110,
            0x17161514,
            0x1B1A1918,
        ]);
        let t = transmute(recon);
        let r1 = vqtbl2q_u8(t, i1);
        let r2 = vqtbl2q_u8(t, i2);
        let prec: S = transmute((r1, r2));

        let mut dup = prec.cmp_eq(new);
        if SKIP_MAX {
            dup |= new.cmp_eq(SIMD_SKIPPED);
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
#[inline(always)]
#[cfg(target_feature = "neon")]
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
        use core::arch::aarch64::{vaddvq_u32, vqtbl2q_u8, vst1_u32_x4};
        use wide::u32x4;

        let new_old_mask = S::new([
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            0,
        ]);
        let recon = new_old_mask.blend(new, old);

        // let rotate_idx = S::new([7, 0, 1, 2, 3, 4, 5, 6]);
        // let idx = rotate_idx * S::splat(0x04_04_04_04) + S::splat(0x03_02_01_00);
        // let (i1, i2) = transmute(idx);
        let (i1, i2) = transmute([
            0x1F1E1D1Cu32,
            0x03020100,
            0x07060504,
            0x0B0A0908,
            0x0F0E0D0C,
            0x13121110,
            0x17161514,
            0x1B1A1918,
        ]);
        let t = transmute(recon);
        let r1 = vqtbl2q_u8(t, i1);
        let r2 = vqtbl2q_u8(t, i2);
        let prec: S = transmute((r1, r2));

        let dup = prec.cmp_eq(new);
        let (d1, d2): (u32x4, u32x4) = transmute(dup);
        let pow1 = u32x4::new([1, 2, 4, 8]);
        let pow2 = u32x4::new([16, 32, 64, 128]);
        let m1 = vaddvq_u32(transmute(d1 & pow1));
        let m2 = vaddvq_u32(transmute(d2 & pow2));
        let mask = (m1 | m2) as usize;

        let numberofnewvalues = L - mask.count_ones() as usize;
        let idx = UNIQSHUF_NEON[mask];
        let (i1, i2) = transmute(idx);
        let t = transmute(vals);
        let r1 = vqtbl2q_u8(t, i1);
        let r2 = vqtbl2q_u8(t, i2);
        let val: S = transmute((r1, r2));
        vst1_u32_x4(v.as_mut_ptr().add(*write_idx), transmute(val));
        let t = transmute(vals2);
        let r1 = vqtbl2q_u8(t, i1);
        let r2 = vqtbl2q_u8(t, i2);
        let val2: S = transmute((r1, r2));
        vst1_u32_x4(v2.as_mut_ptr().add(*write_idx), transmute(val2));
        *write_idx += numberofnewvalues;
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

#[cfg(target_feature = "neon")]
#[allow(clippy::erasing_op, clippy::identity_op)]
#[rustfmt::skip]
const UNIQSHUF_NEON: [wide::u8x32; 256] = unsafe {
const M: u32 = 0x04_04_04_04;
const O: u32 = 0x03_02_01_00;
transmute([
0*M+O,1*M+O,2*M+O,3*M+O,4*M+O,5*M+O,6*M+O,7*M+O,
1*M+O,2*M+O,3*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,
2*M+O,3*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,
1*M+O,3*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
3*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,
1*M+O,2*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
2*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
4*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,5*M+O,6*M+O,7*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
5*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,4*M+O,6*M+O,7*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
4*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,6*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
6*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,4*M+O,5*M+O,7*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
4*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,5*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
5*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,4*M+O,7*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
4*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,7*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
7*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,4*M+O,5*M+O,6*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
4*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,5*M+O,6*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
5*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,4*M+O,6*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
4*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,6*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
6*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,4*M+O,5*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
4*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,5*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
5*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,4*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
4*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,3*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,3*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,3*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,3*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,3*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,3*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,3*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
3*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,2*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,2*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,2*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
2*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,1*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
1*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,0*M+O,
])
};

#[cfg(test)]
mod test {
    use super::*;
    use std::time::Instant;
    use wide::u32x8 as S;
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
