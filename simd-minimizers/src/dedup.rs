use std::{arch::x86_64::__m256i, mem::transmute};

const WIDTH: usize = 256 / 32;

// Slightly modified dedup based on the version of Daniel Lemire's blog.
// https://lemire.me/blog/2017/04/10/removing-duplicates-from-lists-quickly/
// https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/edfd0e8b809d9a57527a7990c4bb44b9d1d05a69/2017/04/10/removeduplicates.cpp
//
// Modification:
// Instead of blending in the 'old' value just before a window and then rotating the lanes,
// we simply do an unaligned read shifted by one value.
// This causes some issues when the first value of that read was overwritten already,
// and so we do this read one iteration ahead, as `nextprev`.
pub fn dedup(v: &mut Vec<u32>) {
    unsafe {
        use std::arch::x86_64::*;
        if v.is_empty() {
            return;
        }
        let mut write_idx = 1;
        let mut read_idx = 1;

        let mut prev = _mm256_loadu_si256(v.as_ptr().add(read_idx - 1) as *const __m256i);
        while read_idx + 2 * WIDTH - 1 <= v.len() {
            // The next prev is read one iteration early to avoid reading data that was just modified.
            let nextprev =
                _mm256_loadu_si256(v.as_ptr().add(read_idx - 1 + WIDTH) as *const __m256i);
            let new = _mm256_loadu_si256(v.as_ptr().add(read_idx) as *const __m256i);
            write_unique_with_prev(prev, new, v, &mut write_idx);
            prev = nextprev;
            read_idx += WIDTH;
        }
        let mut oldv = v[write_idx - 1];
        while read_idx < v.len() {
            let newv = *v.get_unchecked(read_idx);
            if newv != oldv {
                *v.get_unchecked_mut(write_idx) = newv;
                write_idx += 1;
            }
            oldv = newv;
            read_idx += 1;
        }
        v.resize(write_idx, 0);
    }
}

fn write_unique_with_prev(prev: __m256i, new: __m256i, v: &mut [u32], write_idx: &mut usize) {
    unsafe {
        use std::arch::x86_64::*;
        let m = _mm256_movemask_ps(transmute(_mm256_cmpeq_epi32(prev, new))) as usize;
        let numberofnewvalues = WIDTH - m.count_ones() as usize;
        let key = UNIQSHUF[m];
        let val = _mm256_permutevar8x32_epi32(new, key);
        _mm256_storeu_si256(v.as_mut_ptr().add(*write_idx) as *mut __m256i, val);
        *write_idx += numberofnewvalues;
    }
}

#[allow(unused)]
#[inline(always)]
pub fn write_unique_with_old(old: __m256i, new: __m256i, v: &mut [u32], write_idx: &mut usize) {
    write_unique_with_old_distinct_vals(old, new, new, v, write_idx);
}

/// Compare adjacent `new` values (starting with the last element of `old`), and
/// if an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]`.
#[inline(always)]
pub fn write_unique_with_old_distinct_vals(
    old: __m256i,
    new: __m256i,
    vals: __m256i,
    v: &mut [u32],
    write_idx: &mut usize,
) {
    unsafe {
        use std::arch::x86_64::*;

        let recon = _mm256_blend_epi32(old, new, 0b01111111);
        let movebyone_mask = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7); // rotate shuffle
        let vec_tmp = _mm256_permutevar8x32_epi32(recon, movebyone_mask);

        let m = _mm256_movemask_ps(transmute(_mm256_cmpeq_epi32(vec_tmp, new))) as usize;
        let numberofnewvalues = WIDTH - m.count_ones() as usize;
        let key = UNIQSHUF[m];
        let val = _mm256_permutevar8x32_epi32(vals, key);
        _mm256_storeu_si256(v.as_mut_ptr().add(*write_idx) as *mut __m256i, val);
        *write_idx += numberofnewvalues;
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    #[test]
    fn dedup() {
        let len = 10_000_000;
        for max in [len / 10, len / 3, len, len * 3, len * 10] {
            // 1M random numbers up to max.
            let mut v: Vec<u32> = (0..len).map(|_| rand::random::<u32>() % max).collect();
            v.sort();

            let mut v2 = v.clone();
            let start = Instant::now();
            v2.dedup();
            eprintln!("dedup_std {} in {:?}", max, start.elapsed());

            let mut v3 = v.clone();
            let start = Instant::now();
            super::dedup(&mut v3);
            eprintln!("dedup_new {} in {:?}", max, start.elapsed());
            assert_eq!(v2, v3, "Failure for\n      : {v:?}");
        }
    }
}

/// For each of 256 masks of which elements are different than their predecessor,
/// a shuffle that sends those new elements to the beginning.
#[rustfmt::skip]
const UNIQSHUF: [__m256i; 256] = unsafe {transmute([
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
