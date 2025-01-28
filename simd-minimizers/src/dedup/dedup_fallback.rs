use wide::u32x8 as S;

/// Dedup adjacent `new` values (starting with the last element of `old`).
/// If an element is different from the preceding element, append the corresponding element of `vals` to `v[write_idx]`.
#[inline(always)]
pub unsafe fn append_unique_vals(old: S, new: S, vals: S, v: &mut [u32], write_idx: &mut usize) {
    let old = old.to_array();
    let new = new.to_array();
    let vals = vals.to_array();
    let mut prec = old[7];
    for (i, &curr) in new.iter().enumerate() {
        if curr != prec {
            v.as_mut_ptr().add(*write_idx).write(vals[i]);
            *write_idx += 1;
            prec = curr;
        }
    }
}
