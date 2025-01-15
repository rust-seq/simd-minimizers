#![cfg_attr(
    not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx2"
        ),
        all(target_arch = "aarch64", target_feature = "neon"),
        feature = "hide-simd-warning"
    )),
    deprecated(
        note = "This implementation uses SIMD, make sure you are compiling using `-C target-cpu=native` to get the expected performance. You can hide this warning by enabling the `hide-simd-warning` feature."
    )
)]

mod intrinsics;
pub mod nthash;
pub mod sliding_min;

mod alex;
mod canonical;
pub mod collect;
mod dedup;
mod linearize;
pub mod minimizers;

pub trait Captures<U> {}
impl<T: ?Sized, U> Captures<U> for T {}

// Export a select few functions here.
pub use collect::{collect, collect_and_dedup};
pub use wide::u32x8;

// TODO
use minimizers::*;
use packed_seq::Seq;
pub fn minimizers_collect<'s>(seq: impl Seq<'s>, k: usize, w: usize) -> Vec<u32> {
    let head_tail = minimizers_seq_simd(seq, k, w);
    collect(head_tail)
}

/// Prefer `minimizers_collect_and_dedup`
#[doc(hidden)]
pub fn minimizers_dedup<'s>(seq: impl Seq<'s>, k: usize, w: usize) -> Vec<u32> {
    let head_tail = minimizers_seq_simd(seq, k, w);
    let mut positions = collect(head_tail);
    dedup::dedup(&mut positions);
    positions
}

pub fn minimizers_collect_and_dedup<'s, const SUPER: bool>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    let head_tail = minimizers_seq_simd(seq, k, w);
    collect_and_dedup::<SUPER>(head_tail, out_vec);
}

pub fn canonical_minimizer_collect_and_dedup<'s, const SUPER: bool>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    let head_tail = canonical_minimizers_seq_simd(seq, k, w);
    collect_and_dedup::<SUPER>(head_tail, out_vec);
}
