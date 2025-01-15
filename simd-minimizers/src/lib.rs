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

// Private modules.
mod dedup;
mod intrinsics;

// Public modules.
pub mod canonical;
pub mod collect;
pub mod minimizers;
pub mod nthash;
pub mod sliding_min;

// TODO: Old and in-development modules.
// mod anti_lex;
// mod linearize;

// Export a select few functions here.
pub use collect::{collect, collect_and_dedup};
use itertools::Itertools;
use minimizers::{
    canonical_minimizers_seq_scalar, canonical_minimizers_seq_simd, minimizers_seq_scalar,
    minimizers_seq_simd,
};
pub use packed_seq;
use packed_seq::{PackedSeq, Seq};

/// Deduplicated positions of all minimizers in the sequence, using SIMD.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
pub fn minimizer_positions<'s>(seq: PackedSeq<'s>, k: usize, w: usize, out_vec: &mut Vec<u32>) {
    let head_tail = minimizers_seq_simd(seq, k, w);
    collect_and_dedup::<false>(head_tail, out_vec);
}

/// Deduplicated positions of all canonical minimizers in the sequence, using SIMD.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
/// l=w+k-1 must be odd to determine the strand of each window.
pub fn canonical_minimizer_positions<'s>(
    seq: PackedSeq<'s>,
    k: usize,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    let head_tail = canonical_minimizers_seq_simd(seq, k, w);
    collect_and_dedup::<false>(head_tail, out_vec);
}

/// Deduplicated positions of all minimizers in the sequence, not using SIMD.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
/// This scalar version can be faster for sequences known to be short.
pub fn minimizer_positions_scalar<'s>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    out_vec.extend(minimizers_seq_scalar(seq, k, w).dedup());
}

/// Deduplicated positions of all canonical minimizers in the sequence, not using SIMD.
///
/// Positions are appended to a reusable `out_vec` to avoid allocations.
/// l=w+k-1 must be odd to determine the strand of each window.
/// This scalar version can be faster for sequences known to be short.
pub fn canonical_minimizer_positions_scalar<'s>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
    out_vec: &mut Vec<u32>,
) {
    out_vec.extend(canonical_minimizers_seq_scalar(seq, k, w).dedup());
}
