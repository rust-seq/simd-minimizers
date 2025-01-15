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
pub mod minimizer;

pub trait Captures<U> {}
impl<T: ?Sized, U> Captures<U> for T {}

// Export a select few functions here.
pub use collect::{collect, collect_and_dedup};
pub use wide::u32x8;
