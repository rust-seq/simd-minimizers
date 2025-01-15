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

mod alex;
mod canonical;
pub mod collect;
mod dedup;
mod intrinsics;
mod linearize;
pub mod minimizer;
pub mod nthash;
mod sliding_min;

pub use dedup::dedup;

pub trait Captures<U> {}
impl<T: ?Sized, U> Captures<U> for T {}
