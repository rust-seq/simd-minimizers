#![allow(dead_code)]

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx"
))]
mod dedup_avx;
// #[cfg(not(any(
//     all(
//         any(target_arch = "x86", target_arch = "x86_64"),
//         target_feature = "avx"
//     ),
//     all(target_arch = "aarch64", target_feature = "neon")
// )))]
// mod dedup_fallback;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod dedup_neon;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx"
))]
pub use dedup_avx::*;
// #[cfg(not(any(
//     all(
//         any(target_arch = "x86", target_arch = "x86_64"),
//         target_feature = "avx"
//     ),
//     all(target_arch = "aarch64", target_feature = "neon")
// )))]
// pub use dedup_fallback::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use dedup_neon::*;
