#![feature(portable_simd)]

use itertools::Itertools;

pub mod counting;
pub mod fxhash;
pub mod hash;
pub mod jumping;
pub mod minimizer;
pub mod naive;
pub mod nthash;
pub mod queue;
pub mod queue_igor;
pub mod rescan;
pub mod rescan_daniel;
pub mod ringbuf;
pub mod sliding_min;
pub mod split;

pub use fxhash::*;
pub use hash::*;
pub use jumping::*;
pub use minimizer::*;
pub use naive::*;
pub use nthash::*;
pub use queue::*;
pub use queue_igor::*;
pub use rescan::*;
pub use rescan_daniel::*;
pub use ringbuf::*;
pub use sliding_min::*;
pub use split::*;

pub trait Max {
    const MAX: Self;
}

impl Max for usize {
    const MAX: usize = usize::MAX;
}
impl Max for u64 {
    const MAX: u64 = u64::MAX;
}
