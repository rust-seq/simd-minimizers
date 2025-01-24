#![feature(portable_simd)]

use itertools::Itertools;

pub mod counting;
pub mod fxhash;
pub mod hash;
pub mod jumping;
pub mod minimizer;
pub mod naive;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx"
))]
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
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx"
))]
pub use nthash::*;
pub use queue::*;
pub use queue_igor::*;
pub use rescan::*;
pub use rescan_daniel::*;
pub use ringbuf::*;
pub use sliding_min::*;
pub use split::*;

use packed_seq::{PackedSeqVec, SeqVec};

pub(crate) const MAXIMUM_K_SIZE: usize = u32::max_value() as usize;

pub trait Max {
    const MAX: Self;
}

impl Max for usize {
    const MAX: usize = usize::MAX;
}
impl Max for u64 {
    const MAX: u64 = u64::MAX;
}

pub fn read_human_genome(chromosomes: usize) -> PackedSeqVec {
    eprintln!("Reading..");
    let start = std::time::Instant::now();
    let mut packed_text = PackedSeqVec::default();
    let Ok(mut reader) = needletail::parse_fastx_file("human-genome.fa") else {
        eprintln!("Did not find human-genome.fa. Add/symlink it to test runtime on it.");
        return PackedSeqVec::default();
    };
    let mut i = 0;
    while let Some(r) = reader.next() {
        let r = r.unwrap();
        eprintln!(
            "Read {:?} of len {:?}",
            std::str::from_utf8(r.id()),
            r.raw_seq().len()
        );
        packed_text.push_ascii(r.raw_seq());
        i += 1;
        if i == chromosomes {
            break;
        }
    }
    eprintln!("Reading & packing took {:?}", start.elapsed());
    packed_text
}
