//! NtHash the kmers in a sequence.
use std::array::from_fn;
use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};

use super::intrinsics;
use crate::S;
use packed_seq::complement_base;
use packed_seq::Seq;
use wide::u32x8;

pub trait Captures<U> {}
impl<T: ?Sized, U> Captures<U> for T {}

/// Original ntHash seed values.
// TODO: Update to guarantee unique hash values for k<=16?
const HASHES_F: [u32; 4] = [
    0x3c8b_fbb3_95c6_0474u64 as u32,
    0x3193_c185_62a0_2b4cu64 as u32,
    0x2032_3ed0_8257_2324u64 as u32,
    0x2955_49f5_4be2_4456u64 as u32,
];

const R: u32 = 25;

type SeedHasher = BuildHasherDefault<DefaultHasher>;

pub trait CharHasher: Clone {
    #[inline(always)]
    fn new_from_val<'s, SEQ: Seq<'s>>(k: usize, _: SEQ) -> Self {
        Self::new::<SEQ>(k)
    }
    #[inline(always)]
    fn new<'s, SEQ: Seq<'s>>(k: usize) -> Self {
        Self::new_with_seed::<SEQ>(k, None)
    }
    fn new_with_seed<'s, SEQ: Seq<'s>>(k: usize, seed: Option<u32>) -> Self;
    fn f(&self, b: u8) -> u32;
    fn c(&self, b: u8) -> u32;
    fn f_rot(&self, b: u8) -> u32;
    fn c_rot(&self, b: u8) -> u32;
    fn simd_f(&self, b: u32x8) -> u32x8;
    fn simd_c(&self, b: u32x8) -> u32x8;
    fn simd_f_rot(&self, b: u32x8) -> u32x8;
    fn simd_c_rot(&self, b: u32x8) -> u32x8;
}

#[derive(Clone)]
pub struct NtHasher {
    f: [u32; 4],
    c: [u32; 4],
    f_rot: [u32; 4],
    c_rot: [u32; 4],
    simd_f: u32x8,
    simd_c: u32x8,
    simd_f_rot: u32x8,
    simd_c_rot: u32x8,
}

impl CharHasher for NtHasher {
    fn new_with_seed<'s, SEQ: Seq<'s>>(k: usize, seed: Option<u32>) -> Self {
        assert_eq!(SEQ::BITS_PER_CHAR, 2);

        let rot = (k as u32 - 1) * R;
        let hasher = SeedHasher::new();
        let f = match seed {
            None => HASHES_F,
            Some(seed) => from_fn(|i| hasher.hash_one(HASHES_F[i] ^ seed) as u32),
        };
        let c = from_fn(|i| f[complement_base(i as u8) as usize]);
        let f_rot = f.map(|h| h.rotate_left(rot));
        let c_rot = c.map(|h| h.rotate_left(rot));
        let idx = [0, 1, 2, 3, 0, 1, 2, 3];
        let simd_f = idx.map(|i| f[i]).into();
        let simd_c = idx.map(|i| c[i]).into();
        let simd_f_rot = idx.map(|i| f_rot[i]).into();
        let simd_c_rot = idx.map(|i| c_rot[i]).into();

        Self {
            f,
            c,
            f_rot,
            c_rot,
            simd_f,
            simd_c,
            simd_f_rot,
            simd_c_rot,
        }
    }

    #[inline(always)]
    fn f(&self, b: u8) -> u32 {
        unsafe { *self.f.get_unchecked(b as usize) }
    }
    #[inline(always)]
    fn c(&self, b: u8) -> u32 {
        unsafe { *self.c.get_unchecked(b as usize) }
    }
    #[inline(always)]
    fn f_rot(&self, b: u8) -> u32 {
        unsafe { *self.f_rot.get_unchecked(b as usize) }
    }
    #[inline(always)]
    fn c_rot(&self, b: u8) -> u32 {
        unsafe { *self.c_rot.get_unchecked(b as usize) }
    }

    #[inline(always)]
    fn simd_f(&self, b: u32x8) -> u32x8 {
        intrinsics::table_lookup(self.simd_f, b)
    }
    #[inline(always)]
    fn simd_c(&self, b: u32x8) -> u32x8 {
        intrinsics::table_lookup(self.simd_c, b)
    }
    #[inline(always)]
    fn simd_f_rot(&self, b: u32x8) -> u32x8 {
        intrinsics::table_lookup(self.simd_f_rot, b)
    }
    #[inline(always)]
    fn simd_c_rot(&self, b: u32x8) -> u32x8 {
        intrinsics::table_lookup(self.simd_c_rot, b)
    }
}

#[derive(Clone)]
pub struct MulHasher {
    rot: u32,
    mul: u32,
}

// Mixing constant.
const C: u32 = 0x517cc1b727220a95u64 as u32;

impl CharHasher for MulHasher {
    fn new_with_seed<'s, SEQ: Seq<'s>>(k: usize, seed: Option<u32>) -> Self {
        Self {
            rot: ((k as u32 - 1) * R) % 32,
            mul: match seed {
                None => C,
                Some(seed) => C ^ ((SeedHasher::new().hash_one(seed) as u32) << 1), // don't change parity,
            },
        }
    }

    #[inline(always)]
    fn f(&self, b: u8) -> u32 {
        (b as u32).wrapping_mul(self.mul)
    }
    #[inline(always)]
    fn c(&self, b: u8) -> u32 {
        (complement_base(b) as u32).wrapping_mul(self.mul)
    }
    #[inline(always)]
    fn f_rot(&self, b: u8) -> u32 {
        (b as u32).wrapping_mul(self.mul).rotate_left(self.rot)
    }
    #[inline(always)]
    fn c_rot(&self, b: u8) -> u32 {
        (complement_base(b) as u32)
            .wrapping_mul(self.mul)
            .rotate_left(self.rot)
    }

    #[inline(always)]
    fn simd_f(&self, b: u32x8) -> u32x8 {
        b * self.mul.into()
    }
    #[inline(always)]
    fn simd_c(&self, b: u32x8) -> u32x8 {
        packed_seq::complement_base_simd(b) * self.mul.into()
    }
    #[inline(always)]
    fn simd_f_rot(&self, b: u32x8) -> u32x8 {
        let r = b * self.mul.into();
        (r << self.rot) | (r >> (32 - self.rot))
    }
    #[inline(always)]
    fn simd_c_rot(&self, b: u32x8) -> u32x8 {
        let r = packed_seq::complement_base_simd(b) * self.mul.into();
        (r << self.rot) | (r >> (32 - self.rot))
    }
}

/// Naively compute the 32-bit NT hash of a single k-mer.
/// When `RC` is false, compute a forward hash.
/// When `RC` is true, compute a canonical hash.
/// TODO: Investigate if we can use CLMUL instruction for speedup.
pub fn nthash_kmer<'s, const RC: bool, H: CharHasher>(seq: impl Seq<'s>) -> u32 {
    let hasher = H::new_from_val(seq.len(), seq);

    let k = seq.len();
    let mut hfw: u32 = 0;
    let mut hrc: u32 = 0;
    seq.iter_bp().for_each(|a| {
        hfw = hfw.rotate_left(R) ^ hasher.f(a);
        if RC {
            hrc = hrc.rotate_right(R) ^ hasher.c(a);
        }
    });
    hfw.wrapping_add(hrc.rotate_left((k as u32 - 1) * R))
}

/// Returns a scalar iterator over the 32-bit NT hashes of all k-mers in the sequence.
/// Prefer `hash_seq_simd`.
///
/// Set `RC` to true for canonical ntHash.
pub fn nthash_seq_scalar<'s, const RC: bool, H: CharHasher>(
    seq: impl Seq<'s>,
    k: usize,
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> + Clone {
    assert!(k > 0);
    let hasher = H::new_from_val(k, seq);

    let mut hfw: u32 = 0;
    let mut hrc: u32 = 0;
    let mut add = seq.iter_bp();
    let remove = seq.iter_bp();
    add.by_ref().take(k - 1).for_each(|a| {
        hfw = hfw.rotate_left(R) ^ hasher.f(a);
        if RC {
            hrc = hrc.rotate_right(R) ^ hasher.c_rot(a);
        }
    });
    add.zip(remove).map(move |(a, r)| {
        let hfw_out = hfw.rotate_left(R) ^ hasher.f(a);
        hfw = hfw_out ^ hasher.f_rot(r);
        if RC {
            let hrc_out = hrc.rotate_right(R) ^ hasher.c_rot(a);
            hrc = hrc_out ^ hasher.c(r);
            hfw_out.wrapping_add(hrc_out)
        } else {
            hfw_out
        }
    })
}

/// Returns a simd-iterator over the 8 chunks 32-bit ntHashes of all k-mers in the sequence.
/// The tail is returned separately.
/// Returned chunks overlap by w-1 hashes. Set w=1 for non-overlapping chunks.
///
/// Set `RC` to true for canonical ntHash.
pub fn nthash_seq_simd<'s, const RC: bool, SEQ: Seq<'s>, H: CharHasher>(
    seq: impl Seq<'s>,
    k: usize,
    w: usize,
    seed: Option<u32>,
) -> (
    impl ExactSizeIterator<Item = S> + Captures<&'s ()> + Clone,
    usize,
) {
    let (add_remove, padding) = seq.par_iter_bp_delayed(k + w - 1, k - 1);

    let mut it = add_remove.map(nthash_mapper::<RC, SEQ, H>(k, w, seed));
    it.by_ref().take(k - 1).for_each(drop);

    (it, padding)
}

/// A function that 'eats' added and removed bases, and returns the updated hash.
/// The distance between them must be k-1, and the first k-1 removed bases must be 0.
/// The first k-1 returned values will be useless.
///
/// Set `RC` to true for canonical ntHash.
pub fn nthash_mapper<'s, const RC: bool, SEQ: Seq<'s>, H: CharHasher>(
    k: usize,
    w: usize,
    seed: Option<u32>,
) -> impl FnMut((S, S)) -> S + Clone {
    let hasher = H::new_with_seed::<SEQ>(k, seed);

    assert!(k > 0);
    assert!(w > 0);
    // Each 128-bit half has a copy of the 4 32-bit hashes.

    let mut fw = 0u32;
    let mut rc = 0u32;
    for _ in 0..k - 1 {
        fw = fw.rotate_left(R) ^ hasher.f(0);
        rc = rc.rotate_right(R) ^ hasher.c_rot(0);
    }

    let mut h_fw = S::splat(fw);
    let mut h_rc = S::splat(rc);

    move |(a, r)| {
        let hfw_out = ((h_fw << R) | (h_fw >> (32 - R))) ^ hasher.simd_f(a);
        h_fw = hfw_out ^ hasher.simd_f_rot(r);
        if RC {
            let hrc_out = ((h_rc >> R) | (h_rc << (32 - R))) ^ hasher.simd_c_rot(a);
            h_rc = hrc_out ^ hasher.simd_c(r);
            // Wrapping SIMD add
            hfw_out + hrc_out
        } else {
            hfw_out
        }
    }
}
