//! This code is adapted from the nthash crate:
use std::{
    arch::x86_64::_mm256_permutevar_ps,
    array::from_fn,
    simd::{ptr::SimdConstPtr, Simd},
};

use super::*;

#[derive(Clone, Copy, Debug)]
pub struct NtHash;
impl Hasher for NtHash {
    type Out = u64;
    #[inline(always)]
    fn hash(&self, t: &[u8]) -> u64 {
        ::nthash::ntf64(t, 0, t.len())
    }
    #[inline(always)]
    fn hash_kmers(&self, k: usize, t: &[u8]) -> impl Iterator<Item = Self::Out> {
        NtHashIt::new(t, k).unwrap()
    }
}

pub(crate) const MAXIMUM_K_SIZE: usize = u32::max_value() as usize;

const H_LOOKUP: [u64; 256] = {
    let mut lookup = [1; 256];
    lookup[b'A' as usize] = 0x3c8b_fbb3_95c6_0474;
    lookup[b'C' as usize] = 0x3193_c185_62a0_2b4c;
    lookup[b'G' as usize] = 0x2032_3ed0_8257_2324;
    lookup[b'T' as usize] = 0x2955_49f5_4be2_4456;
    lookup[b'N' as usize] = 0;
    lookup
};

#[inline(always)]
fn h(c: u8) -> u64 {
    H_LOOKUP[c as usize]
}

#[derive(Debug)]
pub struct NtHashIt<'a> {
    seq: &'a [u8],
    k: usize,
    fh: u64,
    current_idx: usize,
    max_idx: usize,
    rot_k: [u64; 256],
}

impl<'a> NtHashIt<'a> {
    /// Creates a new NtHashIt with internal state properly initialized.
    #[inline(always)]
    pub fn new(seq: &'a [u8], k: usize) -> Option<Self> {
        if k > seq.len() {
            return None;
        }
        if k > MAXIMUM_K_SIZE {
            return None;
        }

        let mut fh = 0;
        for (i, v) in seq[0..k].iter().enumerate() {
            fh ^= h(*v).rotate_left((k - i - 1) as u32);
        }

        Some(Self {
            seq,
            k,
            fh,
            current_idx: 0,
            max_idx: seq.len() - k + 1,
            rot_k: H_LOOKUP.map(|x| x.rotate_left(k as u32)),
        })
    }
}

impl<'a> Iterator for NtHashIt<'a> {
    type Item = u64;

    #[inline(always)]
    fn next(&mut self) -> Option<u64> {
        if self.current_idx == self.max_idx {
            return None;
        };

        if self.current_idx != 0 {
            let i = self.current_idx - 1;
            let seqi = unsafe { *self.seq.get_unchecked(i) };
            let seqk = unsafe { *self.seq.get_unchecked(i + self.k) };

            self.fh = self.fh.rotate_left(1) ^ self.rot_k[seqi as usize] ^ h(seqk);
        }

        self.current_idx += 1;
        Some(self.fh)
    }
}

pub struct NtHashPar<const L: usize>;

impl<const L: usize> ParHasher<L> for NtHashPar<L> {
    type Out = u64;
    #[inline(always)]
    fn hash_kmers(&self, k: usize, t: &[u8]) -> impl Iterator<Item = [Self::Out; L]> {
        NtHashParIt::new(t, k).unwrap()
    }
}

#[derive(Debug)]
pub struct NtHashParIt<'a, const L: usize> {
    seq: &'a [u8],
    n: usize,
    k: usize,
    fh: [u64; L],
    current_idx: usize,
    rot_k: [u64; 256],
}

impl<'a, const L: usize> NtHashParIt<'a, L> {
    /// Creates a new NtHashIt with internal state properly initialized.
    #[inline(always)]
    pub fn new(seq: &'a [u8], k: usize) -> Option<Self> {
        if k > seq.len() {
            return None;
        }
        if k > MAXIMUM_K_SIZE {
            return None;
        }
        let num_kmers = seq.len() - k + 1;
        if num_kmers % L != 0 {
            return None;
        }
        let n = num_kmers / L;

        let fh = from_fn(|l| {
            let mut fh = 0;
            for (i, v) in seq[l * n..l * n + k].iter().enumerate() {
                fh ^= h(*v).rotate_left((k - i - 1) as u32);
            }
            fh
        });

        Some(Self {
            seq,
            n,
            k,
            fh,
            current_idx: 0,
            rot_k: H_LOOKUP.map(|x| x.rotate_left(k as u32)),
        })
    }
}

impl<'a, const L: usize> Iterator for NtHashParIt<'a, L> {
    type Item = [u64; L];

    #[inline(always)]
    fn next(&mut self) -> Option<[u64; L]> {
        if self.current_idx == self.n {
            return None;
        };

        if self.current_idx != 0 {
            let i = self.current_idx - 1;
            // self.fh = from_fn(|l| {
            //     let seqi = unsafe { *self.seq.get_unchecked(l * self.n + i) };
            //     let seqk = unsafe { *self.seq.get_unchecked(l * self.n + i + self.k) };
            //     self.fh[l].rotate_left(1) ^ self.rot_k[seqi as usize] ^ h(seqk)
            // });
            let seqi: [u8; L] = from_fn(|l| unsafe { *self.seq.get_unchecked(l * self.n + i) });
            let seqk: [u8; L] =
                from_fn(|l| unsafe { *self.seq.get_unchecked(l * self.n + i + self.k) });
            self.fh =
                from_fn(|l| self.fh[l].rotate_left(1) ^ self.rot_k[seqi[l] as usize] ^ h(seqk[l]));
        }

        self.current_idx += 1;
        Some(self.fh)
    }
}

//////////////////////////// SIMD ////////////////////////////

const L: usize = 8;
type T = u32;
type S = Simd<T, L>;

pub struct NtHashSimd;

impl ParHasher<L> for NtHashSimd {
    type Out = T;
    #[inline(always)]
    fn hash_kmers(&self, k: usize, t: &[u8]) -> impl Iterator<Item = [Self::Out; L]> {
        NtHashSimdIt::new(t, k).unwrap().map(|x| x.into())
    }
}

#[derive(Debug)]
pub struct NtHashSimdIt<'a> {
    seq: &'a [u8],
    n: usize,
    k: usize,
    fh: S,
    current_idx: usize,
    table: S,
    table_rot_k: S,
    offsets: Simd<*const u8, 4>,
    offsets_next: Simd<*const u8, 4>,
    chars_i: S,
    chars_k: S,
    chars_i_next: S,
    chars_k_next: S,
}

impl<'a> NtHashSimdIt<'a> {
    /// Creates a new NtHashIt with internal state properly initialized.
    #[inline(always)]
    pub fn new(seq: &'a [u8], k: usize) -> Option<Self> {
        if k > seq.len() {
            return None;
        }
        if k > MAXIMUM_K_SIZE {
            return None;
        }
        if seq.len() > u32::MAX as _ {
            return None;
        }
        let num_kmers = seq.len() - k + 1;
        if num_kmers % L != 0 {
            return None;
        }
        let n = num_kmers / L;

        // Each 128-bit half has a copy of the 4 32-bit hashes.
        let table: S = b"ACTGACTG".map(|c| h(c) as T).into();
        let table_rot_k: S = b"ACTGACTG".map(|c| h(c).rotate_left(k as u32) as T).into();

        let fh = from_fn(|l| {
            let mut fh = 0;
            for (i, v) in seq[l * n..l * n + k].iter().enumerate() {
                fh ^= (h(*v) as T).rotate_left((k - i - 1) as u32);
            }
            fh
        })
        .into();

        Some(Self {
            seq,
            n,
            k,
            fh,
            current_idx: 0,
            table,
            table_rot_k,
            offsets: from_fn(|l| unsafe { seq.as_ptr().add(l * n) }).into(),
            offsets_next: from_fn(|l| unsafe { seq.as_ptr().add((4 + l) * n) }).into(),
            chars_i: S::splat(0),
            chars_k: S::splat(0),
            chars_i_next: S::splat(0),
            chars_k_next: S::splat(0),
        })
    }
}

impl<'a> Iterator for NtHashSimdIt<'a> {
    type Item = S;

    #[inline(always)]
    fn next(&mut self) -> Option<S> {
        if self.current_idx == self.n {
            return None;
        };

        if self.current_idx != 0 {
            let i = self.current_idx - 1;
            unsafe {
                if i % 8 == 0 {
                    let oi1 = self.offsets.wrapping_add(Simd::splat(i));
                    let oi2 = self.offsets_next.wrapping_add(Simd::splat(i));
                    let ok1 = oi1.wrapping_add(Simd::splat(self.k));
                    let ok2 = oi2.wrapping_add(Simd::splat(self.k));

                    // i1, k1: The next 64bits for the first 4 chunks.
                    // i2, k2: The next 64bits for the next 4 chunks.
                    // Assumes little-endian.
                    // The inner transmute makes the *u8 into a *u32.
                    // The outer transmute makes the [u64, 4] into a [u32, 8].
                    let chars_i1: Simd<u32, 8> =
                        transmute(Simd::<u64, 4>::gather_ptr(transmute(oi1)));
                    let chars_i2: Simd<u32, 8> =
                        transmute(Simd::<u64, 4>::gather_ptr(transmute(oi2)));
                    let chars_k1: Simd<u32, 8> =
                        transmute(Simd::<u64, 4>::gather_ptr(transmute(ok1)));
                    let chars_k2: Simd<u32, 8> =
                        transmute(Simd::<u64, 4>::gather_ptr(transmute(ok2)));

                    // The upcoming 32 bits for each chunk.
                    // The 32 bits after that for each chunk.
                    (self.chars_i, self.chars_i_next) = chars_i1.deinterleave(chars_i2);
                    (self.chars_k, self.chars_k_next) = chars_k1.deinterleave(chars_k2);
                }
                if i % 8 == 4 {
                    self.chars_i = self.chars_i_next;
                    self.chars_k = self.chars_k_next;
                }
                // Extract the last 2 bits of each character.
                let seqi = self.chars_i & S::splat(0x03);
                let seqk = self.chars_k & S::splat(0x03);
                // Shift remaining characters to the right.
                self.chars_i >>= S::splat(8);
                self.chars_k >>= S::splat(8);

                use std::mem::transmute;
                let permutevar_epi32 =
                    |a: S, b: S| transmute(_mm256_permutevar_ps(transmute(a), transmute(b)));
                let hi: S = permutevar_epi32(self.table_rot_k, seqi);
                let hk: S = permutevar_epi32(self.table, seqk);

                self.fh = ((self.fh << 1) | (self.fh >> 31)) ^ hi ^ hk;
            }
        }

        self.current_idx += 1;
        Some(self.fh)
    }
}
