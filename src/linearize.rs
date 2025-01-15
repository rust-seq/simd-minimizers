// TODO: The linearized functions are quite slow.

use std::marker::PhantomData;

use packed_seq::Seq;
use packed_seq::{L, S};

use crate::Captures;

/// Take a sequence and a function to parallel iterate over it, and return a buffered linear iterator.
pub fn linearize<
    's,
    BI: Seq<'s>,
    I1: ExactSizeIterator<Item = S>,
    I2: ExactSizeIterator<Item = u32>,
>(
    seq: BI,
    context: usize,
    par_it: impl Fn(BI) -> (I1, I2),
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    Linear::<'s, false, _, _, _, _>::new(seq, context, par_it)
}

pub fn linearize_with_offset<
    's,
    BI: Seq<'s>,
    I1: ExactSizeIterator<Item = S>,
    I2: ExactSizeIterator<Item = u32>,
>(
    seq: BI,
    context: usize,
    par_it: impl Fn(BI) -> (I1, I2),
) -> impl ExactSizeIterator<Item = u32> + Captures<&'s ()> {
    Linear::<'s, true, _, _, _, _>::new(seq, context, par_it)
}

struct Linear<
    's,
    const OFFSET: bool,
    BI: Seq<'s>,
    I1: ExactSizeIterator<Item = S>,
    I2: ExactSizeIterator<Item = u32>,
    ParIt: Fn(BI) -> (I1, I2),
> {
    _s: PhantomData<&'s ()>,
    seq: BI,
    par_it: ParIt,
    context: usize,

    chunk_size: usize,
    par_size: usize,
    num_chunks: usize,

    cache: Vec<S>,
    tail_cache: Vec<u32>,

    /// current chunk
    c: usize,
    /// current lane in chunk
    l: usize,
    /// current index
    i: usize,

    /// number of lanes in the current chunk.
    max_l: usize,
    /// length of the current lane
    max_i: usize,
}

impl<
        's,
        const OFFSET: bool,
        BI: Seq<'s>,
        I1: ExactSizeIterator<Item = S>,
        I2: ExactSizeIterator<Item = u32>,
        ParIt: Fn(BI) -> (I1, I2),
    > Linear<'s, OFFSET, BI, I1, I2, ParIt>
{
    fn new(seq: BI, context: usize, par_it: ParIt) -> Linear<'s, OFFSET, BI, I1, I2, ParIt> {
        if OFFSET {
            assert!(
                seq.len() < u32::MAX as usize,
                "sequence of length {} is too long for u32 indices",
                seq.len()
            );
        }
        assert!(context > 0);
        let len = seq.len();
        let chunk_size = 1 << 13;
        assert!(chunk_size % L == 0);
        let par_size = chunk_size / L;

        let num_chunks = len.saturating_sub(context - 1).div_ceil(chunk_size);

        let cache = vec![S::splat(0); par_size];
        let tail_cache = vec![];

        let mut this = Self {
            _s: PhantomData,
            seq,
            par_it,
            context,
            cache,
            tail_cache,

            chunk_size,
            par_size,
            num_chunks,

            c: usize::MAX,
            i: par_size - 1,
            l: L - 1,

            max_i: par_size,
            max_l: L,
        };
        if num_chunks == 1 {
            this.buffer_chunk(0, seq);
        }
        this
    }

    /// Fill a buffer with NT hash values by using `nthash32_par_it`.
    fn buffer_chunk(&mut self, mut offset: usize, seq: BI) {
        let (mut par_it, mut tail) = (self.par_it)(seq);
        self.cache.resize(par_it.size_hint().0, S::splat(0));
        if !OFFSET {
            offset = 0;
        }
        let offset_simd = S::splat(offset as u32);
        for i in 0..self.cache.len() {
            unsafe {
                *self.cache.get_unchecked_mut(i) = par_it.next().unwrap() + offset_simd;
            }
        }
        let offset = offset as u32;
        self.tail_cache.resize(tail.size_hint().0, 0);
        for i in 0..self.tail_cache.len() {
            unsafe {
                *self.tail_cache.get_unchecked_mut(i) = tail.next().unwrap() + offset;
            }
        }
    }

    /// Returns true when iterator is exhausted.
    #[inline(always)]
    fn next_line(&mut self) -> bool {
        self.i = 0;
        self.l += 1;
        if self.l == self.max_l {
            self.l = 0;
            self.c = self.c.wrapping_add(1);
            if self.c < self.num_chunks {
                let offset = self.c * self.chunk_size;
                self.buffer_chunk(
                    offset,
                    self.seq.slice(
                        offset..(offset + self.chunk_size + self.context - 1).min(self.seq.len()),
                    ),
                );
                if self.c + 1 < self.num_chunks {
                    assert_eq!(self.cache.len(), self.par_size);
                    assert!(self.tail_cache.is_empty());
                } else {
                    assert!(self.cache.len() <= self.par_size);
                    self.max_i = self.cache.len();
                    if self.max_i == 0 {
                        self.c += 1;
                    }
                }
            }
            if self.c == self.num_chunks {
                // Run one extra iteration on the tail.
                self.max_l = 1;
                self.max_i = self.tail_cache.len();
                assert!(self.tail_cache.len() <= self.par_size);
                self.cache.resize(self.max_i, S::splat(0));
                for i in 0..self.max_i {
                    self.cache[i].as_array_mut()[0] = self.tail_cache[i];
                }
                if self.max_i == 0 {
                    self.c += 1;
                }
            }
            if self.c == self.num_chunks + 1 {
                return true;
            }
        }
        false
    }
}

impl<
        's,
        const OFFSET: bool,
        BI: Seq<'s>,
        I1: ExactSizeIterator<Item = S>,
        I2: ExactSizeIterator<Item = u32>,
        ParIt: Fn(BI) -> (I1, I2),
    > Iterator for Linear<'s, OFFSET, BI, I1, I2, ParIt>
{
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        if self.i == self.max_i {
            if self.next_line() {
                return None;
            }
        }

        Some(unsafe {
            *self
                .cache
                .get_unchecked(self.i)
                .as_array_ref()
                .get_unchecked(self.l)
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = if self.c >= self.num_chunks + 1 {
            0
        } else if self.c == self.num_chunks {
            self.max_i - self.i
        } else {
            let total_kmers = self.seq.len() - self.context + 1;
            let processed_kmers = 1 + self.c * L * self.par_size + self.l * self.max_i + self.i;
            total_kmers - processed_kmers
        };
        (len, Some(len))
    }
}

impl<
        's,
        const OFFSET: bool,
        BI: Seq<'s>,
        I1: ExactSizeIterator<Item = S>,
        I2: ExactSizeIterator<Item = u32>,
        ParIt: Fn(BI) -> (I1, I2),
    > ExactSizeIterator for Linear<'s, OFFSET, BI, I1, I2, ParIt>
{
}
