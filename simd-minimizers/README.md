# simd-minimizers

[![crates.io](https://img.shields.io/crates/v/simd-minimizers)](https://crates.io/crates/simd-minimizers)
[![docs](https://img.shields.io/docsrs/simd-minimizers)](https://docs.rs/simd-minimizers)

A SIMD-accelerated library to compute random minimizers.

It can compute all the minimizers of a human genome in 4 seconds using a single thread.
It also provides a *canonical* version that ensures that a sequence and its reverse-complement always select the same positions, which takes 6 seconds on a human genome.

The underlying algorithm is described in the following [preprint](https://doi.org/10.1101/2025.01.27.634998):

-   SimdMinimizers: Computing random minimizers, fast.
    Ragnar Groot Koerkamp, Igor Martayan
    bioRxiv 2025.01.27 [doi.org/10.1101/2025.01.27.634998](https://doi.org/10.1101/2025.01.27.634998)


## Requirements

This library supports AVX2 and NEON instruction sets.
Make sure to set `RUSTFLAGS="-C target-cpu=native"` when compiling to use the instruction sets available on your architecture.

    RUSTFLAGS="-C target-cpu=native" cargo run --release



## Usage example

Full documentation can be found on [docs.rs](https://docs.rs/simd-minimizers).

```rust
// Packed SIMD version.
use packed_seq::{complement_char, PackedSeqVec, SeqVec};
let seq = b"ACGTGCTCAGAGACTCAG";
let k = 5;
let w = 7;

let packed_seq = PackedSeqVec::from_ascii(seq);
let mut minimizer_positions = Vec::new();
simd_minimizers::canonical_minimizer_positions(packed_seq.as_slice(), k, w, &mut minimizer_positions);
assert_eq!(minimizer_positions, vec![3, 5, 12]);
```
