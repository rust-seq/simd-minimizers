# Changelog

## 1.3
- Update to `packed-seq` `3.2.1` for `u128` kmer value support.
- Add `iter_{canonical}_minimizer_values_u128` to iterate over `u128` kmer values.

## 1.2
- Fix #10: Add `iter_{canonical}_minimizer_values` to convert positions into `u64` kmer values.
- Update to `packed-seq` `3.0`.
- Fix to properly initialize arrays when collecting super-k-mers.
- Update `packed-seq` to support non-byte offsets.

## 1.1
- Update `packed-seq` to `2.0`, which uses tuples of (simd iterator, padding),
  instead of a separate scalar iterator over the tail.

## 1.0
- Initial release.
