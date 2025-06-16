# Changelog

## Git
- Fix #10: Add `extract_{canonical}_minimizer_values` to convert positions into `u64` kmer values.
- Update to `packed-seq` `3.0`.
- Fix to properly initialize arrays when collecting super-k-mers.
- Update `packed-seq` to support non-byte offsets.

## 1.1
- Update `packed-seq` to `2.0`, which uses tuples of (simd iterator, padding),
  instead of a separate scalar iterator over the tail.

## 1.0
- Initial release.
