use packed_seq::SeqVec;
use seq_hash::{NtHasher, SeqHasher};

fn main() {
    let k = 9;
    let w = 19;

    eprintln!("      n   fwd simd/scalar   can simd/scalar");
    for n in [50, 100, 150, 500, 10000, 100000, 1000000] {
        let seq = packed_seq::PackedSeqVec::random(n);
        eprint!("{n:>7}: ");

        let time = bench(w, n, &seq, &NtHasher::<false>::new(k), false, true);
        eprint!("  {:5.2}", time);
        let time = bench(w, n, &seq, &NtHasher::<false>::new(k), false, false);
        eprint!(" {:5.2}", time);

        let time = bench(w, n, &seq, &NtHasher::<true>::new(k), true, true);
        eprint!("        {:5.2}", time);
        let time = bench(w, n, &seq, &NtHasher::<true>::new(k), true, false);
        eprint!(" {:5.2}", time);
        eprintln!();
    }
}

fn bench(
    w: usize,
    n: usize,
    seq: &packed_seq::PackedSeqVec,
    hasher: &impl SeqHasher,
    canonical: bool,
    simd: bool,
) -> f32 {
    let total = 100_000_000;
    let samples = total / n;

    let poss = &mut vec![];
    let mut times = vec![];
    for _ in 0..samples {
        poss.clear();
        let s = std::time::Instant::now();
        if simd {
            if canonical {
                simd_minimizers::canonical_minimizer_positions(seq.as_slice(), hasher, w, poss);
            } else {
                simd_minimizers::minimizer_positions(seq.as_slice(), hasher, w, poss);
            }
        } else {
            if canonical {
                simd_minimizers::scalar::canonical_minimizer_positions_scalar(seq.as_slice(), hasher, w, poss);
            } else {
                simd_minimizers::scalar::minimizer_positions_scalar(seq.as_slice(), hasher, w, poss);
            }
        }
        times.push(s.elapsed().as_nanos());
    }
    times.sort();
    times[0] as f32 / n as f32
}
