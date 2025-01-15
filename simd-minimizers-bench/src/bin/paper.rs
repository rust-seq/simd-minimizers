use packed_seq::{AsciiSeqVec, PackedSeqVec, Seq, SeqVec};
use rand::Rng;
use simd_minimizers::{nthash::hash_mapper, sliding_min::sliding_min_mapper};
use simd_minimizers_bench::*;
use std::hint::black_box;
use wide::u32x8;

fn main() {
    bench_minimizers(5, 31); // kraken
    bench_minimizers(11, 21); // sshash
    bench_minimizers(19, 19); // minimap

    // bench_sliding_min();

    let results = RESULTS.with(|r| std::mem::take(&mut *r.borrow_mut()));
    let json = serde_json::to_string(&results).unwrap();
    std::fs::write("results.json", json).unwrap();
}

thread_local! {
    static EXPERIMENT: std::cell::RefCell<String> = std::cell::RefCell::new("".to_string());
    static RESULTS: std::cell::RefCell<Vec<Result>> = std::cell::RefCell::new(vec![]);
}

#[derive(Clone, Copy)]
struct Params {
    n: usize,
    w: usize,
    k: usize,
}

#[derive(serde::Serialize)]
struct Result {
    experiment: String,
    name: String,
    n: usize,
    k: usize,
    w: usize,
    time: f64,
}

fn bench_minimizers(w: usize, k: usize) {
    let n = 100_000_000;

    let params = Params { n, w, k };

    let ascii_seq = AsciiSeqVec::random(n);
    let plain_seq = &ascii_seq.seq;
    let packed_seq = PackedSeqVec::from_ascii(&ascii_seq.seq);
    let packed_seq = packed_seq.as_slice();

    eprintln!("Minimizers w = {w} k = {k}");

    eprintln!("\nSIMD\n");

    let v = &mut vec![];
    let v2 = &mut vec![];
    // warmup
    {
        v.extend(packed_seq.par_iter_bp(k + w - 1).0);
        v.clear();

        simd_minimizers::collect::collect_into(
            simd_minimizers::minimizers::canonical_minimizers_seq_simd(packed_seq, k, w),
            v2,
        );
        v2.clear();
        simd_minimizers::collect::collect_and_dedup_into::<false>(
            simd_minimizers::minimizers::canonical_minimizers_seq_simd(packed_seq, k, w),
            v2,
        );
        v2.clear();
    }

    // INCREMENTAL
    if true {
        EXPERIMENT.with(|e| {
            *e.borrow_mut() = "incremental".to_string();
        });

        // All as both sum and collect-to-vec.

        // 1. gather 1
        // 2. gather 2
        // 3. + nthash
        // 4. + sliding_min
        // 5. + minimizers
        // 6. + canonical nthash
        // 7. + canonical strand
        // 8. + collect
        // 9. + dedup

        v.clear();
        time("gather (sum)", params, || {
            packed_seq.par_iter_bp(k + w - 1).0.sum::<u32x8>()
        });
        time_v(v, "gather (vec)", params, || {
            packed_seq.par_iter_bp(k + w - 1).0
        });
        time_v(v, "gather2", params, || {
            packed_seq
                .par_iter_bp_delayed(k + w - 1, k - 1)
                .0
                .map(|(a, r)| a + r)
        });
        time_v(v, "nthash", params, || {
            simd_minimizers::nthash::hash_seq_simd::<false>(packed_seq, k, w).0
        });
        time_v(v, "sliding_min", params, || {
            simd_minimizers::minimizers::minimizers_seq_simd(packed_seq, k, w).0
        });

        time("fwd-collect", params, || {
            simd_minimizers::collect::collect_into(
                simd_minimizers::minimizers::minimizers_seq_simd(packed_seq, k, w),
                v2,
            )
        });
        v2.clear();
        time("fwd-dedup", params, || {
            simd_minimizers::minimizer_positions(packed_seq, k, w, v2);
        });
        v2.clear();

        time_v(v, "canonical nthash", params, || {
            // Inline minimizers_seq_simd.
            let add_remove = packed_seq.par_iter_bp_delayed(k + w - 1, k - 1).0;
            // True instead of default false here.
            let mut nthash = hash_mapper::<true>(k, w);
            let mut sliding_min = sliding_min_mapper::<true>(w, k, add_remove.len());
            add_remove.map(move |(a, rk)| sliding_min(nthash((a, rk))))
        });
        time_v(v, "canonical strand", params, || {
            simd_minimizers::minimizers::canonical_minimizers_seq_simd(packed_seq, k, w).0
        });

        time("canonical-collect", params, || {
            simd_minimizers::collect::collect_into(
                simd_minimizers::minimizers::canonical_minimizers_seq_simd(packed_seq, k, w),
                v2,
            )
        });
        v2.clear();
        time("canonical-dedup", params, || {
            simd_minimizers::canonical_minimizer_positions(packed_seq, k, w, v2);
        });
        v2.clear();
    }

    {
        EXPERIMENT.with(|e| {
            *e.borrow_mut() = "external".to_string();
        });
        eprintln!("\nFinal functions\n");
        time("simd-minimizer", params, || {
            simd_minimizers::minimizer_positions(packed_seq, k, w, v2);
        });
        v2.clear();
        time("canonical simd-minimizer", params, || {
            simd_minimizers::canonical_minimizer_positions(packed_seq, k, w, v2);
        });
        v2.clear();
    }

    if false {
        eprintln!("\nSCALAR\n");
        time("positions scalar", params, || {
            simd_minimizers::minimizer_positions_scalar(packed_seq, k, w, v2);
        });
        v2.clear();
        time("canonical positions scalar", params, || {
            simd_minimizers::canonical_minimizer_positions_scalar(packed_seq, k, w, v2);
        });
        v2.clear();
    }

    {
        eprintln!("\nEXTERNAL\n");

        time_v(v2, "minimizer-iter", params, || {
            minimizer_iter::MinimizerBuilder::<u64>::new()
                .minimizer_size(k)
                .width(w as u16)
                .iter_pos(plain_seq)
                .map(|x| x as u32)
        });
        v2.clear();
        time_v(v2, "canonical minimizer-iter", params, || {
            minimizer_iter::MinimizerBuilder::<u64>::new()
                .canonical()
                .minimizer_size(k)
                .width(w as u16)
                .iter_pos(plain_seq)
                .map(|x| x.0 as u32)
        });
        v2.clear();
        time("rescan-daniel", params, || {
            minimizers_callback::<true>(plain_seq, k + w - 1, k, |pos| {
                v2.push(pos as u32);
            });
        });
        v2.clear();
    }
}

fn bench_sliding_min() {
    EXPERIMENT.with(|e| {
        *e.borrow_mut() = "sliding_min".to_string();
    });

    let n = 10_000_000;
    let mut rng = rand::thread_rng();
    let vals = (0..n).map(|_| rng.gen::<usize>()).collect::<Vec<_>>();

    for w in [1, 2, 4, 8, 16, 32, 64, 128] {
        let params = Params { n, w, k: 0 };

        eprintln!("\nSliding window min, w = {w}\n");
        let v = &mut vec![1; n];
        // warmup
        v.clear();
        v.extend(
            BufferedOpt
                .sliding_min(w, vals.iter().copied())
                .map(|elem| elem.pos),
        );
        v.clear();

        time_v(v, "naive", params, || {
            BufferedOpt
                .sliding_min(w, vals.iter().copied())
                .map(|elem| elem.pos)
        });
        time_v(v, "queue", params, || {
            Queue
                .sliding_min(w, vals.iter().copied())
                .map(|elem| elem.pos)
        });
        time_v(v, "rescan", params, || {
            RescanOpt
                .sliding_min(w, vals.iter().copied())
                .map(|elem| elem.pos)
        });
        time_v(v, "two-stacks", params, || {
            SplitOpt
                .sliding_min(w, vals.iter().copied())
                .map(|elem| elem.pos)
        });
    }
}

fn time_v<T: std::iter::Sum, I: Iterator<Item = T>>(
    v: &mut Vec<T>,
    name: &str,
    params: Params,
    mut f: impl FnMut() -> I + Clone,
) {
    v.clear();
    time(&format!("{name}"), params, || v.extend(f()));
    v.clear();
}

fn time<T>(name: &str, params: Params, mut f: impl FnMut() -> T) {
    let start = std::time::Instant::now();
    black_box(f());
    let elapsed = start.elapsed().as_secs_f64() * 1_000_000_000. / params.n as f64;
    println!("{name:<20}: {:6.2} ns/elem", elapsed);
    RESULTS.with(|r| {
        r.borrow_mut().push(Result {
            experiment: EXPERIMENT.with(|e| e.borrow().clone()),
            name: name.to_string(),
            n: params.n,
            k: params.k,
            w: params.w,
            time: elapsed,
        })
    });
}
