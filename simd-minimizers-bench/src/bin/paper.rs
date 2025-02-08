use itertools::Itertools;
use packed_seq::{unpack_base, AsciiSeq, AsciiSeqVec, PackedSeq, PackedSeqVec, Seq, SeqVec};
use rand::{random_range, Rng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use simd_minimizers::{
    canonical_minimizer_positions, minimizer_positions, mul_hash,
    private::{
        nthash::{self, nthash_mapper, NtHasher},
        sliding_min::sliding_min_mapper,
        *,
    },
};
use simd_minimizers_bench::*;
use std::{cell::RefCell, hint::black_box};
use wide::u32x8;

fn main() {
    // Experiments for the (w,k) plot.
    // Written to results-plot.json.
    // plot();

    // Experiments for the main result tables.
    bench_minimizers(5, 31); // kraken
    bench_minimizers(11, 21); // sshash
    bench_minimizers(19, 19); // minimap

    let results = RESULTS.with(|r| std::mem::take(&mut *r.borrow_mut()));
    let json = serde_json::to_string(&results).unwrap();
    std::fs::write("results.json", json).unwrap();

    // Additional experiments for human genome density and multithreaded results.
    bench_human_genome();

    // Experiment to test speed on short sequences.
    // Not in the paper.
    // bench_short(11, 21);

    // Experiment to compare sliding window minima algorithms.
    // Not in the paper.
    // bench_sliding_min();
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

#[allow(unused)]
fn bench_short(w: usize, k: usize) {
    let total_len = 1 << 20;
    EXPERIMENT.with(|e| {
        *e.borrow_mut() = "short".to_string();
    });
    eprintln!("\nShort n\n");
    let v = &mut vec![];
    for n in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        let num = total_len / n;

        let packed_seqs = (0..num)
            .map(|_| {
                let len = random_range(n..2 * n);
                PackedSeqVec::random(len)
            })
            .collect_vec();

        let params = Params { n: total_len, w, k };

        time(&format!("simd-minimizers {n}"), params, || {
            for s in &packed_seqs {
                v.clear();
                minimizer_positions(s.as_slice(), k, w, v);
            }
        });
        time(&format!("canonical simd-minimizers {n}"), params, || {
            for s in &packed_seqs {
                v.clear();
                canonical_minimizer_positions(s.as_slice(), k, w, v);
            }
        });
    }
}

fn plot() {
    let n = 10_000_000;
    // let n = 100_000_000;

    for k in [5, 11, 19, 31] {
        for w in (1..16)
            .step_by(2)
            .chain((17..32).step_by(4))
            .chain((33..50).step_by(8))
        {
            // for k in [5, 9, 15, 21, 31] {
            //     for w in (1..100).step_by(2) {
            let params = Params { n, w, k };

            let ascii_seq = AsciiSeqVec::random(n);
            let plain_seq = &ascii_seq.seq;
            let packed_seq = PackedSeqVec::from_ascii(&ascii_seq.seq);
            let packed_seq = packed_seq.as_slice();

            eprintln!("\nMinimizers w = {w} k = {k}");

            type H = NtHasher;

            let v2 = &mut vec![];
            // warmup
            {
                collect::collect_into(
                    minimizers::canonical_minimizers_seq_simd::<_, H>(packed_seq, k, w),
                    v2,
                );
                v2.clear();
                collect::collect_and_dedup_into::<false>(
                    minimizers::canonical_minimizers_seq_simd::<_, H>(packed_seq, k, w),
                    v2,
                );
                v2.clear();
            }

            {
                time("simd-minimizers", params, || {
                    v2.clear();
                    minimizer_positions(packed_seq, k, w, v2);
                });
                time("canonical simd-minimizers", params, || {
                    v2.clear();
                    canonical_minimizer_positions(packed_seq, k, w, v2);
                });

                time_v(v2, "minimizer-iter", params, || {
                    minimizer_iter::MinimizerBuilder::<u64>::new()
                        .minimizer_size(k)
                        .width(w as u16)
                        .iter_pos(plain_seq)
                        .map(|x| x as u32)
                });
                time_v(v2, "canonical minimizer-iter", params, || {
                    minimizer_iter::MinimizerBuilder::<u64>::new()
                        .canonical()
                        .minimizer_size(k)
                        .width(w as u16)
                        .iter_pos(plain_seq)
                        .map(|x| x.0 as u32)
                });
                time("rescan", params, || {
                    v2.clear();
                    minimizers_callback::<true, false>(plain_seq, k + w - 1, k, |pos| {
                        v2.push(pos as u32);
                    });
                });
            }
        }
    }
    let results = RESULTS.with(|r| std::mem::take(&mut *r.borrow_mut()));
    let json = serde_json::to_string(&results).unwrap();
    std::fs::write("results-plot.json", json).unwrap();
}

fn bench_minimizers(w: usize, k: usize) {
    let n = 100_000_000;

    let params = Params { n, w, k };

    let ascii_seq = AsciiSeqVec::random(n);
    let plain_seq = &ascii_seq.seq;
    let packed_seq = PackedSeqVec::from_ascii(&ascii_seq.seq);
    let packed_seq = packed_seq.as_slice();

    eprintln!("\nMinimizers w = {w} k = {k}");

    type H = NtHasher;

    let v = &mut vec![];
    let v2 = &mut vec![];
    // warmup
    {
        v.extend(packed_seq.par_iter_bp(k + w - 1).0);
        v.clear();

        collect::collect_into(
            minimizers::canonical_minimizers_seq_simd::<_, H>(packed_seq, k, w),
            v2,
        );
        v2.clear();
        collect::collect_and_dedup_into(
            minimizers::canonical_minimizers_seq_simd::<_, H>(packed_seq, k, w),
            v2,
        );
        v2.clear();
    }

    // INCREMENTAL
    if k == 21 {
        eprintln!("\nIncremental\n");

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
            nthash::nthash_seq_simd::<false, PackedSeq, H>(packed_seq, k, w).0
        });
        time_v(v, "sliding_min", params, || {
            minimizers::minimizers_seq_simd::<_, H>(packed_seq, k, w).0
        });

        time("fwd-collect", params, || {
            v2.clear();
            collect::collect_into(
                minimizers::minimizers_seq_simd::<_, H>(packed_seq, k, w),
                v2,
            )
        });
        time("fwd-dedup", params, || {
            v2.clear();
            minimizer_positions(packed_seq, k, w, v2);
        });

        time_v(v, "canonical-nthash", params, || {
            // Inline minimizers_seq_simd.
            let add_remove = packed_seq.par_iter_bp_delayed(k + w - 1, k - 1).0;
            // True instead of default false here.
            let mut nthash = nthash_mapper::<true, PackedSeq, H>(k, w);
            let mut sliding_min = sliding_min_mapper::<true>(w, k, add_remove.len());
            add_remove.map(move |(a, rk)| sliding_min(nthash((a, rk))))
        });
        time_v(v, "canonical-strand", params, || {
            minimizers::canonical_minimizers_seq_simd::<_, H>(packed_seq, k, w).0
        });

        time("canonical-collect", params, || {
            v2.clear();
            collect::collect_into(
                minimizers::canonical_minimizers_seq_simd::<_, H>(packed_seq, k, w),
                v2,
            )
        });
        time("canonical-dedup", params, || {
            v2.clear();
            canonical_minimizer_positions(packed_seq, k, w, v2);
        });
    }

    {
        EXPERIMENT.with(|e| {
            *e.borrow_mut() = "external".to_string();
        });
        eprintln!("\nFinal functions\n");
        time("simd-minimizers", params, || {
            v2.clear();
            minimizer_positions(packed_seq, k, w, v2);
        });
        time("canonical simd-minimizers", params, || {
            v2.clear();
            canonical_minimizer_positions(packed_seq, k, w, v2);
        });
        time("mul simd-minimizers", params, || {
            v2.clear();
            mul_hash::minimizer_positions(packed_seq, k, w, v2);
        });
        time("mul canonical simd-minimizers", params, || {
            v2.clear();
            mul_hash::canonical_minimizer_positions(packed_seq, k, w, v2);
        });
        let seq = packed_seq.iter_bp().map(|x| unpack_base(x)).collect_vec();
        let ascii_seq = AsciiSeq(&seq);

        if false {
            let mut packed_seq = PackedSeqVec::from_ascii(&seq);
            time("pack simd-minimizers", params, || {
                v2.clear();
                packed_seq.len = 0;
                packed_seq.push_ascii(&seq);
                minimizer_positions(packed_seq.as_slice(), k, w, v2);
            });
            time("pack canonical simd-minimizers", params, || {
                v2.clear();
                packed_seq.len = 0;
                packed_seq.push_ascii(&seq);
                canonical_minimizer_positions(packed_seq.as_slice(), k, w, v2);
            });
            time("pack mul simd-minimizers", params, || {
                v2.clear();
                packed_seq.len = 0;
                packed_seq.push_ascii(&seq);
                mul_hash::minimizer_positions(packed_seq.as_slice(), k, w, v2);
            });
            time("pack mul canonical simd-minimizers", params, || {
                v2.clear();
                packed_seq.len = 0;
                packed_seq.push_ascii(&seq);
                mul_hash::canonical_minimizer_positions(packed_seq.as_slice(), k, w, v2);
            });
        }

        time("ascii-dna simd-minimizers", params, || {
            v2.clear();
            minimizer_positions(ascii_seq, k, w, v2);
        });
        time("ascii-dna canonical simd-minimizers", params, || {
            v2.clear();
            canonical_minimizer_positions(ascii_seq, k, w, v2);
        });

        time("ascii-dna mul simd-minimizers", params, || {
            v2.clear();
            mul_hash::minimizer_positions(ascii_seq, k, w, v2);
        });
        time("ascii-dna mul canonical simd-minimizers", params, || {
            v2.clear();
            mul_hash::canonical_minimizer_positions(ascii_seq, k, w, v2);
        });

        time("ascii mul simd-minimizers", params, || {
            v2.clear();
            mul_hash::minimizer_positions(&seq[..], k, w, v2);
        });
        time("ascii mul canonical simd-minimizers", params, || {
            v2.clear();
            mul_hash::canonical_minimizer_positions(&seq[..], k, w, v2);
        });
    }

    //
    if false {
        eprintln!("\nENGLISH\n");
        let seq = &std::fs::read("english.200MB").unwrap()[..n];

        time("ascii mul simd-minimizers EN", params, || {
            v2.clear();
            mul_hash::minimizer_positions(seq, k, w, v2);
        });
        time("ascii mul canonical simd-minimizers EN", params, || {
            v2.clear();
            mul_hash::canonical_minimizer_positions(seq, k, w, v2);
        });
        eprintln!("#minis: {}", v2.len());

        eprintln!("\nSOURCES\n");
        let seq = &std::fs::read("sources.200MB").unwrap()[..n];

        time("ascii mul simd-minimizers SRC", params, || {
            v2.clear();
            mul_hash::minimizer_positions(seq, k, w, v2);
        });
        time("ascii mul canonical simd-minimizers SRC", params, || {
            v2.clear();
            mul_hash::canonical_minimizer_positions(seq, k, w, v2);
        });
        eprintln!("#minis: {}", v2.len());
    }

    if false {
        eprintln!("\nSCALAR\n");
        time("positions scalar", params, || {
            v2.clear();
            simd_minimizers::minimizer_positions_scalar(packed_seq, k, w, v2);
        });
        time("canonical positions scalar", params, || {
            v2.clear();
            simd_minimizers::canonical_minimizer_positions_scalar(packed_seq, k, w, v2);
        });
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
        time_v(v2, "canonical minimizer-iter", params, || {
            minimizer_iter::MinimizerBuilder::<u64>::new()
                .canonical()
                .minimizer_size(k)
                .width(w as u16)
                .iter_pos(plain_seq)
                .map(|x| x.0 as u32)
        });
        time("rescan", params, || {
            v2.clear();
            minimizers_callback::<true, false>(plain_seq, k + w - 1, k, |pos| {
                v2.push(pos as u32);
            });
        });
        time("mul rescan", params, || {
            v2.clear();
            minimizers_callback::<true, true>(plain_seq, k + w - 1, k, |pos| {
                v2.push(pos as u32);
            });
        });
    }
}

fn bench_human_genome() {
    let seqs = read_human_genome(usize::MAX);
    let n = seqs.iter().map(|s| s.len()).sum::<usize>();

    let mut v = Vec::new();

    for (w, k) in [(11, 21), (19, 19)] {
        let mut minis = 0;
        time("hg-fwd", Params { n, k, w }, || {
            let mut c = 0;
            for seq in &seqs {
                minimizer_positions(seq.as_slice(), k, w, &mut v);
                c += v.len();
                black_box(&mut v).clear();
            }
            minis = c;
        });
        eprintln!(
            "Total fwd minimizers: {minis}, density = {}",
            minis as f64 / n as f64
        );
        time("hg-canonical", Params { n, k, w }, || {
            let mut c = 0;
            for seq in &seqs {
                canonical_minimizer_positions(seq.as_slice(), k, w, &mut v);
                c += v.len();
                black_box(&mut v).clear();
            }
            minis = c;
        });
        eprintln!(
            "Total can minimizers: {minis}, density = {}",
            minis as f64 / n as f64
        );

        thread_local! {
            static V: RefCell<Vec<u32>> = RefCell::new(vec![]);
        }
        time("hg-fwd-par", Params { n, k, w }, || {
            seqs.par_iter().for_each(|seq| {
                V.with_borrow_mut(|v| {
                    minimizer_positions(seq.as_slice(), k, w, v);
                    black_box(v).clear();
                });
            });
        });
        time("hg-canonical-par", Params { n, k, w }, || {
            seqs.par_iter().for_each(|seq| {
                V.with_borrow_mut(|v| {
                    canonical_minimizer_positions(seq.as_slice(), k, w, v);
                    black_box(v).clear();
                });
            });
        });
    }
}

#[allow(unused)]
fn bench_sliding_min() {
    EXPERIMENT.with(|e| {
        *e.borrow_mut() = "sliding_min".to_string();
    });

    let n = 10_000_000;
    let mut rng = rand::rng();
    let vals = (0..n).map(|_| rng.random::<u64>()).collect::<Vec<_>>();

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
    time(&format!("{name}"), params, || {
        v.clear();
        v.extend(f())
    });
    v.clear();
}

const REPEATS: usize = 5;

fn time<T>(name: &str, params: Params, mut f: impl FnMut() -> T) {
    for _ in 0..REPEATS {
        let start = std::time::Instant::now();
        black_box(f());
        let elapsed = start.elapsed().as_secs_f64();
        let elapsed_per = elapsed * 1_000_000_000. / params.n as f64;
        println!("{name:<40}: {:6.2} s {:6.2} ns/elem", elapsed, elapsed_per);
        RESULTS.with(|r| {
            r.borrow_mut().push(Result {
                experiment: EXPERIMENT.with(|e| e.borrow().clone()),
                name: name.to_string(),
                n: params.n,
                k: params.k,
                w: params.w,
                time: elapsed_per,
            })
        });
    }
}
