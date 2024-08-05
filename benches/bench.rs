#![feature(portable_simd)]
#![allow(dead_code)]
mod blog;
use blog::*;
use itertools::Itertools;
use minimizers::par::{
    minimizer::minimizer_par_it,
    nthash::{nthash32c_par_it, nthash32f_par_it},
    packed::Packed,
};
use std::{cell::LazyCell, simd::Simd, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

criterion_group!(
    name = group;
    config = Criterion::default()
        // Make sure that benchmarks are fast.
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_millis(2000))
        .sample_size(10);
    targets = initial_runtime_comparison,
        blog::counting::count_comparisons_bench,
        optimized, ext_nthash, buffered, local_nthash,
        simd_minimizer, human_genome
);
criterion_main!(group);

fn initial_runtime_comparison(c: &mut Criterion) {
    // Create a random string of length 1Mbp.
    let text = &(0..1000000)
        .map(|_| b"ACGT"[rand::random::<u8>() as usize % 4])
        .collect::<Vec<_>>();

    let hasher = FxHash;

    let w = 11;
    let k = 21;

    #[rustfmt::skip]
    let minimizers: &mut [(&str, &mut dyn Minimizer, bool)] = &mut [
        ("naive", &mut NaiveMinimizer { w, k, hasher }, true),
        ("buffered", &mut SlidingWindowMinimizer { w, k, alg: Buffered, hasher }, true),
        ("queue", &mut SlidingWindowMinimizer { w, k, alg: Queue, hasher }, true),
        ("jumping", &mut JumpingMinimizer { w, k, hasher }, false),
        ("rescan", &mut SlidingWindowMinimizer { w, k, alg: Rescan, hasher }, true),
        ("split", &mut SlidingWindowMinimizer { w, k, alg: Split, hasher }, true),
        ("queue_igor", &mut QueueIgor { w, k }, false),
        ("rescan_daniel", &mut RescanDaniel { w, k }, true),
    ];

    let mut g = c.benchmark_group("g");
    for (name, m, window_minimizers) in minimizers {
        g.bench_function(*name, move |b| {
            if *window_minimizers {
                b.iter(|| m.window_minimizers(text));
            } else {
                b.iter(|| m.minimizer_positions(text));
            }
        });
    }
}

fn optimized(c: &mut Criterion) {
    // Create a random string of length 1Mbp.
    let text = &(0..1000000)
        .map(|_| b"ACGT"[rand::random::<u8>() as usize % 4])
        .collect::<Vec<_>>();

    let hasher = FxHash;

    let w = 11;
    let k = 21;

    #[rustfmt::skip]
    let minimizers: &mut [(&str, &mut dyn Minimizer, bool)] = &mut [
        ("buffered_opt", &mut SlidingWindowMinimizer { w, k, alg: BufferedOpt, hasher }, true),
        ("rescan_opt", &mut SlidingWindowMinimizer { w, k, alg: RescanOpt, hasher }, true),
        ("split_opt", &mut SlidingWindowMinimizer { w, k, alg: SplitOpt, hasher }, true),
    ];

    let mut g = c.benchmark_group("g");
    for (name, m, window_minimizers) in minimizers {
        g.bench_function(*name, |b| {
            if *window_minimizers {
                b.iter(|| m.window_minimizers(text));
            } else {
                b.iter(|| m.minimizer_positions(text));
            }
        });
    }
}

fn ext_nthash(c: &mut Criterion) {
    // Create a random string of length 1Mbp.
    let text = &(0..1000000)
        .map(|_| b"ACGT"[rand::random::<u8>() as usize % 4])
        .collect::<Vec<_>>();

    let w = 11;
    let k = 21;

    let hasher = ExtNtHash;

    #[rustfmt::skip]
    let minimizers: &mut [(&str, &mut dyn Minimizer, bool)] = &mut [
        ("buffered_nt", &mut SlidingWindowMinimizer { w, k, alg: BufferedOpt, hasher }, true),
        ("queue_nt", &mut SlidingWindowMinimizer { w, k, alg: Queue, hasher }, true),
        ("jumping_nt", &mut JumpingMinimizer { w, k, hasher }, false),
        ("rescan_nt", &mut SlidingWindowMinimizer { w, k, alg: RescanOpt, hasher }, true),
        ("split_nt", &mut SlidingWindowMinimizer { w, k, alg: SplitOpt, hasher }, true),
    ];

    let mut g = c.benchmark_group("g");
    for (name, m, window_minimizers) in minimizers {
        g.bench_function(*name, |b| {
            if *window_minimizers {
                b.iter(|| m.window_minimizers(text));
            } else {
                b.iter(|| m.minimizer_positions(text));
            }
        });
    }
}

fn buffered(c: &mut Criterion) {
    // Create a random string of length 1Mbp.
    let text = &(0..1000000)
        .map(|_| b"ACGT"[rand::random::<u8>() as usize % 4])
        .collect::<Vec<_>>();

    let w = 11;
    let k = 21;

    let fxhasher = Buffer { hasher: FxHash };
    let nthasher = Buffer { hasher: ExtNtHash };

    #[rustfmt::skip]
    let minimizers: &mut[(&str, &mut dyn Minimizer, bool)] = &mut [
        ("buffered_buf", &mut SlidingWindowMinimizer { w, k, alg: BufferedOpt, hasher: fxhasher }, true),
        ("queue_buf", &mut SlidingWindowMinimizer { w, k, alg: Queue, hasher: fxhasher }, true),
        ("jumping_buf", &mut JumpingMinimizer { w, k, hasher: fxhasher }, false),
        ("rescan_buf", &mut SlidingWindowMinimizer { w, k, alg: RescanOpt, hasher: fxhasher }, true),
        ("split_buf", &mut SlidingWindowMinimizer { w, k, alg: SplitOpt, hasher: fxhasher }, true),
        ("buffered_nt_buf", &mut SlidingWindowMinimizer { w, k, alg: BufferedOpt, hasher: nthasher }, true),
        ("queue_nt_buf", &mut SlidingWindowMinimizer { w, k, alg: Queue, hasher: nthasher }, true),
        ("jumping_nt_buf", &mut JumpingMinimizer { w, k, hasher: nthasher }, false),
        ("rescan_nt_buf", &mut SlidingWindowMinimizer { w, k, alg: RescanOpt, hasher: nthasher }, true),
        ("split_nt_buf", &mut SlidingWindowMinimizer { w, k, alg: SplitOpt, hasher: nthasher }, true),
    ];

    let mut g = c.benchmark_group("g");
    for (name, m, window_minimizers) in minimizers {
        g.bench_function(*name, |b| {
            if *window_minimizers {
                b.iter(|| m.window_minimizers(text));
            } else {
                b.iter(|| m.minimizer_positions(text));
            }
        });
    }
}

fn local_nthash(c: &mut Criterion) {
    // Create a random string of length 1Mbp.
    let text = &(0..1000000)
        .map(|_| b"ACGT"[rand::random::<u8>() as usize % 4])
        .collect::<Vec<_>>();

    let k = 21;

    let mut g = c.benchmark_group("g");
    g.bench_function("ext_nthash", |b| {
        b.iter(|| ExtNtHash.hash_kmers(k, text).collect::<Vec<_>>());
    });
    g.bench_function("nthash", |b| {
        b.iter(|| NtHash.hash_kmers(k, text).collect::<Vec<_>>());
    });
    g.bench_with_input("nthash_buf", text, |b, text| {
        b.iter(|| Buffer { hasher: NtHash }.hash_kmers(k, text).collect_vec());
    });
    g.bench_with_input("nthash_bufopt", text, |b, text| {
        b.iter(|| {
            BufferOpt { hasher: NtHash }
                .hash_kmers(k, text)
                .collect_vec()
        });
    });
    g.bench_with_input("nthash_bufdouble", text, |b, text| {
        b.iter(|| BufferDouble::new(NtHash).hash_kmers(k, text).collect_vec());
    });
    g.bench_with_input("nthash_bufpar1", text, |b, text| {
        b.iter(|| {
            BufferPar { hasher: NtHashPar::<1> }
                .hash_kmers(k, text)
                .collect_vec()
        });
    });
    g.bench_with_input("nthash_bufpar2", text, |b, text| {
        b.iter(|| {
            BufferPar { hasher: NtHashPar::<2> }
                .hash_kmers(k, text)
                .collect_vec()
        });
    });
    g.bench_with_input("nthash_bufpar3", text, |b, text| {
        b.iter(|| {
            BufferPar { hasher: NtHashPar::<3> }
                .hash_kmers(k, text)
                .collect_vec()
        });
    });
    g.bench_with_input("nthash_bufpar4", text, |b, text| {
        b.iter(|| {
            BufferPar { hasher: NtHashPar::<4> }
                .hash_kmers(k, text)
                .collect_vec()
        });
    });

    let packed_text = &(0..1000000 / 4)
        .map(|_| rand::random::<u8>())
        .collect::<Vec<_>>();

    g.bench_with_input("nthash_simd_ksmall", packed_text, |b, packed_text| {
        b.iter(|| {
            NtHashSimd::<true>
                .hash_kmers(k, packed_text)
                .map(|x| Simd::<u32, 8>::from(x))
                .sum::<Simd<u32, 8>>()
        });
    });

    g.bench_with_input("nthash_bufsimd_ksmall", packed_text, |b, packed_text| {
        b.iter(|| {
            BufferPar { hasher: NtHashSimd::<true> }
                .hash_kmers(k, packed_text)
                .collect_vec()
        });
    });
    g.bench_with_input("nthash_bufsimd_klarge", packed_text, |b, packed_text| {
        b.iter(|| {
            BufferPar { hasher: NtHashSimd::<false> }
                .hash_kmers(k, packed_text)
                .collect_vec()
        });
    });

    let mut hasher = BufferParCached::new(NtHashSimd::<true>);
    g.bench_with_input("nthash_bufsimd_cached", packed_text, |b, packed_text| {
        b.iter(|| drop(black_box(hasher.hash_kmers(k, packed_text))));
    });

    let k = 15;
    g.bench_with_input("fxhash_simd", packed_text, |b, packed_text| {
        b.iter(|| {
            FxHashSimd
                .hash_kmers(k, packed_text)
                .map(|x| Simd::<u32, 8>::from(x))
                .sum::<Simd<u32, 8>>()
        });
    });

    let mut hasher = BufferParCached::new(FxHashSimd);
    g.bench_with_input("fxhash_bufsimd_cached", packed_text, |b, packed_text| {
        b.iter(|| drop(black_box(hasher.hash_kmers(k, packed_text))));
    });

    let packed_text = Packed { seq: packed_text, len_in_bp: packed_text.len() * 4 };
    g.bench_with_input("nthash_par_it_sum", &packed_text, |b, packed_text| {
        b.iter(|| nthash32f_par_it(*packed_text, k, 1).0.sum::<Simd<u32, 8>>());
    });
    g.bench_with_input("nthash_par_it_vec", &packed_text, |b, packed_text| {
        b.iter(|| nthash32f_par_it(*packed_text, k, 1).0.collect_vec());
    });
    g.bench_with_input("nthash_par_it_sum_c", &packed_text, |b, packed_text| {
        b.iter(|| nthash32c_par_it(*packed_text, k, 1).0.sum::<Simd<u32, 8>>());
    });
}

fn simd_minimizer(c: &mut Criterion) {
    // Create a random string of length 1Mbp.
    let packed_text = &(0..1000000 / 4)
        .map(|_| rand::random::<u8>())
        .collect::<Vec<_>>();

    let w = 11;
    let k = 21;

    let mut g = c.benchmark_group("g");
    let mut hasher = NtHashSimd::<true>;
    g.bench_function("split_simd_sum", |b| {
        b.iter(|| {
            SplitSimd
                .sliding_min(w, hasher.hash_kmers(k, packed_text))
                .map(|x| Simd::<u32, 8>::from(x))
                .sum::<Simd<u32, 8>>()
        });
    });

    let hasher = NtHashSimd::<true>;
    let mut hasher = BufferParCached::new(hasher);
    g.bench_function("split_simd_buf_sum", |b| {
        b.iter(|| {
            SplitSimd
                .sliding_min(w, hasher.hash_kmers(k, packed_text))
                .map(|x| Simd::<u32, 8>::from(x))
                .sum::<Simd<u32, 8>>()
        });
    });

    let mut hasher = NtHashSimd::<true>;
    g.bench_function("split_simd_collect", |b| {
        b.iter(|| {
            SplitSimd
                .sliding_min(w, hasher.hash_kmers(k, packed_text))
                .collect_vec()
        });
    });

    let hasher = NtHashSimd::<true>;
    let mut hasher = BufferParCached::new(hasher);
    g.bench_function("split_simd_buf_collect", |b| {
        b.iter(|| {
            SplitSimd
                .sliding_min(w, hasher.hash_kmers(k, packed_text))
                .collect_vec()
        });
    });

    let packed_text = Packed { seq: packed_text, len_in_bp: packed_text.len() * 4 };
    g.bench_function("minimizer_par_it_sum", |b| {
        b.iter(|| minimizer_par_it(packed_text, k, w).0.sum::<Simd<u32, 8>>());
    });
    g.bench_function("minimizer_par_it_vec", |b| {
        b.iter(|| minimizer_par_it(packed_text, k, w).0.collect_vec());
    });
}

fn human_genome(c: &mut Criterion) {
    let w = 11;
    let k = 21;

    let packed_text = LazyCell::new(|| {
        eprintln!("Reading..");
        let start = std::time::Instant::now();
        let mut packed_text = vec![];
        let Ok(mut reader) = needletail::parse_fastx_file("human-genome.fa") else {
            eprintln!("Did not find human-genome.fa. Add/symlink it to test runtime on it.");
            return vec![];
        };
        while let Some(r) = reader.next() {
            let r = r.unwrap();
            eprintln!(
                "Read {:?} of len {:?}",
                std::str::from_utf8(r.id()),
                r.raw_seq().len()
            );
            pack(&r.raw_seq(), &mut packed_text);
            eprintln!("Packed len {:?}", packed_text.len());
        }
        eprintln!("Packing took {:?}", start.elapsed());
        packed_text
    });

    let mut hasher = NtHashSimd::<true>;
    c.bench_function("human_genome", |b| {
        let packed_text = &*packed_text;
        b.iter(|| {
            if packed_text.is_empty() {
                return Default::default();
            }
            SplitSimd
                .sliding_min(w, hasher.hash_kmers(k, &packed_text))
                .map(|x| Simd::<u32, 8>::from(x))
                .sum::<Simd<u32, 8>>()
        });
    });
}

fn pack(text: &[u8], packed: &mut Vec<u8>) {
    let mut packed_byte = 0;
    let mut packed_len = 0;
    for &base in text {
        packed_byte |= match base {
            b'a' | b'A' => 0,
            b'c' | b'C' => 1,
            b'g' | b'G' => 2,
            b't' | b'T' => 3,
            b'\r' | b'\n' => continue,
            _ => panic!(),
        } << (packed_len * 2);
        packed_len += 1;
        if packed_len == 4 {
            packed.push(packed_byte);
            packed_byte = 0;
            packed_len = 0;
        }
    }
}
