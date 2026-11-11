//! Buckets distintos por board y calle del pool postflop (con 256
//! buckets, sus tablas usan ~80 por board en el flop, ~61 en el turn y ~19 en el river).
//!
//! `cargo run --release --example poolstats -- [ochs|percentile] [boards] [buckets]`
use preflop_engine::abstraction::board_pool::{BoardPool, PoolConfig, BLOCKED};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ochs = args.get(1).is_none_or(|s| s == "ochs");
    let boards: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let b: u16 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let cfg = PoolConfig { boards, buckets: [b, b, b], threads: 16, river_ochs: ochs, ..PoolConfig::default() };
    let t0 = std::time::Instant::now();
    let pool = BoardPool::load_or_build(&cfg, std::path::Path::new(".cache"));
    println!("pool: {} boards, buckets {:?}, river {} ({:.1}s)", pool.boards.len(), pool.num_buckets, if ochs { "OCHS" } else { "percentil" }, t0.elapsed().as_secs_f64());
    for (s, name) in ["flop", "turn", "river"].iter().enumerate() {
        let counts: Vec<usize> = pool
            .boards
            .iter()
            .map(|bd| {
                let mut seen = std::collections::HashSet::new();
                for &x in bd.buckets[s].iter() {
                    if x != BLOCKED {
                        seen.insert(x);
                    }
                }
                seen.len()
            })
            .collect();
        let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
        let mut sorted = counts.clone();
        sorted.sort();
        println!("{name:>5}: buckets distintos por board: media {mean:.1}, mediana {}, mín {}, máx {}", sorted[sorted.len() / 2], sorted[0], sorted[sorted.len() - 1]);
    }
}
