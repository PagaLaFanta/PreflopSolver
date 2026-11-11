//! Microbenchmark de los componentes calientes del solver:
//! `cargo run --release --example bench`.

use preflop_engine::abstraction::board_pool::{BoardPool, PoolConfig};
use preflop_engine::cfr::{full_range, PostflopEngine, PostflopEstimator, StrategyMode};
use preflop_engine::eval::equity::EquityMatrix;
use preflop_engine::tree::{build_tree, Payoff, PreflopConfig};
use std::time::Instant;

fn main() {
    let eq = EquityMatrix::load_or_build(std::path::Path::new(".cache/equity_20000.bin"), 20000, 0x5EED, 4);
    let r = full_range();
    let t = Instant::now();
    for _ in 0..100 {
        std::hint::black_box(eq.products(&r[..]));
    }
    println!("products: {:.3} ms", t.elapsed().as_secs_f64() * 10.0);
    let tree = build_tree(PreflopConfig::standard(vec![30.0; 6], Payoff::ChipEv));
    let pool = BoardPool::build(&PoolConfig { boards: 64, buckets: [32, 32, 32], ..PoolConfig::default() });
    let est = PostflopEngine::new(pool, 20, 1);
    let mut out = full_range();
    let hu: Vec<usize> = (0..tree.leaves.len()).filter(|&k| tree.leaves[k].multi.is_none()).collect();
    let t = Instant::now();
    preflop_engine::with_pool!(1, |pool| {
        for k in 0..100 {
            est.hero_values(&tree, hu[k % hu.len()], false, &r, &r, StrategyMode::Current, true, k as u64, &mut out, pool);
        }
    });
    println!("postflop hero_values (Current): {:.3} ms", t.elapsed().as_secs_f64() * 10.0);
    let mut sv = vec![0.0; 6];
    let t = Instant::now();
    preflop_engine::with_pool!(1, |pool| {
        for k in 0..100 {
            est.observer_values(&tree, hu[k % hu.len()], &r, &r, StrategyMode::Current, k as u64, &mut sv, pool);
        }
    });
    println!("postflop observer (Current): {:.3} ms", t.elapsed().as_secs_f64() * 10.0);
    let n_allin = tree.terminals.iter().filter(|t| t.kind == preflop_engine::tree::TerminalKind::AllInRunout).count();
    let n_mw = tree.terminals.iter().filter(|t| t.kind == preflop_engine::tree::TerminalKind::Multiway).count();
    let multi: Vec<usize> = (0..tree.leaves.len()).filter(|&k| tree.leaves[k].multi.is_some()).collect();
    if !multi.is_empty() {
        let t = Instant::now();
        preflop_engine::with_pool!(1, |pool| {
            for k in 0..30 {
                std::hint::black_box(est.values3(&tree, multi[k % multi.len()], k % 3, [&r, &r, &r], StrategyMode::Current, true, k as u64, &[0], pool));
            }
        });
        println!("postflop values3 (Current): {:.3} ms", t.elapsed().as_secs_f64() * 1000.0 / 30.0);
    }
    println!("nodos máx por hoja (HU, 3-way): {:?}", leaf_sizes(&tree));
    println!("terminales: all-in HU {n_allin}, multiway {n_mw}, hojas {}", tree.leaves.len());
    let mw = preflop_engine::cfr::multiway::MultiwayEval::new(2048, 8, 128, 1, 4);
    let t_mw = tree.terminals.iter().find(|t| t.kind == preflop_engine::tree::TerminalKind::Multiway).unwrap();
    let t = Instant::now();
    for k in 0..100 {
        std::hint::black_box(mw.values(t_mw, &tree.utils, 6, 0, &[t_mw.players[0] as usize, 3, 4], [&r, &r], StrategyMode::Current, k));
    }
    println!("multiway values (8 boards): {:.3} ms", t.elapsed().as_secs_f64() * 10.0);
}

#[allow(dead_code)]
pub fn leaf_sizes(tree: &preflop_engine::tree::PreflopTree) -> (usize, usize) {
    let hu: usize = tree.leaves.iter().filter(|l| l.multi.is_none()).map(|l| l.nodes.len()).max().unwrap_or(0);
    let m: usize = tree.leaves.iter().filter_map(|l| l.multi.as_ref()).map(|m| m.nodes.len()).max().unwrap_or(0);
    (hu, m)
}
