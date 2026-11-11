//! Perfilado del recorrido postflop a 3: `valgrind --tool=callgrind target/release/examples/pfprof`.
use preflop_engine::abstraction::board_pool::{BoardPool, PoolConfig};
use preflop_engine::cfr::{full_range, PostflopEngine, PostflopEstimator, StrategyMode};
use preflop_engine::tree::{build_tree, Payoff, PreflopConfig};

fn main() {
    let tree = build_tree(PreflopConfig::standard(vec![30.0; 6], Payoff::ChipEv));
    let pool = BoardPool::build(&PoolConfig { boards: 8, buckets: [16, 16, 16], kmeans_samples: 2000, kmeans_iters: 3, threads: 1, ..PoolConfig::default() });
    let est = PostflopEngine::new(pool, 18, 1);
    let r = full_range();
    let multi: Vec<usize> = (0..tree.leaves.len()).filter(|&k| tree.leaves[k].multi.is_some()).collect();
    preflop_engine::with_pool!(1, |pool| {
        for k in 0..6 {
            std::hint::black_box(est.values3(&tree, multi[k % multi.len()], k % 3, [&r, &r, &r], StrategyMode::Current, true, k as u64, &[0], pool));
        }
    });
}
