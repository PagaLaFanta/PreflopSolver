//! Perfilado del solver completo (preflop + postflop) con N jugadores:
//! `valgrind --tool=callgrind target/release/examples/solveprof 4 [iteraciones] [hilos] [eval]`.
use preflop_engine::abstraction::board_pool::{BoardPool, PoolConfig};
use preflop_engine::cfr::{PostflopEngine, Solver, SolverConfig};
use preflop_engine::eval::equity::EquityMatrix;
use preflop_engine::tree::{build_tree, Payoff, PreflopConfig};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    let iters: u32 = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(2);
    let threads: usize = std::env::args().nth(3).and_then(|a| a.parse().ok()).unwrap_or(1);
    let eq = Arc::new(EquityMatrix::load_or_build(std::path::Path::new(".cache/equity_20000.bin"), 20000, 0x5EED, 4));
    let tree = build_tree(PreflopConfig::standard(vec![30.0; n], Payoff::ChipEv));
    let pool = BoardPool::build(&PoolConfig { boards: 128, threads: 4, ..PoolConfig::default() });
    let bpv: u32 = std::env::args().find_map(|a| a.strip_prefix("bpv=").and_then(|v| v.parse().ok())).unwrap_or(16);
    let mut est = PostflopEngine::new(pool, 22, bpv);
    est.adaptive_boards = !std::env::args().any(|a| a == "fijo");
    let mut s = Solver::new(tree, eq, est, SolverConfig { threads, ..SolverConfig::default() });
    let t = Instant::now();
    s.run(iters);
    let t_cfr = t.elapsed().as_secs_f64();
    eprintln!("{n} jugadores: {:.3} s/iteración", t_cfr / iters as f64);
    if std::env::args().any(|a| a == "eval") {
        let t = Instant::now();
        let g: f64 = s.best_response_gains().iter().sum();
        eprintln!("evaluación {:.2} s, NashConv {g:.3}", t.elapsed().as_secs_f64());
    }
}
