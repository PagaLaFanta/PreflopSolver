//! Banco de convergencia: tiempo de CFR (sin contar las evaluaciones) hasta cada NashConv
//! en los 3 spots de referencia.
//!
//! `nashbench <hu|3p|6m> [iteraciones] [cada] [clave=valor…]`
//!
//! * `hu`: HU push/fold 15 bb, chip EV.
//! * `3p`: 3 jugadores estándar 30 bb, ICM 50/30/20, postflop MCCFR.
//! * `6m`: 6-max estándar 30 bb, chip EV, postflop en check-down.
//!
//! Claves: `avg=p` (peso t^p de la media), `pfavg=p` (ídem postflop), `rp=q` (peso t^q de los regrets), `ramp=n` (rampa de boards por visita),
//! `bpv=n` (boards por visita), `prune=x` (poda de hojas), `updates=alt|sim`, `eval=0` (sin NashConv), `threads=n`,
//! `par=d`.
use preflop_engine::abstraction::board_pool::{BoardPool, PoolConfig};
use preflop_engine::cfr::{CheckDown, PostflopEngine, PostflopEstimator, Solver, SolverConfig};
use preflop_engine::eval::equity::EquityMatrix;
use preflop_engine::tree::{build_tree, Payoff, PreflopConfig};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let spot = args.first().cloned().unwrap_or_else(|| "hu".into());
    let iters: u32 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(200);
    let every: u32 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(10);
    let kv = |k: &str| args.iter().find_map(|a| a.strip_prefix(&format!("{k}=")).map(String::from));
    let eq = Arc::new(EquityMatrix::load_or_build(std::path::Path::new(".cache/equity_20000.bin"), 20000, 0x5EED, 4));
    let mut cfg = SolverConfig::default();
    if let Some(v) = kv("avg") {
        cfg.avg_power = v.parse().unwrap();
    }
    if let Some(v) = kv("threads") {
        cfg.threads = v.parse().unwrap();
    }
    cfg.simultaneous = kv("updates").map(|v| v == "sim");
    cfg.bunching = kv("bunch").is_none_or(|v| v != "0");
    cfg.regret_power = kv("rp").map(|v| v.parse().unwrap());
    let eval = kv("eval").is_none_or(|v| v != "0");
    // `warm=S`: antes, resuelve el mismo spot con stacks S (`warmiters`, 150 por defecto) y
    // arranca en caliente desde ahí.
    let warm = kv("warm").map(|v| v.parse::<f64>().unwrap());
    let warm_iters: u32 = kv("warmiters").map(|v| v.parse().unwrap()).unwrap_or(150);
    let spot_eq = eq.clone();
    match spot.as_str() {
        "hu" => {
            let make = |st: f64| {
                let tree = build_tree(PreflopConfig::push_fold(vec![st; 2], Payoff::ChipEv));
                Solver::new(tree, spot_eq.clone(), CheckDown { eq: spot_eq.clone() }, cfg)
            };
            run(make, 15.0, warm, warm_iters, iters, every, true);
        }
        "6m" => {
            let make = |st: f64| {
                let tree = build_tree(PreflopConfig { postflop: None, ..PreflopConfig::standard(vec![st; 6], Payoff::ChipEv) });
                Solver::new(tree, spot_eq.clone(), CheckDown { eq: spot_eq.clone() }, cfg)
            };
            run(make, 30.0, warm, warm_iters, iters, every, eval);
        }
        _ => {
            let bpv: u32 = kv("bpv").map(|v| v.parse().unwrap()).unwrap_or(16);
            let prune: f64 = kv("prune").map(|v| v.parse().unwrap()).unwrap_or(1e-3);
            let ramp: u32 = kv("ramp").map(|v| v.parse().unwrap()).unwrap_or(0);
            let pfavg = kv("pfavg").map(|v| v.parse().unwrap()).unwrap_or(3.0);
            let make = |st: f64| {
                // `hus`: comprobación extra, HU estándar 30 bb chip EV con MCCFR.
                let tree = if spot == "hus" {
                    build_tree(PreflopConfig::standard(vec![st; 2], Payoff::ChipEv))
                } else if spot == "4p" {
                    // Comprobación extra: 4 jugadores estándar 30 bb chip EV con MCCFR.
                    build_tree(PreflopConfig::standard(vec![st; 4], Payoff::ChipEv))
                } else {
                    build_tree(PreflopConfig::standard(vec![st; 3], Payoff::Icm { payouts: vec![50.0, 30.0, 20.0] }))
                };
                let pool = BoardPool::load_or_build(&PoolConfig { boards: 128, threads: cfg.threads, ..PoolConfig::default() }, std::path::Path::new(".cache"));
                let mut est = PostflopEngine::new(pool, 22, bpv);
                // Mismos valores por defecto que `solve`.
                est.adaptive_boards = true;
                est.prune_mass = prune;
                est.avg_power = pfavg;
                est.boards_ramp = ramp;
                Solver::new(tree, spot_eq.clone(), est, cfg)
            };
            // `precd=N`: presolución del mismo spot con el postflop en check-down (mucho más
            // barata) durante N iteraciones y arranque en caliente del MCCFR desde ella; su
            // tiempo cuenta en la tabla.
            if let Some(n) = kv("precd").map(|v| v.parse::<u32>().unwrap()) {
                let t = Instant::now();
                let cd_cfg = PreflopConfig { postflop: None, ..make(30.0).tree.cfg.clone() };
                let mut pre = Solver::new(build_tree(cd_cfg), spot_eq.clone(), CheckDown { eq: spot_eq.clone() }, cfg);
                pre.run(n);
                let mut s = make(30.0);
                let copied = s.warm_start_from(&pre);
                s.iteration = kv("preit").map(|v| v.parse().unwrap()).unwrap_or(s.iteration);
                let secs = t.elapsed().as_secs_f64();
                eprintln!("presolución check-down: {n} iteraciones, {copied} nodos, {secs:.2} s");
                bench_from(s, s_iters(iters, n), every, eval, secs);
                return;
            }
            run(make, 30.0, warm, warm_iters, iters, every, eval);
        }
    }
}

fn s_iters(iters: u32, pre: u32) -> u32 {
    iters + pre
}

#[allow(clippy::too_many_arguments)]
fn run<E: PostflopEstimator>(make: impl Fn(f64) -> Solver<E>, stack: f64, warm: Option<f64>, warm_iters: u32, iters: u32, every: u32, eval: bool) {
    let mut s = make(stack);
    if let Some(ws) = warm {
        let t = Instant::now();
        let mut src = make(ws);
        src.run(warm_iters);
        let n = s.warm_start_from(&src);
        eprintln!("arranque en caliente desde {ws} bb: {n}/{} nodos ({:.1} s)", s.tree.paths.len(), t.elapsed().as_secs_f64());
        if std::env::args().any(|a| a == "check") {
            let g0: f64 = src.best_response_gains().iter().sum();
            let g1: f64 = s.best_response_gains().iter().sum();
            eprintln!("NashConv origen {g0:.5}, copia {g1:.5}");
        }
    }
    bench(s, iters, every, eval);
}

fn bench<E: PostflopEstimator>(s: Solver<E>, iters: u32, every: u32, eval: bool) {
    bench_from(s, iters, every, eval, 0.0);
}

fn bench_from<E: PostflopEstimator>(mut s: Solver<E>, iters: u32, every: u32, eval: bool, start: f64) {
    let mut cfr = start;
    println!("iter\tsegundos\tnashconv");
    while s.iteration < iters {
        let t = Instant::now();
        s.run(every.min(iters - s.iteration));
        cfr += t.elapsed().as_secs_f64();
        let g: f64 = if eval { s.best_response_gains().iter().sum() } else { f64::NAN };
        println!("{}\t{cfr:.2}\t{g:.5}", s.iteration);
    }
}
