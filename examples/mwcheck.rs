//! Comprobación de los botes a 3: equity exacta (Monte Carlo con bloqueo completo entre
//! los tres jugadores) de cada clase del héroe contra dos rangos.
//! `mwcheck "rango A" "rango B" AKs Q9s … [muestras]`
use preflop_engine::cfr::report::{class_name, grid_cell};
use preflop_engine::eval::evaluator::rank_mask;
use preflop_engine::eval::{COMBO_MASKS, NUM_COMBOS};
use preflop_engine::io::ranges::parse_range;
use preflop_engine::math::rng::SplitMix64;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (ra, rb) = (parse_range(&args[0]).unwrap(), parse_range(&args[1]).unwrap());
    let samples: usize = args.last().and_then(|a| a.parse().ok()).unwrap_or(200_000);
    let mut rng = SplitMix64::new(7);
    let indep = std::env::var("INDEP").is_ok();
    let pick = |r: &[f32], dead: u64, rng: &mut SplitMix64| -> Option<usize> {
        let tot: f32 = (0..NUM_COMBOS).filter(|&h| COMBO_MASKS[h] & dead == 0).map(|h| r[h]).sum();
        if tot <= 0.0 {
            return None;
        }
        let mut x = rng.next_f32() * tot;
        for h in 0..NUM_COMBOS {
            if COMBO_MASKS[h] & dead == 0 {
                x -= r[h];
                if x <= 0.0 {
                    return Some(h);
                }
            }
        }
        None
    };
    for class in args[2..].iter().filter(|a| a.parse::<usize>().is_err()) {
        let combos: Vec<usize> = (0..NUM_COMBOS).filter(|&h| { let (i, j) = grid_cell(h); &class_name(i, j) == class }).collect();
        let mut eq = 0.0f64;
        let mut n = 0usize;
        for s in 0..samples {
            let h = combos[s % combos.len()];
            let dead = COMBO_MASKS[h];
            let Some(a) = pick(&ra, dead, &mut rng) else { continue };
            // `indep`: B solo evita las cartas del héroe (el modelo de independencia del
            // solver); puede compartir carta con A.
            let Some(b) = pick(&rb, if indep { dead } else { dead | COMBO_MASKS[a] }, &mut rng) else { continue };
            let mut used = dead | COMBO_MASKS[a] | COMBO_MASKS[b];
            let mut board = 0u64;
            while board.count_ones() < 5 {
                let c = 1u64 << (rng.next_u64() % 52);
                if used & c == 0 {
                    board |= c;
                    used |= c;
                }
            }
            let (x, y, z) = (rank_mask(board | dead), rank_mask(board | COMBO_MASKS[a]), rank_mask(board | COMBO_MASKS[b]));
            let best = x.max(y).max(z);
            if x == best {
                eq += 1.0 / [x, y, z].iter().filter(|&&v| v == best).count() as f64;
            }
            n += 1;
        }
        println!("{class}\t{:.4}", eq / n as f64);
    }
}
