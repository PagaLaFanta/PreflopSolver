//! Equity de un combo contra un rango: matriz de equity frente a Monte Carlo directo.
use preflop_engine::cfr::report::{class_name, grid_cell};
use preflop_engine::eval::equity::EquityMatrix;
use preflop_engine::eval::evaluator::rank_mask;
use preflop_engine::eval::{COMBO_MASKS, NUM_COMBOS};
use preflop_engine::io::ranges::parse_range;
use preflop_engine::math::rng::SplitMix64;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let r = parse_range(&a[0]).unwrap();
    let eq = EquityMatrix::load(std::path::Path::new(".cache/equity_exact.bin")).unwrap();
    let mut rng = SplitMix64::new(9);
    for class in &a[1..] {
        let h = (0..NUM_COMBOS).find(|&h| { let (i, j) = grid_cell(h); &class_name(i, j) == class }).unwrap();
        let (mut mw, mut mt, mut den) = (0.0f64, 0.0f64, 0.0f64);
        let (mut sw, mut st, mut sd) = (0.0f64, 0.0f64, 0.0f64);
        for v in (0..NUM_COMBOS).filter(|&v| COMBO_MASKS[v] & COMBO_MASKS[h] == 0 && r[v] > 0.0) {
            let x = r[v] as f64;
            mw += x * eq.win_row(h)[v] as f64;
            mt += x * eq.tie_row(h)[v] as f64;
            den += x;
            for _ in 0..2000 {
                let base = COMBO_MASKS[h] | COMBO_MASKS[v];
                let mut b = 0u64;
                while b.count_ones() < 5 {
                    let c = 1u64 << (rng.next_u64() % 52);
                    if (base | b) & c == 0 {
                        b |= c;
                    }
                }
                let (p, q) = (rank_mask(b | COMBO_MASKS[h]), rank_mask(b | COMBO_MASKS[v]));
                sw += x * (p > q) as u8 as f64;
                st += x * (p == q) as u8 as f64;
                sd += x;
            }
        }
        println!("{class}: matriz win {:.4} tie {:.4} | MC win {:.4} tie {:.4}", mw / den, mt / den, sw / sd, st / sd);
    }
}
