//! Árbitro del card bunching: EV exacto (Monte Carlo con reparto real) de empujar frente a
//! foldear para la SB en `HJ:F CO:F BTN:F` del spot 10/12/5/15/7 bb, 50/30/20. Cada
//! reparto de los que foldearon pondera por la probabilidad de que foldeen con sus manos;
//! las manos de la BB se enumeran todas (un board por mano), lo que quita casi todo el
//! ruido de `P(call)`.
//! `bunchcheck fold_HJ fold_CO fold_BTN call_BB 72o 84o …` (rangos con pesos)
//! Entorno: `NOBUNCH` (manos independientes), `NOBOARD` (el board no ve las cartas de los
//! que foldearon), `DECK=δ0,…,δ12` (board del mazo ponderado por rango, el modelo del
//! solver sin linealizar), `N` repartos (20 000), `SEED`.
use preflop_engine::cfr::report::{class_name, grid_cell};
use preflop_engine::eval::evaluator::rank_mask;
use preflop_engine::eval::{COMBO_MASKS, NUM_COMBOS};
use preflop_engine::io::ranges::parse_range;
use preflop_engine::math::icm_equity_precise;
use preflop_engine::math::rng::SplitMix64;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let r: Vec<Vec<f32>> = a[..4].iter().map(|s| parse_range(s).unwrap()).collect();
    let bunch = std::env::var("NOBUNCH").is_err();
    let noboard = std::env::var("NOBOARD").is_ok();
    let deck: Option<Vec<f64>> = std::env::var("DECK").ok().map(|d| d.split(',').map(|x| 1.0 + x.parse::<f64>().unwrap()).collect());
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let payouts = [50.0, 30.0, 20.0];
    let icm = |st: &[f64]| -> f64 {
        let mut o = vec![0.0; 5];
        icm_equity_precise(st, &payouts, &mut o);
        o[3]
    };
    // Fold de la SB: 14.5 / 7.5. Push y la BB foldea: 16 / 6. Push y paga: gana / pierde / empata.
    let (fold, steal) = (icm(&[10.0, 12.0, 5.0, 14.5, 7.5]), icm(&[10.0, 12.0, 5.0, 16.0, 6.0]));
    let (win, lose, tie) = (icm(&[10.0, 12.0, 5.0, 22.0, 0.0]), icm(&[10.0, 12.0, 5.0, 8.0, 14.0]), icm(&[10.0, 12.0, 5.0, 15.0, 7.0]));
    let mut rng = SplitMix64::new(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(3));
    let rand_combo = |dead: u64, rng: &mut SplitMix64| loop {
        let h = (rng.next_u64() % NUM_COMBOS as u64) as usize;
        if COMBO_MASKS[h] & dead == 0 {
            return h;
        }
    };
    for class in &a[4..] {
        let combos: Vec<usize> = (0..NUM_COMBOS).filter(|&h| { let (i, j) = grid_cell(h); &class_name(i, j) == class }).collect();
        let (mut num, mut den, mut num_call) = (0.0f64, 0.0f64, 0.0f64);
        for s in 0..n {
            let h = combos[s % combos.len()];
            let mut dead = COMBO_MASKS[h];
            let mut w = 1.0f64;
            for k in 0..3 {
                let x = rand_combo(dead, &mut rng);
                if bunch {
                    dead |= COMBO_MASKS[x];
                }
                w *= r[k][x] as f64;
            }
            if w == 0.0 {
                continue;
            }
            let (mut ev, mut calls, mut cnt) = (0.0f64, 0.0f64, 0.0f64);
            for bb in (0..NUM_COMBOS).filter(|&v| COMBO_MASKS[v] & dead == 0) {
                let call = r[3][bb] as f64;
                cnt += 1.0;
                calls += call;
                ev += (1.0 - call) * steal;
                if call == 0.0 {
                    continue;
                }
                let base = if noboard || deck.is_some() { COMBO_MASKS[h] | COMBO_MASKS[bb] } else { dead | COMBO_MASKS[bb] };
                let board = loop {
                    let mut b = 0u64;
                    while b.count_ones() < 5 {
                        let c = 1u64 << (rng.next_u64() % 52);
                        if (base | b) & c == 0 {
                            b |= c;
                        }
                    }
                    let Some(d) = &deck else { break b };
                    let (mut p, mut m) = (1.0, b);
                    while m != 0 {
                        p *= d[(m.trailing_zeros() >> 2) as usize];
                        m &= m - 1;
                    }
                    if (rng.next_f32() as f64) * 1.7 < p {
                        break b;
                    }
                };
                let (x, y) = (rank_mask(board | COMBO_MASKS[h]), rank_mask(board | COMBO_MASKS[bb]));
                ev += call * match x.cmp(&y) {
                    std::cmp::Ordering::Greater => win,
                    std::cmp::Ordering::Less => lose,
                    _ => tie,
                };
            }
            num += w * (ev / cnt - fold);
            num_call += w * calls / cnt;
            den += w;
        }
        println!("{class}\tpush−fold {:+.3}  P(call) {:.4}", num / den, num_call / den);
    }
}
