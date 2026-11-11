//! EV exacto (Monte Carlo con reparto real, side pots e ICM) de pagar o foldear para la BB
//! en `HJ:F CO:A BTN:F SB:C` del spot 10/12/5/15/7 bb, 50/30/20. Árbitro de los botes a 3.
//! `icm3check "rango push CO" "rango call SB" AKs 72o …`
use preflop_engine::cfr::report::{class_name, grid_cell};
use preflop_engine::eval::evaluator::rank_mask;
use preflop_engine::eval::{COMBO_MASKS, NUM_COMBOS};
use preflop_engine::io::ranges::parse_range;
use preflop_engine::math::icm_equity_precise;
use preflop_engine::math::rng::SplitMix64;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (rco, rsb) = (parse_range(&a[0]).unwrap(), parse_range(&a[1]).unwrap());
    let payouts = [50.0, 30.0, 20.0];
    // Asientos: HJ 0, CO 1, BTN 2, SB 3, BB 4.
    let start = [10.0, 12.0, 5.0, 15.0, 7.0];
    let mut rng = SplitMix64::new(9);
    let pick = |r: &[f32], dead: u64, rng: &mut SplitMix64| -> usize {
        let tot: f32 = (0..NUM_COMBOS).filter(|&h| COMBO_MASKS[h] & dead == 0).map(|h| r[h]).sum();
        let mut x = rng.next_f32() * tot;
        for h in 0..NUM_COMBOS {
            if COMBO_MASKS[h] & dead == 0 {
                x -= r[h];
                if x <= 0.0 {
                    return h;
                }
            }
        }
        (0..NUM_COMBOS).rev().find(|&h| COMBO_MASKS[h] & dead == 0 && r[h] > 0.0).unwrap()
    };
    let icm = |st: &[f64]| -> f64 {
        let mut o = vec![0.0; 5];
        icm_equity_precise(st, &payouts, &mut o);
        o[4]
    };
    for class in &a[2..] {
        let combos: Vec<usize> = (0..NUM_COMBOS).filter(|&h| { let (i, j) = grid_cell(h); &class_name(i, j) == class }).collect();
        let (mut call, mut fold, n) = (0.0f64, 0.0f64, 200_000usize);
        for s in 0..n {
            let h = combos[s % combos.len()];
            let co = pick(&rco, COMBO_MASKS[h], &mut rng);
            // INDEP: la SB solo evita las cartas de la BB (modelo de independencia).
            let sb = pick(&rsb, if std::env::var("INDEP").is_ok() { COMBO_MASKS[h] } else { COMBO_MASKS[h] | COMBO_MASKS[co] }, &mut rng);
            let mut used = COMBO_MASKS[h] | COMBO_MASKS[co] | COMBO_MASKS[sb];
            let mut board = 0u64;
            while board.count_ones() < 5 {
                let c = 1u64 << (rng.next_u64() % 52);
                if used & c == 0 {
                    board |= c;
                    used |= c;
                }
            }
            let (rb, rc, rs) = (rank_mask(board | COMBO_MASKS[h]), rank_mask(board | COMBO_MASKS[co]), rank_mask(board | COMBO_MASKS[sb]));
            // Fold: la BB pierde 1; CO y SB juegan 12 cada uno (bote 25 con la ciega de la BB).
            let mut st = start;
            st[4] -= 1.0;
            st[1] -= 12.0;
            st[3] -= 12.0;
            match rc.cmp(&rs) {
                std::cmp::Ordering::Greater => st[1] += 25.0,
                std::cmp::Ordering::Less => st[3] += 25.0,
                _ => {
                    st[1] += 12.5;
                    st[3] += 12.5;
                }
            }
            fold += icm(&st);
            // Call: bote principal 21 entre los tres; side pot 10 entre CO y SB.
            let mut st = start;
            st[4] -= 7.0;
            st[1] -= 12.0;
            st[3] -= 12.0;
            let best = rb.max(rc).max(rs);
            let win: Vec<usize> = [(4, rb), (1, rc), (3, rs)].iter().filter(|x| x.1 == best).map(|x| x.0).collect();
            for &w in &win {
                st[w] += 21.0 / win.len() as f64;
            }
            match rc.cmp(&rs) {
                std::cmp::Ordering::Greater => st[1] += 10.0,
                std::cmp::Ordering::Less => st[3] += 10.0,
                _ => {
                    st[1] += 5.0;
                    st[3] += 5.0;
                }
            }
            call += icm(&st);
        }
        println!("{class}\tcall−fold {:+.3}  (call {:.3}, fold {:.3})", (call - fold) / n as f64, call / n as f64, fold / n as f64);
    }
}
