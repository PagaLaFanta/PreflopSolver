//! Error del modelo de card bunching de rangos, aislado del board: EV del push de la SB en
//! `HJ:F CO:F BTN:F` (10/12/5/15/7 bb, 50/30/20) con la matriz de equity (board uniforme)
//! y distintos pesos para la mano de la BB dada la del héroe:
//! exacto (Monte Carlo de repartos de los que foldearon), el modelo del solver (producto de
//! `m_k(v)/m̄_k`) y variantes.
use preflop_engine::cfr::report::{class_name, grid_cell};
use preflop_engine::eval::equity::EquityMatrix;
use preflop_engine::eval::{COMBO_CARDS, COMBO_MASKS, NUM_COMBOS};
use preflop_engine::io::ranges::parse_range;
use preflop_engine::math::icm_equity_precise;
use preflop_engine::math::rng::SplitMix64;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let r: Vec<Vec<f32>> = a[..4].iter().map(|s| parse_range(s).unwrap()).collect();
    let eq = EquityMatrix::load(std::path::Path::new(".cache/equity_exact.bin")).unwrap();
    let payouts = [50.0, 30.0, 20.0];
    let icm = |st: &[f64]| -> f64 {
        let mut o = vec![0.0; 5];
        icm_equity_precise(st, &payouts, &mut o);
        o[3]
    };
    let (fold, steal) = (icm(&[10.0, 12.0, 5.0, 14.5, 7.5]), icm(&[10.0, 12.0, 5.0, 16.0, 6.0]));
    let (win, lose, tie) = (icm(&[10.0, 12.0, 5.0, 22.0, 0.0]), icm(&[10.0, 12.0, 5.0, 8.0, 14.0]), icm(&[10.0, 12.0, 5.0, 15.0, 7.0]));
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    // Masa de un rango compatible con un conjunto de cartas (exacta, inclusión-exclusión).
    let avoid = |r: &[f32], dead: u64| -> f64 { (0..NUM_COMBOS).filter(|&x| COMBO_MASKS[x] & dead == 0).map(|x| r[x] as f64).sum() };
    let ev_of = |h: usize, w: &dyn Fn(usize) -> f64| -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for v in (0..NUM_COMBOS).filter(|&v| COMBO_MASKS[v] & COMBO_MASKS[h] == 0) {
            let wv = w(v);
            let c = r[3][v] as f64;
            let (pw, pt) = (eq.win_row(h)[v] as f64, eq.tie_row(h)[v] as f64);
            num += wv * ((1.0 - c) * steal + c * (pw * win + pt * tie + (1.0 - pw - pt) * lose));
            den += wv;
        }
        num / den - fold
    };
    let mut rng = SplitMix64::new(7);
    for class in &a[4..] {
        let combos: Vec<usize> = (0..NUM_COMBOS).filter(|&h| { let (i, j) = grid_cell(h); &class_name(i, j) == class }).collect();
        let (mut ex, mut md, mut m1, mut un) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for &h in &combos {
            // Exacto: repartos de los tres (exclusión mutua) ponderados por foldear.
            let mut wv = vec![0.0f64; NUM_COMBOS];
            for _ in 0..n / combos.len() {
                let mut dead = COMBO_MASKS[h];
                let mut w = 1.0f64;
                for k in 0..3 {
                    let x = loop {
                        let x = (rng.next_u64() % NUM_COMBOS as u64) as usize;
                        if COMBO_MASKS[x] & dead == 0 {
                            break x;
                        }
                    };
                    dead |= COMBO_MASKS[x];
                    w *= r[k][x] as f64;
                }
                if w > 0.0 {
                    for v in 0..NUM_COMBOS {
                        if COMBO_MASKS[v] & dead == 0 {
                            wv[v] += w;
                        }
                    }
                }
            }
            ex += ev_of(h, &|v| wv[v]);
            un += ev_of(h, &|_| 1.0);
            // Modelo del solver: Π_k m_k(v).
            md += ev_of(h, &|v| (0..3).map(|k| avoid(&r[k], COMBO_MASKS[v])).product());
            // Cada folder exacto con h y v juntos, independientes entre sí.
            m1 += ev_of(h, &|v| (0..3).map(|k| avoid(&r[k], COMBO_MASKS[v] | COMBO_MASKS[h])).product());
        }
        let k = combos.len() as f64;
        println!("{class}\texacto {:+.3}  modelo {:+.3}  folder exacto {:+.3}  sin bunching {:+.3}", ex / k, md / k, m1 / k, un / k);
        let _ = COMBO_CARDS;
    }
}
