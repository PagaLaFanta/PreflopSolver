//! Probabilidades de los 13 resultados de un bote a 3 (`PATTERNS3`) según `board_orders`,
//! para comparar con `mwcheck`. `mworders "rango A" "rango B" Q9s …`
use preflop_engine::cfr::multiway::board_orders;
use preflop_engine::cfr::report::{class_name, grid_cell};
use preflop_engine::eval::showdown::sample_ranked_boards;
use preflop_engine::eval::{HAND_STRIDE, NUM_COMBOS};
use preflop_engine::io::ranges::parse_range;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut ra = preflop_engine::cfr::zero_range();
    let mut rb = preflop_engine::cfr::zero_range();
    ra[..HAND_STRIDE].copy_from_slice(&parse_range(&args[0]).unwrap());
    rb[..HAND_STRIDE].copy_from_slice(&parse_range(&args[1]).unwrap());
    let boards = sample_ranked_boards(2000, 11, 4);
    let mut probs = vec![[0.0f32; 13]; HAND_STRIDE];
    let mut weight = vec![0.0f32; HAND_STRIDE];
    let mut num = vec![[0.0f64; 13]; HAND_STRIDE];
    let mut den = vec![0.0f64; HAND_STRIDE];
    for b in &boards {
        board_orders(b, &ra, &rb, &mut probs, &mut weight);
        for &h in b.order.iter() {
            let h = h as usize;
            if weight[h] > 0.0 {
                den[h] += weight[h] as f64;
                for k in 0..13 {
                    num[h][k] += (weight[h] * probs[h][k]) as f64;
                }
            }
        }
    }
    for class in &args[2..] {
        let (mut p, mut n) = ([0.0f64; 13], 0.0);
        for h in (0..NUM_COMBOS).filter(|&h| { let (i, j) = grid_cell(h); &class_name(i, j) == class }) {
            for k in 0..13 {
                p[k] += num[h][k] / den[h];
            }
            n += 1.0;
        }
        let p: Vec<String> = p.iter().map(|x| format!("{:.4}", x / n)).collect();
        println!("{class}\tórdenes {}", p.join(" "));
    }
}
