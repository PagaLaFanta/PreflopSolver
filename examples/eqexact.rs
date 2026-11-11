//! Construye la matriz de equity exacta (y la guarda en `.cache/equity_exact.bin`).
use preflop_engine::eval::equity::EquityMatrix;
use preflop_engine::eval::{Card, Combo};
fn main() {
    let t0 = std::time::Instant::now();
    let m = EquityMatrix::load_or_build(std::path::Path::new(".cache/equity_exact.bin"), 0, 0, 4);
    println!("{:.1}s", t0.elapsed().as_secs_f64());
    let c = |a: (u8, u8), b: (u8, u8)| Combo::from_cards(Card::new(a.0, a.1), Card::new(b.0, b.1)).0 as usize;
    let (aa, kk, aks, qq) = (c((12, 0), (12, 1)), c((11, 2), (11, 3)), c((12, 2), (11, 2)), c((10, 0), (10, 1)));
    println!("AhAs vs KdKc win {:.5} tie {:.5}", m.win_row(aa)[kk], m.tie_row(aa)[kk]);
    println!("AKs vs QQ win {:.5} tie {:.5}", m.win_row(aks)[qq], m.tie_row(aks)[qq]);
}
