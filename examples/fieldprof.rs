//! Tiempos y precisión del ICM de campo según el paso: `cargo run --release --example fieldprof`.
use preflop_engine::math::field_icm::{FieldIcm, DEFAULT_STEP};
use std::time::Instant;
fn main() {
    let payouts: Vec<f64> = (0..900).map(|k| 20000.0 / (k as f64 + 3.0).powf(1.2)).collect();
    let mut off = vec![40.0; 1500];
    off.extend(vec![25.0; 2000]);
    off.extend(vec![60.0; 1492]);
    let table = [30.0, 45.0, 12.0, 60.0, 25.0, 8.0, 33.0, 50.0];
    let mut reference = vec![0.0; 8];
    FieldIcm::with_step(&off, table.iter().sum(), 0.05, &payouts, 4, DEFAULT_STEP / 4.0).equity(&table, 900, &mut reference);
    for mult in [0.5, 1.0, 1.5, 2.0] {
        let t = Instant::now();
        let f = FieldIcm::with_step(&off, table.iter().sum(), 0.05, &payouts, 4, DEFAULT_STEP * mult);
        let pre = t.elapsed();
        let t = Instant::now();
        let mut e = vec![0.0; 8];
        for _ in 0..20 {
            f.equity(&table, 900, &mut e);
        }
        let err = e.iter().zip(&reference).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
        println!("paso ×{mult}: {} nodos, precomputación {:?}, evaluación {:?}, error relativo máx {:.2e}", f.nodes(), pre, t.elapsed() / 20, err);
    }
}
