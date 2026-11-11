//! Equity HU de cada clase contra un rango (matriz de equity del solver) y umbral:
//! `hueq "rango" umbral` lista las clases por encima del umbral y su % de combos.
use preflop_engine::cfr::report::{class_name, grid_cell};
use preflop_engine::eval::equity::EquityMatrix;
use preflop_engine::eval::NUM_COMBOS;
use preflop_engine::io::ranges::parse_range;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = parse_range(&args[0]).unwrap();
    let thr: f32 = args[1].parse().unwrap();
    let eq = EquityMatrix::load_or_build(std::path::Path::new(".cache/equity_20000.bin"), 20000, 0x5EED, 4);
    let p = eq.products(&r);
    let mut by: std::collections::BTreeMap<String, (f32, u32)> = Default::default();
    for h in 0..NUM_COMBOS {
        if p.all[h] <= 0.0 {
            continue;
        }
        let e = (p.w[h] + 0.5 * p.t[h]) / p.all[h];
        let (i, j) = grid_cell(h);
        let x = by.entry(class_name(i, j)).or_default();
        x.0 += e;
        x.1 += 1;
        if std::env::var("PC").is_ok_and(|c| c == class_name(i, j)) {
            eprintln!("{} combo {h}: compat {:.1} (pc {:.4}), W {:.2} T {:.2}, eq {:.4}", class_name(i, j), p.all[h], p.all[h] / 1225.0, p.w[h], p.t[h], e);
        }
    }
    let (mut combos, mut list) = (0u32, Vec::new());
    for (c, (s, n)) in &by {
        let e = s / *n as f32;
        if e > thr {
            combos += n;
            list.push(format!("{c}({:.1})", e * 100.0));
        }
    }
    println!("{:.1}% combos: {}", combos as f32 / 13.26, list.join(" "));
}
