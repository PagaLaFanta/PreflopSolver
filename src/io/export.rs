//! Exportaciones de una solución: EV por clase (TSV), estrategia completa (JSON) y rangos
//! de cada flop heads-up para llevarlos a un solver postflop (formato PioSolver).

use crate::cfr::report::{class_grid, class_name, grid_cell};
use crate::cfr::{PostflopEstimator, Solver};
use crate::eval::{HAND_STRIDE, NUM_COMBOS};
use crate::io::ranges::format_range;
use crate::tree::builder::seat_names;
use crate::tree::nodes::{NodeKind, NodeRef, TerminalKind};
use crate::tree::Payoff;

/// Cadena JSON con escapes.
pub fn json_string(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_num(x: f64) -> String {
    if x.is_finite() {
        format!("{}", (x * 1e4).round() / 1e4)
    } else {
        "null".into()
    }
}

/// EV medio por clase de mano (169 valores, fila a fila del grid) de una fila
/// `[combo]` de EV; `NaN` en las clases sin combos definidos.
pub fn class_ev(row: &[f32]) -> [f64; 169] {
    let (mut sum, mut cnt) = ([0.0f64; 169], [0u32; 169]);
    for (h, &v) in row[..NUM_COMBOS].iter().enumerate() {
        if v.is_finite() {
            let (i, j) = grid_cell(h);
            sum[i * 13 + j] += v as f64;
            cnt[i * 13 + j] += 1;
        }
    }
    std::array::from_fn(|c| if cnt[c] > 0 { sum[c] / cnt[c] as f64 } else { f64::NAN })
}

fn path_or_root(p: &str) -> &str {
    if p.is_empty() {
        "(raíz)"
    } else {
        p
    }
}

fn class_cells(row: &[f32], offset: &[f64; 169]) -> String {
    class_ev(row).iter().zip(offset).map(|(v, o)| if v.is_finite() { format!("{:.4}", v - o) } else { String::new() }).collect::<Vec<_>>().join(",")
}

/// `--export-ev`: `ruta \t asiento \t acción \t EV del modelo (169) \t EV en fichas (169)`,
/// vacío donde no hay EV. Ambos **relativos a foldear ese mismo nodo** (foldear
/// siempre da 0; el resto es la ganancia/pérdida frente a foldear), no el valor absoluto
/// del stack final — CFR converge igual con cualquiera de los dos (es invariante a sumar
/// una constante), pero comparar a simple vista con otras herramientas necesita la misma referencia. Si
/// el nodo no tiene foldear como opción (p.ej. BB tras limps, sin nadie que pague), se deja
/// en absoluto ($) o neto del stack inicial (fichas), como antes. `evs`:
/// [`Solver::action_evs`]; `chips`: lo mismo con la tabla de fichas ([`Solver::with_utils`])
/// o `evs` si el modelo ya es chip EV.
pub fn ev_tsv<E: PostflopEstimator>(s: &Solver<E>, evs: &[Vec<f32>], chips: &[Vec<f32>]) -> String {
    let names = seat_names(s.tree.n);
    let mut out = String::new();
    for k in 0..s.tree.actions.len() {
        let p = s.tree.actions[k].player as usize;
        let stack = s.tree.cfg.stacks[p];
        let fold_a = s.tree.labels[k].iter().position(|l| l == "F").filter(|_| std::env::var("EV_ABS").is_err());
        let ev_base = fold_a.map_or([0.0; 169], |a| class_ev(&evs[k][a * HAND_STRIDE..(a + 1) * HAND_STRIDE]));
        let chip_base = fold_a.map_or([stack; 169], |a| class_ev(&chips[k][a * HAND_STRIDE..(a + 1) * HAND_STRIDE]));
        for (a, label) in s.tree.labels[k].iter().enumerate() {
            let r = a * HAND_STRIDE..(a + 1) * HAND_STRIDE;
            out.push_str(&format!("{}\t{}\t{label}\t{}\t{}\n", path_or_root(&s.tree.paths[k]), names[p], class_cells(&evs[k][r.clone()], &ev_base), class_cells(&chips[k][r], &chip_base)));
        }
    }
    out
}

/// Flop heads-up alcanzable: ruta, asientos, bote y stack efectivo (bb) y el alcance de
/// cada jugador por combo (producto de sus frecuencias a lo largo de la ruta).
#[derive(Clone, Debug)]
pub struct FlopSpot {
    pub path: String,
    pub oop: usize,
    pub ip: usize,
    pub pot: f64,
    pub eff_stack: f64,
    pub oop_range: Vec<f32>,
    pub ip_range: Vec<f32>,
}

/// Todos los flops heads-up del árbol (los multiway no tienen equivalente en Pio).
pub fn flop_spots<E: PostflopEstimator>(s: &Solver<E>) -> Vec<FlopSpot> {
    fn rec<E: PostflopEstimator>(s: &Solver<E>, node: NodeRef, path: &str, reach: &[Vec<f32>], out: &mut Vec<FlopSpot>) {
        match node.kind() {
            NodeKind::Action => {
                let k = node.index();
                let an = &s.tree.actions[k];
                let p = an.player as usize;
                let seat = seat_names(s.tree.n)[p];
                let sigma = s.average_strategy(k);
                for a in 0..an.num_actions as usize {
                    let mut r = reach.to_vec();
                    for h in 0..NUM_COMBOS {
                        r[p][h] *= sigma[a * HAND_STRIDE + h];
                    }
                    let step = format!("{seat}:{}", s.tree.labels[k][a]);
                    let child = if path.is_empty() { step } else { format!("{path} {step}") };
                    rec(s, an.children[a], &child, &r, out);
                }
            }
            NodeKind::Terminal => {
                let t = &s.tree.terminals[node.index()];
                if t.kind != TerminalKind::FlopLeaf {
                    return;
                }
                let l = &s.tree.leaves[t.leaf as usize];
                if l.multi.is_some() {
                    return;
                }
                let (oop, ip) = (l.oop as usize, l.ip as usize);
                out.push(FlopSpot { path: path.to_string(), oop, ip, pot: l.pot, eff_stack: l.eff_stack, oop_range: reach[oop].clone(), ip_range: reach[ip].clone() });
            }
            _ => {}
        }
    }
    let reach = vec![vec![1.0f32; HAND_STRIDE]; s.tree.n];
    let mut out = Vec::new();
    rec(s, s.tree.root, "", &reach, &mut out);
    out
}

/// `--export-pio`: `ruta \t OOP \t IP \t bote \t stack efectivo \t rango OOP \t rango IP`
/// (bb; rangos en notación estándar con pesos, que PioSolver acepta tal cual).
pub fn pio_tsv<E: PostflopEstimator>(s: &Solver<E>) -> String {
    let names = seat_names(s.tree.n);
    flop_spots(s)
        .iter()
        .map(|f| format!("{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\n", f.path, names[f.oop], names[f.ip], f.pot, f.eff_stack, format_range(&f.oop_range), format_range(&f.ip_range)))
        .collect()
}

/// Script de PioSolver para un flop de [`pio_tsv`] (1 bb = 100 fichas). Falta el board,
/// que se elige en Pio.
pub fn pio_script(path: &str, oop: &str, ip: &str, pot: f64, eff: f64, oop_range: &str, ip_range: &str) -> String {
    let chips = |x: f64| (x * 100.0).round() as i64;
    format!(
        "# preflop_engine → PioSolver\n# Ruta preflop: {path}\n# OOP = {oop}, IP = {ip}; bote {pot:.2} bb, stack efectivo {eff:.2} bb (1 bb = 100 fichas)\n\
         set_range OOP {oop_range}\nset_range IP {ip_range}\nset_pot 0 0 {}\nset_eff_stack {}\n",
        chips(pot),
        chips(eff)
    )
}

/// `--export-json`: configuración, EV por asiento, y por nodo y acción la frecuencia
/// global, la frecuencia y el EV por clase de mano (`{"AA": …}`), más los flops HU con
/// sus rangos. Pensado para herramientas externas.
pub fn solution_json<E: PostflopEstimator>(s: &Solver<E>, evs: &[Vec<f32>], chips: &[Vec<f32>], chip_seats: &[f64]) -> String {
    let names = seat_names(s.tree.n);
    let cfg = &s.tree.cfg;
    let list = |v: &[f64]| v.iter().map(|&x| json_num(x)).collect::<Vec<_>>().join(",");
    let (model, payouts, bounties) = match &cfg.payoff {
        Payoff::ChipEv => ("chip_ev", vec![], vec![]),
        Payoff::Icm { payouts } => ("icm", payouts.clone(), vec![]),
        Payoff::Pko { payouts, bounties, .. } => ("pko", payouts.clone(), bounties.clone()),
        Payoff::PkoChipEv { bounties, .. } => ("pko_chip_ev", vec![], bounties.clone()),
    };
    let ev = s.expected_values();
    let chip_net: Vec<f64> = chip_seats.iter().zip(&cfg.stacks).map(|(e, st)| e - st).collect();
    let mut o = String::new();
    o.push_str(&format!(
        "{{\"format\":\"preflop_engine/1\",\"iterations\":{},\"config\":{{\"seats\":[{}],\"stacks\":[{}],\"sb\":{},\"bb\":{},\"ante\":{},\"bb_ante\":{},\"model\":\"{model}\",\"payouts\":[{}],\"bounties\":[{}],\"offtable\":[{}]}},\"ev\":[{}],\"ev_chips\":[{}],\"nodes\":[",
        s.iteration,
        names.iter().map(|n| json_string(n)).collect::<Vec<_>>().join(","),
        list(&cfg.stacks),
        json_num(cfg.sb),
        json_num(cfg.bb),
        json_num(cfg.ante),
        cfg.bb_ante,
        list(&payouts),
        list(&bounties),
        list(&cfg.offtable_stacks),
        list(&ev),
        list(&chip_net)
    ));
    for k in 0..s.tree.actions.len() {
        if k > 0 {
            o.push(',');
        }
        let strat = s.average_strategy(k);
        let stack = cfg.stacks[s.tree.actions[k].player as usize];
        // Relativo a foldear ese mismo nodo, como `ev_tsv`: foldear da 0.
        let fold_a = s.tree.labels[k].iter().position(|l| l == "F");
        let ev_base = fold_a.map_or([0.0; 169], |a| class_ev(&evs[k][a * HAND_STRIDE..(a + 1) * HAND_STRIDE]));
        let chip_base = fold_a.map_or([stack; 169], |a| class_ev(&chips[k][a * HAND_STRIDE..(a + 1) * HAND_STRIDE]));
        o.push_str(&format!("{{\"path\":{},\"seat\":{},\"actions\":[", json_string(&s.tree.paths[k]), json_string(names[s.tree.actions[k].player as usize])));
        for (a, label) in s.tree.labels[k].iter().enumerate() {
            if a > 0 {
                o.push(',');
            }
            let g = class_grid(&strat, a);
            let e = class_ev(&evs[k][a * HAND_STRIDE..(a + 1) * HAND_STRIDE]);
            let ec = class_ev(&chips[k][a * HAND_STRIDE..(a + 1) * HAND_STRIDE]);
            let freq = strat[a * HAND_STRIDE..a * HAND_STRIDE + NUM_COMBOS].iter().map(|&x| x as f64).sum::<f64>() / NUM_COMBOS as f64;
            let (mut fs, mut es, mut cs) = (Vec::with_capacity(169), Vec::with_capacity(169), Vec::with_capacity(169));
            for i in 0..13 {
                for j in 0..13 {
                    let c = class_name(i, j);
                    fs.push(format!("\"{c}\":{}", json_num(g[i][j] as f64)));
                    es.push(format!("\"{c}\":{}", json_num(e[i * 13 + j] - ev_base[i * 13 + j])));
                    cs.push(format!("\"{c}\":{}", json_num(ec[i * 13 + j] - chip_base[i * 13 + j])));
                }
            }
            o.push_str(&format!("{{\"label\":{},\"freq\":{},\"strategy\":{{{}}},\"ev\":{{{}}},\"ev_chips\":{{{}}}}}", json_string(label), json_num(freq), fs.join(","), es.join(","), cs.join(",")));
        }
        o.push_str("]}");
    }
    o.push_str("],\"flops\":[");
    let flops: Vec<String> = flop_spots(s)
        .iter()
        .map(|f| {
            format!(
                "{{\"path\":{},\"oop\":{},\"ip\":{},\"pot\":{},\"eff_stack\":{},\"oop_range\":{},\"ip_range\":{}}}",
                json_string(&f.path),
                json_string(names[f.oop]),
                json_string(names[f.ip]),
                json_num(f.pot),
                json_num(f.eff_stack),
                json_string(&format_range(&f.oop_range)),
                json_string(&format_range(&f.ip_range))
            )
        })
        .collect();
    o.push_str(&flops.join(","));
    o.push_str("]}\n");
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::{CheckDown, SolverConfig};
    use crate::eval::equity::EquityMatrix;
    use crate::tree::{build_tree, PreflopConfig};
    use std::sync::Arc;

    #[test]
    fn escapes_json() {
        assert_eq!(json_string("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
    }

    /// HU 20 bb estándar en check-down: hay flops HU, sus alcances son subrangos del
    /// rango completo y el bote del limp-check es 2 bb; el JSON trae los 169 valores.
    #[test]
    fn flop_spots_and_json() {
        let eq = Arc::new(EquityMatrix::monte_carlo(2000, 7, 2));
        let mut cfg = PreflopConfig::standard(vec![20.0, 20.0], Payoff::ChipEv);
        cfg.postflop = None;
        let mut s = Solver::new(build_tree(cfg), eq.clone(), CheckDown { eq }, SolverConfig::default());
        s.run(30);
        let flops = flop_spots(&s);
        assert!(!flops.is_empty());
        for f in &flops {
            assert!(f.oop_range[..NUM_COMBOS].iter().chain(&f.ip_range[..NUM_COMBOS]).all(|&x| (0.0..=1.0 + 1e-6).contains(&x)));
            assert!(f.pot >= 2.0 - 1e-9 && f.eff_stack <= 20.0 + 1e-9, "{} {} {}", f.path, f.pot, f.eff_stack);
        }
        let limp = flops.iter().find(|f| f.path == "SB:C BB:X").expect("limp-check");
        assert!((limp.pot - 2.0).abs() < 1e-9 && (limp.eff_stack - 19.0).abs() < 1e-9);
        let evs = s.action_evs();
        let seats = s.expected_values();
        let j = solution_json(&s, &evs, &evs, &seats);
        assert!(j.starts_with("{\"format\":\"preflop_engine/1\""));
        assert_eq!(j.matches("\"AA\":").count(), 3 * s.tree.labels.iter().map(|l| l.len()).sum::<usize>());
        assert!(ev_tsv(&s, &evs, &evs).lines().all(|l| l.split('\t').count() == 5));
        assert!(j.contains("\"flops\":[{"));
        let script = pio_script("SB:C BB:X", "BB", "SB", 2.0, 19.0, "AA", "KK");
        assert!(script.contains("set_pot 0 0 200") && script.contains("set_eff_stack 1900"));
    }
}
