//! Calibrado de las tablas push/fold que usa FGS para jugar las manos futuras: se
//! resuelve con el propio solver el Nash chip-EV push/fold de mesas de 2 a 6 jugadores
//! con stacks iguales sobre una malla de profundidades, con las mismas ciegas y antes del
//! torneo. Resultado cacheado en disco (≈ 30-60 s la primera vez).

use super::report::overall_frequency;
use super::{CheckDown, Solver, SolverConfig};
use crate::eval::equity::EquityMatrix;
use crate::math::fgs::{PushFoldCharts, CHART_MAX_PLAYERS};
use crate::tree::builder::seat_names;
use crate::tree::{build_tree, Payoff, PreflopConfig};
use std::path::Path;
use std::sync::Arc;

pub const DEFAULT_X_GRID: [f64; 8] = [3.0, 5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0];

#[derive(Debug, Clone)]
pub struct ChartSpec {
    pub sb: f64,
    pub bb: f64,
    pub ante: f64,
    pub bb_ante: bool,
    pub x_grid: Vec<f64>,
    pub max_players: usize,
    pub iterations: u32,
}

impl ChartSpec {
    pub fn new(sb: f64, bb: f64, ante: f64, bb_ante: bool) -> Self {
        ChartSpec { sb, bb, ante, bb_ante, x_grid: DEFAULT_X_GRID.to_vec(), max_players: CHART_MAX_PLAYERS, iterations: 150 }
    }

    pub fn cache_name(&self) -> String {
        format!("pushfold_{}_{}_{}_{}_{}_{}.bin", self.sb, self.bb, self.ante, self.bb_ante as u8, self.x_grid.len(), self.iterations)
    }
}

/// Resuelve y extrae las tablas. Tamaños de mesa por encima de `max_players` (hasta 6)
/// replican la tabla más grande calculada.
pub fn build_pushfold_charts(eq: Arc<EquityMatrix>, spec: &ChartSpec) -> PushFoldCharts {
    let mut push = Vec::new();
    let mut call = Vec::new();
    for n in 2..=CHART_MAX_PLAYERS {
        let solve_n = n.min(spec.max_players.max(2));
        let names = seat_names(solve_n);
        let (mut ptab, mut ctab) = (Vec::new(), Vec::new());
        for &x in &spec.x_grid {
            let cfg = PreflopConfig {
                sb: spec.sb,
                bb: spec.bb,
                ante: spec.ante,
                bb_ante: spec.bb_ante,
                max_players_in_pot: 2,
                ..PreflopConfig::push_fold(vec![x * spec.bb; solve_n], Payoff::ChipEv)
            };
            let mut s = Solver::new(build_tree(cfg), eq.clone(), CheckDown { eq: eq.clone() }, SolverConfig { threads: 1, ..SolverConfig::default() });
            s.run(spec.iterations);
            let freq = |path: &str, label: &str| -> f32 {
                match s.tree.find(path) {
                    Some(k) => match s.tree.labels[k].iter().position(|l| l == label) {
                        Some(a) => overall_frequency(&s.average_strategy(k), a),
                        None => 0.0,
                    },
                    None => 0.0,
                }
            };
            let folds = |from: usize, to: usize| -> Vec<String> { (from..to).map(|j| format!("{}:F", names[j])).collect() };
            let mut prow = vec![0.0f32; n];
            let mut crow = vec![0.0f32; n * n];
            // Posiciones de la tabla de `n` sobre la resuelta de `solve_n` (desde la BB).
            let off = n - solve_n;
            for k in 0..solve_n - 1 {
                let path = folds(0, k).join(" ");
                prow[k + off] = freq(&path, "A");
                for m in k + 1..solve_n {
                    let mut p = folds(0, k);
                    p.push(format!("{}:A", names[k]));
                    p.extend(folds(k + 1, m));
                    crow[(k + off) * n + m + off] = freq(&p.join(" "), "C");
                }
            }
            for k in 0..off {
                prow[k] = prow[off];
                for m in k + 1..n {
                    crow[k * n + m] = crow[off * n + m.max(off + 1)];
                }
            }
            ptab.push(prow);
            ctab.push(crow);
        }
        push.push(ptab);
        call.push(ctab);
    }
    PushFoldCharts { x_grid: spec.x_grid.clone(), push, call }
}

/// Ranking de manos "Nash combinado" (calculado con
/// nuestro propio solver): Nash push/fold heads-up chip EV en la malla de stacks, y cada clase
/// puntúa la suma de su frecuencia de push (SB) y de call (BB) en todas las profundidades.
/// Desempate por equity contra una mano al azar. Devuelve los 1326 combos de más fuerte a más
/// débil (clase a clase), para los rangos "top x %" del FGS.
pub fn nash_ranking(eq: Arc<EquityMatrix>, spec: &ChartSpec) -> Vec<u16> {
    use crate::cfr::report::CLASS_COMBOS;
    use crate::eval::{COMBO_MASKS, NUM_COMBOS};
    let mut score = vec![0.0f64; NUM_COMBOS];
    for &x in &spec.x_grid {
        let cfg = PreflopConfig { sb: spec.sb, bb: spec.bb, ante: spec.ante, bb_ante: spec.bb_ante, max_players_in_pot: 2, ..PreflopConfig::push_fold(vec![x * spec.bb; 2], Payoff::ChipEv) };
        let mut s = Solver::new(build_tree(cfg), eq.clone(), CheckDown { eq: eq.clone() }, SolverConfig { threads: 1, ..SolverConfig::default() });
        s.run(spec.iterations);
        let names = seat_names(2);
        for (path, label) in [(String::new(), "A"), (format!("{}:A", names[0]), "C")] {
            if let Some(k) = s.tree.find(&path) {
                if let Some(a) = s.tree.labels[k].iter().position(|l| l == label) {
                    let st = s.average_strategy(k);
                    for h in 0..NUM_COMBOS {
                        score[h] += st[a * crate::eval::HAND_STRIDE + h] as f64;
                    }
                }
            }
        }
    }
    let equity = |h: usize| -> f64 {
        let (w, t) = (eq.win_row(h), eq.tie_row(h));
        let s: f64 = (0..NUM_COMBOS).map(|v| w[v] as f64 + 0.5 * t[v] as f64).sum();
        let cnt = (0..NUM_COMBOS).filter(|&v| COMBO_MASKS[h] & COMBO_MASKS[v] == 0).count();
        s / cnt as f64
    };
    let mut classes: Vec<(f64, f64, usize)> = CLASS_COMBOS
        .iter()
        .enumerate()
        .map(|(c, combos)| {
            let sc = combos.iter().map(|&h| score[h as usize]).sum::<f64>() / combos.len() as f64;
            (sc, equity(combos[0] as usize), c)
        })
        .collect();
    classes.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.total_cmp(&a.1)));
    classes.iter().flat_map(|&(_, _, c)| CLASS_COMBOS[c].iter().copied()).collect()
}

/// [`nash_ranking`] cacheado en disco.
pub fn load_or_build_nash_ranking(dir: &Path, eq: Arc<EquityMatrix>, spec: &ChartSpec) -> Vec<u16> {
    let path = dir.join(format!("nashrank_{}", spec.cache_name()));
    if let Ok(bytes) = std::fs::read(&path) {
        if bytes.len() == 2 * 1326 {
            return bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        }
    }
    let r = nash_ranking(eq, spec);
    let _ = std::fs::create_dir_all(dir);
    let bytes: Vec<u8> = r.iter().flat_map(|x| x.to_le_bytes()).collect();
    if let Err(e) = std::fs::write(&path, bytes) {
        eprintln!("warning: could not save {}: {e}", path.display());
    }
    r
}

pub fn load_or_build_charts(dir: &Path, eq: Arc<EquityMatrix>, spec: &ChartSpec) -> PushFoldCharts {
    let path = dir.join(spec.cache_name());
    if let Ok(bytes) = std::fs::read(&path) {
        if let Some(c) = PushFoldCharts::from_bytes(&bytes) {
            return c;
        }
    }
    let c = build_pushfold_charts(eq, spec);
    let _ = std::fs::create_dir_all(dir);
    if let Err(e) = std::fs::write(&path, c.to_bytes()) {
        eprintln!("warning: could not save {}: {e}", path.display());
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charts_match_known_heads_up_nash() {
        let eq = Arc::new(EquityMatrix::monte_carlo(3000, 21, 4));
        let spec = ChartSpec { x_grid: vec![10.0, 20.0], max_players: 3, iterations: 200, ..ChartSpec::new(0.5, 1.0, 0.0, false) };
        let c = build_pushfold_charts(eq, &spec);
        let (push, call) = (c.push_freq(2, 0, 10.0), c.call_freq(2, 0, 1, 10.0));
        assert!((push - 0.58).abs() < 0.04 && (call - 0.37).abs() < 0.04, "{push} {call}");
        // Más profundo ⇒ más tight; UTG de 3 más tight que el SB.
        assert!(c.push_freq(2, 0, 20.0) < push);
        assert!(c.push_freq(3, 0, 10.0) < c.push_freq(3, 1, 10.0));
        let back = PushFoldCharts::from_bytes(&c.to_bytes()).expect("roundtrip");
        assert_eq!(back.push, c.push);
    }

    /// Ranking Nash combinado: cubre los 1326 combos una vez, AA primero y 72o abajo;
    /// las parejas bajas suben respecto a la equity contra una mano al azar.
    #[test]
    fn nash_ranking_orders_hands_sensibly() {
        use crate::cfr::report::CLASS_COMBOS;
        let eq = Arc::new(EquityMatrix::monte_carlo(3000, 21, 4));
        let spec = ChartSpec { x_grid: vec![5.0, 10.0, 20.0], iterations: 150, ..ChartSpec::new(0.5, 1.0, 0.0, false) };
        let r = nash_ranking(eq, &spec);
        let mut seen = r.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 1326);
        let pos = |cls: usize| r.iter().position(|&h| h == CLASS_COMBOS[cls][0]).unwrap();
        // Celdas de la rejilla (fila·13 + columna, As arriba a la izquierda).
        let (aa, o72, a2o) = (0, 12 * 13 + 7, 12 * 13);
        assert_eq!(pos(aa), 0, "AA primero");
        assert!(pos(o72) > 900, "72o en el tercio de abajo: {}", pos(o72));
        assert!(pos(a2o) < pos(o72));
    }
}
