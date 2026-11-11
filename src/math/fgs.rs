//! # FGS — Future Game Simulation
//!
//! ICM valora un reparto de fichas como si el torneo se decidiera "a sorteo" desde ya:
//! ignora que el juego continúa y que la posición respecto a las ciegas importa (quien
//! pondrá la BB la mano siguiente está peor que su stack sugiere). FGS corrige esto: la utilidad de un estado de stacks es
//!
//! ```text
//! U(s) = E[ ICM(s tras `depth` manos futuras) ]
//! ```
//!
//! simulando las próximas manos con el botón rotando, ciegas y antes reales, y juego
//! push/fold:
//!
//! * **Rangos**: tablas push/fold (`PushFoldCharts`) calibradas con el propio solver
//!   (Nash chip-EV de mesas de 2 a 6 jugadores sobre una malla de stacks), indexadas por
//!   número de jugadores, posición y stack efectivo en bb (interpolación log-lineal).
//!   Mesas de 7+ jugadores usan la tabla de 6 contando posiciones desde la BB.
//! * **All-ins**: equity de "top p %" contra "top c %" (`eval::equity::RangeEquity`).
//! * **Árbol de la mano futura**: quién empuja primero, quién paga primero (HU) o si todos
//!   foldean (walk). Coste por mano O(n²); ramas con probabilidad < `prune` se asignan
//!   al walk (la probabilidad total se conserva exactamente, y con ella la suma de premios).
//!
//! Solo los jugadores de esta mesa juegan manos futuras; los de otras mesas mantienen su
//! stack y cuentan en el ICM final.

use super::field_icm::FieldIcm;
use super::icm::icm_equity;
use crate::eval::equity::RangeEquity;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Rangos push/fold por tamaño de mesa. `push[n-2][xi][pos]`, `call[n-2][xi][k * n + m]`
/// (frecuencia de call de `m` frente al shove de `k` con todos los intermedios foldeando).
#[derive(Debug, Clone)]
pub struct PushFoldCharts {
    pub x_grid: Vec<f64>,
    pub push: Vec<Vec<Vec<f32>>>,
    pub call: Vec<Vec<Vec<f32>>>,
}

pub const CHART_MAX_PLAYERS: usize = 6;

impl PushFoldCharts {
    fn interp(&self, x: f64, f: impl Fn(usize) -> f32) -> f64 {
        let g = &self.x_grid;
        let lx = x.max(g[0]).min(g[g.len() - 1]).ln();
        let i = g.iter().rposition(|&v| v.ln() <= lx).unwrap_or(0).min(g.len() - 2);
        let t = ((lx - g[i].ln()) / (g[i + 1].ln() - g[i].ln())).clamp(0.0, 1.0);
        f(i) as f64 * (1.0 - t) + f(i + 1) as f64 * t
    }

    /// Mapea (jugadores, posición) a la tabla disponible: con más de 6 jugadores, las
    /// posiciones tempranas se asimilan al UTG de 6-max.
    fn map(&self, n: usize, pos: usize) -> (usize, usize) {
        if n <= CHART_MAX_PLAYERS {
            (n, pos)
        } else {
            (CHART_MAX_PLAYERS, pos.saturating_sub(n - CHART_MAX_PLAYERS))
        }
    }

    pub fn push_freq(&self, n: usize, pos: usize, x: f64) -> f64 {
        let (n, pos) = self.map(n, pos);
        self.interp(x, |i| self.push[n - 2][i][pos])
    }

    pub fn call_freq(&self, n: usize, k: usize, m: usize, x: f64) -> f64 {
        let (n2, k2) = self.map(n, k);
        let (_, m2) = self.map(n, m);
        let m2 = m2.max(k2 + 1);
        self.interp(x, |i| self.call[n2 - 2][i][k2 * n2 + m2])
    }

    /// Serialización simple (f32 little-endian) para la caché en disco.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend((self.x_grid.len() as u32).to_le_bytes());
        for &x in &self.x_grid {
            v.extend(x.to_le_bytes());
        }
        for tab in self.push.iter().chain(self.call.iter()) {
            for row in tab {
                v.extend((row.len() as u32).to_le_bytes());
                for &f in row {
                    v.extend(f.to_le_bytes());
                }
            }
        }
        v
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut pos = 0usize;
        let mut u32_ = |b: &[u8]| -> Option<u32> {
            let r = u32::from_le_bytes(b.get(pos..pos + 4)?.try_into().ok()?);
            pos += 4;
            Some(r)
        };
        let ng = u32_(b)? as usize;
        let mut x_grid = Vec::with_capacity(ng);
        for _ in 0..ng {
            x_grid.push(f64::from_le_bytes(b.get(pos..pos + 8)?.try_into().ok()?));
            pos += 8;
        }
        let read_tab = |pos: &mut usize| -> Option<Vec<Vec<f32>>> {
            let mut tab = Vec::with_capacity(ng);
            for _ in 0..ng {
                let len = u32::from_le_bytes(b.get(*pos..*pos + 4)?.try_into().ok()?) as usize;
                *pos += 4;
                let row: Option<Vec<f32>> = (0..len)
                    .map(|i| Some(f32::from_le_bytes(b.get(*pos + 4 * i..*pos + 4 * i + 4)?.try_into().ok()?)))
                    .collect();
                *pos += 4 * len;
                tab.push(row?);
            }
            Some(tab)
        };
        let tables = CHART_MAX_PLAYERS - 1;
        let mut push = Vec::new();
        for _ in 0..tables {
            push.push(read_tab(&mut pos)?);
        }
        let mut call = Vec::new();
        for _ in 0..tables {
            call.push(read_tab(&mut pos)?);
        }
        (pos == b.len()).then_some(PushFoldCharts { x_grid, push, call })
    }
}

/// Cómo se juegan las manos futuras.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FgsMode {
    /// Rangos fijos de las tablas push/fold chip-EV (`PushFoldCharts`), interpolados.
    Charts,
    /// En cada estado futuro se resuelve el push/fold de la mano siguiente con
    /// las utilidades de ese estado (ICM o FGS de la mano de después), por fictitious play
    /// sobre umbrales de un ranking de manos. Las tablas solo sirven de arranque.
    Nash,
}

/// Reglas de profundidad automática:
/// 2-4 jugadores en la mesa → 3 manos, 5-7 → 2, el resto → 1.
pub fn auto_depth(table_players: usize) -> u8 {
    match table_players {
        2..=4 => 3,
        5..=7 => 2,
        _ => 1,
    }
}

/// Caché de valores de estados futuros (compartida entre todos los terminales del árbol:
/// muchos llegan a los mismos stacks). Repartida en trozos para no serializar los hilos.
#[derive(Debug)]
pub struct FgsCache {
    shards: Vec<Mutex<HashMap<Vec<i64>, Arc<[f64]>>>>,
}

impl Default for FgsCache {
    fn default() -> Self {
        FgsCache { shards: (0..64).map(|_| Mutex::new(HashMap::new())).collect() }
    }
}

impl FgsCache {
    fn shard(&self, key: &[i64]) -> &Mutex<HashMap<Vec<i64>, Arc<[f64]>>> {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &k in key {
            h = (h ^ k as u64).wrapping_mul(0x0100_0000_01b3);
        }
        &self.shards[(h >> 58) as usize % self.shards.len()]
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.lock().unwrap().len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub struct Fgs {
    /// Manos futuras simuladas (1-3 en la práctica; el coste es O((2n²)^depth)).
    pub depth: u8,
    pub sb: f64,
    pub bb: f64,
    pub ante: f64,
    pub bb_ante: bool,
    /// Ramas con probabilidad acumulada menor se asignan al walk (solo `FgsMode::Charts`).
    pub prune: f64,
    pub charts: Arc<PushFoldCharts>,
    pub ranges: Arc<RangeEquity>,
    /// Campo grande (`math::field_icm`): `stacks` son solo los de la mesa y las otras
    /// mesas están precomputadas en el ICM de campo, que valora el final de cada rama.
    pub field: Option<Arc<FieldIcm>>,
    pub mode: FgsMode,
    /// Iteraciones de fictitious play por estado: las dos primeras manos futuras y las
    /// siguientes.
    pub nash_iters: (u32, u32),
    pub cache: Arc<FgsCache>,
    /// Tabla de all-ins a 3 (se construye la primera vez que hace falta).
    pub three: Arc<OnceLock<ThreeWayTable>>,
    /// Subida de ciegas en las manos simuladas.
    pub next_level: Option<NextLevel>,
}

/// Subida de nivel en las manos simuladas del FGS. `at` es la profundidad (1 = primera mano simulada); ciegas y ante en
/// las mismas unidades que los stacks (bb del nivel actual). Solo en `FgsMode::Nash`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NextLevel {
    pub at: u32,
    pub sb: f64,
    pub bb: f64,
    pub ante: f64,
}

impl NextLevel {
    /// `"at@sb/bb[/ante]"`, p. ej. `2@0.75/1.5/0.2`.
    pub fn parse(v: &str) -> Result<NextLevel, String> {
        let (at, rest) = v.split_once('@').ok_or("formato at@sb/bb[/ante]")?;
        let at: u32 = at.trim().parse().map_err(|_| format!("invalid depth: {at}"))?;
        let x: Vec<f64> = rest.split('/').map(|t| t.trim().parse::<f64>().map_err(|_| format!("invalid number: {t}"))).collect::<Result<_, _>>()?;
        match x[..] {
            [sb, bb] => Ok(NextLevel { at, sb, bb, ante: 0.0 }),
            [sb, bb, ante] => Ok(NextLevel { at, sb, bb, ante }),
            _ => Err("formato at@sb/bb[/ante]".into()),
        }
    }
}

impl Fgs {
    /// Equity de premios de los jugadores vivos.
    /// * `table_order`: para cada jugador de esta mesa, su índice en `stacks`, en el orden
    ///   de acción de la mano que **acaba de jugarse** (el botón se moverá una posición).
    /// * `stacks`: todos los jugadores vivos (mesa + otras mesas).
    pub fn equity(&self, table_order: &[usize], stacks: &[f64], payouts: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        let next = rotate(table_order, stacks);
        match self.mode {
            FgsMode::Charts => {
                let mut s = stacks.to_vec();
                self.sim(self.depth, &next, &mut s, 1.0, payouts, out);
            }
            FgsMode::Nash => {
                let v = self.nash_value(self.depth, 0, &next, stacks, payouts);
                out.copy_from_slice(&v);
            }
        }
    }

    /// Jugadores vivos en total (mesa + otras mesas, también las del ICM de campo).
    fn alive_total(&self, stacks: &[f64]) -> usize {
        stacks.iter().filter(|&&x| x > 0.0).count() + self.field.as_ref().map_or(0, |f| f.n_off)
    }

    /// ICM de los vivos (sin premios de eliminados: en `Nash` se pagan al caer).
    fn icm_alive(&self, stacks: &[f64], payouts: &[f64]) -> Vec<f64> {
        let idx: Vec<usize> = (0..stacks.len()).filter(|&i| stacks[i] > 0.0).collect();
        let alive: Vec<f64> = idx.iter().map(|&i| stacks[i]).collect();
        let mut eq = vec![0.0; alive.len()];
        let off = self.field.as_ref().map_or(0, |f| f.n_off);
        match &self.field {
            Some(f) => f.equity(&alive, payouts.len().min(alive.len() + off), &mut eq),
            None => icm_equity(&alive, &payouts[..payouts.len().min(alive.len())], &mut eq),
        }
        let mut out = vec![0.0; stacks.len()];
        for (k, &i) in idx.iter().enumerate() {
            out[i] = eq[k];
        }
        out
    }

    /// Valor de cada jugador tras una mano simulada que deja los stacks `after` (antes,
    /// `before`): los que caen cobran ya su puesto (el de mayor stack inicial queda mejor;
    /// empates se reparten, como `math::terminal`) y el resto sigue jugando.
    fn settle(&self, depth: u8, level: u32, order: &[usize], before: &[f64], after: &[f64], payouts: &[f64]) -> Vec<f64> {
        let s: Vec<f64> = after.iter().map(|&x| if x > 1e-9 { x } else { 0.0 }).collect();
        let mut busted: Vec<usize> = (0..s.len()).filter(|&i| before[i] > 0.0 && s[i] <= 0.0).collect();
        let next = rotate(order, &s);
        let mut out: Vec<f64> = if next.len() >= 2 && depth > 0 {
            self.nash_value(depth, level, &next, &s, payouts).to_vec()
        } else {
            self.icm_alive(&s, payouts)
        };
        if !busted.is_empty() {
            let survivors = self.alive_total(&s);
            busted.sort_unstable_by(|&x, &y| before[y].total_cmp(&before[x]));
            let mut r = 0;
            while r < busted.len() {
                let mut e = r + 1;
                while e < busted.len() && before[busted[e]] == before[busted[r]] {
                    e += 1;
                }
                let share = (r..e).map(|k| payouts.get(survivors + k).copied().unwrap_or(0.0)).sum::<f64>() / (e - r) as f64;
                for &v in &busted[r..e] {
                    out[v] = share;
                }
                r = e;
            }
        }
        out
    }

    /// Antes y ciegas que pone cada jugador en una mano con `order` (acotados por el stack).
    fn post_blinds(&self, order: &[usize], stacks: &[f64], (sb, bb, ante_amt): (f64, f64, f64)) -> (Vec<f64>, Vec<f64>) {
        let n = order.len();
        let mut ante = vec![0.0; stacks.len()];
        let mut blind = vec![0.0; stacks.len()];
        for &i in order {
            ante[i] = if self.bb_ante { 0.0 } else { ante_amt.min(stacks[i]) };
        }
        let (sb_i, bb_i) = (order[n - 2], order[n - 1]);
        if self.bb_ante {
            ante[bb_i] = (ante_amt * n as f64).min(stacks[bb_i]);
        }
        blind[sb_i] = sb.min(stacks[sb_i] - ante[sb_i]);
        blind[bb_i] = bb.min(stacks[bb_i] - ante[bb_i]);
        (ante, blind)
    }

    /// Ciegas y ante `(sb, bb, ante)` de la mano simulada número `level` (0 = la primera,
    /// profundidad 1).
    fn blinds_at(&self, level: u32) -> (f64, f64, f64) {
        match self.next_level {
            Some(nl) if level + 1 >= nl.at => (nl.sb, nl.bb, nl.ante),
            _ => (self.sb, self.bb, self.ante),
        }
    }

    /// Valor de un estado (orden de acción de la mano que se va a jugar y stacks): se
    /// resuelve el push/fold de esa mano con las utilidades de cada resultado.
    fn nash_value(&self, depth: u8, level: u32, order: &[usize], stacks: &[f64], payouts: &[f64]) -> Arc<[f64]> {
        let n = order.len();
        if depth == 0 || n < 2 {
            return self.icm_alive(stacks, payouts).into();
        }
        let mut key: Vec<i64> = Vec::with_capacity(stacks.len() + order.len() + 3);
        key.push(depth as i64);
        key.push(level.min(2) as i64);
        // Con subida de ciegas, el valor depende de cuántas de las manos que quedan van ya
        // con el nivel nuevo: basta el nivel hasta `at`.
        if let Some(nl) = self.next_level {
            key.push(level.min(nl.at) as i64);
        }
        key.push(payouts.len() as i64);
        key.extend(order.iter().map(|&i| i as i64));
        key.push(-1);
        key.extend(stacks.iter().map(|&x| (x * 1000.0).round() as i64));
        let shard = self.cache.shard(&key);
        if let Some(v) = shard.lock().unwrap().get(&key) {
            return v.clone();
        }
        let v: Arc<[f64]> = self.solve_hand(depth, level, order, stacks, payouts).into();
        shard.lock().unwrap().insert(key, v.clone());
        v
    }

    /// Push/fold de una mano con utilidades dadas: fictitious play sobre umbrales del
    /// ranking, con shove, call y overcall (botes a
    /// 3 con botes laterales). Devuelve el valor del estado.
    fn solve_hand(&self, depth: u8, level: u32, order: &[usize], stacks: &[f64], payouts: &[f64]) -> Vec<f64> {
        let n = order.len();
        let np = stacks.len();
        let three = self.three.get_or_init(|| ThreeWayTable::load_or_build(&self.ranges, 20, 5000, std::path::Path::new(".cache")));
        // Ciegas y antes del nivel de esta mano.
        let level_blinds = self.blinds_at(level);
        let bb_amt = level_blinds.1;
        let (ante, blind) = self.post_blinds(order, stacks, level_blinds);
        let (sb_i, bb_i) = (order[n - 2], order[n - 1]);
        let dead: f64 = ante.iter().sum();
        let live = |i: usize| stacks[i] - ante[i];
        let base: Vec<f64> = (0..np).map(|i| stacks[i] - ante[i] - blind[i]).collect();
        let next_depth = depth - 1;
        let lvl = level + 1;
        let value_of = |s: &[f64]| self.settle(next_depth, lvl, order, stacks, s, payouts);
        // Botes a 3 en las manos simuladas: valorados con ICM, sin seguir simulando.
        let value_icm = |s: &[f64]| self.settle(0, lvl, order, stacks, s, payouts);

        // --- Resultados posibles y su valor (independiente de la estrategia) ---
        let walk = {
            let mut s = base.clone();
            s[bb_i] += blind[bb_i] + blind[sb_i] + dead;
            value_of(&s)
        };
        let blinds_except = |skip: &[usize]| order.iter().filter(|j| !skip.contains(j)).map(|&j| blind[j]).sum::<f64>();
        let mut uncalled: Vec<Vec<f64>> = Vec::with_capacity(n - 1);
        // win[k][m-k-1]: gana el que empuja (k); lose: gana el que paga (m).
        let mut win: Vec<Vec<Vec<f64>>> = vec![Vec::new(); n - 1];
        let mut lose: Vec<Vec<Vec<f64>>> = vec![Vec::new(); n - 1];
        // tri[k][m][o]: valor por cada orden de llegada (permutación de [k, m, o]).
        let mut tri: Vec<Vec<Vec<[Vec<f64>; 6]>>> = vec![Vec::new(); n - 1];
        for k in 0..n - 1 {
            let ik = order[k];
            let mut s = base.clone();
            s[ik] += blind[ik] + dead + blinds_except(&[ik]);
            uncalled.push(value_of(&s));
            for m in k + 1..n {
                let im = order[m];
                let matched = live(ik).min(live(im));
                let pot = dead + blinds_except(&[ik, im]) + 2.0 * matched;
                let mut sk = base.clone();
                sk[ik] += blind[ik] - matched;
                sk[im] += blind[im] - matched;
                let mut sm = sk.clone();
                sk[ik] += pot;
                sm[im] += pot;
                win[k].push(value_of(&sk));
                lose[k].push(value_of(&sm));
                let mut row = Vec::new();
                for o in m + 1..n {
                    let io = order[o];
                    let ps = [ik, im, io];
                    let contrib: Vec<f64> = (0..3)
                        .map(|a| {
                            let others = (0..3).filter(|&b| b != a).map(|b| live(ps[b])).fold(0.0, f64::max);
                            live(ps[a]).min(others)
                        })
                        .collect();
                    let other_blinds = blinds_except(&ps);
                    let vals: [Vec<f64>; 6] = std::array::from_fn(|q| {
                        let perm = PERMS[q];
                        let mut s = base.clone();
                        for a in 0..3 {
                            s[ps[a]] += blind[ps[a]] - contrib[a];
                        }
                        // Bote principal (3 × la menor aportación) para el mejor de los tres;
                        // el lateral, entre los dos que más aportan, para el mejor de ellos.
                        let cmin = contrib.iter().copied().fold(f64::INFINITY, f64::min);
                        s[ps[perm[0]]] += 3.0 * cmin + dead + other_blinds;
                        let side: f64 = contrib.iter().map(|&c| c - cmin).sum();
                        if side > 1e-12 {
                            let side_winner = perm.iter().copied().find(|&a| contrib[a] - cmin > 1e-12).unwrap_or(perm[0]);
                            s[ps[side_winner]] += side;
                        }
                        value_icm(&s)
                    });
                    row.push(vals);
                }
                tri[k].push(row);
            }
        }

        // --- Estrategia: umbrales sobre el ranking ---
        let eff = |k: usize| {
            let ik = order[k];
            let max_other = order.iter().filter(|&&j| j != ik).map(|&j| live(j)).fold(0.0, f64::max);
            live(ik).min(max_other) / bb_amt
        };
        let mut p: Vec<f64> = (0..n - 1).map(|k| self.charts.push_freq(n, k, eff(k)).clamp(0.0, 1.0)).collect();
        let mut c: Vec<Vec<f64>> = (0..n - 1)
            .map(|k| (k + 1..n).map(|m| self.charts.call_freq(n, k, m, live(order[k]).min(live(order[m])) / bb_amt).clamp(0.0, 1.0)).collect())
            .collect();
        // oc[k][m-k-1][o-m-1]: overcall de o tras el shove de k y el call de m.
        let mut oc: Vec<Vec<Vec<f64>>> = (0..n - 1).map(|k| (k + 1..n).map(|m| (m + 1..n).map(|o| 0.5 * c[k][o - k - 1]).collect()).collect()).collect();

        let steps = self.ranges.steps.max(2);
        let eq = |x: f64, r: f64| self.ranges.equity(x, r);
        let slice_eq = |i: usize, r: f64| {
            let (a, b) = ((i - 1) as f64 / steps as f64, i as f64 / steps as f64);
            let ea = if i == 1 { 0.0 } else { a * eq(a, r) };
            (b * eq(b, r) - ea) / (b - a)
        };
        // Papeles en un bote a 3 (0 = el que empuja k, 1 = el que paga m, 2 = el overcall o):
        // `role_map[héroe][q]` traduce el orden PERMS[q] de (héroe, a, b) al índice de PERMS
        // sobre (k, m, o) en que están guardados los valores `tri`.
        let others = |hero: usize| match hero {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let role_map: [[usize; 6]; 3] = std::array::from_fn(|hero| {
            let (a, b) = others(hero);
            std::array::from_fn(|q| {
                let roles = [hero, a, b];
                let actual = [roles[PERMS[q][0]], roles[PERMS[q][1]], roles[PERMS[q][2]]];
                PERMS.iter().position(|pp| *pp == actual).unwrap()
            })
        });
        // Valor para `who` de un bote a 3 con el héroe en cada fila (rebanada) de la tabla.
        let tri_rows = |k: usize, m: usize, o: usize, hero: usize, ranges: [f64; 3], who: usize| -> Vec<f64> {
            let vals = &tri[k][m - k - 1][o - m - 1];
            let (a, b) = others(hero);
            (0..three.s)
                .map(|si| {
                    let pr = three.interp(si, ranges[a], ranges[b]);
                    (0..6).map(|q| pr[q] * vals[role_map[hero][q]][who]).sum::<f64>()
                })
                .collect()
        };
        let row_of = |i: usize| ((((i as f64 - 0.5) / steps as f64) * three.s as f64) as usize).min(three.s - 1);
        // Valor de un bote a 3 para todos los jugadores, con los tres rangos completos.
        let tri_all = |k: usize, m: usize, o: usize, ranges: [f64; 3]| -> Vec<f64> {
            let vals = &tri[k][m - k - 1][o - m - 1];
            let pr = three.range_orders(ranges[0], ranges[1], ranges[2]);
            let mut out = vec![0.0; np];
            for q in 0..6 {
                if pr[q] > 0.0 {
                    let v = &vals[role_map[0][q]];
                    for j in 0..np {
                        out[j] += pr[q] * v[j];
                    }
                }
            }
            out
        };
        let mix = |a: &[f64], b: &[f64], pa: f64, out: &mut [f64], w: f64| {
            for j in 0..out.len() {
                out[j] += w * (pa * a[j] + (1.0 - pa) * b[j]);
            }
        };
        // Fase de overcalls tras el shove de k y el call de m, desde o0 (los anteriores no).
        let over_stage = |p: &[f64], c: &[Vec<f64>], oc: &[Vec<Vec<f64>>], k: usize, m: usize, o0: usize| -> Vec<f64> {
            let mut out = vec![0.0; np];
            let mut reach = 1.0;
            let cm = c[k][m - k - 1];
            for o in o0..n {
                let w = reach * oc[k][m - k - 1][o - m - 1];
                if w > 0.0 {
                    let v = tri_all(k, m, o, [p[k], cm, oc[k][m - k - 1][o - m - 1]]);
                    for j in 0..np {
                        out[j] += w * v[j];
                    }
                }
                reach *= 1.0 - oc[k][m - k - 1][o - m - 1];
            }
            let e = eq(p[k], cm);
            mix(&win[k][m - k - 1], &lose[k][m - k - 1], e, &mut out, reach);
            out
        };
        let call_stage = |p: &[f64], c: &[Vec<f64>], oc: &[Vec<Vec<f64>>], k: usize, m0: usize| -> Vec<f64> {
            let mut out = vec![0.0; np];
            let mut reach = 1.0;
            for m in m0..n {
                let cm = c[k][m - k - 1];
                let w = reach * cm;
                if w > 0.0 {
                    let v = over_stage(p, c, oc, k, m, m + 1);
                    for j in 0..np {
                        out[j] += w * v[j];
                    }
                }
                reach *= 1.0 - cm;
            }
            for j in 0..np {
                out[j] += reach * uncalled[k][j];
            }
            out
        };
        let push_stage = |p: &[f64], c: &[Vec<f64>], oc: &[Vec<Vec<f64>>], k0: usize| -> Vec<f64> {
            let mut out = vec![0.0; np];
            let mut reach = 1.0;
            for k in k0..n - 1 {
                let w = reach * p[k];
                if w > 0.0 {
                    let v = call_stage(p, c, oc, k, k + 1);
                    for j in 0..np {
                        out[j] += w * v[j];
                    }
                }
                reach *= 1.0 - p[k];
            }
            for j in 0..np {
                out[j] += reach * walk[j];
            }
            out
        };

        let iters = if level <= 1 { self.nash_iters.0 } else { self.nash_iters.1 };
        let mut b = 1.0f64;
        let frac = |good: usize| good as f64 / steps as f64;
        for _ in 0..iters {
            let mut bp = vec![0.0; n - 1];
            let mut bc: Vec<Vec<f64>> = c.iter().map(|r| vec![0.0; r.len()]).collect();
            let mut boc: Vec<Vec<Vec<f64>>> = oc.iter().map(|r| r.iter().map(|x| vec![0.0; x.len()]).collect()).collect();
            for k in 0..n - 1 {
                let ik = order[k];
                // Shove de k.
                let fold = push_stage(&p, &c, &oc, k + 1)[ik];
                // Valor de los botes a 3 por fila de la tabla, una vez por (m, o).
                let rows_k: Vec<Vec<Vec<f64>>> = (k + 1..n)
                    .map(|m| {
                        let cm = c[k][m - k - 1];
                        (m + 1..n)
                            .map(|o| {
                                let x = oc[k][m - k - 1][o - m - 1];
                                if cm > 0.0 && x > 0.0 { tri_rows(k, m, o, 0, [p[k], cm, x], ik) } else { Vec::new() }
                            })
                            .collect()
                    })
                    .collect();
                let mut good = 0usize;
                for i in 1..=steps {
                    let row = row_of(i);
                    let mut ev = 0.0;
                    let mut reach = 1.0;
                    for m in k + 1..n {
                        let cm = c[k][m - k - 1];
                        if cm > 0.0 {
                            let mut r2 = 1.0;
                            let mut sub = 0.0;
                            for o in m + 1..n {
                                let x = oc[k][m - k - 1][o - m - 1];
                                if x > 0.0 {
                                    sub += r2 * x * rows_k[m - k - 1][o - m - 1][row];
                                }
                                r2 *= 1.0 - x;
                            }
                            let e = slice_eq(i, cm);
                            sub += r2 * (e * win[k][m - k - 1][ik] + (1.0 - e) * lose[k][m - k - 1][ik]);
                            ev += reach * cm * sub;
                        }
                        reach *= 1.0 - cm;
                    }
                    ev += reach * uncalled[k][ik];
                    if ev > fold {
                        good += 1;
                    }
                }
                bp[k] = frac(good);
                for m in k + 1..n {
                    let im = order[m];
                    let cm = c[k][m - k - 1];
                    // Call de m (los de en medio han foldeado).
                    let fold_m = call_stage(&p, &c, &oc, k, m + 1)[im];
                    let (wv, lv) = (win[k][m - k - 1][im], lose[k][m - k - 1][im]);
                    let rows_m: Vec<Vec<f64>> = (m + 1..n)
                        .map(|o| {
                            let x = oc[k][m - k - 1][o - m - 1];
                            if x > 0.0 { tri_rows(k, m, o, 1, [p[k], cm.max(1.0 / steps as f64), x], im) } else { Vec::new() }
                        })
                        .collect();
                    let mut good = 0usize;
                    for j in 1..=steps {
                        let row = row_of(j);
                        let mut r2 = 1.0;
                        let mut ev = 0.0;
                        for o in m + 1..n {
                            let x = oc[k][m - k - 1][o - m - 1];
                            if x > 0.0 {
                                ev += r2 * x * rows_m[o - m - 1][row];
                            }
                            r2 *= 1.0 - x;
                        }
                        let e = slice_eq(j, p[k].max(1.0 / steps as f64));
                        ev += r2 * (e * lv + (1.0 - e) * wv);
                        if ev > fold_m {
                            good += 1;
                        }
                    }
                    bc[k][m - k - 1] = frac(good);
                    // Overcall de o tras el call de m.
                    for o in m + 1..n {
                        let io = order[o];
                        let fold_o = over_stage(&p, &c, &oc, k, m, o + 1)[io];
                        let rng = [p[k].max(1.0 / steps as f64), cm.max(1.0 / steps as f64), 0.5];
                        let rows_o = tri_rows(k, m, o, 2, rng, io);
                        let good = (1..=steps).filter(|&l| rows_o[row_of(l)] > fold_o).count();
                        boc[k][m - k - 1][o - m - 1] = frac(good);
                    }
                }
            }
            b += 0.11 * (b + 10.0).ln();
            let w = 1.0 / b;
            for k in 0..n - 1 {
                p[k] += w * (bp[k] - p[k]);
                for (mi, x) in c[k].iter_mut().enumerate() {
                    *x += w * (bc[k][mi] - *x);
                    for (oi, y) in oc[k][mi].iter_mut().enumerate() {
                        *y += w * (boc[k][mi][oi] - *y);
                    }
                }
            }
        }
        push_stage(&p, &c, &oc, 0)
    }

    fn leaf(&self, stacks: &[f64], prob: f64, payouts: &[f64], out: &mut [f64]) {
        for (o, e) in out.iter_mut().zip(self.icm_alive(stacks, payouts)) {
            *o += prob * e;
        }
        // Premios de quien quede eliminado durante la simulación: los puestos bajos
        // restantes, repartidos entre los eliminados (aproximación: orden de caída no
        // rastreado dentro de la simulación).
        let dead: Vec<usize> = (0..stacks.len()).filter(|&i| stacks[i] <= 0.0).collect();
        if !dead.is_empty() {
            let tail: f64 = payouts.iter().skip(self.alive_total(stacks)).take(dead.len()).sum();
            for &i in &dead {
                out[i] += prob * tail / dead.len() as f64;
            }
        }
    }

    fn sim(&self, depth: u8, order: &[usize], stacks: &mut Vec<f64>, prob: f64, payouts: &[f64], out: &mut [f64]) {
        let n = order.len();
        if depth == 0 || n < 2 {
            self.leaf(stacks, prob, payouts, out);
            return;
        }
        let (ante, blind) = self.post_blinds(order, stacks, (self.sb, self.bb, self.ante));
        let (sb_i, bb_i) = (order[n - 2], order[n - 1]);
        let dead: f64 = ante.iter().sum();
        let live = |i: usize| stacks[i] - ante[i];

        let base: Vec<f64> = (0..stacks.len()).map(|i| stacks[i] - ante[i] - blind[i]).collect();
        let mut pruned = 0.0;
        let mut p_none = 1.0;
        let mut branches: Vec<(Vec<f64>, f64)> = Vec::new();
        for k in 0..n - 1 {
            let ik = order[k];
            let max_other = order.iter().filter(|&&j| j != ik).map(|&j| live(j)).fold(0.0, f64::max);
            let pk = self.charts.push_freq(n, k, live(ik).min(max_other) / self.bb);
            let reach = p_none * pk;
            p_none *= 1.0 - pk;
            if prob * reach < self.prune {
                pruned += reach;
                continue;
            }
            let mut p_nocall = 1.0;
            for m in k + 1..n {
                let im = order[m];
                let cm = self.charts.call_freq(n, k, m, live(ik).min(live(im)) / self.bb);
                let pr = reach * p_nocall * cm;
                p_nocall *= 1.0 - cm;
                if prob * pr < self.prune {
                    pruned += pr;
                    continue;
                }
                let e = self.ranges.equity(pk, cm);
                let matched = live(ik).min(live(im));
                let others_blinds: f64 = order.iter().filter(|&&j| j != ik && j != im).map(|&j| blind[j]).sum();
                let pot = dead + others_blinds + 2.0 * matched;
                for (w, pw) in [(ik, e), (im, 1.0 - e)] {
                    let mut s = base.clone();
                    // k y m: se devuelven sus ciegas y pierden lo igualado.
                    s[ik] += blind[ik] - matched;
                    s[im] += blind[im] - matched;
                    s[w] += pot;
                    branches.push((s, pr * pw));
                }
            }
            // Nadie paga: el shover se lleva antes y ciegas ajenas.
            let pr = reach * p_nocall;
            let mut s = base.clone();
            s[ik] += blind[ik] + dead + order.iter().filter(|&&j| j != ik).map(|&j| blind[j]).sum::<f64>();
            branches.push((s, pr));
        }
        // Walk (+ ramas podadas): la BB se lleva antes y la SB.
        let mut s = base.clone();
        s[bb_i] += blind[bb_i] + blind[sb_i] + dead;
        branches.push((s, p_none + pruned));
        for (mut s, p) in branches {
            if p <= 0.0 {
                continue;
            }
            s.iter_mut().for_each(|x| *x = x.max(0.0));
            let next = rotate(order, &s);
            self.sim(depth - 1, &next, &mut s, prob * p, payouts, out);
        }
    }
}

/// Permutaciones de 3 papeles: orden de llegada, del mejor al peor.
const PERMS: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/// Probabilidad de cada orden de llegada en un all-in a 3 entre un héroe (rebanada del
/// ranking) y dos rangos "top x" (tablas de botes a 3, pero
/// directamente por umbrales del ranking). Monte Carlo con bloqueo de cartas, una vez por
/// proceso.
#[derive(Debug)]
pub struct ThreeWayTable {
    s: usize,
    cells: Vec<[f32; 6]>,
}

impl ThreeWayTable {
    /// [`ThreeWayTable::build`] con caché en `dir` (la huella incluye el ranking).
    pub fn load_or_build(ranges: &RangeEquity, s: usize, trials: usize, dir: &std::path::Path) -> Self {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for x in ranges.order.iter().map(|&x| x as u64).chain([s as u64, trials as u64, 1]) {
            h = (h ^ x).wrapping_mul(0x0100_0000_01b3);
        }
        let path = dir.join(format!("threeway_{h:016x}.bin"));
        let n = s * (s + 1) * (s + 1);
        if let Ok(bytes) = std::fs::read(&path) {
            if bytes.len() == n * 6 * 4 {
                let f: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                let cells = f.chunks_exact(6).map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]]).collect();
                return ThreeWayTable { s, cells };
            }
        }
        let t = Self::build(ranges, s, trials);
        let bytes: Vec<u8> = t.cells.iter().flat_map(|c| c.iter().flat_map(|x| x.to_le_bytes())).collect();
        let _ = std::fs::create_dir_all(dir);
        if let Err(e) = std::fs::write(&path, bytes) {
            eprintln!("warning: could not save {}: {e}", path.display());
        }
        t
    }

    pub fn build(ranges: &RangeEquity, s: usize, trials: usize) -> Self {
        use crate::eval::evaluator::rank_mask;
        use crate::eval::COMBO_MASKS;
        use crate::math::rng::SplitMix64;
        let order = &ranges.order;
        let nc = order.len();
        let cut = |x: f64| ((x * nc as f64).round() as usize).clamp(1, nc);
        let side = s + 1;
        let mut cells = vec![[0.0f32; 6]; s * side * side];
        let threads = std::thread::available_parallelism().map_or(4, |x| x.get()).min(s);
        let chunks: Vec<Vec<usize>> = (0..threads).map(|t| (t..s).step_by(threads).collect()).collect();
        let results: Vec<Vec<(usize, [f32; 6])>> = std::thread::scope(|sc| {
            let hs: Vec<_> = chunks
                .iter()
                .map(|sis| {
                    sc.spawn(move || {
                        let mut out = Vec::new();
                        for &si in sis {
                            let (h0, h1) = (cut(si as f64 / s as f64).min(nc - 1) * (si > 0) as usize, cut((si + 1) as f64 / s as f64));
                            let hero = &order[h0..h1.max(h0 + 1)];
                            for ai in 1..=s {
                                for bi in 1..=s {
                                    let ra = &order[..cut(ai as f64 / s as f64)];
                                    let rb = &order[..cut(bi as f64 / s as f64)];
                                    let mut rng = SplitMix64::new(0x3_3A7 ^ ((si * 1000 + ai) * 1000 + bi) as u64);
                                    let mut acc = [0.0f64; 6];
                                    let mut done = 0usize;
                                    let mut tries = 0usize;
                                    while done < trials && tries < trials * 20 {
                                        tries += 1;
                                        let h = hero[(rng.next_u64() % hero.len() as u64) as usize] as usize;
                                        let a = ra[(rng.next_u64() % ra.len() as u64) as usize] as usize;
                                        let b = rb[(rng.next_u64() % rb.len() as u64) as usize] as usize;
                                        let (mh, ma, mb) = (COMBO_MASKS[h], COMBO_MASKS[a], COMBO_MASKS[b]);
                                        if mh & ma != 0 || mh & mb != 0 || ma & mb != 0 {
                                            continue;
                                        }
                                        let mut used = mh | ma | mb;
                                        let mut board = 0u64;
                                        let mut k = 0;
                                        while k < 5 {
                                            let c = rng.next_u64() % 52;
                                            if used & (1 << c) == 0 {
                                                used |= 1 << c;
                                                board |= 1 << c;
                                                k += 1;
                                            }
                                        }
                                        let r = [rank_mask(board | mh), rank_mask(board | ma), rank_mask(board | mb)];
                                        let ok: Vec<usize> = (0..6).filter(|&q| r[PERMS[q][0]] >= r[PERMS[q][1]] && r[PERMS[q][1]] >= r[PERMS[q][2]]).collect();
                                        for &q in &ok {
                                            acc[q] += 1.0 / ok.len() as f64;
                                        }
                                        done += 1;
                                    }
                                    let mut cell = [0.0f32; 6];
                                    if done > 0 {
                                        for q in 0..6 {
                                            cell[q] = (acc[q] / done as f64) as f32;
                                        }
                                    } else {
                                        cell = [1.0 / 6.0; 6];
                                    }
                                    out.push(((si * side + ai) * side + bi, cell));
                                }
                            }
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for r in results {
            for (i, c) in r {
                cells[i] = c;
            }
        }
        // Rango vacío (índice 0): copia del primer punto de la malla.
        for si in 0..s {
            for k in 0..side {
                cells[(si * side) * side + k] = cells[(si * side + 1) * side + k.max(1)];
                cells[(si * side + k) * side] = cells[(si * side + k.max(1)) * side + 1];
            }
        }
        ThreeWayTable { s, cells }
    }

    fn interp(&self, si: usize, a: f64, b: f64) -> [f64; 6] {
        let s = self.s;
        let side = s + 1;
        let (x, y) = ((a.clamp(0.0, 1.0) * s as f64).max(1.0), (b.clamp(0.0, 1.0) * s as f64).max(1.0));
        let (i, j) = ((x as usize).min(s - 1), (y as usize).min(s - 1));
        let (fx, fy) = (x - i as f64, y - j as f64);
        let g = |p: usize, q: usize| &self.cells[(si * side + p) * side + q];
        let mut out = [0.0f64; 6];
        for k in 0..6 {
            out[k] = g(i, j)[k] as f64 * (1.0 - fx) * (1.0 - fy) + g(i + 1, j)[k] as f64 * fx * (1.0 - fy) + g(i, j + 1)[k] as f64 * (1.0 - fx) * fy + g(i + 1, j + 1)[k] as f64 * fx * fy;
        }
        out
    }

    /// Órdenes de llegada con el héroe en todo su rango "top p".
    pub fn range_orders(&self, p: f64, a: f64, b: f64) -> [f64; 6] {
        let f = (p.clamp(0.0, 1.0) * self.s as f64).max(1e-9);
        let full = (f as usize).min(self.s);
        let mut out = [0.0f64; 6];
        for si in 0..full {
            let v = self.interp(si, a, b);
            for k in 0..6 {
                out[k] += v[k];
            }
        }
        let rest = f - full as f64;
        if rest > 1e-9 && full < self.s {
            let v = self.interp(full, a, b);
            for k in 0..6 {
                out[k] += rest * v[k];
            }
        }
        for x in out.iter_mut() {
            *x /= f;
        }
        out
    }
}

/// Orden de acción de la mano siguiente (`[UTG…BTN, SB, BB]`) con el botón móvil:
/// se retiran los eliminados y los supervivientes rotan una
/// posición **más una por cada ciega (SB o BB) eliminada**, y una más si quedan dos (el
/// botón pasa al siguiente superviviente). Con dos jugadores al empezar, una sola rotación.
fn rotate(order: &[usize], stacks: &[f64]) -> Vec<usize> {
    let n = order.len();
    let mut alive: Vec<usize> = order.iter().copied().filter(|&i| stacks[i] > 0.0).collect();
    if alive.is_empty() {
        return alive;
    }
    let steps = if n > 2 {
        let dead_blinds = order[n - 2..].iter().filter(|&&i| stacks[i] <= 0.0).count();
        dead_blinds + 1 + usize::from(alive.len() == 2)
    } else {
        1
    };
    let k = steps % alive.len();
    alive.rotate_left(k);
    alive
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::equity::EquityMatrix;

    /// Tablas sintéticas: empuja el 40 %, paga el 25 % en toda la malla.
    fn charts() -> Arc<PushFoldCharts> {
        let x_grid = vec![2.0, 10.0, 50.0];
        let push = (2..=CHART_MAX_PLAYERS).map(|n| vec![vec![0.4f32; n]; 3]).collect();
        let call = (2..=CHART_MAX_PLAYERS).map(|n| vec![vec![0.25f32; n * n]; 3]).collect();
        Arc::new(PushFoldCharts { x_grid, push, call })
    }

    fn fgs(depth: u8, bb: f64) -> Fgs {
        let m = EquityMatrix::monte_carlo(200, 1, 4);
        Fgs { depth, sb: bb / 2.0, bb, ante: 0.0, bb_ante: false, prune: 1e-6, charts: charts(), ranges: Arc::new(RangeEquity::from_matrix(&m, 20)), field: None, mode: FgsMode::Charts, nash_iters: (200, 50), cache: Arc::default(), three: Arc::default(), next_level: None }
    }

    /// Botón móvil: sin eliminados, una rotación; por cada ciega eliminada,
    /// una más; si quedan dos, otra más.
    #[test]
    fn moving_button_rotation() {
        // [UTG, HJ, CO, BTN, SB, BB] = jugadores 0..5.
        let order = [0, 1, 2, 3, 4, 5];
        let st = |dead: &[usize]| (0..6).map(|i| if dead.contains(&i) { 0.0 } else { 10.0 }).collect::<Vec<_>>();
        assert_eq!(rotate(&order, &st(&[])), vec![1, 2, 3, 4, 5, 0]);
        // Cae la SB: el botón pasa a la antigua BB, UTG pone la SB y HJ la BB.
        assert_eq!(rotate(&order, &st(&[4])), vec![2, 3, 5, 0, 1]);
        // Cae la BB: igual que la SB (dos rotaciones).
        assert_eq!(rotate(&order, &st(&[5])), vec![2, 3, 4, 0, 1]);
        // Cae un jugador que no es ciega: una rotación.
        assert_eq!(rotate(&order, &st(&[2])), vec![1, 3, 4, 5, 0]);
        // A 3, cae el BTN y quedan dos: dos rotaciones (la SB vuelve a ser SB/botón).
        assert_eq!(rotate(&[0, 1, 2], &[0.0, 10.0, 10.0]), vec![1, 2]);
        // Heads-up: una rotación.
        assert_eq!(rotate(&[0, 1], &[10.0, 10.0]), vec![1, 0]);
    }

    /// FGS con el ICM de campo (solo la mesa en los arrays, otras mesas precomputadas)
    /// coincide con FGS sobre el campo completo con ICM exacto (22 jugadores).
    #[test]
    fn fgs_with_field_icm_matches_full_field() {
        let table = vec![12.0, 30.0, 7.0, 22.0, 15.0, 40.0];
        let off: Vec<f64> = (0..16).map(|k| 10.0 + 2.5 * k as f64).collect();
        let payouts = vec![40.0, 25.0, 15.0, 10.0, 6.0, 4.0];
        let plain = fgs(1, 1.0);
        let all: Vec<f64> = table.iter().chain(&off).copied().collect();
        let order: Vec<usize> = (0..table.len()).collect();
        let mut full = vec![0.0; all.len()];
        plain.equity(&order, &all, &payouts, &mut full);
        let field = Arc::new(FieldIcm::new(&off, table.iter().sum(), 0.05, &payouts, 2));
        let with_field = Fgs { field: Some(field), ..plain };
        let mut got = vec![0.0; table.len()];
        with_field.equity(&order, &table, &payouts, &mut got);
        for i in 0..table.len() {
            assert!((got[i] - full[i]).abs() < 1e-5 * 100.0, "asiento {i}: {} vs {}", got[i], full[i]);
        }
    }

    #[test]
    fn depth_zero_is_icm_and_prizes_are_conserved() {
        let stacks = [10.0, 20.0, 30.0, 25.0];
        let payouts = [50.0, 30.0, 20.0];
        let (mut a, mut b) = ([0.0; 4], [0.0; 4]);
        let f0 = fgs(0, 1.0);
        f0.equity(&[0, 1, 2], &stacks, &payouts, &mut a);
        icm_equity(&stacks, &payouts, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-9);
        }
        let f2 = fgs(2, 2.0);
        f2.equity(&[0, 1, 2], &stacks, &payouts, &mut a);
        assert!((a.iter().sum::<f64>() - 100.0).abs() < 1e-6, "{a:?}");
    }

    fn nash(depth: u8, bb: f64) -> Fgs {
        Fgs { mode: FgsMode::Nash, nash_iters: (60, 20), ..fgs(depth, bb) }
    }

    /// Modo Nash: los premios se conservan, también con eliminaciones dentro
    /// de las manos simuladas, y la profundidad 0 es el ICM.
    #[test]
    fn nash_mode_conserves_prizes() {
        let stacks = [3.0, 20.0, 9.0, 25.0];
        let payouts = [50.0, 30.0, 20.0];
        let mut a = [0.0; 4];
        nash(0, 1.0).equity(&[0, 1, 2, 3], &stacks, &payouts, &mut a);
        let mut b = [0.0; 4];
        icm_equity(&stacks, &payouts, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-9);
        }
        for depth in 1..=2 {
            nash(depth, 2.0).equity(&[0, 1, 2, 3], &stacks, &payouts, &mut a);
            assert!((a.iter().sum::<f64>() - 100.0).abs() < 1e-6, "profundidad {depth}: {a:?}");
            assert!(a.iter().all(|&x| x > 0.0), "{a:?}");
        }
    }

    /// Modo Nash con el ICM de campo coincide con el campo completo con ICM exacto.
    #[test]
    fn nash_mode_with_field_icm_matches_full_field() {
        let table = vec![12.0, 30.0, 7.0, 22.0];
        let off: Vec<f64> = (0..18).map(|k| 10.0 + 2.0 * k as f64).collect();
        let payouts = vec![40.0, 25.0, 15.0, 10.0, 6.0, 4.0];
        let plain = nash(1, 1.0);
        let all: Vec<f64> = table.iter().chain(&off).copied().collect();
        let order: Vec<usize> = (0..table.len()).collect();
        let mut full = vec![0.0; all.len()];
        plain.equity(&order, &all, &payouts, &mut full);
        let field = Arc::new(FieldIcm::new(&off, table.iter().sum(), 0.05, &payouts, 2));
        let with_field = Fgs { field: Some(field), cache: Arc::default(), ..plain };
        let mut got = vec![0.0; table.len()];
        with_field.equity(&order, &table, &payouts, &mut got);
        for i in 0..table.len() {
            assert!((got[i] - full[i]).abs() < 1e-5 * 100.0, "asiento {i}: {} vs {}", got[i], full[i]);
        }
    }

    #[test]
    fn upcoming_big_blind_costs_equity() {
        // Tres stacks iguales, ciegas grandes. Orden de la mano jugada [BTN, SB, BB] =
        // [0, 1, 2]; en la siguiente, el BTN (0) pasa a ser BB y el BB (2) a BTN.
        let stacks = [10.0, 10.0, 10.0];
        let mut out = [0.0; 3];
        fgs(1, 2.0).equity(&[0, 1, 2], &stacks, &[50.0, 30.0, 20.0], &mut out);
        assert!(out[0] < out[2], "{out:?}: quien pondrá la BB vale menos que el futuro BTN");
        assert!((out.iter().sum::<f64>() - 100.0).abs() < 1e-6);
    }
}
