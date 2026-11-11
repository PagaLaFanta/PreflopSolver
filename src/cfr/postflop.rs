//! # MCCFR postflop: public chance sampling sobre el árbol geométrico
//!
//! En cada visita a una hoja de flop se muestrea un runout completo del [`BoardPool`]
//! (chance sampling del nodo de azar público) y se recorre el subárbol geométrico con
//! **rangos completos** de ambos jugadores (vectores de 1326 combos), como en PCS-CFR
//! (Johanson et al. 2012). Ventajas frente a external sampling por mano:
//! * una sola pasada evalúa los 1326 combos contra todo el rango rival (showdown O(n));
//! * el valor devuelto al preflop es un vector completo, justo lo que necesita el CFR+
//!   preflop vectorizado.
//!
//! ## Imperfect recall
//! Los combos que caen en el mismo bucket (en esa calle, en ese board) comparten
//! infoset: sus regrets contrafactuales se **suman** y se aplican con un único CAS por
//! bucket a `InfosetTable`. Como la clave no incluye la hoja ni el board, el mismo
//! infoset recibe actualizaciones de todas las hojas preflop compatibles y de todos los
//! boards → convergencia rápida con memoria mínima.
//!
//! ## Estimador del valor preflop
//! Un board bloquea a unos combos y no a otros. El valor devuelto al preflop para `h` es
//! el cociente autonormalizado `Σ_b N_b(h) / Σ_b D_b(h) · M(h)`: `N` = valor contra la
//! masa rival válida en `b`, `D` = esa masa, `M` = masa rival compatible real. Es decir,
//! la utilidad media de `h` contra el rango rival sobre los boards compatibles, escalada
//! a la probabilidad correcta de que el rival llegue. (Un estimador con la constante
//! `C(52,5)/C(48,5)` sería insesgado solo sobre boards uniformes; con un pool finito
//! sesga los pares que el pool bloquea más o menos.) En entrenamiento se muestrean
//! boards hasta cubrir todos los combos con alcance del héroe.

use super::lanes::{self, lbuf, LBuf, Lane, Sums, M};
use super::{PostflopEstimator, RangeVec, StrategyMode};
use crate::abstraction::board_pool::{BoardPool, SampledBoard};
use crate::eval::equity::BlockerSums;
use crate::eval::showdown::showdown_wtl;
use crate::eval::{COMPATIBLE_COMBOS, NUM_COMBOS};
use crate::math::rng::SplitMix64;
use crate::tree::postflop::{FlopLeaf, PfNode, PfNode3, OOP};
use crate::tree::{InfosetTable, PreflopTree};
use crate::tree::nodes::{factorial, nth_perm, perm_index};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Actualizaciones pendientes de la visita en curso (con `batch_updates`): se suman las
    /// de todos sus boards y se aplican juntas al terminar, con la estrategia fija durante
    /// la visita. El suelo de CFR+ ve la suma (ruido ~1/√boards) y no cada muestra.
    static BATCH: RefCell<Option<HashMap<u64, [f32; 4]>>> = const { RefCell::new(None) };
}


#[derive(Debug)]
pub struct PostflopEngine {
    pub pool: BoardPool,
    pub table: InfosetTable,
    pub boards_per_visit: u32,
    /// Bunching (card removal de los rangos foldeados): pondera cada board por la
    /// probabilidad de que salga dado que los jugadores retirados foldearon esas manos.
    pub bunching: bool,
    /// Boards (deterministas, los primeros del pool) al evaluar con la estrategia media
    /// (EV y mejor respuesta). Por defecto el pool entero: con un subconjunto la
    /// evaluación es más barata pero mide un juego ligeramente distinto del que se
    /// entrena (el entrenamiento muestrea todo el pool), y la NashConv se estanca.
    pub eval_boards: usize,
    /// Boards por visita proporcionales a la probabilidad de llegar a la hoja (hasta
    /// `boards_per_visit` con masa ≥ [`ADAPTIVE_MASS_REF`]); con `false`, siempre
    /// `boards_per_visit`.
    pub adaptive_boards: bool,
    /// Poda de hojas despreciables: en entrenamiento, una hoja cuya probabilidad conjunta
    /// de llegar (producto de masas de alcance) es menor que esto se evalúa pasando check
    /// hasta el showdown (el final de la línea de checks del mismo árbol) en vez de
    /// recorrer el árbol entero. Es dinámico: si la estrategia empieza a llegar, vuelve el
    /// MCCFR completo. 0 = desactivada. La evaluación (EV/BR) nunca poda.
    pub prune_mass: f64,
    /// Peso `t^avg_power` de la iteración preflop `t` en las sumas de estrategia postflop
    /// (0 = media uniforme).
    pub avg_power: f32,
    /// Suelo CFR+ en los regrets postflop (`false`: regret matching con regrets negativos,
    /// lo estándar en MCCFR: el suelo sobre muestras ruidosas sesga).
    pub regret_plus: bool,
    /// Peso `t^q` de los regrets de la iteración `t` (0 = sin descuento; 1 = Linear CFR).
    pub regret_power: f32,
    /// Suma las actualizaciones de todos los boards de una visita antes de aplicarlas.
    pub batch_updates: bool,
    /// Rampa de boards por visita: en la iteración `t < boards_ramp` se usa la fracción
    /// `t / boards_ramp` del objetivo (mínimo 1). Con la media ponderada `t²` las primeras
    /// iteraciones apenas cuentan, así que pueden ser más baratas y ruidosas. 0 = sin rampa.
    pub boards_ramp: u32,
    /// Hilos totales (presupuesto para repartir los boards de una hoja entre varios).
    pub threads: usize,
    /// Iteración preflop en curso (ver `PostflopEstimator::set_iteration`).
    iteration: std::sync::atomic::AtomicU32,
    /// Reducción de varianza (VR-MCCFR) en el valor de las hojas heads-up al entrenar: se
    /// resta, board a board, una línea base por bucket del river de la mano y se suma su
    /// media sobre el pool (variable de control: insesgado). Así el valor que llega a los
    /// regrets preflop apenas depende de qué boards salieron en la visita.
    pub vr: bool,
    /// Boards (submuestra sistemática con desplazamiento al azar en cada visita) con los que
    /// se calcula la media de la línea base. 1024 bastan: con 2048 o el pool entero la
    /// NashConv no mejora y cuesta más.
    pub vr_boards: usize,
    /// Líneas base por (hoja, jugador): por bucket del river, `[Σ peso·valor, Σ peso]` con
    /// decaimiento [`VR_DECAY`] por visita.
    baselines: std::sync::Mutex<HashMap<(usize, usize), Vec<[f64; 2]>>>,
}

/// Decaimiento por visita de las líneas base de VR (sigue a la estrategia, que cambia).
pub const VR_DECAY: f64 = 0.8;

/// Calle cuyo bucket indexa la línea base de VR. Cada board del pool es un runout entero y
/// el bucket del river predice el valor mucho mejor que el del flop (4-max 25 bb chip EV,
/// 16 boards, 300 iteraciones: NashConv 0,040 con el del flop, 0,017 con el del river).
const VR_STREET: usize = 2;

/// Masa conjunta de alcance a partir de la cual una hoja recibe todos los boards por
/// visita. Con 0,01 en vez de 0,02 (más boards para las hojas de masa media) la NashConv
/// baja antes a igual tiempo: 3 jugadores 30 bb ICM 0,030 en ~6,9 s en vez de ~8,4;
/// 4 jugadores 0,196 en 22 s en vez de ≈0,21. Con 0,005 ya cuesta más de lo que aporta.
pub const ADAPTIVE_MASS_REF: f64 = 0.01;

/// Suma parcial de un trozo de boards de una hoja (ver `joint_values`).
struct Part {
    num: Vec<Box<RangeVec>>,
    den: Vec<Box<RangeVec>>,
    obs_num: Vec<f64>,
    obs_den: f64,
    /// Con VR: por jugador y bucket del river (`VR_STREET`), `[Σ peso·valor, Σ peso·masa]` de la visita.
    bstat: Vec<Vec<[f64; 2]>>,
}

fn zero() -> Box<RangeVec> {
    super::zero_range()
}

impl PostflopEngine {
    pub fn new(pool: BoardPool, log2_capacity: u32, boards_per_visit: u32) -> Self {
        PostflopEngine { pool, table: InfosetTable::with_log2_capacity(log2_capacity), boards_per_visit: boards_per_visit.max(1), bunching: true, eval_boards: usize::MAX, adaptive_boards: false, prune_mass: 0.0, avg_power: 0.0, regret_plus: true, regret_power: 0.0, batch_updates: false, boards_ramp: 0, threads: 1, iteration: std::sync::atomic::AtomicU32::new(1), vr: false, vr_boards: 1024, baselines: std::sync::Mutex::new(HashMap::new()) }
    }

    /// Probabilidad de la acción 0 (check/fold) por combo, resolviendo cada bucket una vez.
    fn sigma0(&self, board: &SampledBoard, street: usize, key: u64, mode: StrategyMode, out: &mut RangeVec) {
        let buckets = &board.buckets[street];
        let nb = self.pool.num_buckets[street] as usize;
        let mut cache = vec![f32::NAN; nb];
        for &h in board.order.iter() {
            let b = buckets[h as usize] as usize;
            if cache[b].is_nan() {
                cache[b] = match self.table.get(key | b as u64) {
                    Some(info) => match mode {
                        StrategyMode::Current => info.strategy()[0],
                        StrategyMode::Average => info.average_strategy()[0],
                    },
                    None => 0.5,
                };
            }
            out[h as usize] = cache[b];
        }
    }

    /// Nodo final de la línea "todos pasan" (hijo 0 en cada decisión desde la raíz: sin
    /// apuestas previas es siempre check): el showdown del check-down.
    fn checkdown_node(l: &FlopLeaf) -> u32 {
        let mut ni = 0u32;
        match &l.multi {
            Some(m) => while let PfNode3::Decision { children, .. } = m.nodes[ni as usize] {
                ni = children[0];
            },
            None => while let PfNode::Decision { children, .. } = l.nodes[ni as usize] {
                ni = children[0];
            },
        }
        ni
    }

    /// Probabilidad de la acción 0 por carril, resolviendo cada bucket una vez.
    fn sigma0_lanes(&self, board: &SampledBoard, street: usize, key: u64, mode: StrategyMode, out: &mut Lane) {
        let nb = self.pool.num_buckets[street] as usize;
        let mut cache = vec![f32::NAN; nb];
        for (o, &b) in out[..M].iter_mut().zip(board.lane_buckets[street].iter()) {
            let c = &mut cache[b as usize];
            if c.is_nan() {
                *c = match self.table.get(key | b as u64) {
                    Some(info) => match mode {
                        StrategyMode::Current => info.strategy()[0],
                        StrategyMode::Average => info.average_strategy()[0],
                    },
                    None => 0.5,
                };
            }
            *o = *c;
        }
    }

    /// Recorrido PCS **conjunto** sobre carriles (ver [`super::lanes`]): devuelve a la vez
    /// los valores contrafactuales de los `k` jugadores de la hoja (2 en HU, 3 en
    /// multiway), `out[p]` con las utilidades del asiento `seats[p]` (`None` si `p` no
    /// está en `want`). En el nodo de `q` su valor se combina con σ y el de los demás se
    /// suma (sus rangos ya llevan la σ de `q`): CFR simultáneo en un solo recorrido.
    /// `r[p]`: alcances normalizados por 1225. `upd`: jugadores que actualizan.
    ///
    /// `obs`/`obs_acc`: asientos fuera del bote y su acumulador escalar
    /// `Σ_i r0(i)·v_k(i)` sobre los terminales (con el alcance de la posición 0 en ese
    /// terminal, que ya incluye su estrategia postflop). Solo hace falta el escalar: la
    /// capa preflop únicamente usa `E[u_k | todos llegan]`, así que no se propagan vectores
    /// por los nodos de decisión (coste independiente del número de observadores).
    #[allow(clippy::too_many_arguments)]
    fn walk_joint(&self, tree: &PreflopTree, leaf: &FlopLeaf, board: &SampledBoard, ni: u32, r: &[&Lane], ab: &[Option<LBuf>], seats: &[usize], obs: &[usize], obs_acc: &mut [f64], mode: StrategyMode, upd: u8, want: u8, rng: &mut SplitMix64) -> Vec<Option<LBuf>> {
        let k = r.len();
        let n = tree.n;
        let pairs = &board.pairs;
        let dec = match &leaf.multi {
            Some(m) => match m.nodes[ni as usize] {
                PfNode3::Decision { player, street, key, children, .. } => Some((player as usize, street, key, children)),
                _ => None,
            },
            None => match leaf.nodes[ni as usize] {
                PfNode::Decision { role, street, key, children, .. } => Some((role as usize, street, key, children)),
                _ => None,
            },
        };
        if let Some((q, street, key, children)) = dec {
            let s = street as usize - 1;
            let mut s0 = lbuf();
            self.sigma0_lanes(board, s, key, mode, &mut s0);
            let (mut a0, mut a1) = (lbuf(), lbuf());
            for i in 0..M {
                a0[i] = r[q][i] * s0[i];
                a1[i] = r[q][i] - a0[i];
            }
            let (mut r0, mut r1) = (r.to_vec(), r.to_vec());
            r0[q] = &a0;
            r1[q] = &a1;
            let mut v0 = self.walk_joint(tree, leaf, board, children[0], &r0, ab, seats, obs, obs_acc, mode, upd, want, rng);
            let v1 = self.walk_joint(tree, leaf, board, children[1], &r1, ab, seats, obs, obs_acc, mode, upd, want, rng);
            let mut out: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
            for p in (0..k).filter(|&p| want & (1 << p) != 0) {
                let x1 = v1[p].as_ref().unwrap();
                if p == q {
                    let x0 = v0[p].as_ref().unwrap();
                    let mut o = lbuf();
                    for i in 0..M {
                        o[i] = x1[i] + s0[i] * (x0[i] - x1[i]);
                    }
                    if upd & (1 << q) != 0 {
                        self.update_lanes(board, s, key, &o, x0, x1, &a0, &a1, rng);
                    }
                    out[p] = Some(o);
                } else {
                    let mut o = v0[p].take().unwrap();
                    for i in 0..M {
                        o[i] += x1[i];
                    }
                    out[p] = Some(o);
                }
            }
            return out;
        }
        // Sumas de bloqueadores solo de los rangos que hacen de rival de algún jugador pedido.
        let bs: Vec<Sums> = (0..k).map(|j| if (0..k).any(|p| p != j && want & (1 << p) != 0) { Sums::of(r[j], pairs) } else { Sums::none() }).collect();
        let mut out: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
        match &leaf.multi {
            None => match leaf.nodes[ni as usize] {
                PfNode::Fold { util, .. } => {
                    for p in (0..2).filter(|&p| want & (1 << p) != 0) {
                        let o = 1 - p;
                        let mut v = lbuf();
                        bs[o].compat_all(r[o], pairs, &mut v);
                        if p == 0 && !obs.is_empty() {
                            let d: f64 = (0..M).map(|i| (r[0][i] * v[i]) as f64).sum();
                            for (acc, &st) in obs_acc.iter_mut().zip(obs) {
                                *acc += tree.utils[util as usize + st] as f64 * d;
                            }
                        }
                        let u = tree.utils[util as usize + seats[p]];
                        for x in v[..M].iter_mut() {
                            *x *= u;
                        }
                        out[p] = Some(v);
                    }
                }
                PfNode::Showdown { util } if want == 3 => {
                    // Los dos jugadores: una pasada conjunta.
                    let base = util as usize;
                    let u = |k: usize, st: usize| tree.utils[base + k * n + st];
                    let (s0, s1) = (seats[0], seats[1]);
                    // utils: [gana OOP, gana IP, empate] por asiento.
                    let c0 = [u(0, s0), u(2, s0), u(1, s0)];
                    let c1 = [u(1, s1), u(2, s1), u(0, s1)];
                    let (mut o0, mut o1) = (lbuf(), lbuf());
                    let [dw, dt, dl] = lanes::showdown_pair(board, r[0], &bs[0], r[1], &bs[1], c0, c1, &mut o0, &mut o1);
                    for (acc, &st) in obs_acc.iter_mut().zip(obs) {
                        *acc += u(0, st) as f64 * dw + u(2, st) as f64 * dt + u(1, st) as f64 * dl;
                    }
                    out[0] = Some(o0);
                    out[1] = Some(o1);
                }
                PfNode::Showdown { util } => {
                    let base = util as usize;
                    for p in (0..2).filter(|&p| want & (1 << p) != 0) {
                        let o = 1 - p;
                        let (mut w, mut t, mut l) = (lbuf(), lbuf(), lbuf());
                        lanes::showdown(board, r[o], &bs[o], &mut w, &mut t, &mut l);
                        if p == 0 && !obs.is_empty() {
                            let (mut dw, mut dt, mut dl) = (0.0f64, 0.0f64, 0.0f64);
                            for i in 0..M {
                                let x = r[0][i] as f64;
                                dw += x * w[i] as f64;
                                dt += x * t[i] as f64;
                                dl += x * l[i] as f64;
                            }
                            for (acc, &st) in obs_acc.iter_mut().zip(obs) {
                                *acc += tree.utils[base + st] as f64 * dw + tree.utils[base + 2 * n + st] as f64 * dt + tree.utils[base + n + st] as f64 * dl;
                            }
                        }
                        let st = seats[p];
                        let (u_oop, u_ip, u_tie) = (tree.utils[base + st], tree.utils[base + n + st], tree.utils[base + 2 * n + st]);
                        let (uw, ul) = if p == 0 { (u_oop, u_ip) } else { (u_ip, u_oop) };
                        for i in 0..M {
                            w[i] = uw * w[i] + u_tie * t[i] + ul * l[i];
                        }
                        out[p] = Some(w);
                    }
                }
                PfNode::Decision { .. } => unreachable!(),
            },
            Some(m) => match m.nodes[ni as usize] {
                PfNode3::End { alive, util } => self.end_multi(tree, board, alive, util, r, ab, &bs, seats, obs, obs_acc, want, &mut out),
                PfNode3::Decision { .. } => unreachable!(),
            },
        }
        out
    }

    /// Final de una hoja multiway (3 o 4 jugadores) para los jugadores pedidos.
    ///
    /// Con botes laterales el reparto depende del orden de fuerza entero de los vivos, así
    /// que cada final guarda una utilidad por orden (`m!`). Para el héroe del carril `i`:
    /// cada rival vivo `j` queda por encima o por debajo con las masas exactas con bloqueo
    /// del héroe (`showdown` contra `r_j`; los empates, a medias); dentro de cada lado, el
    /// orden entre rivales se reparte con sus distribuciones de fuerza (CDF por grupos,
    /// sin bloqueo entre rivales, como el resto del modelo multiway), condicionadas a
    /// quedar por encima o por debajo del héroe. Los rivales que foldearon aportan su masa
    /// compatible. Con el héroe fuera, el orden de los vivos no depende de su mano.
    #[allow(clippy::too_many_arguments)]
    fn end_multi(&self, tree: &PreflopTree, board: &SampledBoard, alive: u8, util: u32, r: &[&Lane], ab: &[Option<LBuf>], bs: &[Sums], seats: &[usize], obs: &[usize], obs_acc: &mut [f64], want: u8, out: &mut [Option<LBuf>]) {
        end_multi_core(&tree.utils, tree.n, board, alive, util, r, ab, bs, seats, obs, obs_acc, want, out);
    }
    /// Valores de todos los jugadores de una hoja (HU o a 3) con el estimador
    /// autonormalizado por combo (ver doc del módulo), en un único recorrido por board.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn joint_values(&self, tree: &PreflopTree, leaf: usize, reaches: &[&RangeVec], mode: StrategyMode, upd: u8, want: u8, seed: u64, obs: &[usize], folded: &[&RangeVec], pool: super::pool::Pool<'_>) -> Vec<Vec<Box<RangeVec>>> {
        let l = &tree.leaves[leaf];
        let k = reaches.len();
        assert!((k == 2 && l.multi.is_none() && !l.nodes.is_empty()) || (k >= 3 && l.multi.as_ref().is_some_and(|m| m.seats.len() == k)), "leaf incompatible with PostflopEngine");
        let mut rng = SplitMix64::new(seed);
        // Solo hace falta cubrir los combos de los jugadores cuyos valores se piden.
        let wanted: Vec<&RangeVec> = (0..k).filter(|&p| want & (1 << p) != 0).map(|p| reaches[p]).collect();
        // Boards de la visita: con reparto adaptativo, proporcionales al peso de la hoja en
        // los valores pedidos: los de cada jugador van multiplicados por la masa de alcance
        // de los **demás** (no por la suya), así que el ruido que la hoja mete en sus
        // regrets preflop escala con ese producto; se toma el mayor entre los jugadores
        // pedidos. Con el producto de todas las masas, una acción que el jugador aún no
        // juega (su alcance ≈ 0) dejaba la hoja en check-down y sin entrenar justo cuando
        // hace falta valorarla para descubrirla. Siempre con cobertura de combos.
        let masses: Vec<f64> = reaches.iter().map(|r| r[..NUM_COMBOS].iter().map(|&x| x as f64).sum::<f64>() / NUM_COMBOS as f64).collect();
        let mass: f64 = (0..k).filter(|&p| want & (1 << p) != 0).map(|p| (0..k).filter(|&j| j != p).map(|j| masses[j]).product::<f64>()).fold(0.0, f64::max);
        let ramp = if self.boards_ramp > 0 { (self.iteration.load(std::sync::atomic::Ordering::Relaxed) as f64 / self.boards_ramp as f64).min(1.0) } else { 1.0 };
        let full = self.boards_per_visit as f64 * ramp;
        let target = if self.adaptive_boards { ((full * (mass / ADAPTIVE_MASS_REF).min(1.0)).round() as u32).max(1) } else { (full.round() as u32).max(1) };
        // Hoja despreciable en entrenamiento: check-down (sin updates) desde el final de la
        // línea de checks.
        let pruned = mode == StrategyMode::Current && mass < self.prune_mass;
        let (start, upd) = if pruned { (Self::checkdown_node(l), 0) } else { (0, upd) };
        let boards = self.boards_for(mode, &wanted, target, &mut rng);
        let batch = self.batch_updates && upd != 0;
        // VR-MCCFR: solo en hojas heads-up, al entrenar y sin poda. Líneas base (valor medio
        // por unidad de masa rival, por bucket del river) fijas durante la visita.
        let vr = self.vr && k == 2 && mode == StrategyMode::Current && !pruned;
        let vs = VR_STREET;
        let nb = self.pool.num_buckets[vs] as usize;
        let base: Vec<Vec<f32>> = if vr {
            let map = self.baselines.lock().unwrap();
            (0..k)
                .map(|p| match map.get(&(leaf, p)) {
                    Some(v) => v.iter().map(|&[a, b]| if b > 0.0 { (a / b) as f32 } else { 0.0 }).collect(),
                    None => vec![0.0; nb],
                })
                .collect()
        } else {
            Vec::new()
        };
        if batch {
            BATCH.with(|b| *b.borrow_mut() = Some(HashMap::new()));
        }
        let seat_of = |p: usize| -> usize {
            match &l.multi {
                Some(m) => m.seats[p] as usize,
                None if p == 0 => l.oop as usize,
                None => l.ip as usize,
            }
        };
        let seats: Vec<usize> = (0..k).map(seat_of).collect();
        // Los observadores no viajan como vectores: ver `walk_joint`.
        let obs: &[usize] = if want & 1 != 0 { obs } else { &[] };
        // Estimador de razón agregado para los observadores: Σ_b w·S_b / Σ_b w·P_b, con
        // P_b la masa conjunta de los que llegan (posición 0 contra el resto) en el board.
        // Bunching: masa total de cada rango foldeado (para normalizar la parte compatible
        // con cada board). Rangos vacíos no pesan.
        let fold_tot: Vec<f32> = if self.bunching { folded.iter().map(|r| r[..NUM_COMBOS].iter().sum()).collect() } else { Vec::new() };
        // Bunching conjunto: las cartas que se llevan los que foldearon condicionan a la vez el
        // board y las manos de los que llegan. Los rangos que entran ya llevan el factor
        // marginal de cada mano (`P_k(sin cartas de x)`, ver `terminal_all`) y el board su peso
        // `Π_k P_k(sin cartas del board)`; falta la correlación entre ambos: por board y combo,
        // `Π_k P_k(sin cartas del board ni de x) / (P_k(sin cartas del board)·P_k(sin cartas de
        // x))`. El mismo factor para todos los jugadores: pondera sus carriles y, como héroe,
        // su numerador y denominador. Sin él, en un 6-max con cuatro folds el bunching se
        // quedaba en ~80 % del efecto real (BB pagando el shove de la SB con 32o: 0,282 de
        // equity frente a 0,273 exacta).
        let fold_full: Vec<(BlockerSums, &RangeVec, f32)> = if self.bunching {
            folded.iter().zip(&fold_tot).filter(|(_, &t)| t > 0.0).map(|(r, &t)| (BlockerSums::new(&r[..]), *r, t)).collect()
        } else {
            Vec::new()
        };
        // Suma de los boards de `bs` (numeradores y denominadores por combo y los de los
        // observadores): los trozos del reparto en paralelo se suman al final.
        let run = |bs: &[&SampledBoard], rng: &mut SplitMix64| -> Part {
            let mut num: Vec<Box<RangeVec>> = (0..k).map(|_| super::zero_range()).collect();
            let mut den: Vec<Box<RangeVec>> = (0..k).map(|_| super::zero_range()).collect();
            let (mut obs_num, mut obs_den) = (vec![0.0f64; obs.len()], 0.0f64);
            let mut bstat: Vec<Vec<[f64; 2]>> = if vr { vec![vec![[0.0; 2]; nb]; k] } else { Vec::new() };
            for &board in bs {
                let w = self.fold_weight(board, folded, &fold_tot);
                if w <= 0.0 {
                    continue;
                }
                let jf: Option<LBuf> = (!fold_full.is_empty()).then(|| joint_fold_factor(board, &fold_full));
                let rl = board_reaches(board, reaches, jf.as_ref());
                let rs: Vec<&Lane> = rl.iter().map(|b| &**b).collect();
                // A 3: masa exacta de pares de rivales compatibles entre sí (`pair_mass`) frente
                // al producto de compatibles, por héroe pedido, con los rangos de la raíz de la
                // hoja. Escala las masas de los finales sin showdown a 3 y el denominador.
                let ab: Vec<Option<LBuf>> = (0..k)
                    .map(|p| {
                        (k == 3 && want & (1 << p) != 0).then(|| {
                            let (a, b) = rivals3(p);
                            rival_pair_factor(board, rs[a], rs[b], &Sums::of(rs[a], &board.pairs), &Sums::of(rs[b], &board.pairs))
                        })
                    })
                    .collect();
                let mut acc = vec![0.0f64; obs.len()];
                let v = self.walk_joint(tree, l, board, start, &rs, &ab, &seats, obs, &mut acc, mode, upd, want, rng);
                // Masa rival (producto de compatibles) de cada jugador pedido, por carril.
                let sums: Vec<Sums> = rl.iter().map(|r| Sums::of(r, &board.pairs)).collect();
                let mut comp = lbuf();
                for p in (0..k).filter(|&p| want & (1 << p) != 0) {
                    comp[..M].fill(1.0);
                    for j in (0..k).filter(|&j| j != p) {
                        for i in 0..M {
                            comp[i] *= sums[j].compat(rs[j], &board.pairs, i);
                        }
                    }
                    if let Some(q) = &ab[p] {
                        for i in 0..M {
                            comp[i] *= q[i];
                        }
                    }
                    if p == 0 && !obs.is_empty() {
                        let pm: f64 = (0..M).map(|i| (rs[0][i] * comp[i]) as f64).sum();
                        obs_den += w as f64 * pm;
                        for (x, a) in obs_num.iter_mut().zip(&acc) {
                            *x += w as f64 * a;
                        }
                    }
                    let vp = v[p].as_ref().unwrap();
                    let (np, dp) = (&mut num[p], &mut den[p]);
                    if vr {
                        let (bp, st) = (&base[p], &mut bstat[p]);
                        let lb = &board.lane_buckets[vs];
                        for (i, &h) in board.order.iter().enumerate() {
                            let h = h as usize;
                            let wi = jf.as_ref().map_or(w, |f| w * f[i]);
                            let kb = lb[i] as usize;
                            dp[h] += wi * comp[i];
                            np[h] += wi * (vp[i] - comp[i] * bp[kb]);
                            st[kb][0] += (wi * vp[i]) as f64;
                            st[kb][1] += (wi * comp[i]) as f64;
                        }
                    } else {
                        for (i, &h) in board.order.iter().enumerate() {
                            let h = h as usize;
                            let wi = jf.as_ref().map_or(w, |f| w * f[i]);
                            dp[h] += wi * comp[i];
                            np[h] += wi * vp[i];
                        }
                    }
                }
            }
            Part { num, den, obs_num, obs_den, bstat }
        };
        // Casi todo el trabajo está en las hojas y cada una recorre sus boards: con muchos
        // núcleos, antes la CPU esperaba a la hoja más pesada. Los boards se reparten entre
        // el `Pool` compartido de todo `Solver::run` (ver `cfr::pool`): si no hay
        // trabajadores libres ahora mismo, `join_all` corre todo en línea sin más.
        // Con pocos boards por visita (p.ej. 16 de serie) partirlos en tantos trozos como
        // hilos deja 1 board por trozo: el mutex+condvar del pool por tarea pesa más que ese
        // board. `MIN_BOARDS_PER_CHUNK` evita troceos tan finos; el ancho real del recorrido
        // preflop (muchas hojas a la vez) ya reparte el resto entre núcleos.
        const MIN_BOARDS_PER_CHUNK: usize = 4;
        let chunks = (boards.len() / MIN_BOARDS_PER_CHUNK).clamp(1, self.threads.max(1));
        let parts: Vec<Part> = if batch || boards.len() < 2 || chunks <= 1 {
            vec![run(&boards, &mut rng)]
        } else {
            let seeds: Vec<u64> = (0..chunks).map(|_| rng.next_u64()).collect();
            let per = boards.len().div_ceil(chunks);
            let run = &run;
            let tasks: Vec<Box<dyn FnOnce() -> Part + Send + '_>> = boards
                .chunks(per)
                .zip(&seeds)
                .map(|(c, &sd)| Box::new(move || { let mut r = SplitMix64::new(sd); run(c, &mut r) }) as Box<dyn FnOnce() -> Part + Send + '_>)
                .collect();
            pool.join_all(tasks)
        };
        let mut parts = parts.into_iter();
        let first = parts.next().expect("at least one chunk");
        let (mut num, mut den, mut obs_num, mut obs_den, mut bstat) = (first.num, first.den, first.obs_num, first.obs_den, first.bstat);
        for pt in parts {
            for (a, b) in bstat.iter_mut().zip(&pt.bstat) {
                for (x, y) in a.iter_mut().zip(b) {
                    x[0] += y[0];
                    x[1] += y[1];
                }
            }
            for p in 0..k {
                for h in 0..NUM_COMBOS {
                    num[p][h] += pt.num[p][h];
                    den[p][h] += pt.den[p][h];
                }
            }
            for (x, y) in obs_num.iter_mut().zip(&pt.obs_num) {
                *x += y;
            }
            obs_den += pt.obs_den;
        }
        if batch {
            if let Some(map) = BATCH.with(|b| b.borrow_mut().take()) {
                for (key, a) in map {
                    if let Some(info) = self.table.get_or_insert(key) {
                        info.update([a[0], a[1]], [a[2], a[3]], self.regret_plus, &mut rng);
                    }
                }
            }
        }
        let full: Vec<BlockerSums> = reaches.iter().map(|r| BlockerSums::new(&r[..])).collect();
        // VR: media exacta de la línea base sobre el pool entero (solo masas, sin recorrer el
        // árbol) y actualización de las líneas base con lo visto en esta visita.
        let cv: Option<Vec<(Box<RangeVec>, Box<RangeVec>)>> = vr.then(|| self.vr_pool_terms(rng.next_u64(), reaches, want, folded, &fold_tot, &fold_full, &base, pool));
        if vr {
            let mut map = self.baselines.lock().unwrap();
            for p in (0..k).filter(|&p| want & (1 << p) != 0) {
                let e = map.entry((leaf, p)).or_insert_with(|| vec![[0.0; 2]; nb]);
                for (x, y) in e.iter_mut().zip(&bstat[p]) {
                    x[0] = x[0] * VR_DECAY + y[0];
                    x[1] = x[1] * VR_DECAY + y[1];
                }
            }
        }
        let mut out: Vec<Vec<Box<RangeVec>>> = Vec::with_capacity(k);
        for (p, mut np) in num.into_iter().enumerate() {
            for h in 0..NUM_COMBOS {
                let fc = (0..k).filter(|&j| j != p).map(|j| full[j].compatible(&reaches[j][..], h) / COMPATIBLE_COMBOS).product::<f32>();
                let sample = if den[p][h] > 0.0 { np[h] / den[p][h] } else { 0.0 };
                np[h] = match &cv {
                    // Estimador con variable de control: Σ_S w·(v − m·B)/Σ_S w·m + C/D.
                    Some(c) if want & (1 << p) != 0 => {
                        let (cc, dd) = (&c[p].0, &c[p].1);
                        let corr = if dd[h] > 0.0 { cc[h] / dd[h] } else { 0.0 };
                        if den[p][h] > 0.0 || dd[h] > 0.0 { (sample + corr) * fc } else { 0.0 }
                    }
                    _ => sample * fc,
                };
            }
            out.push(vec![np]);
        }
        // Observador k: E[u_k | llegan todos] con la normalización de los valores de la
        // posición 0 (· P(los rivales llegan | h)).
        for x in &obs_num {
            let e = if obs_den > 0.0 { (x / obs_den) as f32 } else { 0.0 };
            let mut v = super::zero_range();
            for h in 0..NUM_COMBOS {
                v[h] = e * (1..k).map(|j| full[j].compatible(&reaches[j][..], h) / COMPATIBLE_COMBOS).product::<f32>();
            }
            out[0].push(v);
        }
        out
    }

    /// Peso de bunching de `board`: probabilidad de que los que foldearon no tengan sus
    /// cartas (producto sobre ellos de su masa compatible / masa total).
    fn fold_weight(&self, board: &SampledBoard, folded: &[&RangeVec], fold_tot: &[f32]) -> f32 {
        if !self.bunching {
            return 1.0;
        }
        folded
            .iter()
            .zip(fold_tot)
            .filter(|(_, &t)| t > 0.0)
            .map(|(r, &t)| {
                let blocked: f32 = board.blocked.iter().map(|&h| r[h as usize]).sum();
                ((t - blocked) / t).max(0.0)
            })
            .product()
    }

    /// VR: por jugador pedido, `C[h] = Σ_b w·m_b(h)·B[bucket_b(h)]` y `D[h] = Σ_b w·m_b(h)`
    /// sobre `vr_boards` boards del pool (`w`: peso de bunching del board y de la mano;
    /// `m`: masa rival compatible), con las mismas definiciones que `joint_values` usa en
    /// los boards muestreados, así que `C/D` es la media exacta de la línea base.
    #[allow(clippy::too_many_arguments)]
    fn vr_pool_terms(&self, offset: u64, reaches: &[&RangeVec], want: u8, folded: &[&RangeVec], fold_tot: &[f32], fold_full: &[(BlockerSums, &RangeVec, f32)], base: &[Vec<f32>], pool: super::pool::Pool<'_>) -> Vec<(Box<RangeVec>, Box<RangeVec>)> {
        let k = reaches.len();
        // Boards para la media de la línea base: el pool entero o una submuestra sistemática.
        // Submuestra sistemática con desplazamiento al azar en cada visita (insesgada; con un
        // subconjunto fijo la media de la línea base quedaba sesgada).
        let len = self.pool.boards.len();
        let n = self.vr_boards.clamp(1, len);
        let all: Vec<&SampledBoard> = (0..n).map(|i| &self.pool.boards[((offset as usize % len) + i * len / n) % len]).collect();
        let vs = VR_STREET;
        let run = |bs: &[&SampledBoard]| -> Vec<(Box<RangeVec>, Box<RangeVec>)> {
            let mut out: Vec<(Box<RangeVec>, Box<RangeVec>)> = (0..k).map(|_| (zero(), zero())).collect();
            for &board in bs {
                let w = self.fold_weight(board, folded, fold_tot);
                if w <= 0.0 {
                    continue;
                }
                let jf: Option<LBuf> = (!fold_full.is_empty()).then(|| joint_fold_factor(board, fold_full));
                let rl = board_reaches(board, reaches, jf.as_ref());
                let rs: Vec<&Lane> = rl.iter().map(|b| &**b).collect();
                let sums: Vec<Sums> = rs.iter().map(|r| Sums::of(r, &board.pairs)).collect();
                let lb = &board.lane_buckets[vs];
                for p in (0..k).filter(|&p| want & (1 << p) != 0) {
                    let (c, d) = &mut out[p];
                    for (i, &h) in board.order.iter().enumerate() {
                        let mut comp = 1.0f32;
                        for j in (0..k).filter(|&j| j != p) {
                            comp *= sums[j].compat(rs[j], &board.pairs, i);
                        }
                        let wi = jf.as_ref().map_or(w, |f| w * f[i]);
                        let h = h as usize;
                        d[h] += wi * comp;
                        c[h] += wi * comp * base[p][lb[i] as usize];
                    }
                }
            }
            out
        };
        let chunks = self.threads.clamp(1, 64);
        let per = all.len().div_ceil(chunks);
        let run = &run;
        let tasks: Vec<Box<dyn FnOnce() -> Vec<(Box<RangeVec>, Box<RangeVec>)> + Send + '_>> =
            all.chunks(per).map(|c| Box::new(move || run(c)) as Box<dyn FnOnce() -> Vec<(Box<RangeVec>, Box<RangeVec>)> + Send + '_>).collect();
        let mut parts = pool.join_all(tasks).into_iter();
        let mut acc = parts.next().unwrap_or_else(|| (0..k).map(|_| (zero(), zero())).collect());
        for pt in parts {
            for (a, b) in acc.iter_mut().zip(pt) {
                for h in 0..NUM_COMBOS {
                    a.0[h] += b.0[h];
                    a.1[h] += b.1[h];
                }
            }
        }
        acc
    }

    /// Agrega regrets y sumas de estrategia por bucket (carriles) y aplica un CAS por
    /// infoset. `v`: valor mezclado; `v0`/`v1`: por acción; `rs0`/`rs1`: alcances propios.
    #[allow(clippy::too_many_arguments)]
    fn update_lanes(&self, board: &SampledBoard, s: usize, key: u64, v: &Lane, v0: &Lane, v1: &Lane, rs0: &Lane, rs1: &Lane, rng: &mut SplitMix64) {
        let nb = self.pool.num_buckets[s] as usize;
        let mut acc = vec![[0.0f32; 4]; nb];
        let mut seen = vec![false; nb];
        let t = self.iteration.load(std::sync::atomic::Ordering::Relaxed) as f32;
        let wt = if self.avg_power == 0.0 { 1.0 } else { t.powf(self.avg_power) };
        // Linear CFR (Pluribus): el regret de la iteración t pesa t^q, así que lo aprendido
        // contra el postflop casi aleatorio del principio se olvida.
        let wr = if self.regret_power == 0.0 { 1.0 } else { t.powf(self.regret_power) };
        for (i, &b) in board.lane_buckets[s].iter().enumerate() {
            let b = b as usize;
            let a = &mut acc[b];
            seen[b] = true;
            a[0] += (v0[i] - v[i]) * wr;
            a[1] += (v1[i] - v[i]) * wr;
            a[2] += rs0[i] * wt;
            a[3] += rs1[i] * wt;
        }
        let batched = BATCH.with(|bt| {
            let mut bt = bt.borrow_mut();
            let Some(map) = bt.as_mut() else { return false };
            for (b, a) in acc.iter().enumerate().filter(|&(b, _)| seen[b]) {
                let e = map.entry(key | b as u64).or_insert([0.0; 4]);
                for j in 0..4 {
                    e[j] += a[j];
                }
            }
            true
        });
        if batched {
            return;
        }
        for (b, a) in acc.iter().enumerate() {
            if seen[b] {
                if let Some(info) = self.table.get_or_insert(key | b as u64) {
                    info.update([a[0], a[1]], [a[2], a[3]], self.regret_plus, rng);
                }
            }
        }
    }

    /// Recorrido de un observador: masas de outcome ponderadas por utilidad de cada asiento.
    #[allow(clippy::too_many_arguments)]
    fn observe(&self, tree: &PreflopTree, leaf: &FlopLeaf, board: &SampledBoard, ni: u32, r_oop: &RangeVec, r_ip: &RangeVec, mode: StrategyMode, acc: &mut [f64]) {
        let n = tree.n;
        match leaf.nodes[ni as usize] {
            PfNode::Fold { util, .. } => {
                let bs = BlockerSums::new(r_ip);
                let mass: f64 = board.order.iter().map(|&a| (r_oop[a as usize] * bs.compatible(r_ip, a as usize)) as f64).sum();
                for (s, x) in acc.iter_mut().enumerate().take(n) {
                    *x += mass * tree.utils[util as usize + s] as f64;
                }
            }
            PfNode::Showdown { util } => {
                let (mut w, mut t, mut l) = (zero(), zero(), zero());
                showdown_wtl(&board.order, &board.ranks, r_ip, &mut w[..], &mut t[..], &mut l[..]);
                let (mut pw, mut pt, mut pl) = (0.0f64, 0.0f64, 0.0f64);
                for &a in board.order.iter() {
                    let a = a as usize;
                    let r = r_oop[a] as f64;
                    pw += r * w[a] as f64;
                    pt += r * t[a] as f64;
                    pl += r * l[a] as f64;
                }
                let u = &tree.utils[util as usize..];
                for (s, x) in acc.iter_mut().enumerate().take(n) {
                    *x += pw * u[s] as f64 + pl * u[n + s] as f64 + pt * u[2 * n + s] as f64;
                }
            }
            PfNode::Decision { role, street, key, children, .. } => {
                let mut s0 = zero();
                self.sigma0(board, street as usize - 1, key, mode, &mut s0);
                let (mut a0, mut a1) = (zero(), zero());
                let src = if role == OOP { r_oop } else { r_ip };
                for &h in board.order.iter() {
                    let h = h as usize;
                    a0[h] = src[h] * s0[h];
                    a1[h] = src[h] * (1.0 - s0[h]);
                }
                if role == OOP {
                    self.observe(tree, leaf, board, children[0], &a0, r_ip, mode, acc);
                    self.observe(tree, leaf, board, children[1], &a1, r_ip, mode, acc);
                } else {
                    self.observe(tree, leaf, board, children[0], r_oop, &a0, mode, acc);
                    self.observe(tree, leaf, board, children[1], r_oop, &a1, mode, acc);
                }
            }
        }
    }

    fn masked(board: &SampledBoard, r: &RangeVec, scale: f32) -> Box<RangeVec> {
        let mut m = zero();
        for &h in board.order.iter() {
            m[h as usize] = r[h as usize] * scale;
        }
        m
    }

    /// Boards a recorrer en una visita.
    ///
    /// * Evaluación (`Average`) o `boards_per_visit ≥ pool`: el pool entero, determinista.
    /// * Entrenamiento: boards distintos al azar hasta tener al menos `boards_per_visit` y
    ///   hasta que **todo combo con alcance** del héroe tenga algún board compatible (así
    ///   el estimador autonormalizado nunca deja un combo sin valor; con P(bloqueo) ≈ 0.185
    ///   bastan ~4-5 boards en media).
    fn boards(&self, mode: StrategyMode, r_hero: &RangeVec, rng: &mut SplitMix64) -> Vec<&SampledBoard> {
        self.boards_for(mode, &[r_hero], self.boards_per_visit, rng)
    }

    /// Como [`Self::boards`] cubriendo los combos con alcance de cualquiera de los rangos.
    fn boards_for(&self, mode: StrategyMode, reaches: &[&RangeVec], target: u32, rng: &mut SplitMix64) -> Vec<&SampledBoard> {
        let pool = &self.pool.boards;
        if mode == StrategyMode::Average {
            // Submuestra sistemática (uno de cada `len/n`): el pool está ordenado por clase de
            // flop, así que los primeros `n` serían solo unas pocas texturas.
            let n = self.eval_boards.clamp(1, pool.len());
            return (0..n).map(|i| &pool[i * pool.len() / n]).collect();
        }
        if target as usize >= pool.len() {
            return pool.iter().collect();
        }
        let mut pending: Vec<usize> = (0..NUM_COMBOS).filter(|&h| reaches.iter().any(|r| r[h] > 0.0)).collect();
        let mut picked = Vec::new();
        let mut used = vec![false; pool.len()];
        // Muestreo sistemático: el pool está ordenado por clase de flop, así que un
        // desplazamiento al azar con paso fijo reparte los `target` boards de la visita
        // entre todas las texturas en su proporción (estratificado: mismo estimador
        // insesgado, menos varianza que `target` boards independientes). Los que falten
        // para cubrir todos los combos, al azar.
        let n = target as usize;
        let off = rng.next_u64() % pool.len() as u64;
        for i in 0..n {
            let k = ((off + (i * pool.len() / n) as u64) % pool.len() as u64) as usize;
            if !used[k] {
                used[k] = true;
                let b = &pool[k];
                pending.retain(|&h| !b.valid(h));
                picked.push(b);
            }
        }
        while picked.len() < pool.len() && (picked.len() < target as usize || !pending.is_empty()) {
            let mut k = (rng.next_u64() % pool.len() as u64) as usize;
            while used[k] {
                k = (k + 1) % pool.len();
            }
            used[k] = true;
            let b = &pool[k];
            pending.retain(|&h| !b.valid(h));
            picked.push(b);
        }
        picked
    }
}

/// Factor de bunching conjunto por carril (ver `joint_values`): `Π_k [P_k(sin cartas del
/// board ni del combo) / P_k(sin cartas del board)] / P_k(sin cartas del combo)`, con `k` los
/// que foldearon (`(sumas del rango completo, rango, masa total)`).
/// Los dos rivales de `p` en un bote a 3.
fn rivals3(p: usize) -> (usize, usize) {
    match p {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Bote a 3: masa exacta de pares de los dos rivales compatibles entre sí (`pair_mass`)
/// frente al producto de sus compatibles, por carril del héroe (≤ 1).
fn rival_pair_factor<B: lanes::LaneBoard + ?Sized>(board: &B, ra: &Lane, rb: &Lane, sa: &Sums, sb: &Sums) -> LBuf {
    let mut q = lbuf();
    lanes::pair_mass(board, ra, rb, &mut q);
    let pairs = board.pairs();
    for i in 0..M {
        let ind = sa.compat(ra, pairs, i) * sb.compat(rb, pairs, i);
        q[i] = if ind > 0.0 { (q[i] / ind).min(1.0) } else { 1.0 };
    }
    q
}

/// Rangos de cada jugador en los carriles de `board` (masa por combo compatible), con el
/// factor conjunto de los que ya foldearon si lo hay.
fn board_reaches(board: &SampledBoard, reaches: &[&RangeVec], jf: Option<&LBuf>) -> Vec<LBuf> {
    reaches
        .iter()
        .map(|r| {
            let mut b = lbuf();
            lanes::gather(board, &r[..], 1.0 / COMPATIBLE_COMBOS, &mut b);
            if let Some(f) = jf {
                for i in 0..M {
                    b[i] *= f[i];
                }
            }
            b
        })
        .collect()
}

pub(crate) fn joint_fold_factor(board: &SampledBoard, folds: &[(BlockerSums, &RangeVec, f32)]) -> LBuf {
    let mut f = lbuf();
    f[..M].fill(1.0);
    for (full, r, tot) in folds {
        let mut l = lbuf();
        lanes::gather(board, &r[..], 1.0, &mut l);
        let s = Sums::of(&l, &board.pairs);
        if s.total <= 0.0 {
            continue;
        }
        for (i, &h) in board.order.iter().enumerate() {
            let on_board = s.compat(&l, &board.pairs, i) / s.total;
            let marginal = full.compatible(&r[..], h as usize) / tot;
            f[i] *= if marginal > 0.0 { on_board / marginal } else { 0.0 };
        }
    }
    f
}

/// Núcleo de [`PostflopEngine::end_multi`] sobre la tabla de utilidades (`utils`, `n`
/// asientos por outcome).
#[allow(clippy::too_many_arguments)]
pub(crate) fn end_multi_core<B: lanes::LaneBoard + ?Sized>(utils: &[f32], n: usize, board: &B, alive: u8, util: u32, r: &[&Lane], ab: &[Option<LBuf>], bs: &[Sums], seats: &[usize], obs: &[usize], obs_acc: &mut [f64], want: u8, out: &mut [Option<LBuf>]) {
    let k = r.len();
    let pairs = board.pairs();
    let list: Vec<usize> = (0..k).filter(|&p| alive & (1 << p) != 0).collect();
    let m = list.len();
    // Forma de los rangos de los vivos como rivales cuando hay jugadores que ya foldearon en
    // la hoja: sus cartas bloquean, combo a combo, a los vivos (foldearon manos flojas en este
    // board, así que lo que queda vivo es más fuerte). Con solo el factor medio de bloqueo,
    // cada vivo veía al otro sin esa corrección y los dos sobrestimaban a la vez su
    // probabilidad de ganar (+0,3 % a 3, +0,8 % a 4: fichas creadas en los botes multiway).
    // Primer orden: el rango de cada vivo se repondera por `P_d(compatible con su combo)` de
    // cada muerto `d`; el nivel de masa (bloqueo con el héroe, `mass_of`) no cambia.
    let dead: Vec<usize> = (0..k).filter(|&p| alive & (1 << p) == 0).collect();
    let mut rw_store: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
    if m >= 2 && !dead.is_empty() {
        let mut f = lbuf();
        f[..M].fill(1.0);
        for &d in &dead {
            let tot: f32 = r[d][..M].iter().sum();
            if tot > 0.0 {
                for i in 0..M {
                    f[i] *= bs[d].compat(r[d], pairs, i) / tot;
                }
            }
        }
        for &j in &list {
            let mut b = lbuf();
            for i in 0..M {
                b[i] = r[j][i] * f[i];
            }
            rw_store[j] = Some(b);
        }
    }
    let rs: Vec<&Lane> = (0..k).map(|j| rw_store[j].as_deref().unwrap_or(r[j])).collect();
    let bsw: Vec<Option<Sums>> = rw_store.iter().map(|b| b.as_ref().map(|b| Sums::of(b, pairs))).collect();
    let base = util as usize;
    let u = |oi: usize, st: usize| utils[base + oi * n + st];
    // Orden (posiciones, de mejor a peor) → outcome.
    let oi_of = |order: &[usize]| -> usize {
        let idx: Vec<u8> = order.iter().map(|p| list.iter().position(|q| q == p).unwrap() as u8).collect();
        perm_index(&idx)
    };
    // Distribuciones de fuerza (masa acumulada a mitad de grupo) de los vivos y, para
    // repartir el orden entre rivales, sus productos acumulados por pares y tríos.
    let lo = |f: &Lane, o: &mut Lane| -> f32 {
        let mut cum = 0.0f32;
        for g in board.groups().windows(2) {
            let (i, j) = (g[0] as usize, g[1] as usize);
            let gm: f32 = f[i..j].iter().sum();
            o[i..j].fill(cum + 0.5 * gm);
            cum += gm;
        }
        cum
    };
    let mut cdf: Vec<Option<(LBuf, f32)>> = (0..k).map(|_| None).collect();
    if m >= 2 {
        for &j in &list {
            let mut o = lbuf();
            let t = lo(rs[j], &mut o);
            cdf[j] = Some((o, t));
        }
    }
    // `pa[a][b]` = Σ_{x<s} r_a(x)·B(x) (a por encima de b, ambos por debajo de s).
    let mut pa: Vec<Vec<Option<(LBuf, f32)>>> = (0..k).map(|_| (0..k).map(|_| None).collect()).collect();
    if m >= 2 {
        for &a in &list {
            for &b in &list {
                if a != b {
                    let cb = &cdf[b].as_ref().unwrap().0;
                    let mut f = lbuf();
                    for i in 0..M {
                        f[i] = rs[a][i] * cb[i];
                    }
                    let mut o = lbuf();
                    let t = lo(&f, &mut o);
                    pa[a][b] = Some((o, t));
                }
            }
        }
    }
    // `tr[(a,b,c)]` = Σ_{x<s} r_a(x)·pa[b][c](x).
    let mut tr: Vec<Option<(LBuf, f32)>> = (0..64).map(|_| None).collect();
    if m >= 3 {
        for &a in &list {
            for &b in &list {
                for &c in &list {
                    if a != b && a != c && b != c {
                        let pbc = &pa[b][c].as_ref().unwrap().0;
                        let mut f = lbuf();
                        for i in 0..M {
                            f[i] = rs[a][i] * pbc[i];
                        }
                        let mut o = lbuf();
                        let t = lo(&f, &mut o);
                        tr[a * 16 + b * 4 + c] = Some((o, t));
                    }
                }
            }
        }
    }
    // Peso de cada orden de `set` (permutaciones en el orden de `nth_perm`) en el carril
    // `i`: por encima del héroe (`Some(true)`), por debajo (`Some(false)`) o sin héroe.
    let perms_of = |set: &[usize]| -> Vec<Vec<usize>> { (0..factorial(set.len())).map(|idx| nth_perm(set.len(), idx).iter().map(|&x| set[x as usize]).collect()).collect() };
    // `perms`: los órdenes del conjunto (jugadores, de mejor a peor), ver `perms_of`.
    let weights = |perms: &[Vec<usize>], side: Option<bool>, i: usize, w: &mut [f32; 6]| -> usize {
        let np = perms.len();
        if perms[0].len() <= 1 {
            w[0] = 1.0;
            return 1;
        }
        let c = |j: usize| cdf[j].as_ref().unwrap().0[i];
        let ct = |j: usize| cdf[j].as_ref().unwrap().1;
        let p2 = |a: usize, b: usize| pa[a][b].as_ref().unwrap();
        for (o, wv) in perms.iter().zip(w.iter_mut()) {
            *wv = if o.len() == 2 {
                let (a, b) = (o[0], o[1]);
                match side {
                    None => p2(a, b).1,
                    Some(false) => p2(a, b).0[i],
                    Some(true) => (p2(a, b).1 - p2(a, b).0[i]) - c(b) * (ct(a) - c(a)),
                }
            } else {
                let (a, b, cc) = (o[0], o[1], o[2]);
                let t3 = tr[a * 16 + b * 4 + cc].as_ref().unwrap();
                match side {
                    None => t3.1,
                    Some(false) => t3.0[i],
                    Some(true) => {
                        let ahi = ct(a) - c(a);
                        ((t3.1 - t3.0[i]) - p2(b, cc).0[i] * ahi) - c(cc) * ((p2(a, b).1 - p2(a, b).0[i]) - c(b) * ahi)
                    }
                }
            }
            .max(0.0);
        }
        let s: f32 = w[..np].iter().sum();
        for wv in w[..np].iter_mut() {
            *wv = if s > 0.0 { *wv / s } else { 1.0 / np as f32 };
        }
        np
    };
    // Showdown (con bloqueo del héroe) contra cada vivo que haga de rival de un héroe.
    let mut wtl: Vec<Option<(LBuf, LBuf)>> = (0..k).map(|_| None).collect();
    if m >= 2 {
        for &j in &list {
            if (0..k).any(|h| h != j && want & (1 << h) != 0 && alive & (1 << h) != 0) {
                let (mut w, mut t, mut l) = (lbuf(), lbuf(), lbuf());
                lanes::showdown(board, rs[j], bsw[j].as_ref().unwrap_or(&bs[j]), &mut w, &mut t, &mut l);
                if let Some(sw) = &bsw[j] {
                    // Misma masa total por carril que sin repondear: solo cambia la forma.
                    for i in 0..M {
                        let (c0, c1) = (bs[j].compat(r[j], pairs, i), sw.compat(rs[j], pairs, i));
                        let x = if c1 > 0.0 { c0 / c1 } else { 0.0 };
                        w[i] *= x;
                        t[i] *= x;
                        l[i] *= x;
                    }
                }
                for i in 0..M {
                    w[i] += 0.5 * t[i];
                    l[i] += 0.5 * t[i];
                }
                wtl[j] = Some((w, l));
            }
        }
    }
    let nout = factorial(m);
    for hero in (0..k).filter(|&p| want & (1 << p) != 0) {
        let st = seats[hero];
        // Bote a 3: masa exacta de pares de los dos rivales del héroe frente al producto de
        // compatibles, con sus rangos **en este final** (no los de la raíz de la hoja: las
        // apuestas cambian los rangos y con ellos el bloqueo entre rivales; con el factor de la
        // raíz cada héroe repartía su masa entre las líneas a su manera y los tres juntos se
        // llevaban ~1-2 % más fichas de las que hay en el bote). La masa de pares es bilineal
        // en los rangos, así que la suma sobre los finales sigue siendo la de la raíz.
        let q3_local: Option<LBuf> = ab[hero].as_ref().map(|_| {
            let (a, b) = rivals3(hero);
            rival_pair_factor(board, r[a], r[b], &bs[a], &bs[b])
        });
        let q3: Option<&Lane> = q3_local.as_deref();
        let add_obs = hero == 0 && !obs.is_empty();
        let mut dsum = vec![0.0f64; nout];
        let mut o = lbuf();
        // Masa compatible de los rivales que no compiten por el bote (o de todos si el
        // héroe no llega al showdown).
        let hero_alive = alive & (1 << hero) != 0;
        let mass_of = |i: usize, skip_alive: bool| -> f32 {
            let mut x = q3.map_or(1.0, |q| q[i]);
            for j in (0..k).filter(|&j| j != hero && !(skip_alive && alive & (1 << j) != 0)) {
                x *= bs[j].compat(r[j], pairs, i);
            }
            x
        };
        if m == 1 || !hero_alive {
            // Orden de los vivos independiente de la mano del héroe.
            let mut w = [0.0f32; 6];
            let lp = perms_of(&list);
            let np = weights(&lp, None, 0, &mut w);
            let mut ev = 0.0f32;
            let mut oi_w = vec![0.0f32; nout];
            for (order, &wv) in lp.iter().zip(&w[..np]) {
                let oi = oi_of(order);
                ev += wv * u(oi, st);
                oi_w[oi] += wv;
            }
            for i in 0..M {
                let x = mass_of(i, false);
                o[i] = x * ev;
                if add_obs {
                    let z = (r[0][i] * x) as f64;
                    for (d, &wv) in dsum.iter_mut().zip(&oi_w) {
                        *d += z * wv as f64;
                    }
                }
            }
        } else {
            let opps: Vec<usize> = list.iter().copied().filter(|&j| j != hero).collect();
            let no = opps.len();
            // Bloqueo entre rivales, por pares: para cada par de rivales vivos, las masas
            // exactas (bloqueo entre los tres, `showdown3_dir`; empates con el héroe a medias)
            // de "los dos por debajo del héroe", "los dos por encima" y "uno a cada lado".
            // Cada reparto arriba/abajo del modelo independiente se corrige por el cociente
            // exacto/independiente de cada par según de qué lado queda cada uno (con dos
            // rivales es exacto; con tres, la corrección de segundo orden). Tratarlos como
            // independientes daba al héroe ~0,5 puntos de más de ganar a dos rivales, siempre
            // en el mismo sentido (+0,1-0,15 $ a quien paga un bote a 3 en PKO).
            let pair_ex: Vec<(usize, usize, LBuf, LBuf, LBuf)> = (0..no)
                .flat_map(|x| (x + 1..no).map(move |y| (x, y)))
                .map(|(x, y)| {
                    let (ra, rb) = (rs[opps[x]], rs[opps[y]]);
                    let (mut lt, mut le, mut nn) = (lbuf(), lbuf(), lbuf());
                    lanes::showdown3_dir(board, ra, rb, false, &mut lt, &mut le, &mut nn);
                    let (mut ht, mut he, mut n2) = (lbuf(), lbuf(), lbuf());
                    lanes::showdown3_dir(board, ra, rb, true, &mut ht, &mut he, &mut n2);
                    for i in 0..M {
                        lt[i] = 0.5 * (lt[i] + le[i]);
                        ht[i] = 0.5 * (ht[i] + he[i]);
                    }
                    (x, y, lt, ht, nn)
                })
                .collect();
            // Para cada subconjunto de rivales por encima: (máscara, órdenes de arriba y
            // de abajo → outcome).
            struct Split {
                above: Vec<usize>,
                below: Vec<usize>,
                pa: Vec<Vec<usize>>,
                pb: Vec<Vec<usize>>,
                oi: Vec<Vec<usize>>,
            }
            let splits: Vec<Split> = (0..1usize << no)
                .map(|sm| {
                    let above: Vec<usize> = (0..no).filter(|&x| sm & (1 << x) != 0).map(|x| opps[x]).collect();
                    let below: Vec<usize> = (0..no).filter(|&x| sm & (1 << x) == 0).map(|x| opps[x]).collect();
                    let (pa, pb) = (perms_of(&above), perms_of(&below));
                    let oi = pa
                        .iter()
                        .map(|x| {
                            pb.iter()
                                .map(|y| {
                                    let mut order = x.clone();
                                    order.push(hero);
                                    order.extend(y);
                                    oi_of(&order)
                                })
                                .collect()
                        })
                        .collect();
                    Split { above, below, pa, pb, oi }
                })
                .collect();
            let uo: Vec<f32> = (0..nout).map(|oi| u(oi, st)).collect();
            let (mut wa, mut wb) = ([0.0f32; 6], [0.0f32; 6]);
            for i in 0..M {
                let dead = mass_of(i, true);
                let mut acc = 0.0f32;
                // Masa de cada reparto (índice = máscara de rivales por encima): modelo
                // independiente × correcciones por pares, con el total del modelo (el que usa
                // el denominador).
                let lohi = |x: usize| -> (f32, f32) {
                    let w = wtl[opps[x]].as_ref().unwrap();
                    (w.0[i], w.1[i])
                };
                let mut parts = [0.0f32; 8];
                if !pair_ex.is_empty() {
                    let mut tot = 0.0f32;
                    for (sm, p) in parts.iter_mut().enumerate().take(1 << no) {
                        let mut v: f32 = (0..no).map(|x| if sm & (1 << x) != 0 { lohi(x).1 } else { lohi(x).0 }).product();
                        for (x, y, bb, aa, nn) in &pair_ex {
                            let (lx, hx) = lohi(*x);
                            let (ly, hy) = (lohi(*y).0, lohi(*y).1);
                            let (ux, uy) = (sm & (1 << x) != 0, sm & (1 << y) != 0);
                            let (ex, ind) = match (ux, uy) {
                                (false, false) => (bb[i].max(0.0), lx * ly),
                                (true, true) => (aa[i].max(0.0), hx * hy),
                                _ => ((nn[i] - bb[i].max(0.0) - aa[i].max(0.0)).max(0.0), hx * ly + hy * lx),
                            };
                            v *= if ind > 0.0 { ex / ind } else { 0.0 };
                        }
                        *p = v;
                        tot += v;
                    }
                    let model: f32 = (0..no).map(|x| lohi(x).0 + lohi(x).1).product();
                    let f = if tot > 0.0 { model / tot } else { 0.0 };
                    for p in parts.iter_mut() {
                        *p *= f;
                    }
                }
                if dead > 0.0 {
                    for (sm, sp) in splits.iter().enumerate() {
                        let mut base = dead;
                        if !pair_ex.is_empty() {
                            base *= parts[sm];
                        } else {
                            for &j in &sp.above {
                                base *= wtl[j].as_ref().unwrap().1[i];
                            }
                            for &j in &sp.below {
                                base *= wtl[j].as_ref().unwrap().0[i];
                            }
                        }
                        if base <= 0.0 {
                            continue;
                        }
                        let na = weights(&sp.pa, Some(true), i, &mut wa);
                        let nb = weights(&sp.pb, Some(false), i, &mut wb);
                        for ia in 0..na {
                            for ib in 0..nb {
                                let x = base * wa[ia] * wb[ib];
                                let oi = sp.oi[ia][ib];
                                acc += x * uo[oi];
                                if add_obs {
                                    dsum[oi] += (r[0][i] * x) as f64;
                                }
                            }
                        }
                    }
                }
                o[i] = acc;
            }
        }
        if add_obs {
            for (a, &s) in obs_acc.iter_mut().zip(obs) {
                *a += dsum.iter().enumerate().map(|(oi, &d)| d * u(oi, s) as f64).sum::<f64>();
            }
        }
        out[hero] = Some(o);
    }
}

impl PostflopEstimator for PostflopEngine {
    fn set_iteration(&self, t: u32) {
        self.iteration.store(t, std::sync::atomic::Ordering::Relaxed);
    }

    fn values_all(&self, tree: &PreflopTree, leaf: usize, reaches: &[&RangeVec], mode: StrategyMode, update_mask: u8, seed: u64, observers: &[usize], folded: &[&RangeVec], pool: super::pool::Pool<'_>) -> Vec<Vec<Box<RangeVec>>> {
        self.joint_values(tree, leaf, reaches, mode, update_mask, (1u8 << reaches.len()) - 1, seed, observers, folded, pool)
    }

    #[allow(clippy::too_many_arguments)]
    fn values3(
        &self,
        tree: &PreflopTree,
        leaf: usize,
        pos: usize,
        reaches: [&RangeVec; 3],
        mode: StrategyMode,
        update: bool,
        seed: u64,
        seats: &[usize],
        pool: super::pool::Pool<'_>,
    ) -> Vec<Box<RangeVec>> {
        assert!(pos == 0 || seats.len() == 1, "observers only from position 0");
        let upd = if update { 1u8 << pos } else { 0 };
        let mut v = self.joint_values(tree, leaf, &reaches, mode, upd, 1 << pos, seed, &seats[1..], &[], pool);
        v.swap_remove(pos)
    }

    fn save_state(&self) -> Vec<u8> {
        let items = self.table.export();
        let mut v = Vec::with_capacity(8 + items.len() * 16);
        v.extend((items.len() as u64).to_le_bytes());
        for (k, d) in items {
            v.extend(k.to_le_bytes());
            v.extend(d.to_le_bytes());
        }
        v
    }

    fn load_state(&self, b: &[u8]) -> Result<(), String> {
        let rd = |o: usize| -> Result<u64, String> { Ok(u64::from_le_bytes(b.get(o..o + 8).ok_or("estado postflop truncado")?.try_into().unwrap())) };
        let n = rd(0)? as usize;
        let items: Result<Vec<(u64, u64)>, String> = (0..n).map(|i| Ok((rd(8 + 16 * i)?, rd(16 + 16 * i)?))).collect();
        match self.table.import(&items?) {
            0 => Ok(()),
            lost => Err(format!("{lost} postflop infosets did not fit in the table")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn values(
        &self,
        tree: &PreflopTree,
        leaf: usize,
        hero_is_ip: bool,
        r_hero: &RangeVec,
        r_opp: &RangeVec,
        mode: StrategyMode,
        update: bool,
        seed: u64,
        seats: &[usize],
        pool: super::pool::Pool<'_>,
    ) -> Vec<Box<RangeVec>> {
        assert!(!hero_is_ip || seats.len() == 1, "observadores solo desde OOP");
        let (reaches, pos) = if hero_is_ip { ([r_opp, r_hero], 1) } else { ([r_hero, r_opp], 0) };
        let upd = if update { 1u8 << pos } else { 0 };
        let mut v = self.joint_values(tree, leaf, &reaches, mode, upd, 1 << pos, seed, &seats[1..], &[], pool);
        v.swap_remove(pos)
    }

    #[allow(clippy::too_many_arguments)]
    fn hero_values(
        &self,
        tree: &PreflopTree,
        leaf: usize,
        hero_is_ip: bool,
        r_hero: &RangeVec,
        r_opp: &RangeVec,
        mode: StrategyMode,
        update: bool,
        seed: u64,
        out: &mut RangeVec,
        pool: super::pool::Pool<'_>,
    ) {
        let l = &tree.leaves[leaf];
        let seat = if hero_is_ip { l.ip } else { l.oop } as usize;
        let v = self.values(tree, leaf, hero_is_ip, r_hero, r_opp, mode, update, seed, &[seat], pool);
        out.copy_from_slice(&v[0][..]);
    }

    fn observer_values(&self, tree: &PreflopTree, leaf: usize, r_oop: &RangeVec, r_ip: &RangeVec, mode: StrategyMode, seed: u64, out: &mut [f64], _pool: super::pool::Pool<'_>) {
        let l = &tree.leaves[leaf];
        let mut rng = SplitMix64::new(seed);
        out.fill(0.0);
        let boards = self.boards(mode, r_oop, &mut rng);
        let mut valid_mass = 0.0f64;
        for board in boards {
            let (a, b) = (Self::masked(board, r_oop, 1.0), Self::masked(board, r_ip, 1.0));
            let bs = BlockerSums::new(&b[..]);
            valid_mass += board.order.iter().map(|&x| (a[x as usize] * bs.compatible(&b[..], x as usize)) as f64).sum::<f64>();
            self.observe(tree, l, board, 0, &a, &b, mode, out);
        }
        let pairs = NUM_COMBOS as f64 * COMPATIBLE_COMBOS as f64;
        let bs = BlockerSums::new(&r_ip[..]);
        let true_mass: f64 = (0..NUM_COMBOS).map(|x| (r_oop[x] * bs.compatible(&r_ip[..], x)) as f64).sum();
        let norm = if valid_mass > 0.0 { true_mass / (valid_mass * pairs) } else { 0.0 };
        out.iter_mut().for_each(|x| *x *= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::board_pool::PoolConfig;

    /// Showdown multiway con botes laterales: el valor de cada héroe (utilidad por orden de
    /// fuerza) frente a la enumeración exacta de manos rivales (con todo el bloqueo y los
    /// empates desempatados al azar). El modelo ignora el bloqueo entre rivales y reparte
    /// el orden entre ellos con sus CDF: se pide un error medio por debajo del 1 % del rango
    /// de utilidades (medido: 0,3-0,6 %).
    #[test]
    #[ignore]
    fn end_multi_share_conservation() {
        // Con utilidades "el mejor se lleva 1" (empates repartidos por los órdenes), la suma
        // sobre los héroes de su probabilidad de ganar debería ser 1.
        // `cargo test --release end_multi_share -- --ignored --nocapture`
        let pool = BoardPool::build(&PoolConfig { boards: 6, buckets: [4, 4, 4], kmeans_samples: 500, kmeans_iters: 2, seed: 9, threads: 2, stratified: true, river_ochs: false });
        let mut rng = SplitMix64::new(5);
        for (k, density, alive) in [(3usize, 0.12f32, 7u8), (3, 0.4, 7), (3, 0.9, 7), (4, 0.4, 15), (3, 0.4, 0b101), (3, 0.4, 0b011), (4, 0.4, 0b1011)] {
            let mut tot_sum = 0.0f64;
            for b in &pool.boards {
                // Con STRENGTH, los que no están vivos tienen manos flojas en el board (foldean)
                // y los vivos, fuertes: rangos correlacionados con la fuerza.
                let strength = std::env::var("STRENGTH").is_ok();
                let rl: Vec<LBuf> = (0..k)
                    .map(|p| {
                        let mut x = lbuf();
                        for i in 0..M {
                            let pos = i as f32 / M as f32;
                            let bias = if !strength { 1.0 } else if alive & (1 << p) == 0 { (1.0 - pos).powi(3) } else { pos };
                            x[i] = if rng.next_f32() < density { (0.2 + rng.next_f32()) * bias } else { 0.0 };
                        }
                        x
                    })
                    .collect();
                let r: Vec<&Lane> = rl.iter().map(|x| &**x).collect();
                let bs: Vec<Sums> = r.iter().map(|x| Sums::of(x, &b.pairs)).collect();
                let n = k;
                let list: Vec<usize> = (0..k).filter(|&p| alive & (1 << p) != 0).collect();
                let m = list.len();
                let mut utils = vec![0.0f32; factorial(m) * n];
                // Reparto por puesto (como con botes laterales: importa el orden entero).
                let place = [0.55f32, 0.3, 0.1, 0.05];
                let place: Vec<f32> = if std::env::var("WINNER").is_ok() { vec![1.0, 0.0, 0.0, 0.0] } else { place[..m].iter().map(|x| x / place[..m].iter().sum::<f32>()).collect() };
                for idx in 0..factorial(m) {
                    let perm = nth_perm(m, idx);
                    for (pl, &x) in perm.iter().enumerate() {
                        utils[idx * n + list[x as usize]] = place[pl];
                    }
                }
                let seats: Vec<usize> = (0..k).collect();
                // Como `joint_values` con k = 3: masa exacta de pares de rivales / producto.
                let ab: Vec<Option<LBuf>> = (0..k)
                    .map(|p| {
                        (k == 3).then(|| {
                            let (a, c) = rivals3(p);
                            rival_pair_factor(b, r[a], r[c], &bs[a], &bs[c])
                        })
                    })
                    .collect();
                let mut out: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
                end_multi_core(&utils, n, b, alive, 0, &r, &ab, &bs, &seats, &[], &mut [], ((1u16 << k) - 1) as u8, &mut out);
                let mut s = 0.0f64;
                for hero in 0..k {
                    let o = out[hero].as_ref().unwrap();
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for i in 0..M {
                        let m: f64 = (0..k).filter(|&j| j != hero).map(|j| bs[j].compat(r[j], &b.pairs, i) as f64).product::<f64>() * ab[hero].as_ref().map_or(1.0, |q| q[i] as f64);
                        num += r[hero][i] as f64 * o[i] as f64;
                        den += r[hero][i] as f64 * m;
                    }
                    s += num / den;
                }
                tot_sum += s / pool.boards.len() as f64;
            }
            eprintln!("k={k} vivos {alive:b} densidad {density}: Σ P(ganar) = {tot_sum:.5}");
        }
    }

    #[test]
    fn end_multi_matches_brute_force() {
        let pool = BoardPool::build(&PoolConfig { boards: 2, buckets: [4, 4, 4], kmeans_samples: 500, kmeans_iters: 2, seed: 7, threads: 2, stratified: true, river_ochs: false });
        let mut rng = SplitMix64::new(21);
        for b in &pool.boards {
            let grp: Vec<usize> = {
                let mut v = vec![0; M];
                for (g, w) in b.groups.windows(2).enumerate() {
                    v[w[0] as usize..w[1] as usize].fill(g);
                }
                v
            };
            let dis = |x: usize, y: usize| {
                let ([a, c], [d, e]) = (b.pairs[x], b.pairs[y]);
                a != d && a != e && c != d && c != e
            };
            for (k, alive, density) in [(3usize, 0b111u8, 0.12f32), (3, 0b101, 0.12), (4, 0b1111, 0.06), (4, 0b1011, 0.06)] {
                let rl: Vec<LBuf> = (0..k)
                    .map(|_| {
                        let mut x = lbuf();
                        for i in 0..M {
                            x[i] = if rng.next_f32() < density { 0.2 + rng.next_f32() } else { 0.0 };
                        }
                        x
                    })
                    .collect();
                let r: Vec<&Lane> = rl.iter().map(|x| &**x).collect();
                let bs: Vec<Sums> = r.iter().map(|x| Sums::of(x, &b.pairs)).collect();
                let m = alive.count_ones() as usize;
                let n = k;
                let utils: Vec<f32> = (0..factorial(m) * n).map(|_| 10.0 * rng.next_f32()).collect();
                let seats: Vec<usize> = (0..k).collect();
                let ab: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
                let mut out: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
                let want = ((1u16 << k) - 1) as u8;
                end_multi_core(&utils, n, b, alive, 0, &r, &ab, &bs, &seats, &[], &mut [], want, &mut out);
                let list: Vec<usize> = (0..k).filter(|&p| alive & (1 << p) != 0).collect();
                for hero in 0..k {
                    let o = out[hero].as_ref().unwrap();
                    let opps: Vec<usize> = (0..k).filter(|&j| j != hero).collect();
                    let (mut err, mut cnt) = (0.0f64, 0usize);
                    for i in (0..M).step_by(29).filter(|&i| r[hero][i] >= 0.0) {
                        // Enumeración de las manos de los rivales.
                        let cand: Vec<Vec<usize>> = opps.iter().map(|&j| (0..M).filter(|&x| r[j][x] > 0.0 && dis(i, x)).collect()).collect();
                        let (mut val, mut mass) = (0.0f64, 0.0f64);
                        // Bucle explícito sobre 2 o 3 rivales.
                        let mut visit = |hs: &[usize]| {
                            for a in 0..hs.len() {
                                for c in a + 1..hs.len() {
                                    if !dis(hs[a], hs[c]) {
                                        return;
                                    }
                                }
                            }
                            let w: f64 = hs.iter().zip(&opps).map(|(&x, &j)| r[j][x] as f64).product();
                            let lane_of = |p: usize| if p == hero { i } else { hs[opps.iter().position(|&q| q == p).unwrap()] };
                            // Órdenes compatibles con la fuerza (empates en todos los órdenes).
                            let mut ev = 0.0f64;
                            let mut cnt = 0usize;
                            for idx in 0..factorial(m) {
                                let perm = nth_perm(m, idx);
                                let order: Vec<usize> = perm.iter().map(|&x| list[x as usize]).collect();
                                if order.windows(2).all(|p| grp[lane_of(p[0])] >= grp[lane_of(p[1])]) {
                                    ev += utils[idx * n + hero] as f64;
                                    cnt += 1;
                                }
                            }
                            val += w * ev / cnt as f64;
                            mass += w;
                        };
                        match opps.len() {
                            2 => {
                                for &x in &cand[0] {
                                    for &y in &cand[1] {
                                        visit(&[x, y]);
                                    }
                                }
                            }
                            _ => {
                                for &x in &cand[0] {
                                    for &y in &cand[1] {
                                        if !dis(x, y) {
                                            continue;
                                        }
                                        for &z in &cand[2] {
                                            visit(&[x, y, z]);
                                        }
                                    }
                                }
                            }
                        }
                        if mass <= 0.0 {
                            continue;
                        }
                        let ours_mass: f64 = opps.iter().map(|&j| bs[j].compat(r[j], &b.pairs, i) as f64).product();
                        let ours = o[i] as f64 / ours_mass;
                        err += (ours - val / mass).abs();
                        cnt += 1;
                    }
                    let mean = err / cnt.max(1) as f64;
                    assert!(cnt > 5 && mean < 0.1, "k={k} vivos={alive:b} héroe={hero}: error medio {mean:.3} ({cnt} carriles)");
                }
            }
        }
    }

    /// Diagnóstico (ignorado): sesgo con signo de P(héroe gana todo) del modelo multiway
    /// frente a la enumeración exacta, con rangos densos tipo call/open.
    #[test]
    #[ignore]
    fn end_multi_win_bias() {
        let pool = BoardPool::build(&PoolConfig { boards: 3, buckets: [4, 4, 4], kmeans_samples: 500, kmeans_iters: 2, seed: 9, threads: 2, stratified: true, river_ochs: false });
        let mut rng = SplitMix64::new(5);
        for (k, dens) in [(3usize, 0.3f32), (4, 0.1)] {
            for b in &pool.boards {
                let grp: Vec<usize> = {
                    let mut v = vec![0; M];
                    for (g, w) in b.groups.windows(2).enumerate() {
                        v[w[0] as usize..w[1] as usize].fill(g);
                    }
                    v
                };
                let dis = |x: usize, y: usize| {
                    let ([a, c], [d, e]) = (b.pairs[x], b.pairs[y]);
                    a != d && a != e && c != d && c != e
                };
                let rl: Vec<LBuf> = (0..k).map(|_| { let mut x = lbuf(); for i in 0..M { x[i] = if rng.next_f32() < dens { 1.0 } else { 0.0 }; } x }).collect();
                let r: Vec<&Lane> = rl.iter().map(|x| &**x).collect();
                let bs: Vec<Sums> = r.iter().map(|x| Sums::of(x, &b.pairs)).collect();
                let n = k;
                let nf = factorial(k);
                let mut utils = vec![0.0f32; nf * n];
                for idx in 0..nf { if nth_perm(k, idx)[0] == 0 { utils[idx * n] = 1.0; } }
                let ab: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
                let mut out: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
                let seats: Vec<usize> = (0..k).collect();
                end_multi_core(&utils, n, b, ((1u16 << k) - 1) as u8, 0, &r, &ab, &bs, &seats, &[], &mut [], 1, &mut out);
                let o = out[0].as_ref().unwrap();
                let (mut sm, mut se, mut c) = (0.0f64, 0.0f64, 0);
                for i in (0..M).step_by(23) {
                    let (mut v, mut mass) = (0.0f64, 0.0f64);
                    let cand: Vec<Vec<usize>> = (1..k).map(|j| (0..M).filter(|&x| r[j][x] > 0.0 && dis(i, x)).collect()).collect();
                    let mut visit = |hs: &[usize]| {
                        for a in 0..hs.len() { for c2 in a + 1..hs.len() { if !dis(hs[a], hs[c2]) { return; } } }
                        mass += 1.0;
                        let top = hs.iter().map(|&x| grp[x]).max().unwrap().max(grp[i]);
                        let nt = hs.iter().filter(|&&x| grp[x] == top).count() + (grp[i] == top) as usize;
                        if grp[i] == top { v += 1.0 / nt as f64; }
                    };
                    if k == 3 {
                        for &x in &cand[0] { for &y in &cand[1] { visit(&[x, y]); } }
                    } else {
                        for &x in &cand[0] { for &y in &cand[1] { if !dis(x, y) { continue; } for &z in &cand[2] { visit(&[x, y, z]); } } }
                    }
                    if mass == 0.0 { continue; }
                    let om: f64 = (1..k).map(|j| bs[j].compat(r[j], &b.pairs, i) as f64).product();
                    sm += o[i] as f64 / om; se += v / mass; c += 1;
                }
                eprintln!("k={k}: P(gana) media modelo {:.4} exacta {:.4} ({} carriles)", sm / c as f64, se / c as f64, c);
            }
        }
    }
}
