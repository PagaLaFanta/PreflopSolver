//! # CFR+ preflop vectorizado sobre rangos
//!
//! * **Full-width, alternating updates**: en la iteración `t`, para cada jugador `i` se
//!   recorre el árbol entero con los vectores de alcance de todos los jugadores (hasta 10). En los nodos
//!   de `i` se actualizan regrets (suelo CFR+ en la arena `u16`) y la suma de estrategia
//!   con peso lineal `t` (linear averaging de CFR+).
//! * **Vectorización**: cada operación actúa sobre filas de 1328 carriles; no hay bucle
//!   por mano en el recorrido del árbol.
//! * **Paralelismo por subárbol**: los hijos de un nodo se reparten entre un
//!   [`super::pool::Pool`] de hilos persistentes, arrancado una vez por `run(iters)` (antes:
//!   un `std::thread::scope` nuevo en cada nodo de acción — ver el módulo `pool` para el
//!   porqué). Como cada nodo se visita una sola vez por recorrido, cada fila de la arena
//!   tiene un único escritor ⇒ ningún lock ni RMW atómico (contrato de
//!   `QuantArena::accumulate`). Los únicos datos compartidos con escritura concurrente son
//!   los infosets postflop (CAS lock-free).
//!
//! ## Valores en terminales con N jugadores
//! Con alcances por jugador `r_k`, para el jugador `i` con mano `h`:
//! `v_i[h] = E[u_i | h] · Π_{k ≠ i} P(k llega | h)`. Los términos de rivales que no
//! intervienen en el showdown se factorizan como masas con bloqueo de `h`
//! (`Σ_{x∩h=∅} r_k[x] / 1225`); el bloqueo *entre* rivales se ignora (aproximación
//! estándar de los solvers multi-jugador por rangos). Si `i` ya no está en el bote, su
//! utilidad sigue dependiendo del resultado (ICM): su valor es la utilidad esperada
//! **condicionada** a que los del bote lleguen, multiplicada por las mismas masas con
//! bloqueo de su mano que el resto de terminales. Así, para cada combo, la probabilidad
//! total de todos los terminales suma exactamente 1 (con chip EV, el valor de foldear es
//! el stack exacto para toda mano; hay un test que lo comprueba).
//!
//! Con 2 jugadores el modelo es exacto (pares compatibles). Con 3 o más, cada jugador
//! descuenta los combos rivales que chocan con **su** mano pero no el choque entre dos
//! rivales, así que cada jugador pondera las manos repartidas de forma ligeramente
//! distinta: la suma de EV de todos los asientos se conserva salvo ~0.4 %.

use super::multiway::{JointFolds, n_mass, weighted_mean, BoardWeight, MultiwayEval};
use super::pool::Pool;
use super::{full_range, zero_range, PostflopEstimator, RangeVec, StrategyMode};
use crate::eval::board_bunch::BoardBunching;
use crate::eval::equity::{BlockerSums, EquityMatrix};
use crate::eval::{Card, Combo, COMPATIBLE_COMBOS, HAND_STRIDE, NUM_COMBOS};
use crate::math::rng::SplitMix64;
use crate::tree::nodes::{NodeKind, NodeRef, TerminalKind, TerminalNode};
use crate::tree::{PreflopTree, QuantArena};
use std::sync::Arc;

type EvCapture = std::sync::Mutex<Vec<Vec<f32>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Cfr,
    BestResponse,
    Evaluate,
}

/// Muestras de Monte Carlo equivalentes a una iteración de CFR de ancho completo,
/// para el CI. Calibrada en un spot push/fold de 5 jugadores ICM para que CI 10 y CI 2
/// marquen la misma distancia a la solución convergida que en la escala de referencia
/// (CI 10 hacia la iteración 86, donde el CI con K = 1 vale
/// 10 084 ⇒ K = 1,02·10⁶ (16-oct-2026). Se puede cambiar con la variable `PF_CI_K`.
pub static CI_SAMPLES_PER_ITERATION_DEFAULT: f64 = 1.0e6;

/// Estado entre dos medidas del CI.
#[derive(Debug, Clone)]
pub struct CiState {
    iteration: u32,
    class_strat: Vec<Vec<f32>>,
    ci: Option<f64>,
    measures: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    pub threads: usize,
    /// Runouts precalculados para los botes a 3. Con 2048 runouts, 8 por visita y 128 al
    /// evaluar, el error de equity por combo (±1 % en entrenamiento, ±3 % al evaluar)
    /// daba la vuelta a decisiones límite: 3 × 10 bb, la BB pagaba un overcall con Q9s,
    /// J9s o 98s que la equity exacta (29-30 % para una cuota del 30 %) no justifica, y
    /// CFR+ convertía el ruido en frecuencias espurias en manos basura. Con 8192 / 32 /
    /// 2048 el overcall pasa de 15,7 % a 14,3 % y la NashConv real de 0,007
    /// a 0,001. 0 = automático según el número de terminales a 3 (ver `Solver::new`).
    pub multiway_pool: usize,
    /// Runouts muestreados por visita a un terminal multiway durante el entrenamiento.
    pub multiway_boards_per_visit: usize,
    /// Boards por visita de los multiway proporcionales a la masa de los rivales.
    pub multiway_adaptive: bool,
    /// Runouts (deterministas) usados al evaluar / calcular mejores respuestas.
    pub multiway_eval_boards: usize,
    /// Updates simultáneos (un recorrido calcula los valores de todos los jugadores) o
    /// alternados (un recorrido por jugador). `None` = simultáneos con 3+ jugadores.
    pub simultaneous: Option<bool>,
    /// Peso de la iteración `t` en la estrategia media: `t^avg_power` (1 = CFR+ lineal; 2
    /// converge el doble de rápido en NashConv por iteración; 3, por defecto, un 13-20 %
    /// más: 6-max check-down 0,10 bb en 139 iteraciones en vez de 160, 3 jugadores con
    /// MCCFR 0,030 en ~95 en vez de ~112, HU push/fold 1e-4 en 25 en vez de 45).
    pub avg_power: f32,
    /// Descuento de regrets al estilo DCFR: el regret instantáneo de la iteración `t` se
    /// acumula con peso `t^regret_power` (equivale a multiplicar los regrets acumulados por
    /// `((t−1)/t)^q` antes de sumar). 0 = CFR+ clásico. `None`: 1.5 heads-up sin postflop
    /// MCCFR (2,5× menos iteraciones hasta NashConv 1e-4 en push/fold) y 0 en el resto:
    /// con 3+ jugadores y updates simultáneos oscila, y con MCCFR el descuento amplifica
    /// el ruido de las hojas (HU estándar 30 bb: 0,127 frente a 0,089 en 100 iteraciones).
    pub regret_power: Option<f32>,
    /// Card bunching preflop por pares: los jugadores que no están en el bote (foldearon
    /// o no llegan) cambian qué manos tienen los demás. Ver `terminal_all`.
    pub bunching: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        SolverConfig { threads, multiway_pool: 0, multiway_boards_per_visit: 0, multiway_adaptive: true, multiway_eval_boards: 0, simultaneous: None, avg_power: 3.0, regret_power: None, bunching: true }
    }
}

#[derive(Debug)]
pub struct Solver<E: PostflopEstimator> {
    pub tree: PreflopTree,
    pub eq: Arc<EquityMatrix>,
    pub est: E,
    regrets: QuantArena,
    strategy: QuantArena,
    pub iteration: u32,
    pub cfg: SolverConfig,
    /// Evaluador de botes a 3 (solo si el árbol tiene terminales `Multiway`).
    pub multiway: Option<MultiwayEval>,
    /// Nodos con estrategia fijada (node locking): `[acción][combo]`, no se actualizan.
    pub locks: std::collections::HashMap<usize, Vec<f32>>,
    /// "Edge" por nodo y acción: constante que se resta al valor contrafactual de
    /// la acción **solo al calcular regrets** (encarece o subsidia la acción sin alterar los
    /// EV que se propagan hacia arriba).
    pub edges: std::collections::HashMap<usize, Vec<f32>>,
    /// Bloqueo de frecuencias: `(acción, frecuencia objetivo)`; el edge de esa acción se
    /// calibra iterativamente hasta que la frecuencia media coincide.
    pub freq_targets: std::collections::HashMap<usize, (usize, f32)>,
    /// Muestreo de subárbol: con `Some(k)` cada
    /// iteración solo recorre el subárbol del nodo de acción `k`, con los alcances de
    /// entrada calculados con la estrategia media (fija) de la ruta hasta él. Todo el
    /// cálculo se concentra en ese spot; el resto del árbol no cambia.
    pub subtree: Option<usize>,
    /// Board bunching (ver [`BoardBunching`]): corrige la equity de los all-in por las
    /// cartas que se llevaron los que foldearon. Solo con el card bunching activo.
    pub board_bunch: Option<Arc<BoardBunching>>,
}

/// `P(a y b llegan)` sin condicionar a ninguna mano: `Σ_x r_a[x]·Σ_{y∩x=∅} r_b[y] / (1326·1225)`.
fn pair_mass(ra: &RangeVec, rb: &RangeVec) -> f64 {
    let bs = BlockerSums::new(&rb[..]);
    (0..NUM_COMBOS).map(|x| (ra[x] * bs.compatible(&rb[..], x)) as f64).sum::<f64>() / (NUM_COMBOS as f64 * COMPATIBLE_COMBOS as f64)
}

/// `P(los tres llegan)` con el mismo modelo que `MultiwayEval` (bloqueo de cada rival con
/// la mano del primero, independencia entre los otros dos).
fn triple_mass(r0: &RangeVec, r1: &RangeVec, r2: &RangeVec) -> f64 {
    let (b1, b2) = (BlockerSums::new(&r1[..]), BlockerSums::new(&r2[..]));
    let c2 = (COMPATIBLE_COMBOS * COMPATIBLE_COMBOS) as f64;
    (0..NUM_COMBOS).map(|x| r0[x] as f64 * (b1.compatible(&r1[..], x) as f64) * (b2.compatible(&r2[..], x) as f64) / c2).sum::<f64>() / NUM_COMBOS as f64
}

#[inline]
fn mix(a: u64, b: u64) -> u64 {
    let mut r = SplitMix64::new(a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    r.next_u64()
}

impl<E: PostflopEstimator> Solver<E> {
    pub fn new(tree: PreflopTree, eq: Arc<EquityMatrix>, est: E, mut cfg: SolverConfig) -> Self {
        // Botes a 3 en automático (0): máxima precisión con pocos terminales a 3 (push/fold
        // de 3-5 jugadores) y más baratos con muchos (6-max estándar: con 32/8192/2048 la
        // NashConv tardaba 43 s en vez de 2, y cada iteración el doble).
        let few = tree.terminals.iter().filter(|t| t.kind == TerminalKind::Multiway).count() <= 64;
        let auto = |v: &mut usize, a: usize, b: usize| {
            if *v == 0 {
                *v = if few { a } else { b };
            }
        };
        auto(&mut cfg.multiway_pool, 8192, 2048);
        // Boards por visita al entrenar (32 en árboles con pocos all-in a 3, como push/fold de
        // 3-5 jugadores, que ya convergían; 256 en el resto), proporcionales a la masa de los rivales
        // (`MultiwayEval::adaptive`). Con 8 fijos el ruido de los all-in multiway en los
        // regrets frenaba CFR+ a ~1/√T (PKO 6-max sin postflop: NashConv 0,149 a 870 s; con
        // 256 adaptativos, 0,037 a 700 s; con postflop, 4,2 % de desacuerdo con la referencia en el
        // tiempo en que antes se llegaba a 5,2 %).
        auto(&mut cfg.multiway_boards_per_visit, 32, 256);
        // La evaluación (EV, NashConv) siempre con 2048 runouts: con 128 fijos, cada combo
        // arrastraba un error propio de varios $ en los all-ins multiway que no se compensa
        // (siempre la misma muestra) y subía por el árbol hasta los cold calls (PKO 6-max
        // contra la referencia: diferencia de EV en lo que se juega 0,88 → 0,22 $ con esto y el
        // cálculo analítico de los botes a 4). Cuesta más NashConv, no más entrenamiento.
        auto(&mut cfg.multiway_eval_boards, 2048, 2048);
        let regrets = QuantArena::new(tree.total_rows, tree.num_blocks());
        let strategy = QuantArena::new(tree.total_rows, tree.num_blocks());
        // El pool de boards también hace falta para la corrección de bunching conjunto de los
        // all-in HU (`MultiwayEval::hu_bunch_correction`) con 3+ jugadores.
        let multiway = tree
            .terminals
            .iter()
            .any(|t| matches!(t.kind, TerminalKind::Multiway | TerminalKind::MultiwayN) || (cfg.bunching && tree.n >= 3 && t.kind == TerminalKind::AllInRunout))
            .then(|| MultiwayEval::new(cfg.multiway_pool, cfg.multiway_boards_per_visit, cfg.multiway_eval_boards, 0x3A7, cfg.threads))
            .map(|mut m| {
                m.eq = Some(eq.clone());
                m.adaptive = cfg.multiway_adaptive;
                m
            });
        Solver { tree, eq, est, regrets, strategy, iteration: 0, cfg, multiway, locks: Default::default(), edges: Default::default(), freq_targets: Default::default(), subtree: None, board_bunch: None }
    }

    fn iterate_with(&mut self, pool: Pool<'_>) {
        self.iteration += 1;
        self.est.set_iteration(self.iteration);
        let (start, reach) = self.start_point();
        if self.cfg.simultaneous.unwrap_or(self.tree.n >= 3) {
            self.walk_all(start, &reach, 0, mix(self.iteration as u64, 0xA11), pool);
        } else {
            for i in 0..self.tree.n {
                self.walk(start, i, &reach, Mode::Cfr, 0, mix(self.iteration as u64, i as u64), pool);
            }
        }
        self.calibrate_edges();
    }

    /// Nodo desde el que empieza cada iteración y alcances de entrada: la raíz con rangos
    /// completos o, con [`Self::subtree`], el nodo elegido con los alcances de la
    /// estrategia media a lo largo de su ruta.
    fn start_point(&self) -> (NodeRef, Vec<Box<RangeVec>>) {
        let mut reach: Vec<Box<RangeVec>> = (0..self.tree.n).map(|_| full_range()).collect();
        let Some(target) = self.subtree else { return (self.tree.root, reach) };
        let mut cur = self.tree.root;
        let path = &self.tree.paths[target];
        for seg in path.split(' ').filter(|s| !s.is_empty()) {
            let idx = cur.index();
            let label = seg.split_once(':').map_or(seg, |(_, l)| l);
            let a = self.tree.labels[idx].iter().position(|l| l == label).expect("ruta del subárbol");
            let sigma = self.average_strategy(idx);
            let p = self.tree.actions[idx].player as usize;
            for (x, &y) in reach[p].iter_mut().zip(&sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE]) {
                *x *= y;
            }
            cur = self.tree.actions[idx].children[a];
        }
        (cur, reach)
    }

    /// Activa el muestreo de subárbol en el nodo de la ruta `path` (`None` = árbol entero).
    pub fn set_subtree(&mut self, path: Option<&str>) -> Result<(), String> {
        self.subtree = match path {
            None => None,
            Some(p) => Some(self.tree.find(p).ok_or_else(|| format!("nodo no encontrado: {p:?}"))?),
        };
        Ok(())
    }

    /// Fija una frecuencia objetivo para una acción: el
    /// solver calibra el edge de esa acción hasta que su frecuencia media (sobre los 1326
    /// combos) es la pedida; CFR elige qué combos la toman.
    pub fn lock_frequency(&mut self, idx: usize, action: usize, target: f32) {
        let na = self.tree.actions[idx].num_actions as usize;
        assert!(action < na);
        self.edges.entry(idx).or_insert_with(|| vec![0.0; na]);
        self.freq_targets.insert(idx, (action, target.clamp(0.0, 1.0)));
    }

    /// Edge manual (positivo encarece la acción, negativo la subsidia), en unidades de
    /// utilidad.
    pub fn set_edge(&mut self, idx: usize, action: usize, edge: f32) {
        let na = self.tree.actions[idx].num_actions as usize;
        self.edges.entry(idx).or_insert_with(|| vec![0.0; na])[action] = edge;
    }

    /// Escala de utilidad del árbol (para el paso del controlador de edges).
    fn utility_span(&self) -> f32 {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &u in &self.tree.utils {
            lo = lo.min(u);
            hi = hi.max(u);
        }
        (hi - lo).max(1e-6)
    }

    /// Controlador proporcional: `edge += k · (frecuencia media actual − objetivo)`. Con
    /// paso decreciente para que la estrategia media converja.
    fn calibrate_edges(&mut self) {
        if self.freq_targets.is_empty() {
            return;
        }
        let span = self.utility_span();
        let targets: Vec<(usize, usize, f32)> = self.freq_targets.iter().map(|(&k, &(a, t))| (k, a, t)).collect();
        for (idx, a, target) in targets {
            let cur = super::report::overall_frequency(&self.current_strategy(idx), a);
            let step = 0.05 * span / (1.0 + self.iteration as f32 / 200.0);
            let e = &mut self.edges.get_mut(&idx).unwrap()[a];
            *e += step * (cur - target);
        }
    }

    /// Reinicio de estrategias: descarta las sumas de estrategia (la media vuelve a
    /// empezar desde la estrategia actual, ya cerca del equilibrio) y conserva los regrets.
    /// El contador de iteraciones se reinicia para que el peso lineal de CFR+ vuelva a
    /// crecer desde 1.
    pub fn reset_strategies(&mut self) {
        self.strategy.clear();
        self.iteration = 0;
    }

    /// Indicador de convergencia anterior: variación media de la estrategia media
    /// preflop (puntos porcentuales por combo y acción) respecto al snapshot anterior,
    /// normalizada a 100 iteraciones. Devuelve `None` en la primera llamada.
    pub fn convergence_indicator(&self, snapshot: &mut Option<(u32, Vec<Vec<f32>>)>) -> Option<f64> {
        // Media sin el respaldo de la estrategia actual (que oscila en las manos que no llegan).
        let current: Vec<Vec<f32>> = (0..self.tree.actions.len())
            .map(|k| {
                let c = &self.tree.cfr[k];
                let mut out = vec![0.0; c.num_actions as usize * HAND_STRIDE];
                self.strategy.normalized(c, &mut out);
                out
            })
            .collect();
        let result = snapshot.as_ref().and_then(|(it, prev)| {
            let iters = self.iteration.saturating_sub(*it).max(1) as f64;
            let (mut sum, mut cnt) = (0.0f64, 0usize);
            for (a, b) in current.iter().zip(prev) {
                let na = a.len() / HAND_STRIDE;
                for act in 0..na {
                    for h in 0..NUM_COMBOS {
                        sum += (a[act * HAND_STRIDE + h] - b[act * HAND_STRIDE + h]).abs() as f64;
                        cnt += 1;
                    }
                }
            }
            (cnt > 0).then(|| sum / cnt as f64 * 100.0 * 100.0 / iters)
        });
        *snapshot = Some((self.iteration, current));
        result
    }

    /// Indicador de convergencia con la fórmula del CI:
    ///
    /// ```text
    /// TV_n  = ½ Σ_clases Σ_a |σ̄_n(a) − σ̄_n,prev(a)|          (169 clases, como sus buckets preflop)
    /// raw   = √( Σ_n √Δvis_n·TV_n / Σ_n √Δvis_n · #nodos / Δmuestras · 1e8 )
    /// CI    = α·CI_prev + (1−α)·raw,  α = min(0,05·k, 0,75)  (k = medidas anteriores)
    /// ```
    ///
    /// En un solver de Monte Carlo las visitas y las muestras son repartos. Aquí, con CFR de
    /// ancho completo, `Δmuestras = K·Δiteraciones` y `Δvis_n = Δmuestras·P(n)`, con P(n)
    /// la probabilidad de llegar al nodo con la estrategia media. `K`
    /// ([`CI_SAMPLES_PER_ITERATION_DEFAULT`]) está calibrada para que CI 10 y CI 2 signifiquen la
    /// misma distancia a la solución convergida.
    pub fn convergence_index(&self, st: &mut Option<CiState>) -> Option<f64> {
        let n_nodes = self.tree.actions.len();
        let mut class_strat: Vec<Vec<f32>> = vec![Vec::new(); n_nodes];
        let mut prob = vec![0.0f64; n_nodes];
        // Recorrido con la estrategia media: alcance de cada jugador y P(nodo) ≈ Π masas.
        fn rec<E: PostflopEstimator>(s: &Solver<E>, node: NodeRef, reach: &[Box<RangeVec>], cs: &mut [Vec<f32>], prob: &mut [f64]) {
            if node.kind() == NodeKind::Terminal {
                return;
            }
            let idx = node.index();
            let an = &s.tree.actions[idx];
            if an.flags & 1 != 0 {
                return; // postflop: el CI solo mira el preflop
            }
            let na = an.num_actions as usize;
            let p = an.player as usize;
            let pn: f64 = reach.iter().map(|r| r.iter().take(NUM_COMBOS).map(|&x| x as f64).sum::<f64>() / NUM_COMBOS as f64).product();
            prob[idx] += pn;
            let sigma = s.average_strategy(idx);
            // Estrategia por clase de mano: media de sus combos ponderada por el alcance de
            // quien decide (uniforme si la clase no llega).
            if cs[idx].is_empty() {
                let mut out = vec![0.0f32; na * 169];
                for (cls, combos) in crate::cfr::report::CLASS_COMBOS.iter().enumerate() {
                    let w: f32 = combos.iter().map(|&h| reach[p][h as usize]).sum();
                    for a in 0..na {
                        let v: f32 = if w > 1e-9 {
                            combos.iter().map(|&h| reach[p][h as usize] * sigma[a * HAND_STRIDE + h as usize]).sum::<f32>() / w
                        } else {
                            combos.iter().map(|&h| sigma[a * HAND_STRIDE + h as usize]).sum::<f32>() / combos.len() as f32
                        };
                        out[a * 169 + cls] = v;
                    }
                }
                cs[idx] = out;
            }
            for a in 0..na {
                let mut r = reach.to_vec();
                for (x, &y) in r[p].iter_mut().zip(&sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE]) {
                    *x *= y;
                }
                rec(s, an.children[a], &r, cs, prob);
            }
        }
        let reach: Vec<Box<RangeVec>> = (0..self.tree.n).map(|_| full_range()).collect();
        rec(self, self.tree.root, &reach, &mut class_strat, &mut prob);
        let it = self.iteration;
        let result = st.as_ref().and_then(|prev| {
            let d_iter = it.saturating_sub(prev.iteration).max(1) as f64;
            let k_samples = std::env::var("PF_CI_K").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(CI_SAMPLES_PER_ITERATION_DEFAULT);
            let d_samples = k_samples * d_iter;
            let (mut num, mut den, mut nodes) = (0.0f64, 0.0f64, 0usize);
            for k in 0..n_nodes {
                if class_strat[k].is_empty() || prev.class_strat[k].len() != class_strat[k].len() {
                    continue;
                }
                nodes += 1;
                let dvis = d_samples * prob[k];
                let tv: f64 = class_strat[k].iter().zip(&prev.class_strat[k]).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / 2.0;
                let w = dvis.sqrt();
                num += w * tv;
                den += w;
            }
            (den > 0.0).then(|| {
                let raw = (num / den * nodes as f64 / d_samples * 1e8).sqrt();
                match prev.ci {
                    None => raw,
                    Some(c) => {
                        let alpha = (0.05 * prev.measures as f64).min(0.75);
                        alpha * c + (1.0 - alpha) * raw
                    }
                }
            })
        });
        let measures = st.as_ref().map_or(0, |p| p.measures + result.is_some() as u32);
        *st = Some(CiState { iteration: it, class_strat, ci: result.or(st.as_ref().and_then(|p| p.ci)), measures });
        result
    }

    pub fn run(&mut self, iterations: u32) {
        let threads = self.cfg.threads;
        crate::with_pool!(threads, |pool| {
            for _ in 0..iterations {
                self.iterate_with(pool);
            }
        });
    }

    /// Como [`Solver::run`], pero comprobando `stop` antes de cada iteración (no solo al
    /// principio y al final del bloque) para poder cortar a mitad de un `--report-every`
    /// grande sin esperar a que termine entero -- el pool de hilos se construye una sola
    /// vez para todo el bloque, igual que `run`, así que cortar antes no cuesta nada extra
    /// en el caso normal (sin parada) frente a llamar a `run` directamente.
    pub fn run_until(&mut self, iterations: u32, mut stop: impl FnMut() -> bool + Send) {
        let threads = self.cfg.threads;
        crate::with_pool!(threads, |pool| {
            for _ in 0..iterations {
                if stop() {
                    break;
                }
                self.iterate_with(pool);
            }
        });
    }

    /// Huella del árbol (estructura, etiquetas y utilidades): una solución guardada solo
    /// se puede cargar sobre un árbol idéntico.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        };
        for (p, l) in self.tree.paths.iter().zip(&self.tree.labels) {
            eat(p.as_bytes());
            eat(l.join(",").as_bytes());
        }
        for u in &self.tree.utils {
            eat(&u.to_bits().to_le_bytes());
        }
        eat(&(self.tree.leaves.iter().map(|l| l.nodes.len()).sum::<usize>() as u64).to_le_bytes());
        h
    }

    /// Guarda la solución (arenas CFR+, iteración, estado postflop y el índice de nodos
    /// preflop, que permite arrancar en caliente otro árbol con [`Self::warm_start_file`]).
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut v = b"PFSOL002".to_vec();
        v.extend(self.fingerprint().to_le_bytes());
        v.extend(self.iteration.to_le_bytes());
        let mut index = String::new();
        for (k, (p, l)) in self.tree.paths.iter().zip(&self.tree.labels).enumerate() {
            let c = &self.tree.cfr[k];
            index.push_str(&format!("{p}\t{}\t{}\t{}\t{}\n", l.join(","), c.row, c.exp_base, c.num_actions));
        }
        for part in [self.regrets.to_bytes(), self.strategy.to_bytes(), self.est.save_state(), index.into_bytes()] {
            v.extend((part.len() as u64).to_le_bytes());
            v.extend(part);
        }
        std::fs::write(path, v)
    }

    /// Partes de un fichero de [`Self::save`]: huella, iteración y secciones (3 en el
    /// formato 001; 4, con el índice de nodos, en el 002).
    fn read_parts(b: &[u8]) -> Result<(u64, u32, Vec<&[u8]>), String> {
        let n = match b.get(..8) {
            Some(b"PFSOL001") => 3,
            Some(b"PFSOL002") => 4,
            _ => return Err("no es un fichero de solución".into()),
        };
        if b.len() < 20 {
            return Err("fichero truncado".into());
        }
        let fp = u64::from_le_bytes(b[8..16].try_into().unwrap());
        let iteration = u32::from_le_bytes(b[16..20].try_into().unwrap());
        let mut off = 20;
        let mut parts = Vec::new();
        for _ in 0..n {
            let len = u64::from_le_bytes(b.get(off..off + 8).ok_or("fichero truncado")?.try_into().unwrap()) as usize;
            off += 8;
            parts.push(b.get(off..off + len).ok_or("fichero truncado")?);
            off += len;
        }
        Ok((fp, iteration, parts))
    }

    /// Carga una solución guardada con [`Self::save`] sobre el mismo árbol.
    pub fn load(&mut self, path: &std::path::Path) -> Result<(), String> {
        let b = std::fs::read(path).map_err(|e| e.to_string())?;
        let (fp, iteration, parts) = Self::read_parts(&b)?;
        if fp != self.fingerprint() {
            return Err("la solución corresponde a otro árbol (stacks, premios o acciones distintos)".into());
        }
        self.regrets.load_bytes(parts[0])?;
        self.strategy.load_bytes(parts[1])?;
        self.est.load_state(parts[2])?;
        self.iteration = iteration;
        Ok(())
    }

    /// Arranque en caliente desde otra solución (p. ej. los mismos jugadores con stacks
    /// vecinos): copia regrets y sumas de estrategia de cada nodo preflop con la misma ruta
    /// y las mismas acciones, y sigue contando desde su iteración. Con todo su peso: con 3+
    /// jugadores la estrategia actual de CFR+ oscila y lo que converge es la media, así que
    /// rebajar el peso de la solución vecina deja que las iteraciones nuevas la estropeen.
    /// 6-max 30 bb desde la solución de 28 bb: NashConv 0,097 sin iterar (en frío, 160
    /// iteraciones). El postflop empieza de cero. Devuelve cuántos nodos se copiaron.
    pub fn warm_start_from<E2: PostflopEstimator>(&mut self, other: &Solver<E2>) -> usize {
        let index = other.tree.paths.iter().zip(&other.tree.labels).zip(&other.tree.cfr).map(|((p, l), c)| ((p.clone(), l.clone()), *c)).collect();
        self.warm_copy(&other.regrets, &other.strategy, &index, other.iteration)
    }

    /// Como [`Self::warm_start_from`] desde un fichero de [`Self::save`] (formato con índice
    /// de nodos) de otro árbol.
    pub fn warm_start_file(&mut self, path: &std::path::Path) -> Result<usize, String> {
        let b = std::fs::read(path).map_err(|e| e.to_string())?;
        let (_, iteration, parts) = Self::read_parts(&b)?;
        let text = std::str::from_utf8(parts.get(3).ok_or("solución sin índice de nodos (guárdala de nuevo con esta versión)")?).map_err(|e| e.to_string())?;
        let mut index = std::collections::HashMap::new();
        for line in text.lines() {
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() != 5 {
                return Err("índice de nodos inválido".into());
            }
            let labels: Vec<String> = if c[1].is_empty() { Vec::new() } else { c[1].split(',').map(String::from).collect() };
            let num = |s: &str| s.parse::<u32>().map_err(|_| "índice de nodos inválido".to_string());
            let node = crate::tree::nodes::CFRNode { row: num(c[2])?, exp_base: num(c[3])?, num_actions: num(c[4])? as u8, ..Default::default() };
            index.insert((c[0].to_string(), labels), node);
        }
        let regrets = QuantArena::from_bytes(parts[0])?;
        let strategy = QuantArena::from_bytes(parts[1])?;
        Ok(self.warm_copy(&regrets, &strategy, &index, iteration))
    }

    fn warm_copy(&mut self, regrets: &QuantArena, strategy: &QuantArena, index: &std::collections::HashMap<(String, Vec<String>), crate::tree::nodes::CFRNode>, iteration: u32) -> usize {
        let mut copied = 0;
        for (k, (p, l)) in self.tree.paths.iter().zip(&self.tree.labels).enumerate() {
            if let Some(src) = index.get(&(p.clone(), l.clone())) {
                if src.num_actions == self.tree.cfr[k].num_actions {
                    self.regrets.copy_node_from(&self.tree.cfr[k], regrets, src);
                    self.strategy.copy_node_from(&self.tree.cfr[k], strategy, src);
                    copied += 1;
                }
            }
        }
        if copied > 0 {
            self.iteration = iteration;
        }
        copied
    }

    /// Fija la estrategia de un nodo.
    /// `freq_of_action`: frecuencia por combo de la acción `action`; el resto de la
    /// probabilidad va a la acción 0 (fold/check). Varias llamadas sobre el mismo nodo se
    /// combinan (una por acción); si las frecuencias fijadas suman más de 1 en un combo,
    /// se reescalan.
    pub fn lock(&mut self, idx: usize, action: usize, freq_of_action: &[f32]) {
        let na = self.tree.actions[idx].num_actions as usize;
        assert!(action < na && action > 0, "la acción bloqueada debe ser distinta de la 0");
        let mut s = self.locks.remove(&idx).unwrap_or_else(|| vec![0.0f32; na * HAND_STRIDE]);
        for h in 0..NUM_COMBOS {
            s[action * HAND_STRIDE + h] = freq_of_action[h].clamp(0.0, 1.0);
            let rest: f32 = (1..na).map(|b| s[b * HAND_STRIDE + h]).sum();
            if rest > 1.0 {
                for b in 1..na {
                    s[b * HAND_STRIDE + h] /= rest;
                }
            }
            s[h] = (1.0 - rest).max(0.0);
        }
        self.locks.insert(idx, s);
    }

    fn current_strategy(&self, idx: usize) -> Vec<f32> {
        if let Some(s) = self.locks.get(&idx) {
            return s.clone();
        }
        let c = &self.tree.cfr[idx];
        let mut out = vec![0.0; c.num_actions as usize * HAND_STRIDE];
        self.regrets.normalized(c, &mut out);
        out
    }

    /// Card bunching activo: 3+ jugadores y updates simultáneos (el recorrido alternado no
    /// lo lleva). Entonces los recorridos devuelven `2n` vectores: valores y, detrás, la
    /// masa de probabilidad corregida de cada jugador (ver `terminal_all`).
    fn bunching_on(&self) -> bool {
        self.cfg.bunching && self.tree.n >= 3 && self.cfg.simultaneous.unwrap_or(self.tree.n >= 3)
    }

    /// Normalización del bunching en el nodo de `p`: el valor de cada acción se reescala
    /// para que su masa sea la exacta sin correlaciones (`P(los demás llegan | h)`, que
    /// es la misma para todas las acciones de `p`). Los factores por pares solo cambian
    /// cómo se reparte esa probabilidad entre los terminales de cada acción, no el total:
    /// así el fold vale exactamente el stack (chip EV) y no hay sesgo entre acciones.
    fn normalize_bunching(&self, reach: &[Box<RangeVec>], p: usize, vals: &mut [&mut [Box<RangeVec>]]) {
        let n = self.tree.n;
        let m0 = self.others_mass(reach, 1 << p);
        for va in vals.iter_mut() {
            for h in 0..NUM_COMBOS {
                let m = va[n + p][h];
                if m > 1e-12 {
                    va[p][h] *= m0[h] / m;
                }
                va[n + p][h] = m0[h];
            }
        }
    }

    /// Coeficientes del board bunching para un bote con los jugadores de `in_pot`: los
    /// demás (los que foldearon) se llevan cartas del mazo según sus rangos.
    fn board_coefs(&self, reach: &[Box<RangeVec>], in_pot: u32) -> Option<(&BoardBunching, Vec<f32>)> {
        let bb = self.board_bunch.as_deref().filter(|_| self.bunching_on())?;
        let outs = (0..self.tree.n).filter(|&k| in_pot & (1 << k) == 0).map(|k| &reach[k][..]);
        let d = BoardBunching::deck_deltas(outs)?;
        Some((bb, bb.coefs(&d)))
    }

    /// Peso exacto de cada board para los botes resueltos por boards: `Π_k P(k no tiene
    /// ninguna carta del board)` sobre los que no están en el bote (como las hojas de flop).
    fn board_weight<'a>(&self, reach: &'a [Box<RangeVec>], in_pot: u32) -> Option<impl Fn(u64) -> f32 + Sync + 'a> {
        self.board_bunch.as_ref().filter(|_| self.bunching_on())?;
        let outs: Vec<(BlockerSums, &RangeVec)> = (0..self.tree.n)
            .filter(|&k| in_pot & (1 << k) == 0)
            .map(|k| (BlockerSums::new(&reach[k][..]), &*reach[k]))
            .filter(|(b, _)| b.total > 0.0)
            .collect();
        (!outs.is_empty()).then_some({
            move |mask: u64| -> f32 {
                let mut cards = [0u8; 5];
                let mut m = mask;
                for c in cards.iter_mut() {
                    *c = m.trailing_zeros() as u8;
                    m &= m - 1;
                }
                outs.iter()
                    .map(|(b, r)| {
                        let mut blocked: f32 = cards.iter().map(|&c| b.card[c as usize]).sum();
                        for i in 0..5 {
                            for j in i + 1..5 {
                                blocked -= r[Combo::from_cards(Card(cards[i]), Card(cards[j])).0 as usize];
                            }
                        }
                        (1.0 - blocked / b.total).max(0.0)
                    })
                    .product()
            }
        })
    }

    fn regret_power(&self) -> f32 {
        self.cfg.regret_power.unwrap_or(if self.tree.n == 2 && self.tree.leaves.is_empty() { 1.5 } else { 0.0 })
    }

    fn weight_regrets(&self, deltas: &mut [f32]) {
        let q = self.regret_power();
        if q != 0.0 {
            let w = (self.iteration as f32).powf(q);
            deltas.iter_mut().for_each(|d| *d *= w);
        }
    }


    /// Estrategia media del nodo de acción `idx`: `[acción][combo]` con stride 1328.
    /// Un combo que nunca llega al nodo (suma de estrategia nula: fuera del rango del
    /// jugador en esa línea) no tiene media; toma la estrategia actual, entrenada con el
    /// alcance de los rivales, en vez de la uniforme. Así el EV de una acción que la mano
    /// no juega (72o que abre) sale de seguir jugando con sentido (foldear al all-in) y no
    /// de pagar la mitad de las veces.
    pub fn average_strategy(&self, idx: usize) -> Vec<f32> {
        if let Some(s) = self.locks.get(&idx) {
            return s.clone();
        }
        let c = &self.tree.cfr[idx];
        let mut out = vec![0.0; c.num_actions as usize * HAND_STRIDE];
        let cur = self.current_strategy(idx);
        self.strategy.normalized_or(c, &mut out, &cur);
        out
    }

    /// Ejecuta `f` con otra tabla de utilidades del mismo árbol (p. ej. la de fichas de
    /// [`crate::tree::builder::chip_utils`] sobre una solución ICM): mide la misma
    /// estrategia en otras unidades. Restaura la tabla original al terminar.
    pub fn with_utils<T>(&mut self, utils: Vec<f32>, f: impl FnOnce(&Self) -> T) -> T {
        assert_eq!(utils.len(), self.tree.utils.len(), "tabla de utilidades de otro árbol");
        let old = std::mem::replace(&mut self.tree.utils, utils);
        let r = f(self);
        self.tree.utils = old;
        r
    }

    /// Valor esperado de cada jugador con el perfil medio (en unidades de utilidad).
    pub fn expected_values(&self) -> Vec<f64> {
        self.evaluate().iter().map(|x| x.0).collect()
    }

    /// Depuración: aportación de cada terminal al EV de cada jugador con la estrategia media,
    /// por el mismo camino que `evaluate` (`terminal_all` + normalización del bunching en
    /// cada decisión). Cada entrada: `(ruta, terminal, Σ_h alcance propio · valor / 1326)`.
    pub fn terminal_breakdown(&self) -> Vec<(String, usize, Vec<f64>)> {
        type Entry = (String, usize, Vec<Box<RangeVec>>, Vec<Box<RangeVec>>);
        let n = self.tree.n;
        fn rec<E: PostflopEstimator>(s: &Solver<E>, node: NodeRef, reach: &[Box<RangeVec>], path: String, pool: Pool<'_>, out: &mut Vec<Entry>) -> Vec<Box<RangeVec>> {
            let n = s.tree.n;
            match node.kind() {
                NodeKind::Terminal => {
                    let idx = node.index();
                    let mut v = s.terminal_all(idx, reach, StrategyMode::Average, mix(0xE7A1, idx as u64), pool);
                    let masses: Vec<Box<RangeVec>> = if v.len() == 2 * n { v.split_off(n) } else { (0..n).map(|k| s.others_mass(reach, 1 << k)).collect() };
                    // Valor ponderado por el alcance propio (lo que suma a la raíz).
                    let vals: Vec<Box<RangeVec>> = v.into_iter().collect();
                    out.push((path, idx, vals, reach.to_vec()));
                    masses
                }
                _ => {
                    let idx = node.index();
                    let an = &s.tree.actions[idx];
                    let p = an.player as usize;
                    let sigma = s.average_strategy(idx);
                    let m0 = s.others_mass(reach, 1 << p);
                    let mut tot: Vec<Box<RangeVec>> = (0..n).map(|_| zero_range()).collect();
                    for a in 0..an.num_actions as usize {
                        let mut r = reach.to_vec();
                        let sa = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
                        for (x, &y) in r[p].iter_mut().zip(sa) {
                            *x *= y;
                        }
                        let start = out.len();
                        let ms = rec(s, an.children[a], &r, format!("{} | {}", s.tree.paths[idx], s.tree.labels[idx][a]), pool, out);
                        // Normalización del bunching de `eval_all` para el que decide.
                        let mut f = full_range();
                        for h in 0..NUM_COMBOS {
                            f[h] = if ms[p][h] > 1e-12 { m0[h] / ms[p][h] } else { 1.0 };
                        }
                        for e in &mut out[start..] {
                            for h in 0..NUM_COMBOS {
                                e.2[p][h] *= f[h];
                            }
                        }
                        for k in 0..n {
                            for h in 0..NUM_COMBOS {
                                tot[k][h] += if k == p { sa[h] * m0[h] } else { ms[k][h] };
                            }
                        }
                    }
                    tot
                }
            }
        }
        let mut entries: Vec<Entry> = Vec::new();
        crate::with_pool!(self.cfg.threads, |pool| {
            let reach: Vec<Box<RangeVec>> = (0..n).map(|_| full_range()).collect();
            rec(self, self.tree.root, &reach, String::new(), pool, &mut entries);
        });
        // El valor del terminal para `i` ya es contrafactual; a la raíz llega ponderado por el
        // alcance propio de `i` hasta el terminal (producto de sus frecuencias en la ruta).
        let mut res = Vec::new();
        for (path, idx, vals, own) in entries {
            res.push((path, idx, (0..n).map(|i| (0..NUM_COMBOS).map(|h| own[i][h] as f64 * vals[i][h] as f64).sum::<f64>() / NUM_COMBOS as f64).collect()));
        }
        res
    }

    /// Ganancia de cada jugador al desviarse a su mejor respuesta **preflop** contra el
    /// perfil medio (el juego postflop se mantiene en su estrategia media). Su suma es la
    /// NashConv preflop; en juegos de suma constante, 2× la explotabilidad.
    pub fn best_response_gains(&self) -> Vec<f64> {
        self.evaluate().iter().map(|(ev, br)| br - ev).collect()
    }

    /// Versión por jugador (un recorrido de BR y otro de evaluación por jugador); se
    /// mantiene como referencia para validar [`Self::evaluate`].
    pub fn best_response_gains_per_player(&self) -> Vec<f64> {
        (0..self.tree.n).map(|i| self.root_value(i, Mode::BestResponse) - self.root_value(i, Mode::Evaluate)).collect()
    }

    fn root_value(&self, i: usize, mode: Mode) -> f64 {
        let reach: Vec<Box<RangeVec>> = (0..self.tree.n).map(|_| full_range()).collect();
        let v = crate::with_pool!(self.cfg.threads, |pool| self.walk(self.tree.root, i, &reach, mode, 0, mix(0xE7A1, i as u64), pool));
        v[..NUM_COMBOS].iter().map(|&x| x as f64).sum::<f64>() / NUM_COMBOS as f64
    }

    fn walk(&self, node: NodeRef, i: usize, reach: &[Box<RangeVec>], mode: Mode, depth: u32, seed: u64, pool: Pool<'_>) -> Box<RangeVec> {
        match node.kind() {
            NodeKind::Action => self.walk_action(node.index(), i, reach, mode, depth, seed, pool),
            NodeKind::Terminal => self.walk_terminal(node.index(), i, reach, mode, seed, pool),
            k => unreachable!("nodo {k:?} en árbol preflop"),
        }
    }

    fn walk_action(&self, idx: usize, i: usize, reach: &[Box<RangeVec>], mode: Mode, depth: u32, seed: u64, pool: Pool<'_>) -> Box<RangeVec> {
        let an = &self.tree.actions[idx];
        let cn = &self.tree.cfr[idx];
        let na = an.num_actions as usize;
        let p = an.player as usize;
        let sigma = match mode {
            Mode::Cfr => self.current_strategy(idx),
            _ => self.average_strategy(idx),
        };
        let child = |a: usize| -> Box<RangeVec> {
            let mut r: Vec<Box<RangeVec>> = reach.to_vec();
            let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
            for (x, &y) in r[p].iter_mut().zip(s) {
                *x *= y;
            }
            self.walk(an.children[a], i, &r, mode, depth + 1, mix(seed, a as u64 + 1), pool)
        };
        let values: Vec<Box<RangeVec>> = if na > 1 {
            pool.join_all((0..na).map(|a| Box::new(move || child(a)) as Box<dyn FnOnce() -> Box<RangeVec> + Send + '_>).collect())
        } else {
            (0..na).map(child).collect()
        };

        let mut v = zero_range();
        if p != i {
            for va in &values {
                for (x, y) in v.iter_mut().zip(va.iter()) {
                    *x += y;
                }
            }
            return v;
        }
        match mode {
            Mode::BestResponse => {
                v.fill(f32::NEG_INFINITY);
                for va in &values {
                    for (x, &y) in v.iter_mut().zip(va.iter()) {
                        *x = x.max(y);
                    }
                }
                v[NUM_COMBOS..].fill(0.0);
            }
            Mode::Evaluate | Mode::Cfr => {
                for (a, va) in values.iter().enumerate() {
                    let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
                    for h in 0..HAND_STRIDE {
                        v[h] += s[h] * va[h];
                    }
                }
            }
        }
        if mode == Mode::Cfr && !self.locks.contains_key(&idx) {
            let mut rng = SplitMix64::new(mix(seed, 0xC0FFEE));
            let mut deltas = vec![0.0f32; na * HAND_STRIDE];
            // Con edges, el valor de referencia es la mezcla de los valores ajustados.
            let edges = self.edges.get(&idx);
            let mut v_adj = vec![0.0f32; HAND_STRIDE];
            if let Some(e) = edges {
                for (a, va) in values.iter().enumerate() {
                    let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
                    for h in 0..HAND_STRIDE {
                        v_adj[h] += s[h] * (va[h] - e[a]);
                    }
                }
            }
            for (a, va) in values.iter().enumerate() {
                let d = &mut deltas[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
                let e = edges.map(|e| e[a]).unwrap_or(0.0);
                for h in 0..HAND_STRIDE {
                    d[h] = if edges.is_some() { va[h] - e - v_adj[h] } else { va[h] - v[h] };
                }
            }
            self.weight_regrets(&mut deltas);
            self.regrets.accumulate(cn, &deltas, &mut rng);
            let t = (self.iteration as f32).powf(self.cfg.avg_power);
            for a in 0..na {
                let d = &mut deltas[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
                let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
                for h in 0..HAND_STRIDE {
                    d[h] = t * reach[i][h] * s[h];
                }
            }
            self.strategy.accumulate(cn, &deltas, &mut rng);
        }
        v
    }

    /// `Π_{k ∉ exclude} P(k llega | h)` con bloqueo de `h`.
    fn others_mass(&self, reach: &[Box<RangeVec>], exclude: u32) -> Box<RangeVec> {
        let mut prod = full_range();
        for (k, r) in reach.iter().enumerate() {
            if exclude & (1 << k) != 0 {
                continue;
            }
            let bs = BlockerSums::new(&r[..]);
            for h in 0..NUM_COMBOS {
                prod[h] *= bs.compatible(&r[..], h) / COMPATIBLE_COMBOS;
            }
        }
        prod
    }

    /// Una sola pasada con el perfil medio que devuelve, para cada jugador, `(EV, BR)`:
    /// el valor esperado y el de su mejor respuesta preflop. Es correcto hacerlo a la vez
    /// porque el alcance de cada jugador con el perfil medio es el mismo en todos los
    /// recorridos de mejor respuesta (el valor de `i` no depende de su propio alcance).
    #[allow(clippy::only_used_in_recursion)]
    fn eval_all(&self, node: NodeRef, reach: &[Box<RangeVec>], depth: u32, seed: u64, cap: Option<&EvCapture>, pool: Pool<'_>) -> Vec<(Box<RangeVec>, Box<RangeVec>)> {
        let n = self.tree.n;
        if node.kind() == NodeKind::Terminal {
            return self.terminal_all(node.index(), reach, StrategyMode::Average, seed, pool).into_iter().map(|v| (v.clone(), v)).collect();
        }
        let idx = node.index();
        let an = &self.tree.actions[idx];
        let na = an.num_actions as usize;
        let p = an.player as usize;
        let sigma = self.average_strategy(idx);
        let child = |a: usize| {
            let mut r: Vec<Box<RangeVec>> = reach.to_vec();
            let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
            for (x, &y) in r[p].iter_mut().zip(s) {
                *x *= y;
            }
            self.eval_all(an.children[a], &r, depth + 1, mix(seed, a as u64 + 1), cap, pool)
        };
        let mut values: Vec<Vec<(Box<RangeVec>, Box<RangeVec>)>> = if na > 1 {
            pool.join_all((0..na).map(|a| Box::new(move || child(a)) as Box<dyn FnOnce() -> Vec<(Box<RangeVec>, Box<RangeVec>)> + Send + '_>).collect())
        } else {
            (0..na).map(child).collect()
        };
        let width = values[0].len();
        if width == 2 * n {
            // Normalización del bunching (ver `normalize_bunching`), en EV y en BR.
            let m0 = self.others_mass(reach, 1 << p);
            for va in values.iter_mut() {
                for h in 0..NUM_COMBOS {
                    let m = va[n + p].0[h];
                    if m > 1e-12 {
                        let r = m0[h] / m;
                        va[p].0[h] *= r;
                        va[p].1[h] *= r;
                    }
                    va[n + p].0[h] = m0[h];
                    va[n + p].1[h] = m0[h];
                }
            }
        }
        if let Some(cap) = cap {
            // Los valores están multiplicados por P(los rivales llegan | h): se deshace.
            let mass = self.others_mass(reach, 1 << p);
            let mut evs = vec![0.0f32; na * HAND_STRIDE];
            for (a, va) in values.iter().enumerate() {
                for h in 0..NUM_COMBOS {
                    evs[a * HAND_STRIDE + h] = if mass[h] > 1e-12 { va[p].0[h] / mass[h] } else { f32::NAN };
                }
            }
            cap.lock().unwrap()[idx] = evs;
        }
        let mut out: Vec<(Box<RangeVec>, Box<RangeVec>)> = (0..width).map(|_| (zero_range(), zero_range())).collect();
        for k in 0..width {
            let own = k % n == p;
            let (ev, br) = &mut out[k];
            if own {
                br.fill(f32::NEG_INFINITY);
            }
            for (a, va) in values.iter().enumerate() {
                let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
                for h in 0..HAND_STRIDE {
                    if own {
                        ev[h] += s[h] * va[k].0[h];
                        br[h] = br[h].max(va[k].1[h]);
                    } else {
                        ev[h] += va[k].0[h];
                        br[h] += va[k].1[h];
                    }
                }
            }
            if own {
                br[NUM_COMBOS..].fill(0.0);
            }
        }
        out
    }

    /// `(EV, BR)` medios por jugador en la raíz, en una sola pasada.
    pub fn evaluate(&self) -> Vec<(f64, f64)> {
        let reach: Vec<Box<RangeVec>> = (0..self.tree.n).map(|_| full_range()).collect();
        let mean = |v: &RangeVec| v[..NUM_COMBOS].iter().map(|&x| x as f64).sum::<f64>() / NUM_COMBOS as f64;
        let mut vals = crate::with_pool!(self.cfg.threads, |pool| self.eval_all(self.tree.root, &reach, 0, mix(0xE7A1, 0), None, pool));
        // Con bunching, los jugadores que no actúan en todas las ramas llegan a la raíz sin
        // normalizar: se divide por su masa (≈ 1).
        let n = self.tree.n;
        if vals.len() == 2 * n {
            for k in 0..n {
                let m = vals[n + k].0.clone();
                for h in 0..NUM_COMBOS {
                    if m[h] > 1e-12 {
                        vals[k].0[h] /= m[h];
                        vals[k].1[h] /= m[h];
                    }
                }
            }
            vals.truncate(n);
        }
        vals.iter().map(|(e, b)| (mean(e), mean(b))).collect()
    }

    /// EV de cada acción por combo en cada nodo de acción, con el perfil medio y
    /// condicionado a la mano (y a que los rivales lleguen al nodo): `out[nodo]` =
    /// `[acción][combo]` con stride 1328, en unidades de utilidad. `NaN` donde los rivales
    /// no llegan nunca con esa mano (bloqueada o rango vacío).
    pub fn action_evs(&self) -> Vec<Vec<f32>> {
        let reach: Vec<Box<RangeVec>> = (0..self.tree.n).map(|_| full_range()).collect();
        let cap: EvCapture = std::sync::Mutex::new(vec![Vec::new(); self.tree.actions.len()]);
        crate::with_pool!(self.cfg.threads, |pool| self.eval_all(self.tree.root, &reach, 0, mix(0xE7A1, 0), Some(&cap), pool));
        cap.into_inner().unwrap()
    }

    fn walk_all(&self, node: NodeRef, reach: &[Box<RangeVec>], depth: u32, seed: u64, pool: Pool<'_>) -> Vec<Box<RangeVec>> {
        match node.kind() {
            NodeKind::Action => self.walk_all_action(node.index(), reach, depth, seed, pool),
            NodeKind::Terminal => self.terminal_all(node.index(), reach, StrategyMode::Current, seed, pool),
            k => unreachable!("nodo {k:?} en árbol preflop"),
        }
    }

    fn walk_all_action(&self, idx: usize, reach: &[Box<RangeVec>], depth: u32, seed: u64, pool: Pool<'_>) -> Vec<Box<RangeVec>> {
        let an = &self.tree.actions[idx];
        let cn = &self.tree.cfr[idx];
        let na = an.num_actions as usize;
        let p = an.player as usize;
        let n = self.tree.n;
        let sigma = self.current_strategy(idx);
        let child = |a: usize| -> Vec<Box<RangeVec>> {
            let mut r: Vec<Box<RangeVec>> = reach.to_vec();
            let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
            for (x, &y) in r[p].iter_mut().zip(s) {
                *x *= y;
            }
            self.walk_all(an.children[a], &r, depth + 1, mix(seed, a as u64 + 1), pool)
        };
        let values: Vec<Vec<Box<RangeVec>>> = if na > 1 {
            pool.join_all((0..na).map(|a| Box::new(move || child(a)) as Box<dyn FnOnce() -> Vec<Box<RangeVec>> + Send + '_>).collect())
        } else {
            (0..na).map(child).collect()
        };
        let mut values = values;
        if values[0].len() == 2 * n {
            let mut refs: Vec<&mut [Box<RangeVec>]> = values.iter_mut().map(|v| v.as_mut_slice()).collect();
            self.normalize_bunching(reach, p, &mut refs);
        }
        let width = values[0].len();
        let mut out: Vec<Box<RangeVec>> = (0..width).map(|_| zero_range()).collect();
        for k in 0..width {
            for (a, va) in values.iter().enumerate() {
                if k % n == p {
                    let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
                    for h in 0..HAND_STRIDE {
                        out[k][h] += s[h] * va[k][h];
                    }
                } else {
                    for h in 0..HAND_STRIDE {
                        out[k][h] += va[k][h];
                    }
                }
            }
        }
        if self.locks.contains_key(&idx) {
            return out;
        }
        let mut rng = SplitMix64::new(mix(seed, 0xC0FFEE));
        let mut deltas = vec![0.0f32; na * HAND_STRIDE];
        let edges = self.edges.get(&idx);
        let mut v_adj = vec![0.0f32; HAND_STRIDE];
        if let Some(e) = edges {
            for (a, va) in values.iter().enumerate() {
                let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
                for h in 0..HAND_STRIDE {
                    v_adj[h] += s[h] * (va[p][h] - e[a]);
                }
            }
        }
        for (a, va) in values.iter().enumerate() {
            let d = &mut deltas[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
            let e = edges.map(|e| e[a]).unwrap_or(0.0);
            for h in 0..HAND_STRIDE {
                d[h] = if edges.is_some() { va[p][h] - e - v_adj[h] } else { va[p][h] - out[p][h] };
            }
        }
        self.weight_regrets(&mut deltas);
        self.regrets.accumulate(cn, &deltas, &mut rng);
        let t = (self.iteration as f32).powf(self.cfg.avg_power);
        for a in 0..na {
            let d = &mut deltas[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
            let s = &sigma[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
            for h in 0..HAND_STRIDE {
                d[h] = t * reach[p][h] * s[h];
            }
        }
        self.strategy.accumulate(cn, &deltas, &mut rng);
        out
    }

    /// Valores de todos los jugadores en un terminal. Con `Current` además aplica updates
    /// MCCFR a los dos jugadores de una hoja de flop; con `Average` solo evalúa.
    fn terminal_all(&self, idx: usize, reach: &[Box<RangeVec>], smode: StrategyMode, seed: u64, pool: Pool<'_>) -> Vec<Box<RangeVec>> {
        let update = smode == StrategyMode::Current;
        let t = &self.tree.terminals[idx];
        if t.kind == TerminalKind::MultiwayN || (t.kind == TerminalKind::FlopLeaf && t.players[2] != 255) {
            let mut out = self.pot_all(t, reach, smode, update, seed, pool);
            // Sin corrección de bunching en estos botes: su masa es la independiente.
            if self.bunching_on() {
                out.extend((0..self.tree.n).map(|k| self.others_mass(reach, 1 << k)));
            }
            return out;
        }
        let n = self.tree.n;
        let u = t.utilities(&self.tree.utils);
        // P(k llega | h) por jugador, calculada una vez.
        let mass: Vec<Box<RangeVec>> = reach
            .iter()
            .map(|r| {
                let bs = BlockerSums::new(&r[..]);
                let mut m = zero_range();
                for h in 0..NUM_COMBOS {
                    m[h] = bs.compatible(&r[..], h) / COMPATIBLE_COMBOS;
                }
                m
            })
            .collect();
        let prod_excl = |excl: u32| -> Box<RangeVec> {
            let mut p = full_range();
            for (k, m) in mass.iter().enumerate() {
                if excl & (1 << k) == 0 {
                    for h in 0..NUM_COMBOS {
                        p[h] *= m[h];
                    }
                }
            }
            p
        };
        let scale = |v: &mut RangeVec, excl: u32| {
            let p = prod_excl(excl);
            for h in 0..NUM_COMBOS {
                v[h] *= p[h];
            }
        };
        let scalar = |s: f64, excl: u32| -> Box<RangeVec> {
            let mut v = prod_excl(excl);
            for h in 0..NUM_COMBOS {
                v[h] *= s as f32;
            }
            v
        };
        // Card bunching por pares. Sin él, los demás
        // jugadores son independientes dada la mano del héroe. Con él, cada par (j, k) de
        // los demás lleva su correlación: `P(j y k llegan) / (P(j)·P(k))`, que refleja que,
        // si k foldeó sus manos flojas, a j le quedan más cartas altas. Donde el rango de
        // un rival entra por combo (showdown, hoja de flop, bote a 3), la corrección va por
        // combo: su rango se repondera por `f_k(v) = P(k llega | v) / P(k llega)` de cada
        // jugador k fuera del bote; el resto de pares, como factor escalar `g_jk`. Así la
        // probabilidad de cada combo sigue repartida entre los terminales a primer orden.
        // 3 × 10 bb ICM, SB tras el fold del BTN con el call de la BB: EV del push
        // de 72o −0,19 → −0,31.
        let bunch = self.bunching_on();
        let tot: Vec<f32> = reach.iter().map(|r| r[..NUM_COMBOS].iter().sum()).collect();
        let f = |k: usize| -> Option<Box<RangeVec>> {
            let mb = tot[k] / NUM_COMBOS as f32;
            (bunch && mb > 0.0).then(|| {
                let mut v = zero_range();
                for h in 0..NUM_COMBOS {
                    v[h] = mass[k][h] / mb;
                }
                v
            })
        };
        let fs: Vec<Option<Box<RangeVec>>> = (0..n).map(f).collect();
        let mut g = vec![1.0f32; n * n];
        if bunch {
            for j in 0..n {
                for k in j + 1..n {
                    if let (Some(fk), true) = (&fs[k], tot[j] > 0.0) {
                        let x = reach[j][..NUM_COMBOS].iter().zip(&fk[..NUM_COMBOS]).map(|(r, f)| r * f).sum::<f32>() / tot[j];
                        g[j * n + k] = x;
                        g[k * n + j] = x;
                    }
                }
            }
        }
        // Producto de `g` sobre los pares de jugadores fuera de `excl`.
        let pairs = |excl: u32| -> f32 {
            let mut x = 1.0f32;
            for j in (0..n).filter(|&j| excl & (1 << j) == 0) {
                for k in (j + 1..n).filter(|&k| excl & (1 << k) == 0) {
                    x *= g[j * n + k];
                }
            }
            x
        };
        // Rango de `p` repondrado por los jugadores fuera de `keep` (y distintos de `p`).
        let weighted = |p: usize, keep: u32| -> Box<RangeVec> {
            let mut w = reach[p].clone();
            for (_, fk) in fs.iter().enumerate().filter(|&(k, _)| k != p && keep & (1 << k) == 0) {
                if let Some(fk) = fk {
                    for h in 0..NUM_COMBOS {
                        w[h] *= fk[h];
                    }
                }
            }
            w
        };
        let times = |v: &mut RangeVec, x: f32| {
            if x != 1.0 {
                for h in 0..NUM_COMBOS {
                    v[h] *= x;
                }
            }
        };
        let mut out: Vec<Box<RangeVec>> = (0..n).map(|_| zero_range()).collect();
        // Con bunching, masa de probabilidad corregida de cada jugador (el valor con u = 1).
        let mut masses: Vec<Box<RangeVec>> = if bunch { (0..n).map(|_| zero_range()).collect() } else { Vec::new() };
        let obs_mass = |k: usize| -> Box<RangeVec> {
            let mut m = scalar(1.0, 1 << k);
            times(&mut m, pairs(1 << k));
            m
        };
        let [a, b] = [t.players[0] as usize, t.players[1] as usize];
        match t.kind {
            TerminalKind::Fold => {
                for k in 0..n {
                    out[k] = scalar(u[k] as f64, 1 << k);
                    times(&mut out[k], pairs(1 << k));
                    if bunch {
                        masses[k] = obs_mass(k);
                    }
                }
            }
            TerminalKind::AllInRunout | TerminalKind::Showdown => {
                let pair = (1 << a) | (1 << b);
                let (wa, wb) = (weighted(a, pair), weighted(b, pair));
                let mut pb = self.eq.products(&wb[..]);
                let mut pa = self.eq.products(&wa[..]);
                if let Some((bb, c)) = self.board_coefs(reach, pair) {
                    bb.apply(&mut pb, &wb[..], &c);
                    bb.apply(&mut pa, &wa[..], &c);
                }
                pb.hero_values(u[a], u[2 * n + a], u[n + a], &mut out[a][..]);
                pa.hero_values(u[n + b], u[2 * n + b], u[b], &mut out[b][..]);
                scale(&mut out[a], pair);
                scale(&mut out[b], pair);
                let gp = pairs(pair);
                times(&mut out[a], gp);
                times(&mut out[b], gp);
                if bunch {
                    pb.hero_values(1.0, 1.0, 1.0, &mut masses[a][..]);
                    pa.hero_values(1.0, 1.0, 1.0, &mut masses[b][..]);
                    for q in [a, b] {
                        scale(&mut masses[q], pair);
                        times(&mut masses[q], gp);
                    }
                }
                // Bunching conjunto board-manos (all-in con los que foldearon): corrección por
                // boards sobre el valor de la matriz, escalada por la masa de cada combo.
                if bunch && t.kind == TerminalKind::AllInRunout {
                    if let Some(mw) = &self.multiway {
                        let folds: Vec<(BlockerSums, &RangeVec, f32)> = (0..n)
                            .filter(|&k| k != a && k != b && tot[k] > 0.0)
                            .map(|k| (BlockerSums::new(&reach[k][..]), &*reach[k], tot[k]))
                            .collect();
                        if !folds.is_empty() {
                            let smode = if update { StrategyMode::Current } else { StrategyMode::Average };
                            let ca = mw.hu_bunch_correction([u[a], u[2 * n + a], u[n + a]], &wb, &folds, smode, mix(seed, 0xB0A));
                            let cb = mw.hu_bunch_correction([u[n + b], u[2 * n + b], u[b]], &wa, &folds, smode, mix(seed, 0xB0B));
                            for h in 0..NUM_COMBOS {
                                out[a][h] += ca[h] * masses[a][h];
                                out[b][h] += cb[h] * masses[b][h];
                            }
                        }
                    }
                }
                let (p_a, p_b, p_t) = pb.outcomes(&wa[..]);
                let tot = p_a + p_b + p_t;
                for k in (0..n).filter(|&k| k != a && k != b) {
                    let s = if tot > 0.0 { (p_a * u[k] as f64 + p_b * u[n + k] as f64 + p_t * u[2 * n + k] as f64) / tot } else { 0.0 };
                    out[k] = scalar(s, 1 << k);
                    times(&mut out[k], pairs(1 << k));
                    if bunch {
                        masses[k] = obs_mass(k);
                    }
                }
            }
            TerminalKind::FlopLeaf => {
                let leaf = t.leaf as usize;
                // Un único recorrido conjunto de la hoja da los valores de ambos jugadores y
                // de los observadores (desde la perspectiva de OOP).
                let observers: Vec<usize> = (0..n).filter(|&k| k != a && k != b).collect();
                let folded: Vec<&RangeVec> = observers.iter().map(|&k| &*reach[k]).collect();
                let pair = (1 << a) | (1 << b);
                let (wa, wb) = (weighted(a, pair), weighted(b, pair));
                let mut vals = self.est.values_all(&self.tree, leaf, &[&wa, &wb], smode, if update { 3 } else { 0 }, mix(seed, 0x1EAF), &observers, &folded, pool);
                let va = std::mem::take(&mut vals[0]);
                out[b] = vals[1][0].clone();
                out[a] = va[0].clone();
                scale(&mut out[a], pair);
                scale(&mut out[b], pair);
                let gp = pairs(pair);
                times(&mut out[a], gp);
                times(&mut out[b], gp);
                if bunch {
                    for (q, w) in [(a, &wb), (b, &wa)] {
                        let bs = BlockerSums::new(&w[..]);
                        for h in 0..NUM_COMBOS {
                            masses[q][h] = bs.compatible(&w[..], h) / COMPATIBLE_COMBOS;
                        }
                        scale(&mut masses[q], pair);
                        times(&mut masses[q], gp);
                    }
                }
                // Observador: E[u_k | ambos llegan] = Σ_h r_a·v_k / Σ_h r_a·M(h)/1225.
                let pm = pair_mass(&wa, &wb) * NUM_COMBOS as f64;
                for (j, &k) in observers.iter().enumerate() {
                    let s = if pm > 0.0 { weighted_mean(&wa, &va[1 + j]) * NUM_COMBOS as f64 / pm } else { 0.0 };
                    out[k] = scalar(s, 1 << k);
                    times(&mut out[k], pairs(1 << k));
                    if bunch {
                        masses[k] = obs_mass(k);
                    }
                }
            }
            TerminalKind::MultiwayN => unreachable!("resuelto en pot_all"),
            TerminalKind::Multiway => {
                let mw = self.multiway.as_ref().expect("evaluador multiway");
                let pl = [t.players[0] as usize, t.players[1] as usize, t.players[2] as usize];
                let in_pot = (1u32 << pl[0]) | (1 << pl[1]) | (1 << pl[2]);
                let observers: Vec<usize> = (0..n).filter(|k| !pl.contains(k)).collect();
                let seed = mix(seed, 0x3A7);
                let bw = self.board_weight(reach, in_pot);
                for pos in 0..3 {
                    let oth: Vec<usize> = (0..3).filter(|&k| k != pos).map(|k| pl[k]).collect();
                    let mut seats = vec![pl[pos]];
                    if pos == 0 {
                        seats.extend(&observers);
                    }
                    // El bloqueo entre los dos rivales lo hace `board_orders` por board (exacto
                    // salvo un término de tercer orden).
                    let (w0, w1) = (weighted(oth[0], in_pot), weighted(oth[1], in_pot));
                    let jf = bunch.then(|| JointFolds { folds: observers.iter().filter(|&&k| tot[k] > 0.0).map(|&k| (BlockerSums::new(&reach[k][..]), &*reach[k], tot[k])).collect(), marginal: true });
                    let vals = mw.values_w(t, &self.tree.utils, n, pos, &seats, [&w0, &w1], smode, seed, bw.as_ref().map(|f| f as &BoardWeight<'_>), jf.as_ref());
                    out[pl[pos]] = vals[0].clone();
                    scale(&mut out[pl[pos]], in_pot);
                    // Los dos rivales del bote se tratan como independientes: su par, escalar.
                    let fac = pairs(in_pot) * g[oth[0] * n + oth[1]];
                    times(&mut out[pl[pos]], fac);
                    if bunch {
                        let (b0, b1) = (BlockerSums::new(&w0[..]), BlockerSums::new(&w1[..]));
                        let m = &mut masses[pl[pos]];
                        for h in 0..NUM_COMBOS {
                            m[h] = b0.compatible(&w0[..], h) / COMPATIBLE_COMBOS * b1.compatible(&w1[..], h) / COMPATIBLE_COMBOS;
                        }
                        scale(m, in_pot);
                        times(m, fac);
                    }
                    if pos == 0 {
                        let wp0 = weighted(pl[0], in_pot);
                        let tm = triple_mass(&wp0, &w0, &w1);
                        for (j, &k) in observers.iter().enumerate() {
                            let s = if tm > 0.0 { weighted_mean(&wp0, &vals[1 + j]) / tm } else { 0.0 };
                            out[k] = scalar(s, 1 << k);
                            times(&mut out[k], pairs(1 << k));
                            if bunch {
                                masses[k] = obs_mass(k);
                            }
                        }
                    }
                }
            }
        }
        out.extend(masses);
        out
    }

    /// Botes de k jugadores resueltos por un evaluador genérico: `MultiwayN` (Monte Carlo
    /// de manos, 4-6 jugadores) y hojas de flop a 3 (MCCFR multiway). Valores de todos los
    /// jugadores; los observadores, condicionados y escalados por sus masas.
    fn pot_all(&self, t: &TerminalNode, reach: &[Box<RangeVec>], smode: StrategyMode, update: bool, seed: u64, pool: Pool<'_>) -> Vec<Box<RangeVec>> {
        let n = self.tree.n;
        let pl = self.tree.pot_players(t);
        let k = pl.len();
        let in_pot: u32 = pl.iter().map(|&p| 1u32 << p).sum();
        let observers: Vec<usize> = (0..n).filter(|p| !pl.contains(p)).collect();
        let mut out: Vec<Box<RangeVec>> = (0..n).map(|_| zero_range()).collect();
        // Poda: el valor de cada jugador va multiplicado por la masa de los demás del bote;
        // si dos o más apenas llegan, todos los valores son despreciables (< 1e-5·U).
        let negligible = pl.iter().filter(|&&p| reach[p][..NUM_COMBOS].iter().sum::<f32>() < 1e-5 * NUM_COMBOS as f32).count();
        if negligible >= 2 {
            return out;
        }
        let leaf3 = t.kind == TerminalKind::FlopLeaf;
        let bw = if leaf3 { None } else { self.board_weight(reach, in_pot) };
        // Bunching conjunto en los all-in a 4+ (rangos sin factor marginal aquí).
        let jf: Option<JointFolds<'_>> = (!leaf3 && self.bunching_on()).then(|| JointFolds {
            folds: observers.iter().map(|&q| (BlockerSums::new(&reach[q][..]), &*reach[q], reach[q][..NUM_COMBOS].iter().sum::<f32>())).filter(|x| x.2 > 0.0).collect(),
            marginal: false,
        });
        let jf_ref = jf.as_ref();
        // Las `k` posiciones son independientes entre sí, salvo en `leaf3`: ahí `pos == 0`
        // calcula las tres juntas de un tirón (más barato que tres llamadas sueltas) y el
        // resto solo recoge su parte, así que ese camino sigue secuencial a propósito. El
        // camino `MultiwayN` (botes de 4-6 vías, Monte Carlo) no tiene ese atajo conjunto:
        // cada posición es una llamada independiente a `values_nw`, que no reparte nada
        // entre hilos por dentro. Antes de esto, un bote a 4+ resolvía sus k evaluaciones
        // una detrás de otra en un solo hilo -- con `--max-in-pot 4` o más, eso es la
        // mayoría del trabajo del árbol sin aprovechar el resto de núcleos.
        let results: Vec<(usize, Vec<Box<RangeVec>>)> = if leaf3 {
            let mut joint: Vec<Vec<Box<RangeVec>>> = Vec::new();
            (0..k)
                .map(|pos| {
                    if pos == 0 {
                        let rk: Vec<&RangeVec> = pl.iter().map(|&q| &*reach[q]).collect();
                        let folded: Vec<&RangeVec> = observers.iter().map(|&q| &*reach[q]).collect();
                        let all = ((1u16 << k) - 1) as u8;
                        joint = self.est.values_all(&self.tree, t.leaf as usize, &rk, smode, if update { all } else { 0 }, mix(seed, 0x3EAF), &observers, &folded, pool);
                    }
                    (pos, std::mem::take(&mut joint[pos]))
                })
                .collect()
        } else {
            let mw = self.multiway.as_ref().expect("evaluador multiway");
            let bw_ref = bw.as_ref().map(|f| f as &BoardWeight<'_>);
            let tasks: Vec<Box<dyn FnOnce() -> (usize, Vec<Box<RangeVec>>) + Send + '_>> = (0..k)
                .map(|pos| {
                    let others: Vec<&RangeVec> = (0..k).filter(|&q| q != pos).map(|q| &*reach[pl[q]]).collect();
                    let mut seats = vec![pl[pos]];
                    if pos == 0 {
                        seats.extend(&observers);
                    }
                    Box::new(move || (pos, mw.values_nw(self.tree.terminal_utils(t), n, k, pos, &seats, &others, smode, mix(seed, 0x4A7), bw_ref, jf_ref))) as Box<dyn FnOnce() -> (usize, Vec<Box<RangeVec>>) + Send + '_>
                })
                .collect();
            pool.join_all(tasks)
        };
        for (pos, vals) in results {
            let prod = self.others_mass(reach, in_pot);
            let mut v = vals[0].clone();
            for h in 0..NUM_COMBOS {
                v[h] *= prod[h];
            }
            out[pl[pos]] = v;
            if pos == 0 && !observers.is_empty() {
                let others: Vec<&RangeVec> = (0..k).filter(|&q| q != 0).map(|q| &*reach[pl[q]]).collect();
                let nm = n_mass(&reach[pl[0]], &others);
                for (j, &obs) in observers.iter().enumerate() {
                    let s = if nm > 0.0 { (weighted_mean(&reach[pl[0]], &vals[1 + j]) / nm) as f32 } else { 0.0 };
                    let prod = self.others_mass(reach, 1 << obs);
                    for h in 0..NUM_COMBOS {
                        out[obs][h] = s * prod[h];
                    }
                }
            }
        }
        out
    }

    fn walk_terminal(&self, idx: usize, i: usize, reach: &[Box<RangeVec>], mode: Mode, seed: u64, pool: Pool<'_>) -> Box<RangeVec> {
        let t = &self.tree.terminals[idx];
        if t.kind == TerminalKind::MultiwayN || (t.kind == TerminalKind::FlopLeaf && t.players[2] != 255) {
            let smode = if mode == Mode::Cfr { StrategyMode::Current } else { StrategyMode::Average };
            return self.pot_all(t, reach, smode, mode == Mode::Cfr, seed, pool).swap_remove(i);
        }
        let n = self.tree.n;
        let u = t.utilities(&self.tree.utils);
        let mut out = zero_range();
        let [a, b] = [t.players[0] as usize, t.players[1] as usize];
        let smode = if mode == Mode::Cfr { StrategyMode::Current } else { StrategyMode::Average };
        match t.kind {
            TerminalKind::MultiwayN => unreachable!("resuelto en pot_all"),
            TerminalKind::Multiway => {
                let mw = self.multiway.as_ref().expect("evaluador multiway");
                let pl = [t.players[0] as usize, t.players[1] as usize, t.players[2] as usize];
                let in_pot = (1u32 << pl[0]) | (1 << pl[1]) | (1 << pl[2]);
                let seed = mix(seed, 0x3A7);
                if let Some(pos) = pl.iter().position(|&k| k == i) {
                    let others: Vec<usize> = (0..3).filter(|&k| k != pos).map(|k| pl[k]).collect();
                    mw.hero_values(t, &self.tree.utils, n, pos, i, [&reach[others[0]], &reach[others[1]]], smode, seed, &mut out);
                    let prod = self.others_mass(reach, in_pot);
                    for h in 0..NUM_COMBOS {
                        out[h] *= prod[h];
                    }
                } else {
                    let tm = triple_mass(&reach[pl[0]], &reach[pl[1]], &reach[pl[2]]);
                    let s = mw.observer_value(t, &self.tree.utils, n, i, [&reach[pl[0]], &reach[pl[1]], &reach[pl[2]]], smode, seed);
                    let s = if tm > 0.0 { (s / tm) as f32 } else { 0.0 };
                    let prod = self.others_mass(reach, 1 << i);
                    for h in 0..NUM_COMBOS {
                        out[h] = s * prod[h];
                    }
                }
            }
            TerminalKind::Fold => {
                let prod = self.others_mass(reach, 1 << i);
                for h in 0..NUM_COMBOS {
                    out[h] = u[i] * prod[h];
                }
            }
            TerminalKind::AllInRunout | TerminalKind::Showdown | TerminalKind::FlopLeaf => {
                let is_leaf = t.kind == TerminalKind::FlopLeaf;
                if i == a || i == b {
                    let j = if i == a { b } else { a };
                    if is_leaf {
                        let leaf = t.leaf as usize;
                        self.est.hero_values(&self.tree, leaf, i == b, &reach[i], &reach[j], smode, mode == Mode::Cfr, mix(seed, 0x1EAF), &mut out, pool);
                    } else {
                        let (uw, ul) = if i == a { (u[i], u[n + i]) } else { (u[n + i], u[i]) };
                        self.eq.hero_values(&reach[j][..], uw, u[2 * n + i], ul, &mut out[..]);
                    }
                    let prod = self.others_mass(reach, (1 << i) | (1 << j));
                    for h in 0..NUM_COMBOS {
                        out[h] *= prod[h];
                    }
                } else {
                    // Utilidad esperada condicionada a que ambos lleguen; la probabilidad de
                    // llegar va en las masas con bloqueo del héroe, como en el resto de
                    // terminales (así la probabilidad total por combo suma exactamente 1).
                    let s = if is_leaf {
                        let mut sv = vec![0.0f64; n];
                        self.est.observer_values(&self.tree, t.leaf as usize, &reach[a], &reach[b], smode, mix(seed, 0x0B5), &mut sv, pool);
                        let pm = pair_mass(&reach[a], &reach[b]);
                        if pm > 0.0 { sv[i] / pm } else { 0.0 }
                    } else {
                        let (pa, pb, pt) = self.eq.pair_outcomes(&reach[a][..], &reach[b][..]);
                        let tot = pa + pb + pt;
                        if tot > 0.0 { (pa * u[i] as f64 + pb * u[n + i] as f64 + pt * u[2 * n + i] as f64) / tot } else { 0.0 }
                    };
                    let prod = self.others_mass(reach, 1 << i);
                    for h in 0..NUM_COMBOS {
                        out[h] = s as f32 * prod[h];
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::board_pool::{BoardPool, PoolConfig};
    use crate::cfr::report::{class_frequency, overall_frequency};
    use crate::cfr::{CheckDown, PostflopEngine};
    use crate::math::PkoParams;
    use crate::tree::{build_tree, Payoff, PreflopConfig, TerminalKind};

    fn matrix() -> Arc<EquityMatrix> {
        use std::sync::OnceLock;
        static M: OnceLock<Arc<EquityMatrix>> = OnceLock::new();
        M.get_or_init(|| Arc::new(EquityMatrix::monte_carlo(3000, 21, 4))).clone()
    }

    fn solve(cfg: PreflopConfig, iters: u32) -> Solver<CheckDown> {
        let eq = matrix();
        let mut s = Solver::new(build_tree(cfg), eq.clone(), CheckDown { eq }, SolverConfig::default());
        s.run(iters);
        s
    }

    /// HU push/fold 10 bb, chip EV. Referencia Nash (sin ante): SB empuja ≈ 58 % de
    /// combos, BB paga ≈ 37 %.
    #[test]
    fn heads_up_push_fold_converges_to_nash() {
        let s = solve(PreflopConfig::push_fold(vec![10.0, 10.0], Payoff::ChipEv), 400);
        let sb = s.average_strategy(0);
        let bb = s.average_strategy(s.tree.find("SB:A").unwrap());
        let (push, call) = (overall_frequency(&sb, 1), overall_frequency(&bb, 1));
        eprintln!("push {push:.3} call {call:.3} gains {:?}", s.best_response_gains());
        assert!((push - 0.58).abs() < 0.04, "push {push}");
        assert!((call - 0.37).abs() < 0.04, "call {call}");
        assert!(class_frequency(&sb, 1, "AA").unwrap() > 0.99);
        assert!(class_frequency(&sb, 1, "72o").unwrap() < 0.05);
        assert!(class_frequency(&bb, 1, "K2o").unwrap() < 0.05);
        let gains: f64 = s.best_response_gains().iter().sum();
        assert!(gains < 0.02, "NashConv {gains} bb");
    }

    /// Burbuja de satélite (2 plazas iguales, 3 jugadores de 10 bb): el ICM debe
    /// estrechar dramáticamente el rango de call de la BB frente al shove de la SB.
    #[test]
    fn icm_bubble_tightens_calling_range() {
        let cfg = |payoff| PreflopConfig::push_fold(vec![10.0, 10.0, 10.0], payoff);
        let chip = solve(cfg(Payoff::ChipEv), 200);
        let icm = solve(cfg(Payoff::Icm { payouts: vec![50.0, 50.0] }), 200);
        let node = "BTN:F SB:A";
        let f = |s: &Solver<CheckDown>| overall_frequency(&s.average_strategy(s.tree.find(node).unwrap()), 1);
        let (c, i) = (f(&chip), f(&icm));
        eprintln!("call chipEV {c:.3} vs ICM {i:.3}");
        assert!(i < 0.5 * c, "ICM {i} vs cEV {c}");
        // Conservación: la suma de $EV es el prize pool.
        let ev: f64 = icm.expected_values().iter().sum();
        assert!((ev - 100.0).abs() < 0.5, "{ev}");
    }

    /// PKO: con la cabeza del shover en juego, la BB (que le cubre) paga más ancho que en
    /// un torneo sin bounties con los mismos premios. Hace falta un tercer jugador (fuera
    /// de la mesa): en un HU puro con α = 1 toda la bounty equity es lineal en fichas y
    /// no altera las decisiones (lo comprueba el segundo bloque).
    #[test]
    fn bounty_widens_calling_range() {
        let payouts = vec![50.0, 30.0, 20.0];
        let with_off = |payoff| PreflopConfig { offtable_stacks: vec![30.0], ..PreflopConfig::push_fold(vec![10.0, 30.0], payoff) };
        let icm = solve(with_off(Payoff::Icm { payouts: payouts.clone() }), 200);
        let pko = solve(with_off(Payoff::Pko { payouts, bounties: vec![20.0, 20.0, 20.0], params: PkoParams::default() }), 200);
        let f = |s: &Solver<CheckDown>| overall_frequency(&s.average_strategy(s.tree.find("SB:A").unwrap()), 1);
        let (a, b) = (f(&icm), f(&pko));
        eprintln!("call ICM {a:.3} vs PKO {b:.3}");
        assert!(b > a + 0.05, "PKO {b} vs ICM {a}");

        let hu = |payoff| solve(PreflopConfig::push_fold(vec![10.0, 30.0], payoff), 200);
        let (x, y) = (f(&hu(Payoff::Icm { payouts: vec![60.0, 40.0] })), f(&hu(Payoff::Pko { payouts: vec![60.0, 40.0], bounties: vec![20.0, 20.0], params: PkoParams::default() })));
        assert!((x - y).abs() < 0.01, "HU: bounty lineal en fichas, {x} vs {y}");
    }

    /// `Payoff::PkoChipEv`: bounties sin ICM, directamente en fichas. Mismo efecto
    /// cualitativo que el PKO con ICM de arriba (ensancha el rango de pago), pero contra
    /// chip EV puro en vez de contra ICM puro.
    #[test]
    fn pko_chip_ev_widens_calling_range_like_regular_pko() {
        let with_off = |payoff| PreflopConfig { offtable_stacks: vec![30.0], ..PreflopConfig::push_fold(vec![10.0, 30.0], payoff) };
        let chip = solve(with_off(Payoff::ChipEv), 200);
        let pko_chip = solve(with_off(Payoff::PkoChipEv { bounties: vec![20.0, 20.0, 20.0], params: PkoParams::default() }), 200);
        let f = |s: &Solver<CheckDown>| overall_frequency(&s.average_strategy(s.tree.find("SB:A").unwrap()), 1);
        let (a, b) = (f(&chip), f(&pko_chip));
        eprintln!("call chip EV {a:.3} vs PKO-fichas {b:.3}");
        // Mismo signo que bounty_widens_calling_range (arriba) pero el margen es menor:
        // sin ICM no hay presión de supervivencia empujando el rango base hacia abajo, así
        // que chip EV solo ya paga más ancho y el bounty tiene menos margen relativo que
        // ensanchar.
        assert!(b > a + 0.03, "PKO-fichas {b} vs chip EV {a}");
    }

    /// Consistencia de la normalización: tras foldear, la utilidad chip-EV es constante,
    /// así que el valor de la rama de fold debe ser el stack exacto para todo combo.
    #[test]
    fn fold_value_is_exact_stack_for_every_combo() {
        let eq = matrix();
        let cfg = PreflopConfig::push_fold(vec![18.75, 28.75, 15.0, 12.0], Payoff::ChipEv);
        for sim in [false, true] {
            let mut s = Solver::new(build_tree(cfg.clone()), eq.clone(), CheckDown { eq: eq.clone() }, SolverConfig { simultaneous: Some(sim), ..SolverConfig::default() });
            s.run(50);
            let reach: Vec<Box<RangeVec>> = (0..4).map(|_| full_range()).collect();
            let fold = s.tree.actions[0].children[0];
            let (v, w) = crate::with_pool!(s.cfg.threads, |pool| (s.walk(fold, 0, &reach, Mode::Evaluate, 9, 1, pool), s.walk_all(fold, &reach, 9, 1, pool)));
            // Con bunching el recorrido simultáneo devuelve además las masas: el valor se
            // normaliza en el nodo del héroe (aquí, la raíz del subárbol).
            let wv = |h: usize| if w.len() == 8 { w[0][h] / w[4][h] } else { w[0][h] };
            for h in 0..NUM_COMBOS {
                assert!((v[h] - 18.75).abs() < 2e-3, "{h}: {}", v[h]);
                assert!((wv(h) - 18.75).abs() < 2e-3, "{h}: {}", wv(h));
            }
        }
    }

    /// EV por acción condicionado a la mano: foldear frente al shove deja a la BB con
    /// 9 bb exactos para cualquier combo aunque el rango del SB sea parcial, y en la raíz
    /// la mezcla de las acciones reproduce el EV del asiento.
    #[test]
    fn action_evs_are_conditional_on_the_hand() {
        let eq = matrix();
        let mut s = Solver::new(build_tree(PreflopConfig::push_fold(vec![10.0, 10.0], Payoff::ChipEv)), eq.clone(), CheckDown { eq }, SolverConfig::default());
        s.run(200);
        let evs = s.action_evs();
        let k = s.tree.find("SB:A").unwrap();
        let fold = s.tree.labels[k].iter().position(|l| l == "F").unwrap();
        let call = s.tree.labels[k].iter().position(|l| l == "C").unwrap();
        let aa = (0..NUM_COMBOS).find(|&h| crate::cfr::report::grid_cell(h) == (0, 0)).unwrap();
        for h in 0..NUM_COMBOS {
            let v = evs[k][fold * HAND_STRIDE + h];
            assert!((v - 9.0).abs() < 1e-3, "combo {h}: {v}");
        }
        assert!(evs[k][call * HAND_STRIDE + aa] > 15.0, "AA paga: {}", evs[k][call * HAND_STRIDE + aa]);
        let sigma = s.average_strategy(0);
        let na = s.tree.actions[0].num_actions as usize;
        let mixed: f64 = (0..NUM_COMBOS).map(|h| (0..na).map(|a| (sigma[a * HAND_STRIDE + h] * evs[0][a * HAND_STRIDE + h]) as f64).sum::<f64>()).sum::<f64>() / NUM_COMBOS as f64;
        let ev = s.expected_values()[0];
        assert!((mixed - ev).abs() < 1e-3, "{mixed} vs {ev}");
    }

    /// Una solución ICM medida en fichas con la tabla de fichas del mismo árbol: foldear en
    /// la raíz vale exactamente el stack, las fichas se conservan y la tabla original
    /// vuelve a su sitio.
    #[test]
    fn icm_solution_measured_in_chips() {
        let eq = matrix();
        let cfg = PreflopConfig::push_fold(vec![10.0, 20.0, 15.0], Payoff::Icm { payouts: vec![50.0, 30.0, 20.0] });
        let chips = crate::tree::builder::chip_utils(&cfg);
        let mut s = Solver::new(build_tree(cfg), eq.clone(), CheckDown { eq }, SolverConfig::default());
        s.run(100);
        let before = s.expected_values();
        let (evs, seats) = s.with_utils(chips, |s| (s.action_evs(), s.expected_values()));
        let fold = s.tree.labels[0].iter().position(|l| l == "F").unwrap();
        for h in 0..NUM_COMBOS {
            assert!((evs[0][fold * HAND_STRIDE + h] - 10.0).abs() < 1e-3);
        }
        assert!((seats.iter().sum::<f64>() - 45.0).abs() < 0.05, "{seats:?}");
        assert_eq!(s.expected_values(), before);
        assert!((before.iter().sum::<f64>() - 100.0).abs() < 0.05);
    }

    /// Dos locks sobre el mismo nodo se combinan: cada acción conserva su rango y el
    /// resto va a la acción 0.
    /// Muestreo de subárbol: solo cambian los nodos del subárbol elegido.
    #[test]
    fn subtree_sampling_only_updates_the_subtree() {
        let mut s = solve(PreflopConfig { postflop: None, ..PreflopConfig::standard(vec![20.0; 3], Payoff::ChipEv) }, 60);
        let sub = "BTN:R2";
        let k = s.tree.find(sub).unwrap();
        let before: Vec<Vec<f32>> = (0..s.tree.paths.len()).map(|j| s.average_strategy(j)).collect();
        s.set_subtree(Some(sub)).unwrap();
        s.run(30);
        let mut changed_inside = false;
        for (j, p) in s.tree.paths.iter().enumerate() {
            let inside = j == k || p.starts_with(&format!("{sub} "));
            let same = s.average_strategy(j) == before[j];
            if inside {
                changed_inside |= !same;
            } else {
                assert!(same, "cambió un nodo fuera del subárbol: {p:?}");
            }
        }
        assert!(changed_inside);
        assert!(s.set_subtree(Some("BTN:R99")).is_err());
    }

    #[test]
    fn locks_on_the_same_node_combine() {
        let eq = matrix();
        let mut s = Solver::new(build_tree(PreflopConfig::push_fold(vec![10.0, 10.0, 10.0], Payoff::ChipEv)), eq.clone(), CheckDown { eq }, SolverConfig::default());
        let k = s.tree.find("BTN:A").unwrap();
        assert_eq!(s.tree.labels[k].len(), 2);
        let k = 0;
        let na = s.tree.actions[k].num_actions as usize;
        let a = na - 1;
        let half = vec![0.5f32; HAND_STRIDE];
        let full = vec![1.0f32; HAND_STRIDE];
        s.lock(k, a, &half);
        s.lock(k, a, &full);
        let st = s.average_strategy(k);
        assert!((st[a * HAND_STRIDE] - 1.0).abs() < 1e-6 && st[0].abs() < 1e-6);
        s.lock(k, a, &half);
        let st = s.average_strategy(k);
        assert!((st[a * HAND_STRIDE] - 0.5).abs() < 1e-6 && (st[0] - 0.5).abs() < 1e-6);
    }

    /// Mesa de 10: posiciones con UTG3, push/fold ICM que converge y conserva
    /// el premio total.
    #[test]
    fn ten_handed_table() {
        let stacks = vec![12.0, 25.0, 9.0, 30.0, 15.0, 18.0, 22.0, 7.0, 40.0, 14.0];
        let cfg = PreflopConfig { max_players_in_pot: 2, ..PreflopConfig::push_fold(stacks, Payoff::Icm { payouts: vec![40.0, 25.0, 15.0, 10.0, 6.0, 4.0] }) };
        let s = solve(cfg, 60);
        assert_eq!(crate::tree::builder::seat_names(10)[3], "UTG3");
        let ev = s.expected_values();
        assert!((ev.iter().sum::<f64>() - 100.0).abs() < 0.05, "{ev:?}");
        assert!(s.best_response_gains().iter().sum::<f64>() < 0.5);
    }

    fn fold_value_check<E: PostflopEstimator>(s: &Solver<E>, stack: f32, tol: f32) {
        let reach: Vec<Box<RangeVec>> = (0..s.tree.n).map(|_| full_range()).collect();
        let fold = s.tree.actions[0].children[0];
        let v = crate::with_pool!(s.cfg.threads, |pool| s.walk(fold, 0, &reach, Mode::Evaluate, 9, 1, pool));
        for h in 0..NUM_COMBOS {
            assert!((v[h] - stack).abs() < tol, "combo {h}: {} (esperado {stack})", v[h]);
        }
    }

    /// Botes a 4 (Monte Carlo de manos): consistencia exacta de la normalización,
    /// conservación del EV y convergencia.
    #[test]
    fn four_handed_push_fold_with_four_way_pots() {
        let cfg = PreflopConfig { max_players_in_pot: 4, shove_over_allin: false, ..PreflopConfig::push_fold(vec![8.0, 12.0, 10.0, 15.0], Payoff::ChipEv) };
        let s = solve(cfg, 200);
        assert!(s.tree.terminals.iter().any(|t| t.kind == TerminalKind::MultiwayN));
        fold_value_check(&s, 8.0, 2e-3);
        let ev = s.expected_values();
        let gains: f64 = s.best_response_gains().iter().sum();
        eprintln!("4-way EV {ev:?} NashConv {gains:.4}");
        assert!((ev.iter().sum::<f64>() - 45.0).abs() < 0.1, "{ev:?}");
        assert!(gains < 0.08, "NashConv {gains}");
        let k = s.tree.find("CO:A BTN:C SB:C").unwrap();
        assert!(class_frequency(&s.average_strategy(k), 0, "AA").unwrap() < 0.05);
    }

    /// Flops a 3 con MCCFR multiway: consistencia exacta y suma de fichas.
    #[test]
    fn three_way_postflop_mccfr() {
        let eq = matrix();
        let pool = BoardPool::build(&PoolConfig { boards: 8, buckets: [8, 8, 8], kmeans_samples: 3000, kmeans_iters: 5, seed: 5, threads: 4, stratified: true, river_ochs: false });
        let cfg = PreflopConfig {
            allow_limp: true,
            open_sizes: vec![crate::tree::RaiseSize::bb(2.5)],
            sb_open_sizes: vec![crate::tree::RaiseSize::bb(3.0)],
            reraise_mults: vec![],
            max_raises: 1,
            ..PreflopConfig::standard(vec![20.0, 20.0, 20.0], Payoff::ChipEv)
        };
        let mut s = Solver::new(build_tree(cfg), eq, PostflopEngine::new(pool, 16, 1), SolverConfig::default());
        assert!(s.tree.leaves.iter().any(|l| l.multi.is_some()));
        s.run(40);
        fold_value_check(&s, 20.0, 5e-3);
        let ev = s.expected_values();
        eprintln!("3-way postflop EV {ev:?} (infosets {})", s.est.table.export().len());
        assert!((ev.iter().sum::<f64>() - 60.0).abs() < 0.2, "{ev:?}");
        let multi_keys = s.est.table.export().iter().filter(|(k, _)| k >> 31 != 0 || (k >> 28) & 7 == 4).count();
        assert!(multi_keys > 0, "se entrenan infosets multiway");
    }

    /// ICM a 4 con MCCFR postflop (flops HU y a 3): los que no están en el bote se
    /// evalúan como escalares (ver `PostflopEngine::walk_joint`) y aun así la suma de los
    /// $EV de todos los asientos reproduce el premio total.
    #[test]
    fn icm_observers_conserve_prize_pool_with_postflop() {
        let eq = matrix();
        let pool = BoardPool::build(&PoolConfig { boards: 8, buckets: [8, 8, 8], kmeans_samples: 3000, kmeans_iters: 5, seed: 9, threads: 4, stratified: true, river_ochs: false });
        let cfg = PreflopConfig {
            allow_limp: true,
            open_sizes: vec![crate::tree::RaiseSize::bb(2.5)],
            sb_open_sizes: vec![crate::tree::RaiseSize::bb(3.0)],
            reraise_mults: vec![],
            max_raises: 1,
            ..PreflopConfig::standard(vec![15.0, 25.0, 20.0, 12.0], Payoff::Icm { payouts: vec![50.0, 30.0, 20.0] })
        };
        let mut s = Solver::new(build_tree(cfg), eq, PostflopEngine::new(pool, 16, 1), SolverConfig::default());
        assert!(s.tree.leaves.iter().any(|l| l.multi.is_some()) && s.tree.leaves.iter().any(|l| l.multi.is_none()));
        s.run(30);
        let ev = s.expected_values();
        eprintln!("ICM 4 jugadores con postflop: EV {ev:?}");
        assert!((ev.iter().sum::<f64>() - 100.0).abs() < 0.15, "{ev:?}");
    }

    /// Bunching: si el jugador que foldeó lo hizo solo con AA, los boards con as pesan
    /// menos y una mano como KK vale más en el flop (menos ases por salir). (Con AK/AQ en
    /// el rango foldeado también bajarían los boards con K o Q, que favorecen a KK, y el
    /// signo dependería del pool.)
    #[test]
    fn bunching_weights_boards_by_folded_ranges() {
        use crate::cfr::PostflopEstimator;
        let pool = BoardPool::build(&PoolConfig { boards: 256, buckets: [8, 8, 8], kmeans_samples: 3000, kmeans_iters: 5, seed: 7, threads: 4, stratified: true, river_ochs: false });
        let tree = build_tree(PreflopConfig::standard(vec![25.0; 3], Payoff::ChipEv));
        let leaf = (0..tree.leaves.len()).find(|&k| tree.leaves[k].multi.is_none() && !tree.leaves[k].nodes.is_empty()).unwrap();
        let l = &tree.leaves[leaf];
        let r = full_range();
        let mut engine = PostflopEngine::new(pool, 16, 1);
        let aces = crate::io::ranges::parse_range("AA").unwrap();
        let mut aces_v = zero_range();
        aces_v.copy_from_slice(&aces);
        let kk = crate::cfr::report::class_frequency;
        engine.bunching = false;
        let plain = crate::with_pool!(1, |pool| engine.values_all(&tree, leaf, &[&r, &r], StrategyMode::Average, 0, 1, &[], &[&aces_v], pool));
        engine.bunching = true;
        let bunched = crate::with_pool!(1, |pool| engine.values_all(&tree, leaf, &[&r, &r], StrategyMode::Average, 0, 1, &[], &[&aces_v], pool));
        let (a, b) = (kk(&plain[0][0][..], 0, "KK").unwrap(), kk(&bunched[0][0][..], 0, "KK").unwrap());
        let (c, d) = (kk(&plain[0][0][..], 0, "22").unwrap(), kk(&bunched[0][0][..], 0, "22").unwrap());
        eprintln!("KK {a} → {b}, 22 {c} → {d} (oop seat {})", l.oop);
        assert!(b > a, "KK debe mejorar con menos ases en el mazo: {a} vs {b}");
        assert!((b - a) > (d - c), "el efecto es mayor en KK que en 22");
    }

    /// Conservación de fichas en una hoja de flop a 3: con chip EV, la suma sobre los tres
    /// jugadores de su stack final esperado (cada uno condicionado a su propia masa) debe
    /// ser la suma de stacks. `cargo test --release three_way_leaf_value -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn three_way_leaf_values_conserve_chips() {
        use crate::cfr::PostflopEstimator;
        let nb: usize = std::env::var("NB").ok().and_then(|x| x.parse().ok()).unwrap_or(512);
        let pool = BoardPool::build(&PoolConfig { boards: nb, buckets: [8, 8, 8], kmeans_samples: 3000, kmeans_iters: 5, seed: 7, threads: 4, stratified: true, river_ochs: false });
        let tree = build_tree(PreflopConfig::standard(vec![25.0, 40.0, 15.0], Payoff::ChipEv));
        // Con un solo board, los rangos no tienen combos que choquen con él (conservación exacta
        // por board).
        let bmask = if nb == 1 { pool.boards[0].mask } else { 0 };
        let mut engine = PostflopEngine::new(pool, 16, 1);
        engine.bunching = false;
        let mut rng = SplitMix64::new(3);
        let leaves: Vec<usize> = (0..tree.leaves.len()).filter(|&k| tree.leaves[k].multi.is_some()).collect();
        eprintln!("hojas a 3: {}", leaves.len());
        for &leaf in leaves.iter().take(1) {
            let k = tree.leaves[leaf].multi.as_ref().unwrap().seats.len();
            let seats: Vec<usize> = tree.leaves[leaf].multi.as_ref().unwrap().seats.iter().map(|&x| x as usize).collect();
            for density in [1.0f32] {
                let rs: Vec<Box<RangeVec>> = (0..k)
                    .map(|_| {
                        let mut r = zero_range();
                        for h in 0..NUM_COMBOS {
                            r[h] = if rng.next_f32() < density && crate::eval::COMBO_MASKS[h] & bmask == 0 { 0.1 + rng.next_f32() } else { 0.0 };
                        }
                        r
                    })
                    .collect();
                let rk: Vec<&RangeVec> = rs.iter().map(|r| &**r).collect();
                let v = crate::with_pool!(1, |pool| engine.values_all(&tree, leaf, &rk, StrategyMode::Average, 0, 1, &[], &[], pool));
                let mut tot = 0.0f64;
                let mut parts = Vec::new();
                // Masa de b compatible con un conjunto de cartas (inclusión-exclusión).
                let card_sums = |r: &RangeVec| -> ([f64; 52], f64) {
                    let mut sc = [0.0f64; 52];
                    let mut t = 0.0f64;
                    for h in 0..NUM_COMBOS {
                        let [c1, c2] = crate::eval::COMBO_CARDS[h];
                        sc[c1 as usize] += r[h] as f64;
                        sc[c2 as usize] += r[h] as f64;
                        t += r[h] as f64;
                    }
                    (sc, t)
                };
                let combo_of = |a: u8, b: u8| -> usize { (0..NUM_COMBOS).find(|&h| { let [x, y] = crate::eval::COMBO_CARDS[h]; (x == a && y == b) || (x == b && y == a) }).unwrap() };
                let mut idx = vec![[0usize; 52]; 52];
                for a in 0..52u8 {
                    for b in a + 1..52u8 {
                        let h = combo_of(a, b);
                        idx[a as usize][b as usize] = h;
                        idx[b as usize][a as usize] = h;
                    }
                }
                for p in 0..k {
                    let others: Vec<usize> = (0..k).filter(|&j| j != p).collect();
                    let (ra, rb) = (&rs[others[0]], &rs[others[1]]);
                    let (sc, tb) = card_sums(rb);
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for h in 0..NUM_COMBOS {
                        if rs[p][h] <= 0.0 {
                            continue;
                        }
                        let hm = crate::eval::COMBO_MASKS[h];
                        let mut pm = 0.0f64;
                        for a in 0..NUM_COMBOS {
                            if ra[a] <= 0.0 || crate::eval::COMBO_MASKS[a] & hm != 0 {
                                continue;
                            }
                            let cs: Vec<usize> = crate::eval::COMBO_CARDS[h].iter().chain(crate::eval::COMBO_CARDS[a].iter()).map(|&c| c as usize).collect();
                            let mut c = tb;
                            for &x in &cs {
                                c -= sc[x];
                            }
                            for i in 0..4 {
                                for j in i + 1..4 {
                                    c += rb[idx[cs[i]][cs[j]]] as f64;
                                }
                            }
                            pm += ra[a] as f64 * c;
                        }
                        let m: f64 = others.iter().map(|&j| (BlockerSums::new(&rs[j][..]).compatible(&rs[j][..], h) / COMPATIBLE_COMBOS) as f64).product();
                        if m <= 0.0 {
                            continue;
                        }
                        // Valor condicionado a h por la masa conjunta exacta de los otros dos.
                        let w = rs[p][h] as f64 * pm;
                        num += w * v[p][0][h] as f64 / m;
                        den += w;
                    }
                    parts.push(num / den);
                    tot += num / den;
                }
                let stacks: f64 = seats.iter().map(|&s| tree.cfg.stacks[s]).sum();
                eprintln!("hoja {leaf} asientos {seats:?} densidad {density}: Σ stacks finales {tot:.4} (debe {stacks}) {parts:?}");
            }
        }
    }

    /// Bloqueo de frecuencias: forzar al SB a empujar el 40 % (en vez del 58 % Nash) y que
    /// CFR elija las mejores manos; la BB responde de forma óptima a ese rango.
    #[test]
    fn frequency_lock_via_edge() {
        let eq = matrix();
        let mut s = Solver::new(build_tree(PreflopConfig::push_fold(vec![10.0, 10.0], Payoff::ChipEv)), eq.clone(), CheckDown { eq }, SolverConfig::default());
        s.lock_frequency(0, 1, 0.40);
        s.run(600);
        let sb = s.average_strategy(0);
        let f = overall_frequency(&sb, 1);
        eprintln!("push {f:.3} edge {:?}", s.edges[&0]);
        assert!((f - 0.40).abs() < 0.03, "frecuencia {f}");
        assert!(class_frequency(&sb, 1, "AA").unwrap() > 0.95 && class_frequency(&sb, 1, "K5o").unwrap() < 0.05);
        assert!(s.edges[&0][1] > 0.0, "empujar debe encarecerse");
        // Contra un rango más estrecho, la BB paga menos que frente al Nash (37.7 %).
        let bb = s.average_strategy(s.tree.find("SB:A").unwrap());
        assert!(overall_frequency(&bb, 1) < 0.34);
    }

    /// Node locking: si el SB empuja solo AA-QQ (explotable), la BB paga mucho más
    /// estrecho que en Nash y el SB pierde EV respecto al equilibrio.
    #[test]
    fn node_lock_exploitation() {
        let eq = matrix();
        let mut s = Solver::new(build_tree(PreflopConfig::push_fold(vec![10.0, 10.0], Payoff::ChipEv)), eq.clone(), CheckDown { eq }, SolverConfig::default());
        let tight = crate::io::ranges::parse_range("QQ+").unwrap();
        s.lock(0, 1, &tight);
        s.run(300);
        let bb = s.average_strategy(s.tree.find("SB:A").unwrap());
        assert!(class_frequency(&bb, 1, "AQs").unwrap() < 0.05, "BB solo paga lo que bate QQ+");
        assert!(class_frequency(&bb, 1, "KK").unwrap() > 0.95);
        assert!((overall_frequency(&s.average_strategy(0), 1) - 18.0 / 1326.0).abs() < 1e-6);
        let ev_sb = s.expected_values()[0];
        assert!(ev_sb < 9.9, "empujar solo QQ+ pierde ciegas: {ev_sb}");
    }

    /// Guardar y cargar reproduce exactamente la estrategia y permite continuar.
    #[test]
    fn save_and_load_roundtrip() {
        let s = solve(PreflopConfig::push_fold(vec![10.0, 15.0, 12.0], Payoff::ChipEv), 60);
        let path = std::env::temp_dir().join(format!("pfsol_test_{}.bin", std::process::id()));
        s.save(&path).unwrap();
        let eq = matrix();
        let mut t = Solver::new(build_tree(PreflopConfig::push_fold(vec![10.0, 15.0, 12.0], Payoff::ChipEv)), eq.clone(), CheckDown { eq: eq.clone() }, SolverConfig::default());
        t.load(&path).unwrap();
        assert_eq!(t.iteration, 60);
        for k in 0..s.tree.actions.len() {
            assert_eq!(s.average_strategy(k), t.average_strategy(k));
        }
        let mut other = Solver::new(build_tree(PreflopConfig::push_fold(vec![10.0, 15.0, 13.0], Payoff::ChipEv)), eq.clone(), CheckDown { eq }, SolverConfig::default());
        assert!(other.load(&path).is_err(), "otro árbol");
        let _ = std::fs::remove_file(path);
    }

    /// La pasada única de EV/BR coincide con los recorridos por jugador.
    #[test]
    fn single_pass_evaluation_matches_per_player() {
        let s = solve(PreflopConfig::push_fold(vec![10.0, 15.0, 12.0], Payoff::Icm { payouts: vec![50.0, 30.0, 20.0] }), 100);
        let a = s.best_response_gains();
        let b = s.best_response_gains_per_player();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-3, "{a:?} vs {b:?}");
        }
    }

    /// Push/fold a 3 con overcalls (botes multiway con side pots).
    #[test]
    fn three_handed_push_fold_with_overcalls() {
        let s = solve(PreflopConfig { shove_over_allin: false, ..PreflopConfig::push_fold(vec![10.0, 15.0, 12.0], Payoff::ChipEv) }, 300);
        let ev = s.expected_values();
        let gains: f64 = s.best_response_gains().iter().sum();
        let over = s.tree.find("BTN:A SB:C").unwrap();
        let strat = s.average_strategy(over);
        // Acciones F / C / A (la BB puede resubir para jugar el lateral contra la SB).
        let enter = 1.0 - overall_frequency(&strat, 0);
        eprintln!("EV {ev:?} NashConv {gains:.4} entra {enter:.3}");
        // Con 3+ jugadores el bloqueo solo se modela entre cada jugador y cada rival (no
        // entre rivales): la suma de EV se conserva salvo ~0.4 % (ver doc del módulo).
        assert!((ev.iter().sum::<f64>() - 37.0).abs() < 0.2, "{ev:?}");
        assert!(gains < 0.05, "NashConv {gains}");
        assert!(class_frequency(&strat, 0, "AA").unwrap() < 0.05);
        assert!(class_frequency(&strat, 0, "72o").unwrap() > 0.95);
        // Entrar en un bote a 3 es más estrecho que pagar HU el mismo shove.
        let hu = overall_frequency(&s.average_strategy(s.tree.find("BTN:A SB:F").unwrap()), 1);
        assert!(enter < hu, "{enter} vs {hu}");
    }

    /// VR-MCCFR: con la estrategia fija (sin updates), la media de muchas visitas con 4
    /// boards y variable de control coincide con el valor exacto sobre el pool entero
    /// (insesgado) y varía menos entre visitas que sin ella.
    #[test]
    fn vr_leaf_values_are_unbiased_and_less_noisy() {
        let pool = BoardPool::build(&PoolConfig { boards: 64, buckets: [8, 8, 8], kmeans_samples: 4000, kmeans_iters: 6, seed: 9, threads: 4, stratified: true, river_ochs: false });
        let mut est = PostflopEngine::new(pool, 16, 64);
        let cfg = PreflopConfig::standard(vec![25.0, 25.0], Payoff::ChipEv);
        let tree = build_tree(cfg);
        let leaf = (0..tree.leaves.len()).find(|&l| tree.leaves[l].multi.is_none() && !tree.leaves[l].nodes.is_empty()).expect("hoja HU");
        let mut rng = SplitMix64::new(5);
        let rs: Vec<Box<RangeVec>> = (0..2)
            .map(|_| {
                let mut r = crate::cfr::zero_range();
                for h in 0..NUM_COMBOS {
                    r[h] = if rng.next_f32() < 0.5 { 0.2 + rng.next_f32() } else { 0.0 };
                }
                r
            })
            .collect();
        let reaches: Vec<&RangeVec> = rs.iter().map(|r| &**r).collect();
        crate::with_pool!(4, |pool| {
            // Exacto: todos los boards del pool.
            let exact = est.joint_values(&tree, leaf, &reaches, StrategyMode::Current, 0, 1, 1, &[], &[], pool)[0][0].clone();
            let mut stats = Vec::new();
            for vr in [false, true] {
                est.vr = vr;
                est.boards_per_visit = 4;
                let reps = 400;
                let (mut mean, mut sq) = (vec![0.0f64; NUM_COMBOS], vec![0.0f64; NUM_COMBOS]);
                for s in 0..reps {
                    let v = &est.joint_values(&tree, leaf, &reaches, StrategyMode::Current, 0, 1, 100 + s, &[], &[], pool)[0][0];
                    for h in 0..NUM_COMBOS {
                        mean[h] += v[h] as f64 / reps as f64;
                        sq[h] += (v[h] as f64).powi(2) / reps as f64;
                    }
                }
                let (mut bias, mut var, mut scale) = (0.0f64, 0.0f64, 0.0f64);
                for h in 0..NUM_COMBOS {
                    if rs[0][h] > 0.0 {
                        bias += (mean[h] - exact[h] as f64).abs();
                        var += sq[h] - mean[h] * mean[h];
                        scale += (exact[h] as f64).abs();
                    }
                }
                eprintln!("vr {vr}: sesgo relativo {:.4} %  varianza {var:.4}", 100.0 * bias / scale);
                stats.push((bias / scale, var));
            }
            assert!(stats[1].0 < 0.01, "VR sesgado: {:.4}", stats[1].0);
            assert!(stats[1].1 < 0.7 * stats[0].1, "VR no reduce la varianza: {:?}", stats);
        });
    }

    /// Árbol completo con opens, 3-bets y MCCFR postflop geométrico: comprobaciones de
    /// cordura (valores finitos, suma constante, estrategias razonables).
    #[test]
    fn postflop_engine_end_to_end() {
        let eq = matrix();
        let pool = BoardPool::build(&PoolConfig { boards: 12, buckets: [8, 8, 8], kmeans_samples: 4000, kmeans_iters: 6, seed: 3, threads: 4, stratified: true, river_ochs: false });
        let est = PostflopEngine::new(pool, 16, 1);
        let cfg = PreflopConfig::standard(vec![25.0, 25.0], Payoff::ChipEv);
        let mut s = Solver::new(build_tree(cfg), eq, est, SolverConfig::default());
        s.run(60);
        let ev = s.expected_values();
        eprintln!("EV {ev:?} nodes {} leaves {}", s.tree.actions.len(), s.tree.leaves.len());
        assert!(ev.iter().all(|x| x.is_finite()));
        assert!((ev[0] + ev[1] - 50.0).abs() < 0.5, "suma de fichas {ev:?}");
        let root = s.average_strategy(0);
        let labels = &s.tree.labels[0];
        let fold = labels.iter().position(|l| l == "F").unwrap();
        assert!(class_frequency(&root, fold, "AA").unwrap() < 0.05, "AA no debe foldear el SB");
    }
}

