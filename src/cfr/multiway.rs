//! # Botes a 3: probabilidades de orden por board
//!
//! Un terminal `Multiway` guarda `U[orden][asiento]` para los 6 órdenes de fuerza de las
//! manos de sus tres jugadores (el reparto de side pots y las eliminaciones ya están
//! dentro de `U`). Falta `P(orden | h)` para cada combo del héroe contra los rangos de los
//! otros dos: una tabla 1326³ es inviable, así que se calcula por board en O(n):
//!
//! * **Marginales exactas con bloqueo del héroe**: `P(A < h)` y `P(B < h)` salen del
//!   barrido de `showdown_wtl` (inclusión-exclusión de las cartas de `h`), con empates
//!   repartidos a medias.
//! * **Independencia entre rivales**: `P(A<h, B<h) = P(A<h)·P(B<h)` (se ignora el bloqueo
//!   A↔B, la misma aproximación que el resto del solver multijugador).
//! * **Orden entre los rivales** cuando ambos quedan por debajo (o por encima) del héroe:
//!   se reparte con los términos cruzados `G_AB(h) = Σ_{a<h} p_A(a)·F_B(a)` y sus
//!   sufijos. Son sumas por debajo/encima del héroe de vectores ponderados, así que salen
//!   del mismo barrido `showdown_wtl`, también con bloqueo exacto del héroe.
//!
//! Las seis probabilidades suman 1 exactamente por construcción. El valor por combo se
//! promedia sobre boards con el estimador autonormalizado (peso = masa rival válida en el
//! board) y se escala por la probabilidad real de que ambos rivales lleguen.
//! Entrenamiento: `boards_per_visit` runouts aleatorios del pool (Monte Carlo).
//! Evaluación: los primeros `eval_boards`, deterministas.

use super::{RangeVec, StrategyMode};
use crate::eval::equity::BlockerSums;
use crate::eval::showdown::{sample_ranked_boards, showdown_wtl, RankedBoard};
use crate::eval::{COMBO_CARDS, COMPATIBLE_COMBOS, HAND_STRIDE, NUM_COMBOS};
use crate::math::rng::SplitMix64;
use crate::tree::nodes::{pattern3_index, perm_index, TerminalNode, PATTERNS3};

#[derive(Debug)]
pub struct MultiwayEval {
    pub pool: Vec<RankedBoard>,
    pub boards_per_visit: usize,
    pub eval_boards: usize,
    /// Matriz de equity exacta HU: variable de control al evaluar (ver [`ControlVariate`]).
    pub eq: Option<std::sync::Arc<crate::eval::equity::EquityMatrix>>,
    /// Boards por visita proporcionales a la masa de los rivales (ver [`Self::boards`]).
    pub adaptive: bool,
}

/// Masa de los rivales a partir de la cual un bote multiway recibe todos sus boards por
/// visita en entrenamiento, y mínimo de boards.
pub const MW_MASS_REF: f64 = 0.01;
pub const MW_MIN_BOARDS: usize = 4;

/// Producto de las fracciones de combos con que llegan los rangos.
fn reach_mass(rs: &[&RangeVec]) -> f64 {
    rs.iter().map(|r| r[..NUM_COMBOS].iter().map(|&x| x as f64).sum::<f64>() / NUM_COMBOS as f64).product()
}

/// Buffers de [`board_orders_with`], reutilizables entre boards (evita reservar y poner a
/// cero ~20 vectores de 1328 por board).
#[derive(Debug)]
pub struct OrderScratch {
    fa: Vec<f32>,
    fb: Vec<f32>,
    eb: Vec<f32>,
    x: [Vec<f32>; 3],
    s: [[Vec<f32>; 3]; 5],
    over: [Vec<f32>; 3],
    groups: Vec<(u16, u16)>,
}

impl Default for OrderScratch {
    fn default() -> Self {
        let z = || vec![0.0f32; HAND_STRIDE];
        OrderScratch { fa: z(), fb: z(), eb: z(), x: [z(), z(), z()], s: std::array::from_fn(|_| [z(), z(), z()]), over: [z(), z(), z()], groups: Vec::new() }
    }
}

/// Probabilidades de los 13 resultados de [`PATTERNS3`] (posiciones relativas: 0 =
/// héroe, 1 = A, 2 = B; con empates) para cada combo del héroe en un board, y su peso
/// (masa rival válida).
pub fn board_orders(board: &RankedBoard, ra: &RangeVec, rb: &RangeVec, probs: &mut [[f32; 13]], weight: &mut [f32]) {
    board_orders_with(board, ra, rb, probs, weight, &mut OrderScratch::default());
}

/// Como [`board_orders`] con buffers reutilizables. Solo lee de `ra`/`rb` los combos
/// válidos en el board.
///
/// Frente al héroe, cada rival está por debajo, empatado o por encima (barrido
/// `showdown_wtl`, con el bloqueo exacto del héroe). Cuando los dos quedan del mismo lado,
/// su orden relativo (A > B, A = B, A < B) sale de términos cruzados: sumas por debajo /
/// por encima del héroe de `p_A(x)·P(B < x)`, `p_B(x)·P(A < x)` y `p_A(x)·P(B = x)`
/// (sin bloqueo entre rivales). Los empates cuentan aparte: con ICM, repartir un bote no
/// vale lo mismo que jugárselo a cara o cruz.
pub fn board_orders_with(board: &RankedBoard, ra: &RangeVec, rb: &RangeVec, probs: &mut [[f32; 13]], weight: &mut [f32], sc: &mut OrderScratch) {
    let order = &board.order;
    let ranks = &board.ranks;
    let tot_a: f32 = order.iter().map(|&h| ra[h as usize]).sum();
    let tot_b: f32 = order.iter().map(|&h| rb[h as usize]).sum();
    if tot_a <= 0.0 || tot_b <= 0.0 {
        order.iter().for_each(|&h| weight[h as usize] = 0.0);
        return;
    }
    // CDF estricta de un rival en el rango de cada combo, P(rival < x), y masa de su mismo
    // rango, P(rival = x) (sin bloqueo).
    let cdf = |r: &RangeVec, tot: f32, lt: &mut [f32], eq: Option<&mut [f32]>| {
        let (mut cum, mut i) = (0.0f32, 0);
        let mut eq = eq;
        while i < order.len() {
            let rank = ranks[order[i] as usize];
            let mut j = i;
            let mut g = 0.0;
            while j < order.len() && ranks[order[j] as usize] == rank {
                g += r[order[j] as usize];
                j += 1;
            }
            for &x in &order[i..j] {
                lt[x as usize] = cum / tot;
                if let Some(e) = eq.as_deref_mut() {
                    e[x as usize] = g / tot;
                }
            }
            cum += g;
            i = j;
        }
    };
    cdf(ra, tot_a, &mut sc.fa, None);
    cdf(rb, tot_b, &mut sc.fb, Some(&mut sc.eb));
    let [xab, xba, xeq] = &mut sc.x;
    for &x in order.iter() {
        let x = x as usize;
        xab[x] = ra[x] / tot_a * sc.fb[x]; // Σ_{a<h} = P(B < A < h)
        xba[x] = rb[x] / tot_b * sc.fa[x]; // Σ_{b<h} = P(A < B < h)
        xeq[x] = ra[x] / tot_a * sc.eb[x]; // Σ_{a<h} = P(A = B < h)
    }
    let [sa, sb, sab, sba, seq] = &mut sc.s;
    let wtl = |r: &[f32], o: &mut [Vec<f32>; 3]| {
        let [w, t, l] = o;
        showdown_wtl(order, ranks, r, w, t, l);
    };
    wtl(&ra[..], sa);
    wtl(&rb[..], sb);
    wtl(&sc.x[0], sab);
    wtl(&sc.x[1], sba);
    wtl(&sc.x[2], seq);
    let ([wa, ta, la], [wb, tb, lb]) = (&*sa, &*sb);
    let ([w_ab, _, l_ab], [w_ba, _, l_ba], [w_eq, _, l_eq]) = (&*sab, &*sba, &*seq);
    let (ia, ib) = (1.0 / tot_a, 1.0 / tot_b);
    // Reparto de una región entre (A > B, A = B, B > A).
    let split = |gt: f32, eq: f32, lt: f32| {
        let (gt, eq, lt) = (gt.max(0.0), eq.max(0.0), lt.max(0.0));
        let s = gt + eq + lt;
        if s > 0.0 { (gt / s, eq / s, lt / s) } else { (0.5, 0.0, 0.5) }
    };
    for &h in order.iter() {
        let h = h as usize;
        let (na, nb) = (wa[h] + ta[h] + la[h], wb[h] + tb[h] + lb[h]);
        weight[h] = na * nb;
        if weight[h] <= 0.0 {
            continue;
        }
        let (a_lt, a_eq, a_gt) = (wa[h] / na, ta[h] / na, la[h] / na); // A frente al héroe
        let (b_lt, b_eq, b_gt) = (wb[h] / nb, tb[h] / nb, lb[h] / nb);
        // Ambos por debajo: P(B < A < h), P(A = B < h), P(A < B < h).
        let (d_ab, d_eq, d_ba) = split(w_ab[h], w_eq[h], w_ba[h]);
        // Ambos por encima: P(A > B > h), P(A = B > h), P(B > A > h).
        let (u_ab, u_eq, u_ba) = split(lb[h] * ib - l_ba[h] - l_eq[h], l_eq[h], la[h] * ia - l_ab[h] - l_eq[h]);
        let (below, above) = (a_lt * b_lt, a_gt * b_gt);
        let p = &mut probs[h];
        p[0] = below * d_ab; //   h > A > B
        p[1] = below * d_ba; //   h > B > A
        p[2] = a_gt * b_lt; //    A > h > B
        p[3] = above * u_ab; //   A > B > h
        p[4] = b_gt * a_lt; //    B > h > A
        p[5] = above * u_ba; //   B > A > h
        p[6] = a_eq * b_lt; //    h = A > B
        p[7] = b_eq * a_lt; //    h = B > A
        p[8] = above * u_eq; //   A = B > h
        p[9] = below * d_eq; //   h > A = B
        p[10] = a_gt * b_eq; //   A > h = B
        p[11] = b_gt * a_eq; //   B > h = A
        p[12] = a_eq * b_eq; //   triple
    }
    ab_blocking(board, ra, rb, probs, weight, sc);
}

/// Bloqueo entre los dos rivales, que el cálculo anterior trata como independientes: a la
/// masa de cada patrón se le resta la de los pares `(a, b)` que comparten carta, con el
/// mismo orden de rangos. Esos pares se cuentan por carta compartida (`Σ_c X[c]·Y[c]` con
/// `X`, `Y` las sumas por carta de cada rival en la región del patrón) y se quitan las
/// cartas del héroe, cuyos pares ya estaban excluidos. Queda sin tratar, de tercer orden,
/// un par que comparte una carta cuando uno de los dos lleva además la otra del héroe.
/// Con rangos cargados de ases (push y call), la equity del héroe con un as bajaba hasta
/// 0,5 puntos (A8s contra push y call de 10 bb: 29,99 % → 29,52 %, exacto 29,52 %).
fn ab_blocking(board: &RankedBoard, ra: &RangeVec, rb: &RangeVec, probs: &mut [[f32; 13]], weight: &mut [f32], sc: &mut OrderScratch) {
    let order = &board.order;
    let ranks = &board.ranks;
    // Grupos de combos con el mismo rango, en orden ascendente.
    let groups = &mut sc.groups;
    groups.clear();
    let mut i = 0;
    while i < order.len() {
        let rank = ranks[order[i] as usize];
        let mut j = i;
        while j < order.len() && ranks[order[j] as usize] == rank {
            j += 1;
        }
        groups.push((i as u16, j as u16));
        i = j;
    }
    let cards = |x: u16| {
        let [a, b] = COMBO_CARDS[x as usize];
        (a as usize, b as usize)
    };
    // Sumas por carta del grupo (dispersas: solo se tocan y se limpian sus cartas).
    let (mut ga, mut gb) = ([0.0f32; 52], [0.0f32; 52]);
    let mut touched: Vec<usize> = Vec::with_capacity(64);
    let load = |g: (u16, u16), ga: &mut [f32; 52], gb: &mut [f32; 52], touched: &mut Vec<usize>| {
        for &c in touched.iter() {
            ga[c] = 0.0;
            gb[c] = 0.0;
        }
        touched.clear();
        for &x in &order[g.0 as usize..g.1 as usize] {
            let (a, b) = cards(x);
            for c in [a, b] {
                if !touched.contains(&c) {
                    touched.push(c);
                }
                ga[c] += ra[x as usize];
                gb[c] += rb[x as usize];
            }
        }
    };
    let sdot = |t: &[usize], x: &[f32; 52], y: &[f32; 52]| -> f32 { t.iter().map(|&c| x[c] * y[c]).sum() };
    // Pares iguales (a = b) del grupo, por carta (`ge`) y en total: comparten las dos
    // cartas y Σ_c G_A·G_B los cuenta dos veces. Los que no tocan al héroe:
    // total − ge[c1] − ge[c2] + (el propio h).
    let mut ge = [0.0f32; 52];
    // Añade al acumulador por carta `t` los pares nuevos de un grupo que pasa al lado ya
    // barrido (`sa`, `sb`: sumas por carta de ese lado) y devuelve lo añadido a cada total.
    let absorb = |g: (u16, u16), t: &mut [[f32; 52]; 3], ts: &mut [f32; 3], sa: &mut [f32; 52], sb: &mut [f32; 52], ga: &[f32; 52], gb: &[f32; 52], touched: &[usize], a_first: bool| {
        for &x in &order[g.0 as usize..g.1 as usize] {
            let (a, b) = cards(x);
            let (xa, xb) = (ra[x as usize], rb[x as usize]);
            // `a_first`: barrido ascendente (A del grupo por encima de B ya barrido).
            let (k0, k1) = if a_first { (xa, xb) } else { (xb, xa) };
            let (s0, s1) = if a_first { (&*sb, &*sa) } else { (&*sa, &*sb) };
            for c in [a, b] {
                let (d0, d1, d2) = (k0 * s0[c], k1 * s1[c], -0.5 * xa * xb);
                t[0][c] += d0;
                t[1][c] += d1;
                t[2][c] += d2;
                ts[0] += d0;
                ts[1] += d1;
                ts[2] += d2;
            }
        }
        for &c in touched {
            let d = ga[c] * gb[c];
            t[2][c] += d;
            ts[2] += d;
            sa[c] += ga[c];
            sb[c] += gb[c];
        }
    };
    // Barrido descendente: pares con los dos rivales por encima del héroe.
    // `over[0]`: A > B > h, `over[1]`: B > A > h, `over[2]`: A = B > h.
    {
        let over = &mut sc.over;
        let (mut ua, mut ub) = ([0.0f32; 52], [0.0f32; 52]);
        let mut t = [[0.0f32; 52]; 3];
        let mut ts = [0.0f32; 3];
        for gi in (0..groups.len()).rev() {
            let g = groups[gi];
            load(g, &mut ga, &mut gb, &mut touched);
            for &h in &order[g.0 as usize..g.1 as usize] {
                let (c1, c2) = cards(h);
                for k in 0..3 {
                    over[k][h as usize] = ts[k] - t[k][c1] - t[k][c2];
                }
            }
            absorb(g, &mut t, &mut ts, &mut ua, &mut ub, &ga, &gb, &touched, false);
        }
    }
    // Barrido ascendente: el resto de patrones. `U = total − L − G` sin materializarla:
    // `U_A·L_B = T_A·L_B − L_A·L_B − G_A·L_B`, con `T_A·L_B` y `L_A·L_B` incrementales.
    let (mut tot_a, mut tot_b) = ([0.0f32; 52], [0.0f32; 52]);
    for &h in order.iter() {
        let (a, b) = cards(h);
        tot_a[a] += ra[h as usize];
        tot_a[b] += ra[h as usize];
        tot_b[a] += rb[h as usize];
        tot_b[b] += rb[h as usize];
    }
    let (mut la, mut lb) = ([0.0f32; 52], [0.0f32; 52]);
    let (mut ta_lb, mut tb_la, mut la_lb) = (0.0f32, 0.0f32, 0.0f32);
    let mut t = [[0.0f32; 52]; 3];
    let mut ts = [0.0f32; 3];
    for gi in 0..groups.len() {
        let g = groups[gi];
        for &c in touched.iter() {
            ge[c] = 0.0;
        }
        load(g, &mut ga, &mut gb, &mut touched);
        let mut ge_tot = 0.0f32;
        for &x in &order[g.0 as usize..g.1 as usize] {
            let (a, b) = cards(x);
            let e = ra[x as usize] * rb[x as usize];
            ge[a] += e;
            ge[b] += e;
            ge_tot += e;
        }
        let (ga_lb, gb_la, ga_gb) = (sdot(&touched, &ga, &lb), sdot(&touched, &gb, &la), sdot(&touched, &ga, &gb));
        let (ta_gb, tb_ga) = (sdot(&touched, &tot_a, &gb), sdot(&touched, &tot_b, &ga));
        let (la_gb, lb_ga) = (sdot(&touched, &la, &gb), sdot(&touched, &lb, &ga));
        let ua_lb = ta_lb - la_lb - ga_lb; // U_A·L_B
        let ub_la = tb_la - la_lb - gb_la; // U_B·L_A
        let ua_gb = ta_gb - la_gb - ga_gb; // U_A·G_B
        let ub_ga = tb_ga - lb_ga - ga_gb; // U_B·G_A
        for &h in &order[g.0 as usize..g.1 as usize] {
            let hu = h as usize;
            if weight[hu] <= 0.0 {
                continue;
            }
            let (c1, c2) = cards(h);
            let (ua1, ua2) = (tot_a[c1] - la[c1] - ga[c1], tot_a[c2] - la[c2] - ga[c2]);
            let (ub1, ub2) = (tot_b[c1] - lb[c1] - gb[c1], tot_b[c2] - lb[c2] - gb[c2]);
            let mut o = [0.0f32; 13];
            o[0] = ts[0] - t[0][c1] - t[0][c2]; //                    h > A > B
            o[1] = ts[1] - t[1][c1] - t[1][c2]; //                    h > B > A
            o[2] = ua_lb - ua1 * lb[c1] - ua2 * lb[c2]; //            A > h > B
            o[3] = sc.over[0][hu]; //                                 A > B > h
            o[4] = ub_la - ub1 * la[c1] - ub2 * la[c2]; //            B > h > A
            o[5] = sc.over[1][hu]; //                                 B > A > h
            o[6] = ga_lb - ga[c1] * lb[c1] - ga[c2] * lb[c2]; //      h = A > B
            o[7] = gb_la - gb[c1] * la[c1] - gb[c2] * la[c2]; //      h = B > A
            o[8] = sc.over[2][hu]; //                                 A = B > h
            o[9] = ts[2] - t[2][c1] - t[2][c2]; //                    h > A = B
            o[10] = ua_gb - ua1 * gb[c1] - ua2 * gb[c2]; //           A > h = B
            o[11] = ub_ga - ub1 * ga[c1] - ub2 * ga[c2]; //           B > h = A
            o[12] = ga_gb - ga[c1] * gb[c1] - ga[c2] * gb[c2] - (ge_tot - ge[c1] - ge[c2] + ra[hu] * rb[hu]); // triple
            let w = weight[hu];
            let p = &mut probs[hu];
            let mut tot = 0.0f32;
            for k in 0..13 {
                p[k] = (p[k] * w - o[k]).max(0.0);
                tot += p[k];
            }
            weight[hu] = tot;
            if tot > 0.0 {
                p.iter_mut().for_each(|x| *x /= tot);
            }
        }
        // El grupo pasa a estar por debajo.
        ta_lb += ta_gb;
        tb_la += tb_ga;
        la_lb += la_gb + lb_ga + ga_gb;
        absorb(g, &mut t, &mut ts, &mut la, &mut lb, &ga, &gb, &touched, true);
    }
}

/// Rangos de los que foldearon, para el bunching conjunto board-manos de los all-in
/// multiway: `(sumas del rango completo, rango, masa total)`. `marginal`: los rangos del bote
/// ya llevan el factor marginal `P_k(sin cartas de x)` (entonces se divide por él).
#[allow(missing_debug_implementations)]
pub struct JointFolds<'a> {
    pub folds: Vec<(BlockerSums, &'a RangeVec, f32)>,
    pub marginal: bool,
}

impl JointFolds<'_> {
    /// Factor por combo en un board: `Π_k P_k(sin cartas del board ni de x) / P_k(sin cartas
    /// del board)` (÷ el marginal si ya está aplicado). Los combos que chocan con el board no
    /// se usan.
    pub fn factor(&self, board: &RankedBoard) -> Vec<f32> {
        let mut f = vec![1.0f32; NUM_COMBOS];
        for (full, r, tot) in &self.folds {
            let mut masked = super::zero_range();
            for &h in board.order.iter() {
                masked[h as usize] = r[h as usize];
            }
            let mb = BlockerSums::new(&masked[..]);
            if mb.total <= 0.0 {
                continue;
            }
            for &h in board.order.iter() {
                let h = h as usize;
                let on = mb.compatible(&masked[..], h) / mb.total;
                let x = if self.marginal {
                    let mg = full.compatible(&r[..], h) / tot;
                    if mg > 0.0 { on / mg } else { 0.0 }
                } else {
                    on
                };
                f[h] *= x;
            }
        }
        f
    }
}

/// Variable de control para evaluar botes multiway con boards fijos: por board, la masa de
/// cada rival a la que el héroe gana heads-up (`N_b = Σ_a r(a)·[h gana a a] + ½ empates`,
/// solo bloqueo héroe-rival). Su media exacta sobre boards se conoce por la matriz de equity
/// exacta: `E[N_b | b válido para h] = C(48,5)/C(50,5) · Σ_a r(a)·eq(h, a)`. Lo que se estima
/// (numeradores y denominador del estimador de razón) se corrige por regresión sobre esas
/// variables, `t̄ − β·(x̄ − μ)`, con `β` por mano. Con 2048 boards, el error propio de cada
/// mano en un all-in a 3 era de ±1 $ (T9o pagando un all-in a 3 en PKO: 8,49 frente a 6,85
/// exacto); casi todo es la fuerza de la mano del héroe en cada board, que estas variables
/// capturan. Sin sesgo salvo el de estimar `β` (O(1/n)).
pub(crate) struct ControlVariate {
    k: usize,
    mu: Vec<Vec<f64>>,
    cnt: Vec<f64>,
    sx: Vec<Vec<f64>>,
    sxx: Vec<Vec<f64>>,
    /// Objetivos (`[0]` el denominador, `[1..]` los numeradores): sumas y productos con x.
    st: Vec<Vec<f64>>,
    stx: Vec<Vec<Vec<f64>>>,
    wtl: (Vec<f32>, Vec<f32>, Vec<f32>),
}

/// `C(48,5) / C(50,5)`: probabilidad de que un board que no toca la mano del héroe tampoco
/// toque la del rival.
const BOARD_AVOID: f64 = 1_712_304.0 / 2_118_760.0;

impl ControlVariate {
    pub(crate) fn new(eq: &crate::eval::equity::EquityMatrix, rivals: &[&RangeVec], targets: usize) -> Self {
        let k = rivals.len();
        let mu = rivals
            .iter()
            .map(|r| {
                let p = eq.products(&r[..]);
                (0..HAND_STRIDE).map(|h| if h < NUM_COMBOS { BOARD_AVOID * (p.w[h] as f64 + 0.5 * p.t[h] as f64) } else { 0.0 }).collect()
            })
            .collect();
        ControlVariate {
            k,
            mu,
            cnt: vec![0.0; HAND_STRIDE],
            sx: vec![vec![0.0; HAND_STRIDE]; k],
            sxx: vec![vec![0.0; HAND_STRIDE]; k * k],
            st: vec![vec![0.0; HAND_STRIDE]; targets],
            stx: vec![vec![vec![0.0; HAND_STRIDE]; k]; targets],
            wtl: (vec![0.0; HAND_STRIDE], vec![0.0; HAND_STRIDE], vec![0.0; HAND_STRIDE]),
        }
    }

    /// Controles de un board para todos los combos válidos (`x[j][h]`).
    pub(crate) fn controls(&mut self, board: &RankedBoard, rivals: &[&RangeVec]) -> Vec<Vec<f32>> {
        rivals
            .iter()
            .map(|r| {
                let (w, t, l) = &mut self.wtl;
                showdown_wtl(&board.order, &board.ranks, &r[..], w, t, l);
                let mut x = vec![0.0f32; HAND_STRIDE];
                for &h in board.order.iter() {
                    let h = h as usize;
                    x[h] = w[h] + 0.5 * t[h];
                }
                x
            })
            .collect()
    }

    /// Una observación del combo `h` en un board válido para él: controles y objetivos.
    #[inline]
    pub(crate) fn add(&mut self, h: usize, x: &[f64], t: &[f64]) {
        self.cnt[h] += 1.0;
        for i in 0..self.k {
            self.sx[i][h] += x[i];
            for j in 0..self.k {
                self.sxx[i * self.k + j][h] += x[i] * x[j];
            }
        }
        for (g, &v) in t.iter().enumerate() {
            self.st[g][h] += v;
            for i in 0..self.k {
                self.stx[g][i][h] += v * x[i];
            }
        }
    }

    /// Sumas corregidas de cada objetivo para el combo `h` (en la escala de sumas, como las
    /// acumula el estimador sin control). `None` si hay pocas observaciones.
    pub(crate) fn corrected(&self, h: usize) -> Option<Vec<f64>> {
        let n = self.cnt[h];
        if n < 32.0 {
            return None;
        }
        let k = self.k;
        let xm: Vec<f64> = (0..k).map(|i| self.sx[i][h] / n).collect();
        let mut a = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = self.sxx[i * k + j][h] / n - xm[i] * xm[j];
            }
            a[i * k + i] += 1e-12 * (1.0 + a[i * k + i].abs());
        }
        let d: Vec<f64> = (0..k).map(|i| xm[i] - self.mu[i][h]).collect();
        let mut out = Vec::with_capacity(self.st.len());
        for g in 0..self.st.len() {
            let tm = self.st[g][h] / n;
            let c: Vec<f64> = (0..k).map(|i| self.stx[g][i][h] / n - tm * xm[i]).collect();
            let beta = solve_small(&a, &c, k)?;
            let adj: f64 = (0..k).map(|i| beta[i] * d[i]).sum();
            out.push((tm - adj) * n);
        }
        Some(out)
    }
}

/// Sistema lineal pequeño `a·x = b` (k ≤ 3) por Gauss con pivoteo parcial.
fn solve_small(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut m: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut r = a[i * k..(i + 1) * k].to_vec();
            r.push(b[i]);
            r
        })
        .collect();
    for col in 0..k {
        let piv = (col..k).max_by(|&x, &y| m[x][col].abs().total_cmp(&m[y][col].abs()))?;
        if m[piv][col].abs() < 1e-18 {
            return None;
        }
        m.swap(col, piv);
        for r in 0..k {
            if r != col {
                let f = m[r][col] / m[col][col];
                for c in col..=k {
                    m[r][c] -= f * m[col][c];
                }
            }
        }
    }
    Some((0..k).map(|i| m[i][k] / m[i][i]).collect())
}

impl MultiwayEval {
    pub fn new(pool: usize, boards_per_visit: usize, eval_boards: usize, seed: u64, threads: usize) -> Self {
        MultiwayEval { pool: sample_ranked_boards(pool.max(1), seed, threads), boards_per_visit: boards_per_visit.max(1), eval_boards: eval_boards.max(1), eq: None, adaptive: false }
    }

    /// Boards de una visita. En entrenamiento, con `adaptive`, proporcionales a la masa de
    /// alcance de los rivales (`mass`: producto de sus fracciones de combos): el ruido que el
    /// bote mete en los regrets del héroe escala con ese producto, así que los botes que casi
    /// no se juegan no necesitan tantos boards (como `postflop::ADAPTIVE_MASS_REF`).
    fn boards(&self, mode: StrategyMode, seed: u64, mass: f64) -> Vec<&RankedBoard> {
        match mode {
            StrategyMode::Average => self.pool.iter().take(self.eval_boards).collect(),
            StrategyMode::Current => {
                let n = if self.adaptive { ((self.boards_per_visit as f64 * (mass / MW_MASS_REF).min(1.0)).round() as usize).max(MW_MIN_BOARDS).min(self.boards_per_visit) } else { self.boards_per_visit };
                let mut rng = SplitMix64::new(seed);
                (0..n).map(|_| &self.pool[(rng.next_u64() % self.pool.len() as u64) as usize]).collect()
            }
        }
    }

    /// Valores para el jugador en la posición `pos` del bote (0..3), con las utilidades de
    /// cada asiento de `seats` (el propio jugador y, opcionalmente, observadores: las
    /// probabilidades de orden por board se calculan una sola vez para todos).
    /// `r_others`: rangos de los otros dos, en orden de posición en el bote.
    #[allow(clippy::too_many_arguments)]
    pub fn values(&self, t: &TerminalNode, utils: &[f32], n: usize, pos: usize, seats: &[usize], r_others: [&RangeVec; 2], mode: StrategyMode, seed: u64) -> Vec<Box<RangeVec>> {
        self.values_w(t, utils, n, pos, seats, r_others, mode, seed, None, None)
    }

    /// Como [`Self::values`] con board bunching: cada board del pool (uniforme) pondera por
    /// `board_w(máscara)`, la probabilidad de que los que foldearon no tengan ninguna de
    /// sus cartas (muestreo por importancia; el estimador autonormalizado absorbe la
    /// constante).
    #[allow(clippy::too_many_arguments)]
    pub fn values_w(&self, t: &TerminalNode, utils: &[f32], n: usize, pos: usize, seats: &[usize], r_others: [&RangeVec; 2], mode: StrategyMode, seed: u64, board_w: Option<&BoardWeight<'_>>, joint: Option<&JointFolds<'_>>) -> Vec<Box<RangeVec>> {
        let others: Vec<usize> = (0..3).filter(|&k| k != pos).collect();
        let map = [pos as u8, others[0] as u8, others[1] as u8];
        let u = t.utilities(utils);
        // U de cada asiento pedido para cada orden relativo.
        let u_rel: Vec<[f32; 13]> = seats
            .iter()
            .map(|&seat| {
                let mut r = [0.0f32; 13];
                for (k, rel) in PATTERNS3.iter().enumerate() {
                    // Puestos en las posiciones canónicas del terminal.
                    let mut canon = [0u8; 3];
                    for i in 0..3 {
                        canon[map[i] as usize] = rel[i];
                    }
                    r[k] = u[pattern3_index(canon) * n + seat];
                }
                r
            })
            .collect();
        let mut num: Vec<Vec<f32>> = vec![vec![0.0f32; HAND_STRIDE]; seats.len()];
        let mut den = vec![0.0f32; HAND_STRIDE];
        let mut probs = vec![[0.0f32; 13]; HAND_STRIDE];
        let mut weight = vec![0.0f32; HAND_STRIDE];
        let mut scratch = OrderScratch::default();
        // Evaluación (boards fijos): variable de control con la equity HU exacta.
        let mut cv = match (&self.eq, mode) {
            (Some(eq), StrategyMode::Average) => Some(ControlVariate::new(eq, &r_others, 1 + seats.len())),
            _ => None,
        };
        let (mut xb, mut tb) = (vec![0.0f64; 2], vec![0.0f64; 1 + seats.len()]);
        for board in self.boards(mode, seed, reach_mass(&r_others)) {
            let jf = joint.map(|j| j.factor(board));
            match &jf {
                Some(f) => {
                    let (mut a, mut b) = (super::zero_range(), super::zero_range());
                    for &h in board.order.iter() {
                        let h = h as usize;
                        a[h] = r_others[0][h] * f[h];
                        b[h] = r_others[1][h] * f[h];
                    }
                    board_orders_with(board, &a, &b, &mut probs, &mut weight, &mut scratch);
                }
                None => board_orders_with(board, r_others[0], r_others[1], &mut probs, &mut weight, &mut scratch),
            }
            let bw = board_w.map_or(1.0, |f| f(board.mask));
            let xs = cv.as_mut().map(|c| c.controls(board, &r_others));
            for &h in board.order.iter() {
                let h = h as usize;
                let w = weight[h] * bw * jf.as_ref().map_or(1.0, |f| f[h]);
                if let (Some(c), Some(xs)) = (cv.as_mut(), xs.as_ref()) {
                    // Todas las observaciones válidas para h cuentan (también con w = 0).
                    xb[0] = xs[0][h] as f64;
                    xb[1] = xs[1][h] as f64;
                    tb[0] = w as f64;
                    for (s, ur) in u_rel.iter().enumerate() {
                        tb[1 + s] = (w * probs[h].iter().zip(ur).map(|(p, u)| p * u).sum::<f32>()) as f64;
                    }
                    c.add(h, &xb, &tb);
                }
                if w > 0.0 {
                    den[h] += w;
                    for (s, ur) in u_rel.iter().enumerate() {
                        num[s][h] += w * probs[h].iter().zip(ur).map(|(p, u)| p * u).sum::<f32>();
                    }
                }
            }
        }
        if let Some(c) = &cv {
            for h in 0..NUM_COMBOS {
                if let Some(t) = c.corrected(h) {
                    if t[0] > 0.0 {
                        den[h] = t[0] as f32;
                        for s in 0..seats.len() {
                            num[s][h] = t[1 + s] as f32;
                        }
                    }
                }
            }
        }
        let (ba, bb) = (BlockerSums::new(&r_others[0][..]), BlockerSums::new(&r_others[1][..]));
        let mut reach = vec![0.0f32; NUM_COMBOS];
        for h in 0..NUM_COMBOS {
            if den[h] > 0.0 {
                reach[h] = ba.compatible(&r_others[0][..], h) * bb.compatible(&r_others[1][..], h) / (COMPATIBLE_COMBOS * COMPATIBLE_COMBOS) / den[h];
            }
        }
        num.iter()
            .map(|nm| {
                let mut out = super::zero_range();
                for h in 0..NUM_COMBOS {
                    out[h] = nm[h] * reach[h];
                }
                out
            })
            .collect()
    }

    /// Valores de un único asiento (ver [`Self::values`]).
    #[allow(clippy::too_many_arguments)]
    pub fn hero_values(&self, t: &TerminalNode, utils: &[f32], n: usize, pos: usize, util_seat: usize, r_others: [&RangeVec; 2], mode: StrategyMode, seed: u64, out: &mut RangeVec) {
        let v = self.values(t, utils, n, pos, &[util_seat], r_others, mode, seed);
        out.copy_from_slice(&v[0][..]);
    }

    /// Utilidad esperada de `seat` (que no está en el bote), normalizada por la
    /// probabilidad de que los tres lleguen.
    #[allow(clippy::too_many_arguments)]
    pub fn observer_value(&self, t: &TerminalNode, utils: &[f32], n: usize, seat: usize, r: [&RangeVec; 3], mode: StrategyMode, seed: u64) -> f64 {
        let mut v = super::zero_range();
        self.hero_values(t, utils, n, 0, seat, [r[1], r[2]], mode, seed, &mut v);
        weighted_mean(r[0], &v)
    }
}

impl MultiwayEval {
    /// Botes de `k` jugadores (4 a 6) por **Monte Carlo de manos rivales**: por cada board
    /// se muestrean manos de los `k−1` rivales según sus rangos (rechazando solapes entre
    /// ellos, así que el bloqueo rival↔rival es exacto), se ordenan por fuerza, y cada
    /// combo del héroe compatible con la muestra se inserta en ese orden (empates:
    /// promedio de las posiciones empatadas). El valor es la media condicionada por combo
    /// multiplicada por la probabilidad de que lleguen los rivales, igual que en 3-way.
    ///
    /// `u`: tabla `U[orden Lehmer][asiento]` del terminal; `r_others`: rangos de los
    /// otros jugadores del bote en orden de posición (sin `pos`).
    #[allow(clippy::too_many_arguments)]
    pub fn values_n(&self, u: &[f32], n: usize, k: usize, pos: usize, seats: &[usize], r_others: &[&RangeVec], mode: StrategyMode, seed: u64) -> Vec<Box<RangeVec>> {
        self.values_nw(u, n, k, pos, seats, r_others, mode, seed, None, None)
    }

    /// [`Self::values_n`] con board bunching (ver [`Self::values_w`]).
    #[allow(clippy::too_many_arguments)]
    pub fn values_nw(&self, u: &[f32], n: usize, k: usize, pos: usize, seats: &[usize], r_others: &[&RangeVec], mode: StrategyMode, seed: u64, board_w: Option<&BoardWeight<'_>>, joint: Option<&JointFolds<'_>>) -> Vec<Box<RangeVec>> {
        if k == 4 {
            return self.values_4(u, n, pos, seats, r_others, mode, seed, board_w, joint);
        }
        self.values_mc(u, n, k, pos, seats, r_others, mode, seed, board_w, 8)
    }

    /// Monte Carlo de manos rivales (5-6 jugadores), `samples` repartos por board.
    #[allow(clippy::too_many_arguments)]
    fn values_mc(&self, u: &[f32], n: usize, k: usize, pos: usize, seats: &[usize], r_others: &[&RangeVec], mode: StrategyMode, seed: u64, board_w: Option<&BoardWeight<'_>>, samples: usize) -> Vec<Box<RangeVec>> {
        let others: Vec<u8> = (0..k as u8).filter(|&p| p as usize != pos).collect();
        let boards: Vec<&RankedBoard> = match mode {
            StrategyMode::Current => self.boards(mode, seed, reach_mass(r_others)),
            StrategyMode::Average => self.pool.iter().take(self.eval_boards.min(64)).collect(),
        };
        let mut num: Vec<Vec<f32>> = vec![vec![0.0; NUM_COMBOS]; seats.len()];
        let mut cnt = vec![0.0f32; NUM_COMBOS];
        let mut rng = SplitMix64::new(seed ^ 0x6E57);
        for board in boards {
            let bw = board_w.map_or(1.0, |f| f(board.mask));
            // Distribuciones acumuladas de cada rival sobre los combos válidos del board.
            let cums: Vec<Vec<f32>> = r_others
                .iter()
                .map(|r| {
                    let mut acc = 0.0f32;
                    board.order.iter().map(|&h| {
                        acc += r[h as usize];
                        acc
                    }).collect()
                })
                .collect();
            if cums.iter().any(|c| c.last().copied().unwrap_or(0.0) <= 0.0) {
                continue;
            }
            // Sumas por carta de cada rival sobre los combos válidos del board: masa de su
            // rango compatible con las manos ya repartidas (`z`).
            let card_sums: Vec<[f32; 52]> = r_others
                .iter()
                .map(|r| {
                    let mut cs = [0.0f32; 52];
                    for &h in board.order.iter() {
                        let [a, b] = crate::eval::COMBO_CARDS[h as usize];
                        cs[a as usize] += r[h as usize];
                        cs[b as usize] += r[h as usize];
                    }
                    cs
                })
                .collect();
            let z = |j: usize, cards: &[u8]| -> f32 {
                let r = r_others[j];
                let mut blocked: f32 = cards.iter().map(|&c| card_sums[j][c as usize]).sum();
                for x in 0..cards.len() {
                    for y in x + 1..cards.len() {
                        let hc = crate::eval::Combo::from_cards(crate::eval::Card(cards[x]), crate::eval::Card(cards[y])).0 as usize;
                        if board.mask & crate::eval::COMBO_MASKS[hc] == 0 {
                            blocked -= r[hc];
                        }
                    }
                }
                (cums[j].last().unwrap() - blocked).max(0.0)
            };
            'sample: for _ in 0..samples {
                let mut mask = 0u64;
                let mut drawn: Vec<(u32, u8)> = Vec::with_capacity(k - 1);
                // Muestreo secuencial: cada rival de su rango sin las cartas de los anteriores.
                // La propuesta es r₁(x₁)·r₂(x₂)/Z₂(x₁)·…, no la conjunta ∝ r₁·r₂·…: el peso
                // de importancia Π Z_j(anteriores)/Z_j corrige el sesgo hacia manos que
                // bloquean mucho rango de los demás (rangos de all-in cargados de A, K, Q).
                let mut weight = 1.0f32;
                let mut dealt: Vec<u8> = Vec::with_capacity(2 * (k - 1));
                for (j, c) in cums.iter().enumerate() {
                    let total = *c.last().unwrap();
                    if j > 0 {
                        weight *= z(j, &dealt) / total;
                    }
                    let mut ok = false;
                    for _ in 0..32 {
                        let x = rng.next_f32() * total;
                        let i = c.partition_point(|&v| v <= x).min(c.len() - 1);
                        let h = board.order[i] as usize;
                        if crate::eval::COMBO_MASKS[h] & mask == 0 {
                            mask |= crate::eval::COMBO_MASKS[h];
                            drawn.push((board.ranks[h], others[j]));
                            let [a, b] = crate::eval::COMBO_CARDS[h];
                            dealt.push(a);
                            dealt.push(b);
                            ok = true;
                            break;
                        }
                    }
                    if !ok {
                        continue 'sample;
                    }
                }
                if weight <= 0.0 {
                    continue;
                }
                let bw = bw * weight;
                // Rivales de mejor a peor (empates entre rivales: orden aleatorio).
                let jitter: Vec<u64> = (0..drawn.len()).map(|_| rng.next_u64()).collect();
                let mut idx: Vec<usize> = (0..drawn.len()).collect();
                idx.sort_by(|&a, &b| drawn[b].0.cmp(&drawn[a].0).then(jitter[a].cmp(&jitter[b])));
                let opp_ranks: Vec<u32> = idx.iter().map(|&i| drawn[i].0).collect();
                // Utilidad de cada asiento según la posición de inserción del héroe.
                let ins: Vec<Vec<f32>> = (0..k)
                    .map(|p| {
                        let mut order: Vec<u8> = idx.iter().map(|&i| drawn[i].1).collect();
                        order.insert(p, pos as u8);
                        let o = perm_index(&order);
                        seats.iter().map(|&st| u[o * n + st]).collect()
                    })
                    .collect();
                for &h in board.order.iter() {
                    let h = h as usize;
                    if crate::eval::COMBO_MASKS[h] & mask != 0 {
                        continue;
                    }
                    let rh = board.ranks[h];
                    let (mut better, mut ties) = (0, 0);
                    for &r in &opp_ranks {
                        better += (r > rh) as usize;
                        ties += (r == rh) as usize;
                    }
                    cnt[h] += bw;
                    if ties == 0 {
                        let row = &ins[better];
                        for (sidx, acc) in num.iter_mut().enumerate() {
                            acc[h] += bw * row[sidx];
                        }
                    } else {
                        for (sidx, acc) in num.iter_mut().enumerate() {
                            acc[h] += bw * (better..=better + ties).map(|p| ins[p][sidx]).sum::<f32>() / (ties + 1) as f32;
                        }
                    }
                }
            }
        }
        let blockers: Vec<BlockerSums> = r_others.iter().map(|r| BlockerSums::new(&r[..])).collect();
        let mut reach = vec![0.0f64; NUM_COMBOS];
        for h in 0..NUM_COMBOS {
            if cnt[h] > 0.0 {
                let m: f64 = blockers.iter().zip(r_others).map(|(b, r)| (b.compatible(&r[..], h) / COMPATIBLE_COMBOS) as f64).product();
                reach[h] = m / cnt[h] as f64;
            }
        }
        num.iter()
            .map(|nm| {
                let mut out = super::zero_range();
                for h in 0..NUM_COMBOS {
                    out[h] = (nm[h] as f64 * reach[h]) as f32;
                }
                out
            })
            .collect()
    }
}

impl MultiwayEval {
    /// Bote de 4 jugadores **sin muestrear manos rivales**: en cada board, la probabilidad de
    /// cada orden de fuerza sale analíticamente como en los finales postflop multiway
    /// ([`super::postflop::end_multi_core`]): bloqueo exacto héroe↔rival y orden entre
    /// rivales por sus distribuciones de fuerza. El Monte Carlo de manos (8 muestras por
    /// board) dejaba un ruido de varios $ por combo en la evaluación (p. ej. pagar con KK
    /// por debajo de QQ en un all-in a 4), siempre el mismo con los boards fijos de la
    /// evaluación, que subía por el árbol hasta los cold calls.
    #[allow(clippy::too_many_arguments)]
    fn values_4(&self, u: &[f32], n: usize, pos: usize, seats: &[usize], r_others: &[&RangeVec], mode: StrategyMode, seed: u64, board_w: Option<&BoardWeight<'_>>, joint: Option<&JointFolds<'_>>) -> Vec<Box<RangeVec>> {
        use super::lanes::{gather_order, lbuf, LBuf, LaneLayout, Lane, Sums, M};
        let k = 4;
        let others: Vec<usize> = (0..k).filter(|&p| p != pos).collect();
        let mut num: Vec<Vec<f64>> = vec![vec![0.0; NUM_COMBOS]; seats.len()];
        let mut den = vec![0.0f64; NUM_COMBOS];
        let boards: Vec<&RankedBoard> = self.boards(mode, seed, reach_mass(r_others));
        // Evaluación (boards fijos): variable de control con la equity HU exacta contra cada
        // rival (ver `ControlVariate`). Cuenta cada board válido para el combo, también los
        // que no aportan (peso 0).
        let mut cv = match (&self.eq, mode) {
            (Some(eq), StrategyMode::Average) => Some(ControlVariate::new(eq, r_others, 1 + seats.len())),
            _ => None,
        };
        let mut bden = vec![0.0f64; NUM_COMBOS];
        let mut bnum: Vec<Vec<f64>> = vec![vec![0.0; NUM_COMBOS]; seats.len()];
        let (mut xb, mut tb) = (vec![0.0f64; k - 1], vec![0.0f64; 1 + seats.len()]);
        for board in boards {
            for &h in board.order.iter() {
                bden[h as usize] = 0.0;
                for v in bnum.iter_mut() {
                    v[h as usize] = 0.0;
                }
            }
            'board: {
                let bw = board_w.map_or(1.0, |f| f(board.mask)) as f64;
                if bw <= 0.0 {
                    break 'board;
                }
                let layout = LaneLayout::new(&board.order, &board.ranks);
                let jf: Option<Vec<f32>> = joint.map(|j| j.factor(board));
                let fl = |i: usize| jf.as_ref().map_or(1.0, |f| f[board.order[i] as usize]);
                let rl: Vec<LBuf> = (0..k)
                    .map(|p| {
                        let mut b = lbuf();
                        match others.iter().position(|&q| q == p) {
                            Some(j) => {
                                gather_order(&board.order, r_others[j], 1.0, &mut b);
                                for i in 0..M {
                                    b[i] *= fl(i);
                                }
                            }
                            None => b[..M].fill(0.0),
                        }
                        b
                    })
                    .collect();
                let r: Vec<&Lane> = rl.iter().map(|b| &**b).collect();
                if others.iter().any(|&p| r[p][..M].iter().sum::<f32>() <= 0.0) {
                    break 'board;
                }
                let bs: Vec<Sums> = r.iter().map(|x| Sums::of(x, &layout.pairs)).collect();
                let ab: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
                let mut dm = vec![0.0f32; M];
                for (i, d) in dm.iter_mut().enumerate() {
                    *d = others.iter().map(|&j| bs[j].compat(r[j], &layout.pairs, i)).product();
                }
                for (i, &h) in board.order.iter().enumerate() {
                    bden[h as usize] = bw * (dm[i] * fl(i)) as f64;
                }
                for (si, &st) in seats.iter().enumerate() {
                    let mut sv = vec![0usize; k];
                    sv[pos] = st;
                    let mut out: Vec<Option<LBuf>> = (0..k).map(|_| None).collect();
                    super::postflop::end_multi_core(u, n, &layout, 0b1111, 0, &r, &ab, &bs, &sv, &[], &mut [], 1 << pos, &mut out);
                    let o = out[pos].as_ref().unwrap();
                    for (i, &h) in board.order.iter().enumerate() {
                        bnum[si][h as usize] = bw * (o[i] * fl(i)) as f64;
                    }
                }
            }
            let xs = cv.as_mut().map(|c| c.controls(board, r_others));
            for &h in board.order.iter() {
                let h = h as usize;
                den[h] += bden[h];
                for si in 0..seats.len() {
                    num[si][h] += bnum[si][h];
                }
                if let (Some(c), Some(xs)) = (cv.as_mut(), xs.as_ref()) {
                    for j in 0..k - 1 {
                        xb[j] = xs[j][h] as f64;
                    }
                    tb[0] = bden[h];
                    for si in 0..seats.len() {
                        tb[1 + si] = bnum[si][h];
                    }
                    c.add(h, &xb, &tb);
                }
            }
        }
        if let Some(c) = &cv {
            for h in 0..NUM_COMBOS {
                if let Some(t) = c.corrected(h) {
                    if t[0] > 0.0 {
                        den[h] = t[0];
                        for si in 0..seats.len() {
                            num[si][h] = t[1 + si];
                        }
                    }
                }
            }
        }
        let blockers: Vec<BlockerSums> = r_others.iter().map(|r| BlockerSums::new(&r[..])).collect();
        let mut reach = vec![0.0f64; NUM_COMBOS];
        for h in 0..NUM_COMBOS {
            if den[h] > 0.0 {
                let m: f64 = blockers.iter().zip(r_others).map(|(b, r)| (b.compatible(&r[..], h) / COMPATIBLE_COMBOS) as f64).product();
                reach[h] = m / den[h];
            }
        }
        num.iter()
            .map(|nm| {
                let mut out = super::zero_range();
                for h in 0..NUM_COMBOS {
                    out[h] = (nm[h] * reach[h]) as f32;
                }
                out
            })
            .collect()
    }
}

impl MultiwayEval {
    /// Corrección del all-in heads-up por el bunching **conjunto** de board y manos (ver
    /// `postflop::joint_fold_factor`): la equity de la matriz con la corrección lineal del
    /// board trata por separado "qué cartas quedan para el board" y "qué manos quedan para el
    /// rival", y pierde su correlación (las mismas cartas bajas que se llevan los que
    /// foldearon suben a la vez el board y la mano del rival). Por board: valor con pesos
    /// conjuntos menos valor con los pesos separados, **sobre los mismos boards** (variable de
    /// control: el ruido del muestreo se cancela y solo queda la correlación). Devuelve, por
    /// combo del héroe, la diferencia de utilidad esperada condicionada.
    /// `u`: utilidades del héroe (gana, empata, pierde); `r_v`: rango del rival con el factor
    /// marginal de bunching ya aplicado; `folds`: (sumas, rango, masa) de los que foldearon.
    #[allow(clippy::too_many_arguments)]
    pub fn hu_bunch_correction(&self, u: [f32; 3], r_v: &RangeVec, folds: &[(BlockerSums, &RangeVec, f32)], mode: StrategyMode, seed: u64) -> Box<RangeVec> {
        use super::lanes::{gather_order, lbuf, showdown, LaneLayout, Sums, M};
        let boards: Vec<&RankedBoard> = match mode {
            StrategyMode::Average => self.pool.iter().take(self.eval_boards).collect(),
            StrategyMode::Current => {
                let mut rng = SplitMix64::new(seed);
                (0..HU_BUNCH_BOARDS).map(|_| &self.pool[(rng.next_u64() % self.pool.len() as u64) as usize]).collect()
            }
        };
        let (mut na, mut da, mut nj, mut dj) = (vec![0.0f64; NUM_COMBOS], vec![0.0f64; NUM_COMBOS], vec![0.0f64; NUM_COMBOS], vec![0.0f64; NUM_COMBOS]);
        for board in boards {
            let lay = LaneLayout::new(&board.order, &board.ranks);
            let mut f = lbuf();
            f[..M].fill(1.0);
            let mut w = 1.0f64;
            for (full, r, tot) in folds {
                let mut l = lbuf();
                gather_order(&board.order, &r[..], 1.0, &mut l);
                let s = Sums::of(&l, &lay.pairs);
                if s.total <= 0.0 {
                    w = 0.0;
                    break;
                }
                w *= (s.total / tot) as f64;
                for (i, &h) in board.order.iter().enumerate() {
                    let on = s.compat(&l, &lay.pairs, i) / s.total;
                    let mg = full.compatible(&r[..], h as usize) / tot;
                    f[i] *= if mg > 0.0 { on / mg } else { 0.0 };
                }
            }
            if w <= 0.0 {
                continue;
            }
            let mut rv = lbuf();
            gather_order(&board.order, &r_v[..], 1.0, &mut rv);
            let mut rj = lbuf();
            for i in 0..M {
                rj[i] = rv[i] * f[i];
            }
            let (mut ww, mut tt, mut ll) = (lbuf(), lbuf(), lbuf());
            for (lanes_r, joint) in [(&rv, false), (&rj, true)] {
                let sr = Sums::of(lanes_r, &lay.pairs);
                showdown(&lay, lanes_r, &sr, &mut ww, &mut tt, &mut ll);
                for (i, &h) in board.order.iter().enumerate() {
                    let h = h as usize;
                    let m = (ww[i] + tt[i] + ll[i]) as f64;
                    let v = (u[0] * ww[i] + u[1] * tt[i] + u[2] * ll[i]) as f64;
                    if joint {
                        let wi = w * f[i] as f64;
                        nj[h] += wi * v;
                        dj[h] += wi * m;
                    } else {
                        na[h] += w * v;
                        da[h] += w * m;
                    }
                }
            }
        }
        let mut out = super::zero_range();
        for h in 0..NUM_COMBOS {
            if da[h] > 0.0 && dj[h] > 0.0 {
                out[h] = (nj[h] / dj[h] - na[h] / da[h]) as f32;
            }
        }
        out
    }
}

/// Boards por visita de la corrección de bunching de los all-in HU en entrenamiento.
pub const HU_BUNCH_BOARDS: usize = 64;

/// Peso de un board (por su máscara de cartas) para el board bunching.
pub type BoardWeight<'a> = dyn Fn(u64) -> f32 + Sync + 'a;

/// `P(todos llegan)` en el modelo multijugador: `Σ_x r0[x]·Π_j m_j(x) / 1326`.
pub fn n_mass(r0: &RangeVec, rs: &[&RangeVec]) -> f64 {
    let bs: Vec<BlockerSums> = rs.iter().map(|r| BlockerSums::new(&r[..])).collect();
    (0..NUM_COMBOS)
        .map(|x| r0[x] as f64 * bs.iter().zip(rs).map(|(b, r)| (b.compatible(&r[..], x) / COMPATIBLE_COMBOS) as f64).product::<f64>())
        .sum::<f64>()
        / NUM_COMBOS as f64
}

/// `Σ_x r[x]·v[x] / 1326`: de valores por combo de un jugador a escalar de observador.
pub fn weighted_mean(r: &RangeVec, v: &RangeVec) -> f64 {
    (0..NUM_COMBOS).map(|x| (r[x] * v[x]) as f64).sum::<f64>() / NUM_COMBOS as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::COMBO_MASKS;
    use crate::tree::nodes::TerminalKind;

    /// La media de los controles sobre muchos boards debe coincidir con su valor exacto.
    /// `cargo test --release control_variate_mean -- --ignored --nocapture` (necesita
    /// `.cache/equity_exact.bin`).
    #[test]
    #[ignore]
    fn control_variate_mean_matches_exact() {
        let Ok(eq) = crate::eval::equity::EquityMatrix::load(std::path::Path::new(".cache/equity_exact.bin")) else {
            eprintln!("sin matriz exacta en caché");
            return;
        };
        let mut rng = SplitMix64::new(31);
        let mut r = crate::cfr::zero_range();
        for h in 0..NUM_COMBOS {
            r[h] = if rng.next_f32() < 0.4 { 0.2 + rng.next_f32() } else { 0.0 };
        }
        let rv: Vec<&RangeVec> = vec![&r];
        let mut cv = ControlVariate::new(&eq, &rv, 1);
        let pool = sample_ranked_boards(16384, 0x77, 8);
        let (mut sum, mut cnt) = (vec![0.0f64; NUM_COMBOS], vec![0.0f64; NUM_COMBOS]);
        for b in &pool {
            let x = cv.controls(b, &rv);
            for &h in b.order.iter() {
                sum[h as usize] += x[0][h as usize] as f64;
                cnt[h as usize] += 1.0;
            }
        }
        let (mut rel, mut m) = (0.0f64, 0.0f64);
        for h in 0..NUM_COMBOS {
            let est = sum[h] / cnt[h];
            if h % 197 == 0 { eprintln!("combo {h}: media {est:.4}  exacta {:.4}", cv.mu[0][h]); }
            rel += est - cv.mu[0][h];
            m += cv.mu[0][h];
        }
        eprintln!("sesgo relativo {:+.4} %", 100.0 * rel / m);
    }

    #[test]
    #[ignore]
    fn three_way_training_estimator_bias() {
        // Sesgo del estimador de razón con pocos boards por visita (entrenamiento) frente a
        // la evaluación con muchos boards: `cargo test --release three_way_training -- --ignored --nocapture`.
        let mut rng = SplitMix64::new(23);
        let rs: Vec<Box<RangeVec>> = (0..2)
            .map(|_| {
                let mut r = crate::cfr::zero_range();
                for h in 0..NUM_COMBOS {
                    r[h] = if rng.next_f32() < 0.4 { 0.2 + rng.next_f32() } else { 0.0 };
                }
                r
            })
            .collect();
        let n = 3;
        // Héroe (asiento 0): 100 si gana, 20 segundo, 0 tercero; empates a medias.
        let mut u = vec![0.0f32; 13 * n];
        for (k, p) in crate::tree::nodes::PATTERNS3.iter().enumerate() {
            let better = p[1..].iter().filter(|&&x| x < p[0]).count();
            let ties = p[1..].iter().filter(|&&x| x == p[0]).count();
            let v = match better { 0 => 100.0, 1 => 20.0, _ => 0.0 };
            u[k * n] = if ties > 0 && better == 0 { (100.0 + 20.0) / 2.0 } else { v };
        }
        let t = TerminalNode { contributions: [0; crate::tree::nodes::MAX_PLAYERS], pot: 0, utility_offset: 0, alive_mask: 7, kind: TerminalKind::Multiway, num_outcomes: 13, num_seats: n as u8, players: [0, 1, 2], leaf: 0, _reserved: [0; 4] };
        let reference = MultiwayEval::new(16384, 8, 16384, 0x51, 8).values(&t, &u, n, 0, &[0], [&rs[0], &rs[1]], StrategyMode::Average, 1)[0].clone();
        for bpv in [8usize, 32] {
            let mw = MultiwayEval::new(8192, bpv, 64, 0x52, 8);
            let mut acc = vec![0.0f64; NUM_COMBOS];
            let reps = 600;
            for s in 0..reps {
                let v = mw.values(&t, &u, n, 0, &[0], [&rs[0], &rs[1]], StrategyMode::Current, 1000 + s as u64);
                for h in 0..NUM_COMBOS {
                    acc[h] += v[0][h] as f64 / reps as f64;
                }
            }
            let (mut bias, mut rms, mut m) = (0.0f64, 0.0f64, 0.0f64);
            for h in 0..NUM_COMBOS {
                if reference[h] > 0.0 {
                    let d = acc[h] - reference[h] as f64;
                    bias += d;
                    rms += d * d;
                    m += reference[h] as f64;
                }
            }
            eprintln!("boards/visita {bpv}: sesgo medio relativo {:+.4} %  rms {:.4}", 100.0 * bias / m, (rms / NUM_COMBOS as f64).sqrt());
        }
    }

    /// All-in a 4: el cálculo analítico por board coincide con el Monte Carlo de manos
    /// rivales (con muchas muestras), salvo el ruido de este.
    #[test]
    fn four_way_analytic_matches_monte_carlo() {
        let mw = MultiwayEval::new(256, 8, 64, 0x44, 4);
        let mut rng = SplitMix64::new(17);
        let rs: Vec<Box<RangeVec>> = (0..3)
            .map(|_| {
                let mut r = crate::cfr::zero_range();
                for h in 0..NUM_COMBOS {
                    r[h] = if rng.next_f32() < 0.35 { 0.2 + rng.next_f32() } else { 0.0 };
                }
                r
            })
            .collect();
        let ro: Vec<&RangeVec> = rs.iter().map(|r| &**r).collect();
        let n = 4;
        let u: Vec<f32> = (0..24 * n).map(|_| 10.0 * rng.next_f32()).collect();
        for pos in [0usize, 2] {
            let a = mw.values_4(&u, n, pos, &[pos, 3], &ro, StrategyMode::Average, 1, None, None);
            let b = mw.values_mc(&u, n, 4, pos, &[pos, 3], &ro, StrategyMode::Average, 1, None, 256);
            let bl: Vec<BlockerSums> = ro.iter().map(|r| BlockerSums::new(&r[..])).collect();
            for s in 0..2 {
                let (mut err, mut bias, mut cnt) = (0.0f64, 0.0f64, 0usize);
                for h in 0..NUM_COMBOS {
                    let m: f64 = bl.iter().zip(&ro).map(|(x, r)| (x.compatible(&r[..], h) / COMPATIBLE_COMBOS) as f64).product();
                    if m <= 1e-6 {
                        continue;
                    }
                    let d = (a[s][h] as f64 - b[s][h] as f64) / m;
                    err += d.abs();
                    bias += d;
                    cnt += 1;
                }
                let (err, bias) = (err / cnt as f64, bias / cnt as f64);
                assert!(err < 0.35 && bias.abs() < 0.1, "pos {pos} asiento {s}: |Δ| {err:.3} sesgo {bias:+.3}");
            }
        }
    }

    /// Contra la enumeración exacta: los tres jugadores sin cartas en común.
    #[test]
    fn orders_match_exact_enumeration() {
        let board = RankedBoard::new([2u64, 9, 20, 33, 47].iter().map(|&c| 1u64 << c).sum());
        let mut rng = SplitMix64::new(4);
        let (mut ra, mut rb) = (crate::cfr::zero_range(), crate::cfr::zero_range());
        for &h in board.order.iter() {
            ra[h as usize] = if rng.next_f32() < 0.3 { rng.next_f32() } else { 0.0 };
            rb[h as usize] = if rng.next_f32() < 0.5 { rng.next_f32() } else { 0.0 };
        }
        let mut probs = vec![[0.0f32; 13]; HAND_STRIDE];
        let mut weight = vec![0.0f32; HAND_STRIDE];
        board_orders(&board, &ra, &rb, &mut probs, &mut weight);
        // Puesto de cada uno (0 = mejor, empates con el mismo puesto) → patrón.
        let pattern = |v: [u32; 3]| -> usize {
            let mut r = [0u8; 3];
            for i in 0..3 {
                r[i] = v.iter().filter(|&&x| x > v[i]).collect::<std::collections::BTreeSet<_>>().len() as u8;
            }
            crate::tree::nodes::pattern3_index(r)
        };
        for &h in board.order.iter().step_by(37) {
            let h = h as usize;
            let rh = board.ranks[h];
            let mut p = [0.0f64; 13];
            let mut tot = 0.0f64;
            for &a in board.order.iter() {
                let a = a as usize;
                if ra[a] == 0.0 || COMBO_MASKS[a] & COMBO_MASKS[h] != 0 {
                    continue;
                }
                for &b in board.order.iter() {
                    let b = b as usize;
                    if rb[b] == 0.0 || COMBO_MASKS[b] & (COMBO_MASKS[h] | COMBO_MASKS[a]) != 0 {
                        continue;
                    }
                    let w = (ra[a] * rb[b]) as f64;
                    p[pattern([rh, board.ranks[a], board.ranks[b]])] += w;
                    tot += w;
                }
            }
            let s: f32 = probs[h].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "suma {s}");
            // Masa total (pares sin cartas en común): falta solo el término de tercer orden.
            assert!((weight[h] as f64 / tot - 1.0).abs() < 0.02, "masa {} vs {tot}", weight[h]);
            for k in 0..13 {
                // El orden entre los dos rivales cuando ambos quedan del mismo lado sale de
                // términos cruzados sin el bloqueo del héroe: ±1 punto (no cambia al héroe).
                let tol = if matches!(k, 0 | 1 | 3 | 5) { 0.015 } else { 0.003 };
                assert!((probs[h][k] as f64 - p[k] / tot).abs() < tol, "h={h} k={k}: {} vs {}", probs[h][k], p[k] / tot);
            }
        }
    }
}
