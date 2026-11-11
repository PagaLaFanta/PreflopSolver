//! # Carriles: vectores compactos en el orden de un board
//!
//! Con un runout completo (5 cartas) quedan exactamente `C(47,2) = 1081` combos válidos.
//! El recorrido postflop trabaja sobre vectores de ese tamaño **en el orden de fuerza del
//! board** (`SampledBoard::order`): los bucles son contiguos (vectorizables, sin índices
//! indirectos ni comprobaciones de límites), el showdown recorre grupos de igual fuerza
//! precalculados y los buckets se leen por carril. Solo al entrar y salir de la hoja se
//! convierte entre índices de combo y carriles.

use crate::abstraction::board_pool::SampledBoard;

/// Combos válidos en un runout completo.
pub const M: usize = 1081;
/// Tamaño reservado (múltiplo de 16); los carriles `M..` no se usan.
pub const LANES: usize = 1088;

pub type Lane = [f32; LANES];

thread_local! {
    static POOL: std::cell::RefCell<Vec<Box<Lane>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Carril reciclado del pool del hilo (contenido arbitrario: quien lo pide escribe los
/// `M` carriles antes de leerlos). Sin `malloc` ni `memset` en el recorrido.
#[allow(missing_debug_implementations)]
pub struct LBuf(std::mem::ManuallyDrop<Box<Lane>>);

pub fn lbuf() -> LBuf {
    LBuf(std::mem::ManuallyDrop::new(POOL.with(|p| p.borrow_mut().pop()).unwrap_or_else(|| Box::new([0.0; LANES]))))
}

impl std::ops::Deref for LBuf {
    type Target = Lane;
    #[inline(always)]
    fn deref(&self) -> &Lane {
        &self.0
    }
}

impl std::ops::DerefMut for LBuf {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Lane {
        &mut self.0
    }
}

impl Drop for LBuf {
    fn drop(&mut self) {
        // SAFETY: `self.0` no se vuelve a usar tras tomarlo.
        let b = unsafe { std::mem::ManuallyDrop::take(&mut self.0) };
        POOL.with(|p| p.borrow_mut().push(b));
    }
}

/// Sumas de bloqueadores de un rango en carriles: masa total y masa que contiene cada
/// carta. `compat(i)` = masa del rango compatible con el combo del carril `i`.
#[derive(Debug, Clone)]
pub struct Sums {
    pub total: f32,
    pub card: [f32; 52],
}

impl Sums {
    pub fn of(r: &Lane, pairs: &[[u8; 2]]) -> Self {
        let mut card = [0.0f32; 52];
        let mut total = 0.0f32;
        for (x, &[a, b]) in r[..M].iter().zip(pairs) {
            total += x;
            card[a as usize] += x;
            card[b as usize] += x;
        }
        Sums { total, card }
    }

    /// Vacías (para rangos cuyos bloqueadores no hacen falta).
    pub fn none() -> Self {
        Sums { total: 0.0, card: [0.0; 52] }
    }

    #[inline(always)]
    pub fn compat(&self, r: &Lane, pairs: &[[u8; 2]], i: usize) -> f32 {
        let [a, b] = pairs[i];
        self.total - self.card[a as usize] - self.card[b as usize] + r[i]
    }

    /// `out[i] = compat(i)` para todos los carriles.
    pub fn compat_all(&self, r: &Lane, pairs: &[[u8; 2]], out: &mut Lane) {
        for ((o, &x), &[a, b]) in out[..M].iter_mut().zip(&r[..M]).zip(pairs) {
            *o = self.total - self.card[a as usize] - self.card[b as usize] + x;
        }
    }
}

/// Carriles de un vector por combo (`v[order[i]]·scale`).
pub fn gather(board: &SampledBoard, v: &[f32], scale: f32, out: &mut Lane) {
    gather_order(&board.order, v, scale, out);
}

/// [`gather`] con el orden de combos dado.
pub fn gather_order(order: &[u16], v: &[f32], scale: f32, out: &mut Lane) {
    for (o, &h) in out[..M].iter_mut().zip(order.iter()) {
        *o = v[h as usize] * scale;
    }
}

/// Board en carriles: grupos de igual fuerza (límites, de menor a mayor) y cartas del combo
/// de cada carril. Lo cumplen los runouts del postflop ([`SampledBoard`]) y los de los
/// all-ins multiway ([`LaneLayout`]).
pub trait LaneBoard {
    fn groups(&self) -> &[u16];
    fn pairs(&self) -> &[[u8; 2]];
}

impl LaneBoard for SampledBoard {
    fn groups(&self) -> &[u16] {
        &self.groups
    }
    fn pairs(&self) -> &[[u8; 2]] {
        &self.pairs
    }
}

/// Carriles de un board dado por su orden de combos (ascendente por fuerza) y sus rangos.
#[derive(Debug, Clone)]
pub struct LaneLayout {
    pub groups: Box<[u16]>,
    pub pairs: Box<[[u8; 2]]>,
}

impl LaneLayout {
    pub fn new(order: &[u16], ranks: &[u32]) -> Self {
        assert_eq!(order.len(), M, "runout completo");
        let pairs: Vec<[u8; 2]> = order.iter().map(|&h| crate::eval::COMBO_CARDS[h as usize]).collect();
        let mut groups = vec![0u16];
        for i in 1..M {
            if ranks[order[i] as usize] != ranks[order[i - 1] as usize] {
                groups.push(i as u16);
            }
        }
        groups.push(M as u16);
        LaneLayout { groups: groups.into(), pairs: pairs.into() }
    }
}

impl LaneBoard for LaneLayout {
    fn groups(&self) -> &[u16] {
        &self.groups
    }
    fn pairs(&self) -> &[[u8; 2]] {
        &self.pairs
    }
}

/// Showdown del combo de cada carril contra `r_opp` (masas que pierden, empatan y ganan
/// contra él, sin los combos que comparten carta): la misma fórmula O(n) que
/// `eval::showdown`, recorriendo los grupos de igual fuerza del board.
pub fn showdown<B: LaneBoard + ?Sized>(board: &B, r_opp: &Lane, s: &Sums, w: &mut Lane, t: &mut Lane, l: &mut Lane) {
    let pairs = board.pairs();
    let mut weaker = 0.0f32;
    let mut wc = [0.0f32; 52];
    let mut gc = [0.0f32; 52];
    for g in board.groups().windows(2) {
        let (i, j) = (g[0] as usize, g[1] as usize);
        let mut gm = 0.0f32;
        for x in i..j {
            let [a, b] = pairs[x];
            let r = r_opp[x];
            gm += r;
            gc[a as usize] += r;
            gc[b as usize] += r;
        }
        for x in i..j {
            let [a, b] = pairs[x];
            let (a, b) = (a as usize, b as usize);
            let wv = weaker - wc[a] - wc[b];
            let tv = gm - gc[a] - gc[b] + r_opp[x];
            w[x] = wv;
            t[x] = tv;
            l[x] = s.total - s.card[a] - s.card[b] + r_opp[x] - wv - tv;
        }
        for x in i..j {
            let [a, b] = pairs[x];
            let r = r_opp[x];
            wc[a as usize] += r;
            wc[b as usize] += r;
            gc[a as usize] = 0.0;
            gc[b as usize] = 0.0;
        }
        weaker += gm;
    }
}

/// Showdown heads-up de los dos jugadores en una sola pasada por los grupos: escribe
/// directamente el valor de cada uno, `o_p = c_p[0]·gana + c_p[1]·empata + c_p[2]·pierde`
/// contra el rango del otro (con bloqueo), sin los tres carriles intermedios por jugador.
/// Devuelve `Σ r0·(gana, empata, pierde)` del jugador 0 (para los observadores).
#[allow(clippy::too_many_arguments)]
pub fn showdown_pair(board: &SampledBoard, r0: &Lane, s0: &Sums, r1: &Lane, s1: &Sums, c0: [f32; 3], c1: [f32; 3], o0: &mut Lane, o1: &mut Lane) -> [f64; 3] {
    let pairs = &board.pairs;
    let (mut weaker0, mut weaker1) = (0.0f32, 0.0f32);
    let (mut wc0, mut wc1) = ([0.0f32; 52], [0.0f32; 52]);
    let (mut gc0, mut gc1) = ([0.0f32; 52], [0.0f32; 52]);
    let (mut dw, mut dt, mut dl) = (0.0f32, 0.0f32, 0.0f32);
    // Coeficientes con la derrota despejada: `c·(w, t, compat − w − t)`.
    let (a0, b0, z0) = (c0[0] - c0[2], c0[1] - c0[2], c0[2]);
    let (a1, b1, z1) = (c1[0] - c1[2], c1[1] - c1[2], c1[2]);
    for g in board.groups.windows(2) {
        let (i, j) = (g[0] as usize, g[1] as usize);
        let (mut gm0, mut gm1) = (0.0f32, 0.0f32);
        for x in i..j {
            let [a, b] = pairs[x];
            let (a, b) = (a as usize, b as usize);
            let (x0, x1) = (r0[x], r1[x]);
            gm0 += x0;
            gm1 += x1;
            gc0[a] += x0;
            gc0[b] += x0;
            gc1[a] += x1;
            gc1[b] += x1;
        }
        for x in i..j {
            let [a, b] = pairs[x];
            let (a, b) = (a as usize, b as usize);
            let (x0, x1) = (r0[x], r1[x]);
            // Jugador 0 contra r1.
            let w = weaker1 - wc1[a] - wc1[b];
            let t = gm1 - gc1[a] - gc1[b] + x1;
            let c = s1.total - s1.card[a] - s1.card[b] + x1;
            o0[x] = a0 * w + b0 * t + z0 * c;
            dw += x0 * w;
            dt += x0 * t;
            dl += x0 * (c - w - t);
            // Jugador 1 contra r0.
            let w = weaker0 - wc0[a] - wc0[b];
            let t = gm0 - gc0[a] - gc0[b] + x0;
            let c = s0.total - s0.card[a] - s0.card[b] + x0;
            o1[x] = a1 * w + b1 * t + z1 * c;
        }
        for x in i..j {
            let [a, b] = pairs[x];
            let (a, b) = (a as usize, b as usize);
            let (x0, x1) = (r0[x], r1[x]);
            wc0[a] += x0;
            wc0[b] += x0;
            wc1[a] += x1;
            wc1[b] += x1;
            gc0[a] = 0.0;
            gc0[b] = 0.0;
            gc1[a] = 0.0;
            gc1[b] = 0.0;
        }
        weaker0 += gm0;
        weaker1 += gm1;
    }
    [dw as f64, dt as f64, dl as f64]
}

/// Estado incremental del showdown a 3 con bloqueo exacto entre los dos rivales `a` y `b`
/// sobre el conjunto `S` de carriles insertados: sumas por carta, matrices por par de
/// cartas y los productos cruzados que permiten obtener, para cada héroe `i`, la masa
/// `Σ_{x,y∈S, x⊥i, y⊥i, x⊥y} r_a(x)·r_b(y)` en O(1).
struct Tri {
    ta: f64,
    tb: f64,
    ca: [f64; 52],
    cb: [f64; 52],
    ra: Vec<[f64; 52]>,
    rb: Vec<[f64; 52]>,
    d: f64,
    kab: [f64; 52],
    kba: [f64; 52],
    /// `J[h][h'] = Σ_c RA[c][h]·RB[c][h']` = `j[h][h'] + jt[h'][h]`: lo que suma cada
    /// inserción de a va por filas de `j` y lo de b por filas de `jt` (sin recorrer columnas).
    j: Vec<[f64; 52]>,
    jt: Vec<[f64; 52]>,
    e: f64,
    ec: [f64; 52],
    /// Sin las correcciones por la otra carta del héroe (`kab`, `kba`, `j`): O(1) por
    /// inserción; ver [`showdown3_fast`].
    fast: bool,
}

impl Tri {
    fn new(fast: bool) -> Self {
        let mat = || if fast { Vec::new() } else { vec![[0.0; 52]; 52] };
        Tri { ta: 0.0, tb: 0.0, ca: [0.0; 52], cb: [0.0; 52], ra: mat(), rb: mat(), d: 0.0, kab: [0.0; 52], kba: [0.0; 52], j: mat(), jt: mat(), e: 0.0, ec: [0.0; 52], fast }
    }

    /// Inserta el combo `{p, q}` con masas `m` (rival a) y `n` (rival b).
    fn insert(&mut self, p: usize, q: usize, m: f64, n: f64) {
        if self.fast {
            self.ta += m;
            self.tb += n;
            self.d += m * (self.cb[p] + self.cb[q]);
            self.d += n * (self.ca[p] + self.ca[q] + 2.0 * m);
            self.ca[p] += m;
            self.ca[q] += m;
            self.cb[p] += n;
            self.cb[q] += n;
            self.e += m * n;
            self.ec[p] += m * n;
            self.ec[q] += m * n;
            return;
        }
        if m != 0.0 {
            self.ta += m;
            self.d += m * (self.cb[p] + self.cb[q]);
            for h in 0..52 {
                self.kab[h] += m * (self.rb[p][h] + self.rb[q][h]);
            }
            self.kba[q] += self.cb[p] * m;
            self.kba[p] += self.cb[q] * m;
            let (rbp, rbq) = (self.rb[p], self.rb[q]);
            for h in 0..52 {
                self.j[q][h] += m * rbp[h];
                self.j[p][h] += m * rbq[h];
            }
            self.ca[p] += m;
            self.ca[q] += m;
            self.ra[p][q] += m;
            self.ra[q][p] += m;
        }
        if n != 0.0 {
            self.tb += n;
            self.d += n * (self.ca[p] + self.ca[q]);
            self.kab[q] += self.ca[p] * n;
            self.kab[p] += self.ca[q] * n;
            for h in 0..52 {
                self.kba[h] += n * (self.ra[p][h] + self.ra[q][h]);
            }
            let (rap, raq) = (self.ra[p], self.ra[q]);
            for h in 0..52 {
                self.jt[q][h] += rap[h] * n;
                self.jt[p][h] += raq[h] * n;
            }
            self.cb[p] += n;
            self.cb[q] += n;
            self.rb[p][q] += n;
            self.rb[q][p] += n;
        }
        if m != 0.0 && n != 0.0 {
            self.e += m * n;
            self.ec[p] += m * n;
            self.ec[q] += m * n;
        }
    }

    #[inline(always)]
    fn jj(&self, h: usize, g: usize) -> f64 {
        self.j[h][g] + self.jt[g][h]
    }

    /// Masa de pares `(x, y)` insertados, compatibles con el héroe `{h1, h2}` y entre sí.
    /// `own = (r_a(i), r_b(i))` si el propio combo del héroe ya está insertado.
    fn mass(&self, h1: usize, h2: usize, own: (f64, f64)) -> f64 {
        let a = self.ta - self.ca[h1] - self.ca[h2] + own.0;
        let b = self.tb - self.cb[h1] - self.cb[h2] + own.1;
        // Σ_{c∉i} A_c·B_c con A_c = ca[c] − RA[c][h1] − RA[c][h2] (a con c, sin cartas del héroe).
        if self.fast {
            // Σ_c ca[c]·RB[c][h] con RB[c][h] ≈ cb[h]·cb[c]/(2·tb − cb[h]) (emparejamiento
            // independiente de las cartas del rango), idem con a y b cambiados.
            let k = |h: usize| {
                let dd = self.d - self.ca[h] * self.cb[h];
                let (xa, xb) = (2.0 * self.ta - self.ca[h], 2.0 * self.tb - self.cb[h]);
                (if xb > 0.0 { self.cb[h] * dd / xb } else { 0.0 }) + (if xa > 0.0 { self.ca[h] * dd / xa } else { 0.0 })
            };
            let all = self.d - k(h1) - k(h2);
            let t1 = (self.ca[h1] - own.0) * (self.cb[h1] - own.1);
            let t2 = (self.ca[h2] - own.0) * (self.cb[h2] - own.1);
            let shared = all - t1 - t2;
            let same = self.e - self.ec[h1] - self.ec[h2] + own.0 * own.1;
            return (a * b - (shared - same)).max(0.0);
        }
        let all = self.d - self.kab[h1] - self.kab[h2] - self.kba[h1] - self.kba[h2] + self.jj(h1, h1) + self.jj(h1, h2) + self.jj(h2, h1) + self.jj(h2, h2);
        let t1 = (self.ca[h1] - self.ra[h1][h2]) * (self.cb[h1] - self.rb[h1][h2]);
        let t2 = (self.ca[h2] - self.ra[h2][h1]) * (self.cb[h2] - self.rb[h2][h1]);
        let shared = all - t1 - t2;
        let same = self.e - self.ec[h1] - self.ec[h2] + own.0 * own.1;
        (a * b - (shared - same)).max(0.0)
    }
}

/// Showdown a 3 con bloqueo exacto entre los tres jugadores. Para cada carril `i` del
/// héroe escribe `lt[i]` = masa de pares de rivales `(x, y)` ambos estrictamente peores,
/// `le[i]` = ambos no mejores (empates incluidos) y `n[i]` = masa total compatible.
pub fn showdown3(board: &SampledBoard, ra: &Lane, rb: &Lane, lt: &mut Lane, le: &mut Lane, n: &mut Lane) {
    showdown3_with(board, ra, rb, false, false, lt, le, n)
}

/// Como [`showdown3`] sin restar de las sumas por carta de los pares que comparten carta
/// los combos que además llevan la otra carta del héroe (término de tercer orden, el
/// mismo que deja `ab_blocking` en el preflop): O(M) en vez de O(52·M).
pub fn showdown3_fast(board: &SampledBoard, ra: &Lane, rb: &Lane, lt: &mut Lane, le: &mut Lane, n: &mut Lane) {
    showdown3_with(board, ra, rb, true, false, lt, le, n)
}

/// [`showdown3_fast`] en cualquier board y en el sentido pedido: con `stronger`, `lt` y `le`
/// cuentan los pares de rivales **ambos mejores** (estrictamente / o empatados) que el héroe.
#[allow(clippy::too_many_arguments)]
pub fn showdown3_dir<B: LaneBoard + ?Sized>(board: &B, ra: &Lane, rb: &Lane, stronger: bool, lt: &mut Lane, le: &mut Lane, n: &mut Lane) {
    showdown3_with(board, ra, rb, true, stronger, lt, le, n)
}

#[allow(clippy::too_many_arguments)]
fn showdown3_with<B: LaneBoard + ?Sized>(board: &B, ra: &Lane, rb: &Lane, fast: bool, rev: bool, lt: &mut Lane, le: &mut Lane, n: &mut Lane) {
    let pairs = board.pairs();
    let mut s = Tri::new(fast);
    let groups: Vec<&[u16]> = if rev { board.groups().windows(2).rev().collect() } else { board.groups().windows(2).collect() };
    for g in groups {
        let (i, j) = (g[0] as usize, g[1] as usize);
        for x in i..j {
            let [p, q] = pairs[x];
            lt[x] = s.mass(p as usize, q as usize, (0.0, 0.0)) as f32;
        }
        for x in i..j {
            let [p, q] = pairs[x];
            s.insert(p as usize, q as usize, ra[x] as f64, rb[x] as f64);
        }
        for x in i..j {
            let [p, q] = pairs[x];
            le[x] = s.mass(p as usize, q as usize, (ra[x] as f64, rb[x] as f64)) as f32;
        }
    }
    for x in 0..M {
        let [p, q] = pairs[x];
        n[x] = s.mass(p as usize, q as usize, (ra[x] as f64, rb[x] as f64)) as f32;
    }
}

/// Masa de pares de rivales `(x, y)` compatibles con cada carril y entre sí.
/// Con la aproximación de [`showdown3_fast`].
pub fn pair_mass<B: LaneBoard + ?Sized>(board: &B, ra: &Lane, rb: &Lane, n: &mut Lane) {
    let pairs = board.pairs();
    let mut s = Tri::new(true);
    for x in 0..M {
        let [p, q] = pairs[x];
        s.insert(p as usize, q as usize, ra[x] as f64, rb[x] as f64);
    }
    for x in 0..M {
        let [p, q] = pairs[x];
        n[x] = s.mass(p as usize, q as usize, (ra[x] as f64, rb[x] as f64)) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::board_pool::{BoardPool, PoolConfig};
    use crate::eval::equity::BlockerSums;
    use crate::eval::showdown::showdown_wtl;
    use crate::eval::NUM_COMBOS;
    use crate::math::rng::SplitMix64;

    /// La pasada conjunta da lo mismo que dos `showdown` separados.
    #[test]
    fn showdown_pair_matches_separate_sweeps() {
        let pool = BoardPool::build(&PoolConfig { boards: 2, buckets: [4, 4, 4], kmeans_samples: 500, kmeans_iters: 2, seed: 2, threads: 2, stratified: true, river_ochs: false });
        let mut rng = SplitMix64::new(9);
        for b in &pool.boards {
            let (mut r0, mut r1) = (lbuf(), lbuf());
            for i in 0..M {
                r0[i] = if rng.next_f32() < 0.4 { 0.0 } else { rng.next_f32() };
                r1[i] = if rng.next_f32() < 0.4 { 0.0 } else { rng.next_f32() };
            }
            let (s0, s1) = (Sums::of(&r0, &b.pairs), Sums::of(&r1, &b.pairs));
            let (c0, c1) = ([3.0, 1.0, -2.0], [5.0, 0.5, -1.0]);
            let (mut o0, mut o1) = (lbuf(), lbuf());
            let d = showdown_pair(b, &r0, &s0, &r1, &s1, c0, c1, &mut o0, &mut o1);
            let (mut w, mut t, mut l) = (lbuf(), lbuf(), lbuf());
            showdown(b, &r1, &s1, &mut w, &mut t, &mut l);
            let mut dw = 0.0f64;
            for i in 0..M {
                let v = c0[0] * w[i] + c0[1] * t[i] + c0[2] * l[i];
                assert!((o0[i] - v).abs() < 1e-2 * (1.0 + v.abs()), "{i}: {} vs {v}", o0[i]);
                dw += (r0[i] * w[i]) as f64;
            }
            assert!((d[0] - dw).abs() < 1e-3 * (1.0 + dw.abs()));
            showdown(b, &r0, &s0, &mut w, &mut t, &mut l);
            for i in 0..M {
                let v = c1[0] * w[i] + c1[1] * t[i] + c1[2] * l[i];
                assert!((o1[i] - v).abs() < 1e-2 * (1.0 + v.abs()), "{i}");
            }
        }
    }

    /// El showdown a 3 con bloqueo coincide con la enumeración de pares de rivales.
    #[test]
    fn showdown3_matches_brute_force() {
        let pool = BoardPool::build(&PoolConfig { boards: 2, buckets: [4, 4, 4], kmeans_samples: 500, kmeans_iters: 2, seed: 5, threads: 2, stratified: true, river_ochs: false });
        let mut rng = SplitMix64::new(11);
        for b in &pool.boards {
            let (mut ra, mut rb) = (lbuf(), lbuf());
            for i in 0..M {
                ra[i] = if rng.next_f32() < 0.85 { 0.0 } else { rng.next_f32() };
                rb[i] = if rng.next_f32() < 0.85 { 0.0 } else { rng.next_f32() };
            }
            let grp: Vec<usize> = {
                let mut v = vec![0; M];
                for (k, g) in b.groups.windows(2).enumerate() {
                    v[g[0] as usize..g[1] as usize].fill(k);
                }
                v
            };
            let (mut lt, mut le, mut n) = (lbuf(), lbuf(), lbuf());
            showdown3(b, &ra, &rb, &mut lt, &mut le, &mut n);
            let (mut flt, mut fle, mut fnn) = (lbuf(), lbuf(), lbuf());
            showdown3_fast(b, &ra, &rb, &mut flt, &mut fle, &mut fnn);
            let dis = |x: usize, y: usize| {
                let ([a, b2], [c, d]) = (b.pairs[x], b.pairs[y]);
                a != c && a != d && b2 != c && b2 != d
            };
            let xs: Vec<usize> = (0..M).filter(|&x| ra[x] > 0.0).collect();
            let ys: Vec<usize> = (0..M).filter(|&y| rb[y] > 0.0).collect();
            for i in (0..M).step_by(7) {
                let (mut e_lt, mut e_le, mut e_n) = (0.0f64, 0.0f64, 0.0f64);
                for &x in xs.iter().filter(|&&x| dis(i, x)) {
                    for &y in ys.iter().filter(|&&y| dis(i, y) && dis(x, y)) {
                        let m = (ra[x] * rb[y]) as f64;
                        e_n += m;
                        if grp[x] < grp[i] && grp[y] < grp[i] {
                            e_lt += m;
                        }
                        if grp[x] <= grp[i] && grp[y] <= grp[i] {
                            e_le += m;
                        }
                    }
                }
                let tol = 1e-3 * (1.0 + e_n);
                assert!((lt[i] as f64 - e_lt).abs() < tol, "lt {i}: {} vs {e_lt}", lt[i]);
                assert!((le[i] as f64 - e_le).abs() < tol, "le {i}: {} vs {e_le}", le[i]);
                assert!((n[i] as f64 - e_n).abs() < tol, "n {i}: {} vs {e_n}", n[i]);
                // Aproximación de tercer orden: a menos del 0,5 % de la masa.
                let ftol = 5e-3 * e_n + 1e-3;
                assert!((flt[i] as f64 - e_lt).abs() < ftol && (fle[i] as f64 - e_le).abs() < ftol && (fnn[i] as f64 - e_n).abs() < ftol, "fast {i}: {} {} {} vs {e_lt} {e_le} {e_n}", flt[i], fle[i], fnn[i]);
            }
        }
    }

    /// El showdown y los bloqueadores en carriles coinciden con la versión por combo.
    #[test]
    fn lanes_match_combo_indexed_versions() {
        let pool = BoardPool::build(&PoolConfig { boards: 3, buckets: [4, 4, 4], kmeans_samples: 500, kmeans_iters: 2, seed: 1, threads: 2, stratified: true, river_ochs: false });
        let mut rng = SplitMix64::new(3);
        for b in &pool.boards {
            assert_eq!(b.order.len(), M);
            let r: Vec<f32> = (0..NUM_COMBOS).map(|_| if rng.next_f32() < 0.3 { 0.0 } else { rng.next_f32() }).collect();
            let (mut w, mut t, mut l) = (vec![0.0; NUM_COMBOS], vec![0.0; NUM_COMBOS], vec![0.0; NUM_COMBOS]);
            showdown_wtl(&b.order, &b.ranks, &r, &mut w, &mut t, &mut l);
            let mut rl = lbuf();
            gather(b, &r, 1.0, &mut rl);
            let s = Sums::of(&rl, &b.pairs);
            let (mut wl, mut tl, mut ll) = (lbuf(), lbuf(), lbuf());
            showdown(b, &rl, &s, &mut wl, &mut tl, &mut ll);
            let bs = BlockerSums::from_indices(&r, &b.order);
            for (i, &h) in b.order.iter().enumerate() {
                let h = h as usize;
                assert!((wl[i] - w[h]).abs() < 1e-2 && (tl[i] - t[h]).abs() < 1e-3 && (ll[i] - l[h]).abs() < 1e-2, "{i}");
                assert!((s.compat(&rl, &b.pairs, i) - bs.compatible(&r, h)).abs() < 1e-2);
            }
        }
    }
}
