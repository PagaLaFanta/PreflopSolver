//! # Pool de runouts representativos + buckets por calle
//!
//! El MCCFR postflop (public chance sampling) muestrea un runout completo
//! (flop+turn+river) de este pool en cada visita. Todo lo que depende solo del board se
//! precalcula aquí una vez, de modo que el recorrido del árbol no evalúa ninguna mano:
//!
//! * `ranks` / `order`: fuerza de 7 cartas de cada combo válido y orden ascendente →
//!   showdown O(n) (`eval::showdown`).
//! * `buckets[calle][combo]`:
//!   - **flop**: histograma sobre los runouts de turn y river del percentil de la mano
//!     en el river (fuerza final, con el potencial de los proyectos) → k-means EMD;
//!   - **turn**: histograma sobre las 48 cartas de river del percentil en el river
//!     → k-means EMD;
//!   - **river**: percentil de fuerza (EHS contra rango uniforme) → cuantiles.
//!
//! Los centroides EMD se entrenan con una submuestra de histogramas de todo el pool, así
//! que un bucket significa lo mismo en todos los boards: condición necesaria para que el
//! imperfect recall fusione infosets de boards distintos de forma coherente.

use super::bucketing::{emd_1d, kmeans_emd, EquityHistogram, EQUITY_BINS};
use crate::eval::evaluator::rank_mask;
use crate::eval::{Card, COMBO_MASKS, NUM_COMBOS};
use crate::math::rng::SplitMix64;

pub const BLOCKED: u16 = u16::MAX;

#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub boards: usize,
    /// Buckets de flop, turn y river.
    pub buckets: [u16; 3],
    /// Histogramas usados para entrenar cada k-means.
    pub kmeans_samples: usize,
    pub kmeans_iters: u32,
    pub seed: u64,
    pub threads: usize,
    /// Flops por muestreo sistemático sobre las 1755 clases isomórficas ponderadas (cada
    /// textura en su proporción exacta) en vez de al azar; turn y river al azar.
    pub stratified: bool,
    /// River: rasgo OCHS de 8 dimensiones (fuerza contra 8 grupos de manos rivales
    /// por fuerza preflop) y k-means global, así que cada board usa solo una parte de los
    /// buckets. Si no, percentil de fuerza por board (256 por
    /// board con 256 buckets: mucho más fino).
    pub river_ochs: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig { boards: 256, buckets: [64, 64, 64], kmeans_samples: 20_000, kmeans_iters: 12, seed: 0xB0A2D, threads: 4, stratified: true, river_ochs: false }
    }
}

#[derive(Debug)]
pub struct SampledBoard {
    pub cards: [Card; 5],
    pub mask: u64,
    /// Rango de 7 cartas; 0 = combo bloqueado por el board.
    pub ranks: Box<[u32]>,
    /// Combos válidos ordenados por `ranks` ascendente.
    pub order: Box<[u16]>,
    /// Combos bloqueados por el board (los 245 que no están en `order`).
    pub blocked: Box<[u16]>,
    /// `buckets[0]` flop, `[1]` turn, `[2]` river; [`BLOCKED`] si el combo choca con el board.
    pub buckets: [Box<[u16]>; 3],
    /// Representación compacta (carriles en el orden de `order`, ver `cfr::lanes`):
    /// cartas de cada combo válido,
    pub pairs: Box<[[u8; 2]]>,
    /// inicio de cada grupo de igual fuerza (más un centinela = `order.len()`),
    pub groups: Box<[u16]>,
    /// y bucket de cada calle por carril.
    pub lane_buckets: [Box<[u16]>; 3],
}

impl SampledBoard {
    #[inline(always)]
    pub fn valid(&self, h: usize) -> bool {
        COMBO_MASKS[h] & self.mask == 0
    }
}

#[derive(Debug)]
pub struct BoardPool {
    pub boards: Vec<SampledBoard>,
    pub num_buckets: [u16; 3],
}

/// Percentil (0..1) de cada combo no bloqueado por `board` (5-7 cartas en total).
/// Empates reciben la posición media del grupo.
fn percentiles(board: u64, out: &mut [f32], scratch: &mut Vec<(u32, u16)>) {
    scratch.clear();
    for h in 0..NUM_COMBOS {
        if COMBO_MASKS[h] & board == 0 {
            scratch.push((rank_mask(board | COMBO_MASKS[h]), h as u16));
        }
    }
    scratch.sort_unstable();
    let n = scratch.len();
    let denom = (n.max(2) - 1) as f32;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && scratch[j].0 == scratch[i].0 {
            j += 1;
        }
        let p = (i + j - 1) as f32 * 0.5 / denom;
        for &(_, h) in &scratch[i..j] {
            out[h as usize] = p;
        }
        i = j;
    }
}

#[inline(always)]
fn bin(p: f32) -> usize {
    ((p * EQUITY_BINS as f32) as usize).min(EQUITY_BINS - 1)
}

/// Histograma de la calle siguiente para todos los combos válidos en `full`, partiendo de
/// `partial` (3 o 4 cartas).
fn next_street_histograms(partial: u64, full: u64, hist: &mut [EquityHistogram], scratch: &mut Vec<(u32, u16)>) {
    let mut pct = vec![0.0f32; NUM_COMBOS];
    let mut counts = vec![0u32; NUM_COMBOS];
    hist.iter_mut().for_each(|h| *h = EquityHistogram::default());
    for c in 0..52u64 {
        let cm = 1u64 << c;
        if partial & cm != 0 {
            continue;
        }
        percentiles(partial | cm, &mut pct, scratch);
        for h in 0..NUM_COMBOS {
            if COMBO_MASKS[h] & (full | cm) == 0 {
                hist[h].mass[bin(pct[h])] += 1.0;
                counts[h] += 1;
            }
        }
    }
    for h in 0..NUM_COMBOS {
        if counts[h] > 0 {
            let inv = 1.0 / counts[h] as f32;
            hist[h].mass.iter_mut().for_each(|m| *m *= inv);
        }
    }
}

/// Histograma de fuerza **final** en el flop: sobre los C(47,2) runouts de turn y river,
/// el percentil de la mano en el river (7 cartas). Con el percentil en el turn (6 cartas)
/// un proyecto de color o de escalera que no liga en el turn cuenta como aire aunque
/// tenga ~20-35 % de equity con el river por salir, y el k-means lo mezclaba con aire.
fn runout_histograms(flop: u64, full: u64, hist: &mut [EquityHistogram], scratch: &mut Vec<(u32, u16)>) {
    let mut pct = vec![0.0f32; NUM_COMBOS];
    let mut counts = vec![0u32; NUM_COMBOS];
    hist.iter_mut().for_each(|h| *h = EquityHistogram::default());
    for t in 0..52u64 {
        let tm = 1u64 << t;
        if flop & tm != 0 {
            continue;
        }
        for r in t + 1..52u64 {
            let rm = 1u64 << r;
            if flop & rm != 0 {
                continue;
            }
            percentiles(flop | tm | rm, &mut pct, scratch);
            for h in 0..NUM_COMBOS {
                if COMBO_MASKS[h] & (full | tm | rm) == 0 {
                    hist[h].mass[bin(pct[h])] += 1.0;
                    counts[h] += 1;
                }
            }
        }
    }
    for h in 0..NUM_COMBOS {
        if counts[h] > 0 {
            let inv = 1.0 / counts[h] as f32;
            hist[h].mass.iter_mut().for_each(|m| *m *= inv);
        }
    }
}

struct Raw {
    cards: [Card; 5],
    mask: u64,
    ranks: Box<[u32]>,
    order: Box<[u16]>,
    river: Box<[u16]>,
    /// Rasgo OCHS del river (8 valores por combo), solo con `river_ochs`.
    ochs: Vec<[f32; OCHS_GROUPS]>,
    flop_hist: Vec<EquityHistogram>,
    turn_hist: Vec<EquityHistogram>,
}

fn sample_board(rng: &mut SplitMix64) -> [Card; 5] {
    let mut cards = [Card(0); 5];
    let mut m = 0u64;
    let mut k = 0;
    while k < 5 {
        let c = (rng.next_u64() % 52) as u8;
        if m & (1 << c) == 0 {
            m |= 1 << c;
            cards[k] = Card(c);
            k += 1;
        }
    }
    cards
}

/// Grupos de rivales del rasgo OCHS del river.
pub const OCHS_GROUPS: usize = 8;

/// Grupo OCHS (0 = manos preflop más fuertes) de cada combo: las 169 clases ordenadas por
/// su equity contra una mano al azar (Monte Carlo determinista) y repartidas en
/// `OCHS_GROUPS` grupos de igual número de combos.
fn ochs_groups() -> &'static [u8; NUM_COMBOS] {
    static G: std::sync::OnceLock<[u8; NUM_COMBOS]> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        let classes = &*crate::cfr::report::CLASS_COMBOS;
        let mut rng = SplitMix64::new(0x0C45_5EED);
        let mut strength: Vec<(f64, usize)> = Vec::with_capacity(169);
        for (cls, combos) in classes.iter().enumerate() {
            let hm = COMBO_MASKS[combos[0] as usize];
            let (mut score, trials) = (0.0f64, 6000);
            for _ in 0..trials {
                let mut used = hm;
                let mut draw = |used: &mut u64| loop {
                    let c = (rng.next_u64() % 52) as u8;
                    if *used & (1 << c) == 0 {
                        *used |= 1 << c;
                        return 1u64 << c;
                    }
                };
                let v = draw(&mut used) | draw(&mut used);
                let mut board = 0u64;
                for _ in 0..5 {
                    board |= draw(&mut used);
                }
                let (a, b) = (rank_mask(board | hm), rank_mask(board | v));
                score += if a > b {
                    1.0
                } else if a == b {
                    0.5
                } else {
                    0.0
                };
            }
            strength.push((score / trials as f64, cls));
        }
        strength.sort_by(|x, y| y.0.total_cmp(&x.0));
        let mut g = [0u8; NUM_COMBOS];
        let mut done = 0usize;
        for (_, cls) in strength {
            for &h in &classes[cls] {
                g[h as usize] = ((done * OCHS_GROUPS) / NUM_COMBOS) as u8;
            }
            done += classes[cls].len();
        }
        g
    })
}

/// Rasgo OCHS de cada combo válido en un board de river: fuerza (victorias + ½ empates)
/// contra cada grupo de rivales, con el bloqueo exacto de cartas.
fn ochs_features(ranks: &[u32], order: &[u16]) -> Vec<[f32; OCHS_GROUPS]> {
    let groups = ochs_groups();
    let n = order.len();
    let mut out = vec![[0.5f32; OCHS_GROUPS]; NUM_COMBOS];
    let mut total = [0u32; OCHS_GROUPS];
    for &h in order {
        total[groups[h as usize] as usize] += 1;
    }
    // `below[g]`: rivales del grupo g con fuerza estrictamente menor que el bloque actual.
    let mut below = [0u32; OCHS_GROUPS];
    let mut i = 0;
    while i < n {
        let r = ranks[order[i] as usize];
        let mut j = i;
        let mut same = [0u32; OCHS_GROUPS];
        while j < n && ranks[order[j] as usize] == r {
            same[groups[order[j] as usize] as usize] += 1;
            j += 1;
        }
        for &h in &order[i..j] {
            let h = h as usize;
            let hm = COMBO_MASKS[h];
            // Quitar los rivales que comparten carta con h (incluido h mismo).
            let (mut w, mut t, mut tot) = (below, same, total);
            for &v in order {
                let v = v as usize;
                if COMBO_MASKS[v] & hm != 0 {
                    let g = groups[v] as usize;
                    tot[g] -= 1;
                    let rv = ranks[v];
                    if rv < r {
                        w[g] -= 1;
                    } else if rv == r {
                        t[g] -= 1;
                    }
                }
            }
            for g in 0..OCHS_GROUPS {
                out[h][g] = if tot[g] > 0 { (w[g] as f32 + 0.5 * t[g] as f32) / tot[g] as f32 } else { 0.5 };
            }
        }
        for g in 0..OCHS_GROUPS {
            below[g] += same[g];
        }
        i = j;
    }
    out
}

/// k-means (L2, k-means++) de los rasgos OCHS; centroides ordenados por fuerza media para
/// que el bucket más alto siga siendo el más fuerte.
fn kmeans_l2(samples: &[[f32; OCHS_GROUPS]], k: usize, iters: u32, seed: u64) -> Vec<[f32; OCHS_GROUPS]> {
    let d2 = |a: &[f32; OCHS_GROUPS], b: &[f32; OCHS_GROUPS]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
    let mut rng = SplitMix64::new(seed);
    let k = k.min(samples.len()).max(1);
    let mut cs: Vec<[f32; OCHS_GROUPS]> = vec![samples[(rng.next_u64() % samples.len() as u64) as usize]];
    let mut best = vec![f32::INFINITY; samples.len()];
    while cs.len() < k {
        let last = *cs.last().unwrap();
        let mut sum = 0.0f64;
        for (i, s) in samples.iter().enumerate() {
            best[i] = best[i].min(d2(s, &last));
            sum += best[i] as f64;
        }
        let mut x = (rng.next_u64() as f64 / u64::MAX as f64) * sum;
        let mut pick = samples.len() - 1;
        for (i, &b) in best.iter().enumerate() {
            x -= b as f64;
            if x <= 0.0 {
                pick = i;
                break;
            }
        }
        cs.push(samples[pick]);
    }
    for _ in 0..iters {
        let mut sums = vec![[0.0f64; OCHS_GROUPS]; k];
        let mut cnt = vec![0usize; k];
        for s in samples {
            let c = nearest_l2(s, &cs) as usize;
            cnt[c] += 1;
            for g in 0..OCHS_GROUPS {
                sums[c][g] += s[g] as f64;
            }
        }
        for c in 0..k {
            if cnt[c] > 0 {
                for g in 0..OCHS_GROUPS {
                    cs[c][g] = (sums[c][g] / cnt[c] as f64) as f32;
                }
            }
        }
    }
    cs.sort_by(|a, b| a.iter().sum::<f32>().total_cmp(&b.iter().sum::<f32>()));
    cs
}

fn nearest_l2(x: &[f32; OCHS_GROUPS], cs: &[[f32; OCHS_GROUPS]]) -> u16 {
    let mut best = (f32::INFINITY, 0u16);
    for (i, c) in cs.iter().enumerate() {
        let d: f32 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best.0 {
            best = (d, i as u16);
        }
    }
    best.1
}

fn raw_board(cards: [Card; 5], river_buckets: u16, river_ochs: bool) -> Raw {
    let mask = cards.iter().fold(0u64, |m, c| m | c.mask());
    let flop = cards[..3].iter().fold(0u64, |m, c| m | c.mask());
    let turn = flop | cards[3].mask();
    let mut scratch = Vec::with_capacity(NUM_COMBOS);
    let (ranks, order) = ranked(mask);
    let mut pct = vec![0.0f32; NUM_COMBOS];
    percentiles(mask, &mut pct, &mut scratch);
    let river: Vec<u16> = (0..NUM_COMBOS)
        .map(|h| {
            if ranks[h] == 0 {
                BLOCKED
            } else {
                ((pct[h] * river_buckets as f32) as u16).min(river_buckets - 1)
            }
        })
        .collect();
    let mut flop_hist = vec![EquityHistogram::default(); NUM_COMBOS];
    let mut turn_hist = vec![EquityHistogram::default(); NUM_COMBOS];
    runout_histograms(flop, mask, &mut flop_hist, &mut scratch);
    next_street_histograms(turn, mask, &mut turn_hist, &mut scratch);
    let ochs = if river_ochs { ochs_features(&ranks, &order) } else { Vec::new() };
    Raw { cards, mask, ranks, order, river: river.into(), ochs, flop_hist, turn_hist }
}

fn nearest(h: &EquityHistogram, centroids: &[EquityHistogram]) -> u16 {
    let mut best = (f32::INFINITY, 0u16);
    for (i, c) in centroids.iter().enumerate() {
        let d = emd_1d(h, c);
        if d < best.0 {
            best = (d, i as u16);
        }
    }
    best.1
}

fn par_map<T: Send, U: Send>(items: Vec<T>, threads: usize, f: impl Fn(T) -> U + Sync) -> Vec<U> {
    let threads = threads.max(1);
    let chunk = items.len().div_ceil(threads).max(1);
    let mut chunks: Vec<Vec<T>> = Vec::new();
    let mut it = items.into_iter().peekable();
    while it.peek().is_some() {
        chunks.push(it.by_ref().take(chunk).collect());
    }
    std::thread::scope(|s| {
        let f = &f;
        let hs: Vec<_> = chunks.into_iter().map(|c| s.spawn(move || c.into_iter().map(f).collect::<Vec<U>>())).collect();
        hs.into_iter().flat_map(|h| h.join().expect("hilo del pool")).collect()
    })
}

/// Clases isomórficas de flop (por permutación de palos): representante y número de
/// flops de la clase (1755 clases, 22 100 flops), ordenadas por textura (pareja o trío,
/// número de palos y después carta alta, media y baja) para que el muestreo sistemático
/// estratifique por textura.
pub fn flop_classes() -> Vec<([Card; 3], u32)> {
    const PERMS: [[u8; 4]; 24] = {
        let mut out = [[0u8; 4]; 24];
        let mut k = 0;
        let mut a = 0;
        while a < 4 {
            let mut b = 0;
            while b < 4 {
                let mut c = 0;
                while c < 4 {
                    let d = 6 - a - b - c;
                    if a != b && a != c && b != c && d < 4 && d != a && d != b && d != c {
                        out[k] = [a as u8, b as u8, c as u8, d as u8];
                        k += 1;
                    }
                    c += 1;
                }
                b += 1;
            }
            a += 1;
        }
        out
    };
    let canon = |f: [u8; 3]| -> [u8; 3] {
        let mut best = [u8::MAX; 3];
        for p in &PERMS {
            let mut g = f.map(|c| (c / 4) * 4 + p[(c % 4) as usize]);
            g.sort_unstable();
            if g < best {
                best = g;
            }
        }
        best
    };
    let mut counts: std::collections::HashMap<[u8; 3], u32> = std::collections::HashMap::new();
    for a in 0..52u8 {
        for b in a + 1..52 {
            for c in b + 1..52 {
                *counts.entry(canon([a, b, c])).or_insert(0) += 1;
            }
        }
    }
    let mut v: Vec<([u8; 3], u32)> = counts.into_iter().collect();
    let key = |f: &[u8; 3]| {
        let mut r = f.map(|c| c / 4);
        r.sort_unstable_by(|x, y| y.cmp(x));
        let mut suits = f.map(|c| c % 4).to_vec();
        suits.sort_unstable();
        suits.dedup();
        let pair = (r[0] == r[1]) as u8 + (r[1] == r[2]) as u8;
        (pair, suits.len(), r[0], r[1], r[2], *f)
    };
    v.sort_by_key(|(f, _)| key(f));
    v.into_iter().map(|(f, w)| (f.map(Card), w)).collect()
}

/// `n` runouts con flops por muestreo sistemático estratificado (ver [`flop_classes`]):
/// palos permutados al azar y turn y river uniformes entre las cartas restantes.
fn stratified_boards(n: usize, rng: &mut SplitMix64) -> Vec<[Card; 5]> {
    let classes = flop_classes();
    let total: f64 = classes.iter().map(|c| c.1 as f64).sum();
    let step = total / n as f64;
    let mut t = (1.0 - rng.next_f64_open0()) * step;
    let (mut i, mut cum) = (0usize, classes[0].1 as f64);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        while t >= cum && i + 1 < classes.len() {
            i += 1;
            cum += classes[i].1 as f64;
        }
        let mut perm = [0u8, 1, 2, 3];
        for k in (1..4).rev() {
            perm.swap(k, (rng.next_u64() % (k as u64 + 1)) as usize);
        }
        let flop = classes[i].0.map(|c| Card((c.0 / 4) * 4 + perm[(c.0 % 4) as usize]));
        let mut used = flop.iter().fold(0u64, |m, c| m | c.mask());
        let mut cards = [flop[0], flop[1], flop[2], Card(0), Card(0)];
        for slot in cards.iter_mut().skip(3) {
            loop {
                let c = (rng.next_u64() % 52) as u8;
                if used & (1 << c) == 0 {
                    used |= 1 << c;
                    *slot = Card(c);
                    break;
                }
            }
        }
        out.push(cards);
        t += step;
    }
    out
}

/// Fuerza de cada combo y combos válidos ordenados por fuerza.
fn ranked(mask: u64) -> (Box<[u32]>, Box<[u16]>) {
    let mut ranks = vec![0u32; NUM_COMBOS];
    let mut order: Vec<u16> = Vec::with_capacity(NUM_COMBOS);
    for h in 0..NUM_COMBOS {
        if COMBO_MASKS[h] & mask == 0 {
            ranks[h] = rank_mask(mask | COMBO_MASKS[h]);
            order.push(h as u16);
        }
    }
    order.sort_unstable_by_key(|&h| ranks[h as usize]);
    (ranks.into(), order.into())
}

/// Board completo a partir de sus cartas, fuerzas y buckets (lo barato de recalcular).
fn assemble(cards: [Card; 5], mask: u64, ranks: Box<[u32]>, order: Box<[u16]>, buckets: [Box<[u16]>; 3]) -> SampledBoard {
    let blocked: Box<[u16]> = (0..NUM_COMBOS as u16).filter(|&h| COMBO_MASKS[h as usize] & mask != 0).collect();
    let pairs: Box<[[u8; 2]]> = order.iter().map(|&h| crate::eval::COMBO_CARDS[h as usize]).collect();
    let mut groups: Vec<u16> = (0..order.len()).filter(|&i| i == 0 || ranks[order[i] as usize] != ranks[order[i - 1] as usize]).map(|i| i as u16).collect();
    groups.push(order.len() as u16);
    let lane_buckets = [0, 1, 2].map(|s| order.iter().map(|&h| buckets[s][h as usize]).collect::<Box<[u16]>>());
    SampledBoard { cards, mask, blocked, buckets, ranks, order, pairs, groups: groups.into(), lane_buckets }
}

const POOL_MAGIC: &[u8; 8] = b"PFPOOL1\0";

impl PoolConfig {
    /// Huella de todo lo que determina el pool (para la caché en disco).
    pub fn fingerprint(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in [self.boards as u64, self.buckets[0] as u64, self.buckets[1] as u64, self.buckets[2] as u64, self.kmeans_samples as u64, self.kmeans_iters as u64, self.seed, self.stratified as u64, 2] {
            for b in x.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        // Solo cambia la huella con el river OCHS: las cachés existentes siguen valiendo.
        if self.river_ochs {
            for b in 0x0C45_u64.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        h
    }
}

impl BoardPool {
    /// Pool desde la caché `dir/pool_<huella>.bin` o construido (y guardado) si no está.
    /// Lo caro (histogramas de equity y k-means EMD) se guarda; fuerzas y carriles se
    /// recalculan al cargar.
    pub fn load_or_build(cfg: &PoolConfig, dir: &std::path::Path) -> Self {
        let path = dir.join(format!("pool_{:016x}.bin", cfg.fingerprint()));
        if let Ok(p) = Self::load(&path, cfg) {
            return p;
        }
        let p = Self::build(cfg);
        let _ = std::fs::create_dir_all(dir);
        if let Err(e) = p.save(&path, cfg) {
            eprintln!("aviso: no se pudo guardar la caché del pool {}: {e}", path.display());
        }
        p
    }

    pub fn save(&self, path: &std::path::Path, cfg: &PoolConfig) -> std::io::Result<()> {
        let mut b = Vec::with_capacity(32 + self.boards.len() * (5 + 3 * NUM_COMBOS * 2));
        b.extend_from_slice(POOL_MAGIC);
        b.extend(cfg.fingerprint().to_le_bytes());
        b.extend((self.boards.len() as u32).to_le_bytes());
        for nb in self.num_buckets {
            b.extend(nb.to_le_bytes());
        }
        for board in &self.boards {
            b.extend(board.cards.iter().map(|c| c.0));
            for s in 0..3 {
                for &x in board.buckets[s].iter() {
                    b.extend(x.to_le_bytes());
                }
            }
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, b)?;
        std::fs::rename(tmp, path)
    }

    pub fn load(path: &std::path::Path, cfg: &PoolConfig) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| e.to_string())?;
        if b.len() < 26 || &b[..8] != POOL_MAGIC {
            return Err("no es una caché de pool".into());
        }
        let rd64 = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        if rd64(8) != cfg.fingerprint() {
            return Err("caché de otra configuración".into());
        }
        let n = u32::from_le_bytes(b[16..20].try_into().unwrap()) as usize;
        let num_buckets = [0, 1, 2].map(|s| u16::from_le_bytes([b[20 + 2 * s], b[21 + 2 * s]]));
        let per = 5 + 3 * NUM_COMBOS * 2;
        if b.len() != 26 + n * per {
            return Err("caché truncada".into());
        }
        let raw: Vec<&[u8]> = (0..n).map(|k| &b[26 + k * per..26 + (k + 1) * per]).collect();
        let boards = par_map(raw, cfg.threads, |r| {
            let cards: [Card; 5] = std::array::from_fn(|i| Card(r[i]));
            let mask = cards.iter().fold(0u64, |m, c| m | c.mask());
            let buckets = [0, 1, 2].map(|s| (0..NUM_COMBOS).map(|h| u16::from_le_bytes([r[5 + 2 * (s * NUM_COMBOS + h)], r[6 + 2 * (s * NUM_COMBOS + h)]])).collect::<Box<[u16]>>());
            let (ranks, order) = ranked(mask);
            assemble(cards, mask, ranks, order, buckets)
        });
        Ok(BoardPool { boards, num_buckets })
    }

    pub fn build(cfg: &PoolConfig) -> Self {
        assert!(cfg.boards > 0 && cfg.buckets.iter().all(|&b| b >= 1));
        let mut rng = SplitMix64::new(cfg.seed);
        let boards: Vec<[Card; 5]> = if cfg.stratified { stratified_boards(cfg.boards, &mut rng) } else { (0..cfg.boards).map(|_| sample_board(&mut rng)).collect() };
        let river_b = cfg.buckets[2];
        let raws = par_map(boards, cfg.threads, |c| raw_board(c, river_b, cfg.river_ochs));

        // Entrenamiento de centroides EMD (flop, turn) con una submuestra global.
        let train = |street: usize, k: u16, rng: &mut SplitMix64| -> Vec<EquityHistogram> {
            let mut samples = Vec::with_capacity(cfg.kmeans_samples);
            let total: usize = raws.iter().map(|r| r.order.len()).sum();
            let stride = (total / cfg.kmeans_samples.max(1)).max(1);
            let mut i = (rng.next_u64() % stride as u64) as usize;
            for r in &raws {
                let hist = if street == 0 { &r.flop_hist } else { &r.turn_hist };
                while i < r.order.len() {
                    samples.push(hist[r.order[i] as usize]);
                    i += stride;
                }
                i -= r.order.len();
            }
            let k = (k as usize).min(samples.len());
            kmeans_emd(&samples, k, cfg.kmeans_iters, rng.next_u64())
        };
        let flop_c = train(0, cfg.buckets[0], &mut rng);
        let turn_c = train(1, cfg.buckets[1], &mut rng);
        // River OCHS: centroides L2 globales con una submuestra de todo el pool.
        let river_c: Vec<[f32; OCHS_GROUPS]> = if cfg.river_ochs {
            let mut samples = Vec::with_capacity(cfg.kmeans_samples);
            let total: usize = raws.iter().map(|r| r.order.len()).sum();
            let stride = (total / cfg.kmeans_samples.max(1)).max(1);
            let mut i = (rng.next_u64() % stride as u64) as usize;
            for r in &raws {
                while i < r.order.len() {
                    samples.push(r.ochs[r.order[i] as usize]);
                    i += stride;
                }
                i -= r.order.len();
            }
            kmeans_l2(&samples, river_b as usize, cfg.kmeans_iters, rng.next_u64())
        } else {
            Vec::new()
        };

        let boards = par_map(raws, cfg.threads, |r| {
            let assign = |hist: &[EquityHistogram], cs: &[EquityHistogram]| -> Box<[u16]> {
                (0..NUM_COMBOS).map(|h| if r.ranks[h] == 0 { BLOCKED } else { nearest(&hist[h], cs) }).collect()
            };
            let river = if river_c.is_empty() {
                r.river
            } else {
                (0..NUM_COMBOS).map(|h| if r.ranks[h] == 0 { BLOCKED } else { nearest_l2(&r.ochs[h], &river_c) }).collect()
            };
            let buckets = [assign(&r.flop_hist, &flop_c), assign(&r.turn_hist, &turn_c), river];
            assemble(r.cards, r.mask, r.ranks, r.order, buckets)
        });
        let river_n = if river_c.is_empty() { river_b } else { river_c.len() as u16 };
        BoardPool { boards, num_buckets: [flop_c.len() as u16, turn_c.len() as u16, river_n] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flop_classes_are_the_1755_isomorphs() {
        let c = flop_classes();
        assert_eq!(c.len(), 1755);
        assert_eq!(c.iter().map(|x| x.1).sum::<u32>(), 22100);
    }

    /// El pool estratificado reproduce las proporciones de textura mejor que el azar:
    /// flops con pareja 16,9 %, monocolor 5,2 %.
    #[test]
    fn stratified_flops_match_texture_frequencies() {
        let mut rng = SplitMix64::new(11);
        let b = stratified_boards(256, &mut rng);
        let frac = |f: &dyn Fn(&[Card; 5]) -> bool| b.iter().filter(|x| f(x)).count() as f64 / b.len() as f64;
        let paired = frac(&|x| x[0].rank() == x[1].rank() || x[1].rank() == x[2].rank() || x[0].rank() == x[2].rank());
        let mono = frac(&|x| x[0].0 % 4 == x[1].0 % 4 && x[1].0 % 4 == x[2].0 % 4);
        assert!((paired - 0.1694).abs() < 0.01, "pareja {paired}");
        assert!((mono - 0.0518).abs() < 0.01, "monocolor {mono}");
        for x in &b {
            assert_eq!(x.iter().fold(0u64, |m, c| m | c.mask()).count_ones(), 5);
        }
    }

    /// La caché en disco devuelve exactamente el mismo pool.
    #[test]
    fn pool_cache_roundtrip() {
        let cfg = PoolConfig { boards: 6, buckets: [4, 4, 4], kmeans_samples: 500, kmeans_iters: 2, seed: 3, threads: 2, stratified: true, river_ochs: false };
        let dir = std::env::temp_dir().join(format!("pool_cache_test_{}", std::process::id()));
        let a = BoardPool::load_or_build(&cfg, &dir);
        let b = BoardPool::load_or_build(&cfg, &dir);
        assert_eq!(a.num_buckets, b.num_buckets);
        for (x, y) in a.boards.iter().zip(&b.boards) {
            assert_eq!(x.mask, y.mask);
            assert_eq!(x.order, y.order);
            assert_eq!(x.groups, y.groups);
            assert_eq!(x.buckets, y.buckets);
            assert_eq!(x.lane_buckets, y.lane_buckets);
        }
        let other = PoolConfig { seed: 4, ..cfg.clone() };
        assert!(BoardPool::load(&dir.join(format!("pool_{:016x}.bin", cfg.fingerprint())), &other).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
    use crate::eval::Combo;

    /// River (OCHS + k-means global): buckets válidos, menos buckets distintos por
    /// board que con el percentil y la mano más fuerte arriba.
    #[test]
    fn ochs_river_buckets_are_coarser_per_board() {
        let base = PoolConfig { boards: 24, buckets: [8, 8, 64], kmeans_samples: 6000, kmeans_iters: 8, seed: 7, threads: 4, stratified: true, river_ochs: false };
        let pct = BoardPool::build(&base);
        let ochs = BoardPool::build(&PoolConfig { river_ochs: true, ..base.clone() });
        assert_ne!(base.fingerprint(), PoolConfig { river_ochs: true, ..base.clone() }.fingerprint());
        let distinct = |p: &BoardPool| -> f64 {
            p.boards.iter().map(|b| b.buckets[2].iter().filter(|&&x| x != BLOCKED).collect::<std::collections::HashSet<_>>().len() as f64).sum::<f64>() / p.boards.len() as f64
        };
        let (dp, dochs) = (distinct(&pct), distinct(&ochs));
        assert!(dochs < 0.7 * dp, "OCHS {dochs} frente a percentil {dp}");
        for b in &ochs.boards {
            for h in 0..NUM_COMBOS {
                let x = b.buckets[2][h];
                assert_eq!(x == BLOCKED, !b.valid(h));
                assert!(x == BLOCKED || x < ochs.num_buckets[2]);
            }
            let top = *b.order.last().unwrap() as usize;
            assert!(b.buckets[2][top] >= ochs.num_buckets[2] * 3 / 4, "la mano más fuerte en un bucket bajo: {}", b.buckets[2][top]);
        }
    }

    #[test]
    fn pool_buckets_are_consistent() {
        let pool = BoardPool::build(&PoolConfig { boards: 6, buckets: [8, 8, 8], kmeans_samples: 3000, kmeans_iters: 6, seed: 1, threads: 2, stratified: true, river_ochs: false });
        for b in &pool.boards {
            assert_eq!(b.order.len(), 1081, "C(47,2) combos válidos");
            for h in 0..NUM_COMBOS {
                for s in 0..3 {
                    let x = b.buckets[s][h];
                    assert_eq!(x == BLOCKED, !b.valid(h));
                    assert!(x == BLOCKED || x < pool.num_buckets[s]);
                }
            }
            // El combo más fuerte del river está en el bucket de river más alto.
            let top = *b.order.last().unwrap() as usize;
            assert_eq!(b.buckets[2][top], pool.num_buckets[2] - 1);
            let _ = Combo(top as u16);
        }
    }
}
