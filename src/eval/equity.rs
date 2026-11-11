//! # Matriz de equity preflop 1326 × 1326
//!
//! `win[h][v]` = P(h gana a v con el board completo), `tie[h][v]` = P(empate). Pares que
//! comparten carta valen 0 en ambas. Almacenada con stride [`HAND_STRIDE`] (1328) para
//! que cada fila sea directamente un operando SIMD contra un `RangeVec`.
//!
//! ## Construcción: Monte Carlo compartido + simetrización por palos
//! Enumerar C(48,5) boards por par (~880k pares) es inviable. En su lugar:
//! 1. Se muestrean `B` boards de 5 cartas. En cada uno se evalúan los 1326 combos **una
//!    vez** y se acumulan win/tie/count para **todos** los pares compatibles
//!    (≈584k pares/board). El bucle interno recorre el triángulo inferior de forma
//!    contigua y sin ramas → vectorizable.
//! 2. **Simetrización**: la equity es invariante bajo las 24 permutaciones de palos,
//!    `eq(πh, πv) = eq(h, v)`. Sumar los contadores de toda la órbita equivale a
//!    multiplicar por ~24 los boards efectivos sin evaluar ninguna mano más.
//!    Con B = 20 000 → ~480k boards efectivos → error típico ≈ 0.08 %.
//! 3. Se paraleliza por hilos con acumuladores privados (sin atómicos) y reducción final.
//!
//! El resultado se cachea en disco (≈14 MB) para no recalcular entre ejecuciones.

use super::evaluator::rank_mask;
use super::{Combo, EquityOracle, COMBO_CARDS, COMBO_MASKS, COMPATIBLE_COMBOS, HAND_STRIDE, NUM_COMBOS};
use crate::math::rng::SplitMix64;
use std::io::{Read, Write};
use std::path::Path;

const TRI: usize = NUM_COMBOS * (NUM_COMBOS - 1) / 2;
const MAGIC: &[u8; 8] = b"PFEQ0001";

#[inline(always)]
fn tri(lo: usize, hi: usize) -> usize {
    hi * (hi - 1) / 2 + lo
}

/// Sumas para excluir combos bloqueados en O(1): `Σ_{v ∩ h = ∅} r[v]`.
#[derive(Clone, Debug)]
pub struct BlockerSums {
    pub total: f32,
    pub card: [f32; 52],
}

impl BlockerSums {
    pub fn new(r: &[f32]) -> Self {
        let mut s = BlockerSums { total: 0.0, card: [0.0; 52] };
        for (h, &x) in r[..NUM_COMBOS].iter().enumerate() {
            if x != 0.0 {
                let [a, b] = COMBO_CARDS[h];
                s.total += x;
                s.card[a as usize] += x;
                s.card[b as usize] += x;
            }
        }
        s
    }

    /// Igual que [`Self::new`] pero sumando solo los combos de `idx` (los válidos en un
    /// board): el resto del vector puede contener basura.
    pub fn from_indices(r: &[f32], idx: &[u16]) -> Self {
        let mut s = BlockerSums { total: 0.0, card: [0.0; 52] };
        for &h in idx {
            let x = r[h as usize];
            let [a, b] = COMBO_CARDS[h as usize];
            s.total += x;
            s.card[a as usize] += x;
            s.card[b as usize] += x;
        }
        s
    }

    /// Inclusión-exclusión: `h` se resta dos veces (una por carta) y se suma una.
    #[inline(always)]
    pub fn compatible(&self, r: &[f32], h: usize) -> f32 {
        let [a, b] = COMBO_CARDS[h];
        self.total - self.card[a as usize] - self.card[b as usize] + r[h]
    }
}

#[derive(Debug)]
pub struct EquityMatrix {
    win: Vec<f32>,
    tie: Vec<f32>,
    /// Copias cuantizadas a `u16` (prob × 65535) para los productos matriz-vector del
    /// CFR: la mitad de ancho de banda (7 MB en vez de 14 MB por las dos matrices, que
    /// caben mejor en L3); error de cuantización ≤ 7.6e-6.
    winq: Vec<u16>,
    tieq: Vec<u16>,
    pub boards: u32,
}

fn quantize(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| (x * 65535.0).round().clamp(0.0, 65535.0) as u16).collect()
}

struct Acc {
    win: Vec<u32>,
    tie: Vec<u32>,
    cnt: Vec<u32>,
}

impl Acc {
    fn new() -> Self {
        Acc { win: vec![0; TRI], tie: vec![0; TRI], cnt: vec![0; TRI] }
    }

    fn add_board(&mut self, board: u64, ranks: &mut [u32], masks: &mut [u64]) {
        for h in 0..NUM_COMBOS {
            if COMBO_MASKS[h] & board == 0 {
                ranks[h] = rank_mask(board | COMBO_MASKS[h]);
                masks[h] = COMBO_MASKS[h];
            } else {
                ranks[h] = 0;
                masks[h] = u64::MAX; // se solapa con todo ⇒ par inválido
            }
        }
        for v in 1..NUM_COMBOS {
            let rv = ranks[v];
            if rv == 0 {
                continue;
            }
            let mv = masks[v];
            let base = tri(0, v);
            let (win, tie, cnt) = (&mut self.win[base..base + v], &mut self.tie[base..base + v], &mut self.cnt[base..base + v]);
            for h in 0..v {
                let ok = ((masks[h] & mv) == 0) as u32;
                win[h] += ok & (ranks[h] > rv) as u32;
                tie[h] += ok & (ranks[h] == rv) as u32;
                cnt[h] += ok;
            }
        }
    }

    fn merge(&mut self, o: &Acc) {
        for i in 0..TRI {
            self.win[i] += o.win[i];
            self.tie[i] += o.tie[i];
            self.cnt[i] += o.cnt[i];
        }
    }
}

/// Boards de 5 cartas canónicos bajo permutaciones de palos (máscaras de rango por palo en
/// orden no creciente) con el tamaño de su órbita: 134 459 boards cuyas órbitas cubren los
/// C(52,5) = 2 598 960.
pub(crate) fn canonical_boards() -> Vec<(u64, u32)> {
    let mut out = Vec::with_capacity(140_000);
    for a in 0..48u32 {
        for b in a + 1..49 {
            for c in b + 1..50 {
                for d in c + 1..51 {
                    for e in d + 1..52 {
                        let mut sm = [0u16; 4];
                        for x in [a, b, c, d, e] {
                            sm[(x & 3) as usize] |= 1 << (x >> 2);
                        }
                        if sm[0] < sm[1] || sm[1] < sm[2] || sm[2] < sm[3] {
                            continue;
                        }
                        // Estabilizador: permutaciones de palos con la misma máscara.
                        let mut stab = 1u32;
                        let mut run = 1u32;
                        for i in 1..4 {
                            if sm[i] == sm[i - 1] {
                                run += 1;
                                stab *= run;
                            } else {
                                run = 1;
                            }
                        }
                        out.push(((1u64 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e), 24 / stab));
                    }
                }
            }
        }
    }
    out
}

/// Acumuladores de la enumeración exacta: win/tie/count ponderados por órbita y, para cada
/// función base `s_k(B) = Σ_{c∈B} b_k(rango c)`, las mismas sumas multiplicadas por `s_k`.
struct ExactAcc {
    win: Vec<u32>,
    tie: Vec<u32>,
    cnt: Vec<u32>,
    ws: Vec<Vec<f32>>,
    ts: Vec<Vec<f32>>,
    cs: Vec<Vec<f32>>,
}

/// Un hilo: filas `v` de su rango, con sus trozos (contiguos) de cada acumulador.
struct RowBlock<'a> {
    v0: usize,
    v1: usize,
    win: &'a mut [u32],
    tie: &'a mut [u32],
    cnt: &'a mut [u32],
    ws: Vec<&'a mut [f32]>,
    ts: Vec<&'a mut [f32]>,
    cs: Vec<&'a mut [f32]>,
}

fn split_by<'a, T>(mut v: &'a mut [T], cuts: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(cuts.len() - 1);
    for w in cuts.windows(2) {
        let (a, b) = std::mem::take(&mut v).split_at_mut(w[1] - w[0]);
        out.push(a);
        v = b;
    }
    out
}

/// Enumeración exacta de todos los boards (por órbitas de palos). `basis` vacío: solo la
/// matriz de equity.
fn enumerate_exact(threads: usize, basis: &[[f32; 13]]) -> ExactAcc {
    let boards = canonical_boards();
    let k = basis.len();
    let mut acc = ExactAcc {
        win: vec![0; TRI],
        tie: vec![0; TRI],
        cnt: vec![0; TRI],
        ws: vec![vec![0.0; TRI]; k],
        ts: vec![vec![0.0; TRI]; k],
        cs: vec![vec![0.0; TRI]; k],
    };
    let threads = threads.clamp(1, 64);
    // Filas repartidas por área (la fila v tiene v pares).
    let mut rows: Vec<usize> = (0..=threads).map(|t| ((t as f64 / threads as f64).sqrt() * NUM_COMBOS as f64).round() as usize).collect();
    rows[0] = 1;
    rows[threads] = NUM_COMBOS;
    rows.dedup();
    let cuts: Vec<usize> = rows.iter().map(|&v| tri(0, v)).collect();
    const CHUNK: usize = 256;
    let mut ranks = vec![0u32; CHUNK * NUM_COMBOS];
    let mut svals = vec![0.0f32; CHUNK * k.max(1)];
    for chunk in boards.chunks(CHUNK) {
        // Rango de 7 cartas de cada combo en cada board del bloque (0 = bloqueado).
        std::thread::scope(|s| {
            let per = chunk.len().div_ceil(threads);
            for (bs, rs) in chunk.chunks(per).zip(ranks.chunks_mut(per * NUM_COMBOS)) {
                s.spawn(move || {
                    for (&(board, _), r) in bs.iter().zip(rs.chunks_mut(NUM_COMBOS)) {
                        for h in 0..NUM_COMBOS {
                            r[h] = if COMBO_MASKS[h] & board == 0 { rank_mask(board | COMBO_MASKS[h]) } else { 0 };
                        }
                    }
                });
            }
        });
        for (i, &(board, w)) in chunk.iter().enumerate() {
            for (j, b) in basis.iter().enumerate() {
                let mut m = board;
                let mut x = 0.0f32;
                while m != 0 {
                    x += b[(m.trailing_zeros() >> 2) as usize];
                    m &= m - 1;
                }
                svals[i * k + j] = x * w as f32;
            }
        }
        let (ranks, svals) = (&ranks, &svals);
        let mut blocks: Vec<RowBlock> = {
            let (mut win, mut tie, mut cnt) = (split_by(&mut acc.win, &cuts), split_by(&mut acc.tie, &cuts), split_by(&mut acc.cnt, &cuts));
            let mut ws: Vec<Vec<&mut [f32]>> = acc.ws.iter_mut().map(|v| split_by(v, &cuts)).collect();
            let mut ts: Vec<Vec<&mut [f32]>> = acc.ts.iter_mut().map(|v| split_by(v, &cuts)).collect();
            let mut cs: Vec<Vec<&mut [f32]>> = acc.cs.iter_mut().map(|v| split_by(v, &cuts)).collect();
            (0..rows.len() - 1)
                .rev()
                .map(|t| RowBlock {
                    v0: rows[t],
                    v1: rows[t + 1],
                    win: win.pop().unwrap(),
                    tie: tie.pop().unwrap(),
                    cnt: cnt.pop().unwrap(),
                    ws: ws.iter_mut().map(|x| x.pop().unwrap()).collect(),
                    ts: ts.iter_mut().map(|x| x.pop().unwrap()).collect(),
                    cs: cs.iter_mut().map(|x| x.pop().unwrap()).collect(),
                })
                .collect()
        };
        std::thread::scope(|s| {
            for blk in blocks.iter_mut() {
                s.spawn(move || {
                    let base = tri(0, blk.v0);
                    let (mut gtf, mut eqf, mut okf) = (vec![0.0f32; NUM_COMBOS], vec![0.0f32; NUM_COMBOS], vec![0.0f32; NUM_COMBOS]);
                    for v in blk.v0..blk.v1 {
                        let off = tri(0, v) - base;
                        let mv = COMBO_MASKS[v];
                        for (i, &(_, w)) in chunk.iter().enumerate() {
                            let r = &ranks[i * NUM_COMBOS..(i + 1) * NUM_COMBOS];
                            let rv = r[v];
                            if rv == 0 {
                                continue;
                            }
                            let (win, tie, cnt) = (&mut blk.win[off..off + v], &mut blk.tie[off..off + v], &mut blk.cnt[off..off + v]);
                            if k == 0 {
                                for h in 0..v {
                                    let ok = (((COMBO_MASKS[h] & mv) == 0) & (r[h] != 0)) as u32 * w;
                                    win[h] += ok * (r[h] > rv) as u32;
                                    tie[h] += ok * (r[h] == rv) as u32;
                                    cnt[h] += ok;
                                }
                                continue;
                            }
                            for h in 0..v {
                                let ok = (((COMBO_MASKS[h] & mv) == 0) & (r[h] != 0)) as u32;
                                let (g, e) = (ok * (r[h] > rv) as u32, ok * (r[h] == rv) as u32);
                                win[h] += w * g;
                                tie[h] += w * e;
                                cnt[h] += w * ok;
                                gtf[h] = g as f32;
                                eqf[h] = e as f32;
                                okf[h] = ok as f32;
                            }
                            for j in 0..k {
                                let sv = svals[i * k + j];
                                let (ws, ts, cs) = (&mut blk.ws[j][off..off + v], &mut blk.ts[j][off..off + v], &mut blk.cs[j][off..off + v]);
                                for (x, &g) in ws.iter_mut().zip(&gtf[..v]) {
                                    *x += sv * g;
                                }
                                for (x, &e) in ts.iter_mut().zip(&eqf[..v]) {
                                    *x += sv * e;
                                }
                                for (x, &o) in cs.iter_mut().zip(&okf[..v]) {
                                    *x += sv * o;
                                }
                            }
                        }
                    }
                });
            }
        });
    }
    acc
}

fn random_board(rng: &mut SplitMix64) -> u64 {
    let mut m = 0u64;
    while m.count_ones() < 5 {
        m |= 1 << (rng.next_u64() % 52);
    }
    m
}

/// Las 24 permutaciones de palos aplicadas a índices de combo.
fn suit_permutations() -> Vec<[u16; NUM_COMBOS]> {
    let mut perms = Vec::with_capacity(24);
    for a in 0..4u8 {
        for b in 0..4u8 {
            for c in 0..4u8 {
                for d in 0..4u8 {
                    let p = [a, b, c, d];
                    let mut seen = 0u8;
                    p.iter().for_each(|&x| seen |= 1 << x);
                    if seen != 0b1111 {
                        continue;
                    }
                    let mut map = [0u16; NUM_COMBOS];
                    for (h, m) in map.iter_mut().enumerate() {
                        let [x, y] = COMBO_CARDS[h];
                        let px = (x & !3) | p[(x & 3) as usize];
                        let py = (y & !3) | p[(y & 3) as usize];
                        *m = Combo::from_cards(super::Card(px), super::Card(py)).0;
                    }
                    perms.push(map);
                }
            }
        }
    }
    perms
}

impl EquityMatrix {
    #[inline(always)]
    pub fn win_row(&self, h: usize) -> &[f32] {
        &self.win[h * HAND_STRIDE..(h + 1) * HAND_STRIDE]
    }
    #[inline(always)]
    pub fn tie_row(&self, h: usize) -> &[f32] {
        &self.tie[h * HAND_STRIDE..(h + 1) * HAND_STRIDE]
    }

    /// Monte Carlo con `boards` boards repartidos en `threads` hilos.
    pub fn monte_carlo(boards: u32, seed: u64, threads: usize) -> Self {
        Self::monte_carlo_with(boards, seed, threads, random_board)
    }

    /// Como [`Self::monte_carlo`] con otro generador de boards (debe ser invariante bajo
    /// permutaciones de palos: la simetrización lo supone).
    pub(crate) fn monte_carlo_with(boards: u32, seed: u64, threads: usize, gen: impl Fn(&mut SplitMix64) -> u64 + Sync) -> Self {
        let threads = threads.max(1);
        let per = boards.div_ceil(threads as u32);
        let accs: Vec<Acc> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let n = per.min(boards.saturating_sub(per * t as u32));
                    let gen = &gen;
                    s.spawn(move || {
                        let mut acc = Acc::new();
                        let mut rng = SplitMix64::new(seed ^ (0xA5A5_0000 + t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let mut ranks = vec![0u32; NUM_COMBOS];
                        let mut masks = vec![0u64; NUM_COMBOS];
                        for _ in 0..n {
                            acc.add_board(gen(&mut rng), &mut ranks, &mut masks);
                        }
                        acc
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("equity thread")).collect()
        });
        let mut it = accs.into_iter();
        let mut acc = it.next().unwrap();
        for o in it {
            acc.merge(&o);
        }
        Self::symmetrize(&acc, boards)
    }

    /// Matriz **exacta**: todos los boards, enumerados por órbitas de palos (`boards = 0`).
    pub fn exact(threads: usize) -> Self {
        Self::exact_with_basis(threads, &[]).0
    }

    /// Matriz exacta y, para cada función base `b_k` de rango, las matrices de covarianza
    /// `D_k(h,v) = Cov_B(win(h,v,B), Σ_{c∈B} b_k(rango c))` (y la de empates) del board
    /// bunching (ver `board_bunch`).
    pub(crate) fn exact_with_basis(threads: usize, basis: &[[f32; 13]]) -> (Self, Vec<(Vec<f32>, Vec<f32>)>) {
        let acc = enumerate_exact(threads, basis);
        let perms = suit_permutations();
        let k = basis.len();
        let mut win = vec![0.0f32; NUM_COMBOS * HAND_STRIDE];
        let mut tie = vec![0.0f32; NUM_COMBOS * HAND_STRIDE];
        let mut dm: Vec<(Vec<f32>, Vec<f32>)> = (0..k).map(|_| (vec![0.0f32; NUM_COMBOS * HAND_STRIDE], vec![0.0f32; NUM_COMBOS * HAND_STRIDE])).collect();
        let (mut sw, mut st, mut sc) = (vec![0.0f64; k], vec![0.0f64; k], vec![0.0f64; k]);
        for v in 1..NUM_COMBOS {
            for h in 0..v {
                if COMBO_MASKS[h] & COMBO_MASKS[v] != 0 {
                    continue;
                }
                let (mut w, mut t, mut c) = (0u64, 0u64, 0u64);
                sw.fill(0.0);
                st.fill(0.0);
                sc.fill(0.0);
                for p in &perms {
                    let (a, b) = (p[h] as usize, p[v] as usize);
                    let i = if a < b { tri(a, b) } else { tri(b, a) };
                    let (wi, ti, ci) = (acc.win[i] as u64, acc.tie[i] as u64, acc.cnt[i] as u64);
                    w += if a < b { wi } else { ci - wi - ti };
                    t += ti;
                    c += ci;
                    for j in 0..k {
                        let (x, y, z) = (acc.ws[j][i] as f64, acc.ts[j][i] as f64, acc.cs[j][i] as f64);
                        sw[j] += if a < b { x } else { z - x - y };
                        st[j] += y;
                        sc[j] += z;
                    }
                }
                if c == 0 {
                    continue;
                }
                let inv = 1.0 / c as f64;
                let (pw, pt) = (w as f64 * inv, t as f64 * inv);
                win[h * HAND_STRIDE + v] = pw as f32;
                win[v * HAND_STRIDE + h] = (1.0 - pw - pt) as f32;
                tie[h * HAND_STRIDE + v] = pt as f32;
                tie[v * HAND_STRIDE + h] = pt as f32;
                for (j, (dw, dt)) in dm.iter_mut().enumerate() {
                    let (ew, et, es) = (sw[j] * inv, st[j] * inv, sc[j] * inv);
                    let el = es - ew - et;
                    dw[h * HAND_STRIDE + v] = (ew - pw * es) as f32;
                    dw[v * HAND_STRIDE + h] = (el - (1.0 - pw - pt) * es) as f32;
                    let d = (et - pt * es) as f32;
                    dt[h * HAND_STRIDE + v] = d;
                    dt[v * HAND_STRIDE + h] = d;
                }
            }
        }
        (EquityMatrix::from_parts(win, tie, 0), dm)
    }

    fn symmetrize(acc: &Acc, boards: u32) -> Self {
        let perms = suit_permutations();
        let mut win = vec![0.0f32; NUM_COMBOS * HAND_STRIDE];
        let mut tie = vec![0.0f32; NUM_COMBOS * HAND_STRIDE];
        for v in 1..NUM_COMBOS {
            for h in 0..v {
                if COMBO_MASKS[h] & COMBO_MASKS[v] != 0 {
                    continue;
                }
                let (mut w, mut t, mut c) = (0u64, 0u64, 0u64);
                for p in &perms {
                    let (a, b) = (p[h] as usize, p[v] as usize);
                    let i = if a < b { tri(a, b) } else { tri(b, a) };
                    let (wi, ti, ci) = (acc.win[i] as u64, acc.tie[i] as u64, acc.cnt[i] as u64);
                    // `win[tri(lo,hi)]` cuenta victorias de `lo`; si la permutación invierte
                    // el orden, las victorias de `πh` son las derrotas de `lo`.
                    w += if a < b { wi } else { ci - wi - ti };
                    t += ti;
                    c += ci;
                }
                if c == 0 {
                    continue;
                }
                let inv = 1.0 / c as f64;
                let (pw, pt) = (w as f64 * inv, t as f64 * inv);
                win[h * HAND_STRIDE + v] = pw as f32;
                win[v * HAND_STRIDE + h] = (1.0 - pw - pt) as f32;
                tie[h * HAND_STRIDE + v] = pt as f32;
                tie[v * HAND_STRIDE + h] = pt as f32;
            }
        }
        EquityMatrix::from_parts(win, tie, boards)
    }

    fn from_parts(win: Vec<f32>, tie: Vec<f32>, boards: u32) -> Self {
        let (winq, tieq) = (quantize(&win), quantize(&tie));
        EquityMatrix { win, tie, winq, tieq, boards }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&self.boards.to_le_bytes())?;
        for x in self.win.iter().chain(self.tie.iter()) {
            f.write_all(&x.to_le_bytes())?;
        }
        f.flush()
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
        let n = NUM_COMBOS * HAND_STRIDE;
        if bytes.len() != 12 + 8 * n || &bytes[..8] != MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid equity cache"));
        }
        let boards = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let floats: Vec<f32> = bytes[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let (win, tie) = floats.split_at(n);
        Ok(EquityMatrix::from_parts(win.to_vec(), tie.to_vec(), boards))
    }

    /// Carga la caché si existe y coincide en nº de boards; si no, construye y guarda.
    pub fn load_or_build(path: &Path, boards: u32, seed: u64, threads: usize) -> Self {
        if let Ok(m) = Self::load(path) {
            if m.boards == boards {
                return m;
            }
        }
        if boards == 0 {
            eprintln!("equity matrix: enumerating every board (first time only, ~1 min)…");
        }
        let m = if boards == 0 { Self::exact(threads) } else { Self::monte_carlo(boards, seed, threads) };
        if let Err(e) = m.save(path) {
            eprintln!("warning: could not save the equity cache to {}: {e}", path.display());
        }
        m
    }

    /// `W·r`, `T·r` y la masa compatible de `r` para cada combo, en una sola pasada por
    /// las dos matrices (cada fila de W y T se lee una vez, contigua).
    ///
    /// Con rangos estrechos (menos de la mitad de combos con alcance, lo normal en botes de
    /// 3-bet y 4-bet) recorre solo las filas de los combos del rango: `T` es simétrica y
    /// `W[h][v] = compatible(h, v) − W[v][h] − T[v][h]`, así que
    /// `(W·r)[h] = masa compatible − (T·r)[h] − Σ_v r[v]·W[v][h]`, con coste ∝ tamaño del rango.
    pub fn products(&self, r: &[f32]) -> PairProducts {
        let bs = BlockerSums::new(r);
        let nz = r[..NUM_COMBOS].iter().filter(|&&x| x != 0.0).count();
        if nz * 2 < NUM_COMBOS {
            self.products_sparse(r, &bs)
        } else {
            self.products_dense(r, &bs)
        }
    }

    fn products_dense(&self, r: &[f32], bs: &BlockerSums) -> PairProducts {
        let mut p = PairProducts { w: vec![0.0; HAND_STRIDE], t: vec![0.0; HAND_STRIDE], all: vec![0.0; HAND_STRIDE] };
        for h in 0..NUM_COMBOS {
            let row = h * HAND_STRIDE..(h + 1) * HAND_STRIDE;
            let (w, t) = dot2_q(&self.winq[row.clone()], &self.tieq[row], r);
            p.w[h] = w;
            p.t[h] = t;
            p.all[h] = bs.compatible(r, h);
        }
        p
    }

    fn products_sparse(&self, r: &[f32], bs: &BlockerSums) -> PairProducts {
        let (mut aw, mut at) = (vec![0.0f32; HAND_STRIDE], vec![0.0f32; HAND_STRIDE]);
        for (v, &x) in r[..NUM_COMBOS].iter().enumerate() {
            if x != 0.0 {
                let row = v * HAND_STRIDE..(v + 1) * HAND_STRIDE;
                axpy2_q(x, &self.winq[row.clone()], &self.tieq[row], &mut aw, &mut at);
            }
        }
        const S: f32 = 1.0 / 65535.0;
        let mut p = PairProducts { w: vec![0.0; HAND_STRIDE], t: vec![0.0; HAND_STRIDE], all: vec![0.0; HAND_STRIDE] };
        for h in 0..NUM_COMBOS {
            let all = bs.compatible(r, h);
            p.t[h] = at[h] * S;
            p.w[h] = (all - p.t[h] - aw[h] * S).max(0.0);
            p.all[h] = all;
        }
        p
    }

    /// Valores del héroe contra el rango `r_opp` en un terminal de showdown/all-in:
    /// `out[h] = Σ_v r[v]·(W·u_w + T·u_t + L·u_l) / 1225`, con `L = 1 − W − T` solo en
    /// pares compatibles (vía [`BlockerSums`]).
    pub fn hero_values(&self, r_opp: &[f32], u_win: f32, u_tie: f32, u_lose: f32, out: &mut [f32]) {
        self.products(r_opp).hero_values(u_win, u_tie, u_lose, out);
    }

    /// Masas de outcome `(gana a, gana b, empate)` para dos rangos, normalizadas por
    /// `1326·1225` (probabilidad conjunta de que ambos lleguen, ignorando bloqueo con
    /// terceros).
    pub fn pair_outcomes(&self, r_a: &[f32], r_b: &[f32]) -> (f64, f64, f64) {
        self.products(r_b).outcomes(r_a)
    }
}

/// Como [`dot2`] con filas `u16` (prob × 65535); la conversión a f32 vectoriza.
#[inline]
fn dot2_q(a: &[u16], b: &[u16], r: &[f32]) -> (f32, f32) {
    let (mut x, mut y) = ([0.0f32; 16], [0.0f32; 16]);
    for ((ca, cb), cr) in a.chunks_exact(16).zip(b.chunks_exact(16)).zip(r.chunks_exact(16)) {
        for l in 0..16 {
            x[l] += ca[l] as f32 * cr[l];
            y[l] += cb[l] as f32 * cr[l];
        }
    }
    const S: f32 = 1.0 / 65535.0;
    (x.iter().sum::<f32>() * S, y.iter().sum::<f32>() * S)
}

/// `aw += x·a`, `at += x·b` con filas `u16` (sin escalar).
#[inline]
fn axpy2_q(x: f32, a: &[u16], b: &[u16], aw: &mut [f32], at: &mut [f32]) {
    for ((ca, cb), (cw, ct)) in a.chunks_exact(16).zip(b.chunks_exact(16)).zip(aw.chunks_exact_mut(16).zip(at.chunks_exact_mut(16))) {
        for l in 0..16 {
            cw[l] += x * ca[l] as f32;
            ct[l] += x * cb[l] as f32;
        }
    }
}

/// Dos productos escalares simultáneos (misma lectura de `r`).
#[inline]
#[allow(dead_code)]
fn dot2(a: &[f32], b: &[f32], r: &[f32]) -> (f32, f32) {
    let (mut x, mut y) = ([0.0f32; 16], [0.0f32; 16]);
    for ((ca, cb), cr) in a.chunks_exact(16).zip(b.chunks_exact(16)).zip(r.chunks_exact(16)) {
        for l in 0..16 {
            x[l] += ca[l] * cr[l];
            y[l] += cb[l] * cr[l];
        }
    }
    (x.iter().sum(), y.iter().sum())
}

/// Productos matriz-vector contra el rango de un rival; de ellos salen tanto los valores
/// del otro jugador como las masas de outcome que necesitan los observadores.
#[derive(Debug, Clone)]
pub struct PairProducts {
    pub w: Vec<f32>,
    pub t: Vec<f32>,
    pub all: Vec<f32>,
}

impl PairProducts {
    pub fn hero_values(&self, u_win: f32, u_tie: f32, u_lose: f32, out: &mut [f32]) {
        let (dw, dt) = (u_win - u_lose, u_tie - u_lose);
        let norm = 1.0 / COMPATIBLE_COMBOS;
        for h in 0..NUM_COMBOS {
            out[h] = (dw * self.w[h] + dt * self.t[h] + u_lose * self.all[h]) * norm;
        }
        out[NUM_COMBOS..].fill(0.0);
    }

    /// `(P(héroe gana), P(rival gana), P(empate))` con el rango del héroe `r_hero`.
    pub fn outcomes(&self, r_hero: &[f32]) -> (f64, f64, f64) {
        let (mut pw, mut pt, mut all) = (0.0f64, 0.0f64, 0.0f64);
        for x in 0..NUM_COMBOS {
            let r = r_hero[x] as f64;
            pw += r * self.w[x] as f64;
            pt += r * self.t[x] as f64;
            all += r * self.all[x] as f64;
        }
        let norm = 1.0 / (NUM_COMBOS as f64 * COMPATIBLE_COMBOS as f64);
        (pw * norm, (all - pw - pt) * norm, pt * norm)
    }
}

/// Equity de "top p %" contra "top c %" (orden por equity contra mano aleatoria), en
/// una malla de `steps × steps` con interpolación bilineal. La usa FGS para resolver los
/// all-ins simulados de las manos futuras sin tocar la matriz completa.
#[derive(Debug, Clone)]
pub struct RangeEquity {
    pub steps: usize,
    /// `grid[i * (steps+1) + j]` = equity del top `i/steps` contra el top `j/steps`.
    grid: Vec<f32>,
    /// Combos de más fuerte a más débil.
    pub order: Vec<u16>,
}

impl RangeEquity {
    pub fn from_matrix(m: &EquityMatrix, steps: usize) -> Self {
        let mut strength = vec![0.0f64; NUM_COMBOS];
        for h in 0..NUM_COMBOS {
            let (w, t) = (m.win_row(h), m.tie_row(h));
            let s: f64 = (0..NUM_COMBOS).map(|v| w[v] as f64 + 0.5 * t[v] as f64).sum();
            let cnt = (0..NUM_COMBOS).filter(|&v| COMBO_MASKS[h] & COMBO_MASKS[v] == 0).count();
            strength[h] = s / cnt as f64;
        }
        let mut order: Vec<u16> = (0..NUM_COMBOS as u16).collect();
        order.sort_by(|&a, &b| strength[b as usize].total_cmp(&strength[a as usize]));
        Self::from_matrix_with_order(m, steps, order)
    }

    /// Como [`RangeEquity::from_matrix`], pero con el ranking dado (combos de más fuerte a
    /// más débil) en vez del de equity contra una mano al azar.
    pub fn from_matrix_with_order(m: &EquityMatrix, steps: usize, order: Vec<u16>) -> Self {
        assert_eq!(order.len(), NUM_COMBOS);
        let cut = |k: usize| ((k * NUM_COMBOS) as f64 / steps as f64).round().max(1.0) as usize;
        let side = steps + 1;
        let mut grid = vec![0.5f32; side * side];
        // colsum[b] = Σ_{a < i} eq(o[a], o[b]); se consulta en los cortes de la malla.
        let mut colsum = vec![0.0f64; NUM_COMBOS];
        let mut colcnt = vec![0.0f64; NUM_COMBOS];
        let mut next_row = 1;
        for a in 0..NUM_COMBOS {
            let ha = order[a] as usize;
            let (w, t) = (m.win_row(ha), m.tie_row(ha));
            for (b, &hb) in order.iter().enumerate() {
                let hb = hb as usize;
                if COMBO_MASKS[ha] & COMBO_MASKS[hb] == 0 {
                    colsum[b] += w[hb] as f64 + 0.5 * t[hb] as f64;
                    colcnt[b] += 1.0;
                }
            }
            while next_row <= steps && cut(next_row) == a + 1 {
                let (mut s, mut c, mut next_col) = (0.0f64, 0.0f64, 1);
                for b in 0..NUM_COMBOS {
                    s += colsum[b];
                    c += colcnt[b];
                    while next_col <= steps && cut(next_col) == b + 1 {
                        grid[next_row * side + next_col] = if c > 0.0 { (s / c) as f32 } else { 0.5 };
                        next_col += 1;
                    }
                }
                next_row += 1;
            }
        }
        // Fila/columna 0 (rango vacío): se replica la 1 para interpolar sin saltos.
        for k in 0..side {
            grid[k] = grid[side + k.max(1)];
            grid[k * side] = grid[k * side + 1];
        }
        RangeEquity { steps, grid, order }
    }

    /// Equity del rango "top `p`" (fracción 0..1) contra "top `c`".
    pub fn equity(&self, p: f64, c: f64) -> f64 {
        let side = self.steps + 1;
        let (x, y) = (p.clamp(0.0, 1.0) * self.steps as f64, c.clamp(0.0, 1.0) * self.steps as f64);
        let (i, j) = ((x as usize).min(self.steps - 1), (y as usize).min(self.steps - 1));
        let (fx, fy) = (x - i as f64, y - j as f64);
        let g = |a: usize, b: usize| self.grid[a * side + b] as f64;
        g(i, j) * (1.0 - fx) * (1.0 - fy) + g(i + 1, j) * fx * (1.0 - fy) + g(i, j + 1) * (1.0 - fx) * fy + g(i + 1, j + 1) * fx * fy
    }
}

impl EquityOracle for EquityMatrix {
    fn heads_up_allin(&self, hero: Combo, villain: Combo) -> (f32, f32) {
        let (h, v) = (hero.0 as usize, villain.0 as usize);
        (self.win[h * HAND_STRIDE + v], self.tie[h * HAND_STRIDE + v])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::eval::Card;
    use std::sync::OnceLock;

    /// Matriz compartida por todos los tests del crate (se construye una vez).
    pub(crate) fn shared() -> &'static EquityMatrix {
        static M: OnceLock<EquityMatrix> = OnceLock::new();
        M.get_or_init(|| EquityMatrix::monte_carlo(600, 11, 4))
    }

    #[test]
    fn canonical_boards_cover_every_board() {
        let b = canonical_boards();
        assert_eq!(b.len(), 134_459);
        assert_eq!(b.iter().map(|&(_, w)| w as u64).sum::<u64>(), 2_598_960);
    }

    #[test]
    fn sparse_products_match_dense() {
        let m = shared();
        // Rango estrecho con pesos variados: cada 7.º combo.
        let mut r = vec![0.0f32; HAND_STRIDE];
        for h in (0..NUM_COMBOS).step_by(7) {
            r[h] = 0.2 + (h % 5) as f32 * 0.2;
        }
        let bs = BlockerSums::new(&r);
        let (d, s) = (m.products_dense(&r, &bs), m.products_sparse(&r, &bs));
        for h in 0..NUM_COMBOS {
            assert!((d.w[h] - s.w[h]).abs() < 2e-3, "w {h}: {} vs {}", d.w[h], s.w[h]);
            assert!((d.t[h] - s.t[h]).abs() < 1e-3, "t {h}");
            assert_eq!(d.all[h], s.all[h]);
        }
    }

    fn c(r: u8, s: u8) -> Card {
        Card::new(r, s)
    }

    #[test]
    fn known_matchups() {
        let m = shared();
        let aa = Combo::from_cards(c(12, 0), c(12, 1));
        let kk = Combo::from_cards(c(11, 2), c(11, 3));
        let (w, t) = m.heads_up_allin(aa, kk);
        assert!((w + t / 2.0 - 0.82).abs() < 0.02, "AA vs KK = {w} / {t}");
        // Espejo exacto AKo vs AKo: casi todo empate.
        let ak1 = Combo::from_cards(c(12, 0), c(11, 1));
        let ak2 = Combo::from_cards(c(12, 2), c(11, 3));
        let (w, t) = m.heads_up_allin(ak1, ak2);
        assert!(t > 0.9 && w < 0.05, "AKo vs AKo = {w} / {t}");
        // Complementariedad y bloqueo.
        let (w2, t2) = m.heads_up_allin(kk, aa);
        let (w1, t1) = m.heads_up_allin(aa, kk);
        assert!((w1 + w2 + t1 - 1.0).abs() < 1e-5 && t1 == t2);
        assert_eq!(m.heads_up_allin(aa, Combo::from_cards(c(12, 0), c(3, 3))), (0.0, 0.0));
    }

    #[test]
    fn range_equity_table() {
        let r = RangeEquity::from_matrix(shared(), 50);
        assert!((r.equity(1.0, 1.0) - 0.5).abs() < 0.01);
        assert!(r.equity(0.05, 1.0) > 0.7, "{}", r.equity(0.05, 1.0));
        assert!(r.equity(0.5, 0.1) < 0.4);
        // Rangos simétricos: equity 1/2.
        assert!((r.equity(0.3, 0.3) - 0.5).abs() < 0.01);
    }

    #[test]
    fn hero_values_match_direct_sum() {
        let m = shared();
        let mut rng = SplitMix64::new(3);
        let mut r = vec![0.0f32; HAND_STRIDE];
        r[..NUM_COMBOS].iter_mut().for_each(|x| *x = rng.next_f32());
        let mut out = vec![0.0f32; HAND_STRIDE];
        m.hero_values(&r, 3.0, 1.0, -2.0, &mut out);
        for h in [0usize, 77, 1325] {
            let mut s = 0.0f64;
            for v in 0..NUM_COMBOS {
                if COMBO_MASKS[h] & COMBO_MASKS[v] != 0 {
                    continue;
                }
                let (w, t) = (m.win_row(h)[v] as f64, m.tie_row(h)[v] as f64);
                s += r[v] as f64 * (3.0 * w + t - 2.0 * (1.0 - w - t));
            }
            assert!((out[h] as f64 - s / 1225.0).abs() < 1e-3);
        }
    }
}
