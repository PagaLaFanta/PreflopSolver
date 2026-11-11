//! # Showdown rango-contra-rango en O(n) con bloqueadores
//!
//! Dado un board completo con los combos válidos ordenados por fuerza ascendente, para
//! cada combo del héroe `h` calcula las masas de rango rival que pierden, empatan y ganan
//! contra él, **excluyendo los combos que comparten carta con `h`**:
//!
//! ```text
//! weaker(h) = W − W_c1 − W_c2          (W = masa acumulada de grupos más débiles,
//!                                       W_c = parte de W que contiene la carta c)
//! tie(h)    = G − G_c1 − G_c2 + r[h]   (G = masa del grupo de igual rango; `h` se
//!                                       resta dos veces y se repone una)
//! lose(h)   = compatible(h) − weaker − tie
//! ```
//!
//! Coste O(n + 52·grupos) por board, frente a O(n²) del producto matriz-vector: es lo que
//! hace viable el MCCFR postflop vectorizado sobre rangos.

use super::equity::BlockerSums;
use super::evaluator::rank_mask;
use super::{COMBO_CARDS, COMBO_MASKS, NUM_COMBOS};
use crate::math::rng::SplitMix64;

/// Runout completo con la fuerza de cada combo precalculada.
#[derive(Debug, Clone)]
pub struct RankedBoard {
    pub mask: u64,
    /// Rango de 7 cartas; 0 = combo bloqueado por el board.
    pub ranks: Box<[u32]>,
    /// Combos válidos ordenados por rango ascendente.
    pub order: Box<[u16]>,
}

impl RankedBoard {
    pub fn new(mask: u64) -> Self {
        let mut ranks = vec![0u32; NUM_COMBOS];
        let mut order = Vec::with_capacity(NUM_COMBOS);
        for h in 0..NUM_COMBOS {
            if COMBO_MASKS[h] & mask == 0 {
                ranks[h] = rank_mask(mask | COMBO_MASKS[h]);
                order.push(h as u16);
            }
        }
        order.sort_unstable_by_key(|&h| ranks[h as usize]);
        RankedBoard { mask, ranks: ranks.into(), order: order.into() }
    }

    #[inline(always)]
    pub fn valid(&self, h: usize) -> bool {
        COMBO_MASKS[h] & self.mask == 0
    }
}

/// `n` runouts uniformes, evaluados en paralelo (≈ 60 µs por board).
pub fn sample_ranked_boards(n: usize, seed: u64, threads: usize) -> Vec<RankedBoard> {
    let mut rng = SplitMix64::new(seed);
    let masks: Vec<u64> = (0..n)
        .map(|_| {
            let mut m = 0u64;
            while m.count_ones() < 5 {
                m |= 1 << (rng.next_u64() % 52);
            }
            m
        })
        .collect();
    let threads = threads.max(1);
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let hs: Vec<_> = masks.chunks(chunk).map(|c| s.spawn(move || c.iter().map(|&m| RankedBoard::new(m)).collect::<Vec<_>>())).collect();
        hs.into_iter().flat_map(|h| h.join().expect("boards thread")).collect()
    })
}

/// `order`: combos válidos en el board, ordenados por `ranks` ascendente.
/// Solo se leen de `r_opp` los índices de `order` y solo se escriben esos índices de
/// `w/t/l` (el resto de los vectores puede contener cualquier cosa).
pub fn showdown_wtl(order: &[u16], ranks: &[u32], r_opp: &[f32], w: &mut [f32], t: &mut [f32], l: &mut [f32]) {
    showdown_wtl_with(order, ranks, r_opp, &BlockerSums::from_indices(r_opp, order), w, t, l);
}

/// Como [`showdown_wtl`] con las sumas de bloqueadores de `r_opp` ya calculadas.
pub fn showdown_wtl_with(order: &[u16], ranks: &[u32], r_opp: &[f32], bs: &BlockerSums, w: &mut [f32], t: &mut [f32], l: &mut [f32]) {
    let mut weaker = 0.0f32;
    let mut wc = [0.0f32; 52];
    let mut gc = [0.0f32; 52];
    let mut i = 0;
    while i < order.len() {
        let rank = ranks[order[i] as usize];
        let mut j = i;
        let mut g = 0.0f32;
        while j < order.len() && ranks[order[j] as usize] == rank {
            let x = order[j] as usize;
            let [a, b] = COMBO_CARDS[x];
            g += r_opp[x];
            gc[a as usize] += r_opp[x];
            gc[b as usize] += r_opp[x];
            j += 1;
        }
        for &x in &order[i..j] {
            let x = x as usize;
            let [a, b] = COMBO_CARDS[x];
            let (a, b) = (a as usize, b as usize);
            let wv = weaker - wc[a] - wc[b];
            let tv = g - gc[a] - gc[b] + r_opp[x];
            w[x] = wv;
            t[x] = tv;
            l[x] = bs.compatible(r_opp, x) - wv - tv;
        }
        for &x in &order[i..j] {
            let [a, b] = COMBO_CARDS[x as usize];
            let rx = r_opp[x as usize];
            wc[a as usize] += rx;
            wc[b as usize] += rx;
            gc[a as usize] = 0.0;
            gc[b as usize] = 0.0;
        }
        weaker += g;
        i = j;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::evaluator::rank_mask;
    use crate::eval::{COMBO_MASKS, NUM_COMBOS};
    use crate::math::rng::SplitMix64;

    #[test]
    fn matches_bruteforce() {
        let mut rng = SplitMix64::new(5);
        let board: u64 = [3u64, 17, 22, 40, 51].iter().map(|&c| 1u64 << c).sum();
        let mut ranks = vec![0u32; NUM_COMBOS];
        let mut r = vec![0.0f32; NUM_COMBOS];
        let mut order = Vec::new();
        for h in 0..NUM_COMBOS {
            if COMBO_MASKS[h] & board == 0 {
                ranks[h] = rank_mask(board | COMBO_MASKS[h]);
                r[h] = rng.next_f32();
                order.push(h as u16);
            }
        }
        order.sort_by_key(|&h| ranks[h as usize]);
        let (mut w, mut t, mut l) = (vec![0.0; NUM_COMBOS], vec![0.0; NUM_COMBOS], vec![0.0; NUM_COMBOS]);
        showdown_wtl(&order, &ranks, &r, &mut w, &mut t, &mut l);
        for &h in order.iter().step_by(37) {
            let h = h as usize;
            let (mut bw, mut bt, mut bl) = (0.0f64, 0.0f64, 0.0f64);
            for &v in &order {
                let v = v as usize;
                if COMBO_MASKS[h] & COMBO_MASKS[v] != 0 {
                    continue;
                }
                let x = r[v] as f64;
                match ranks[h].cmp(&ranks[v]) {
                    std::cmp::Ordering::Greater => bw += x,
                    std::cmp::Ordering::Equal => bt += x,
                    std::cmp::Ordering::Less => bl += x,
                }
            }
            assert!((w[h] as f64 - bw).abs() < 1e-2, "w {h}");
            assert!((t[h] as f64 - bt).abs() < 1e-2, "t {h}");
            assert!((l[h] as f64 - bl).abs() < 1e-2, "l {h}");
        }
    }
}
