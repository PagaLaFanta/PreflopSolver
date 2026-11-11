//! # Abstracción de información: bucketing por EHS / EMD
//!
//! ## Qué se agrupa
//! * **River**: la mano ya no tiene potencial ⇒ el descriptor es escalar: EHS = equity
//!   contra rango uniforme (o, mejor, contra rango de oponente clusterizado: OCHS).
//!   Buckets = percentiles de EHS (clustering 1-D óptimo ≈ cuantiles).
//! * **Turn / Flop**: dos manos con la misma EHS media pueden ser radicalmente distintas
//!   (proyecto de color 35% vs par medio 35%). El descriptor es el **histograma de equity
//!   en la calle siguiente** (distribución de EHS sobre todos los runouts de 1 carta) y la
//!   distancia es **EMD** (Earth Mover's Distance / Wasserstein-1). Johanson et al. 2013,
//!   "Evaluating State-Space Abstractions in Extensive-Form Games".
//!
//! ## EMD en 1-D es O(bins)
//! Para histogramas sobre el mismo soporte ordenado,
//! `EMD(p, q) = Σ_k |CDF_p(k) − CDF_q(k)| · Δ`. No hace falta resolver un problema de
//! transporte: es un único barrido acumulado → vectorizable, sin ramas.
//!
//! ## K-means con EMD
//! Asignación por EMD; actualización por **media de histogramas**. Nota de rigor: el
//! baricentro W1 exacto en 1-D es la media de las funciones cuantil, no de las densidades;
//! la media de densidades es el estándar de la literatura (y lo que se evalúa en
//! Johanson 2013), converge y es barato. Iteración 2 puede cambiarlo por media de cuantiles.

use crate::eval::{Card, Combo};
use crate::math::rng::SplitMix64;

/// Resolución del histograma de equity futura.
pub const EQUITY_BINS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(64))]
pub struct EquityHistogram {
    /// Masa normalizada (suma 1). 32 × f32 = 128 B = 2 líneas, alineado para SIMD.
    pub mass: [f32; EQUITY_BINS],
}

impl Default for EquityHistogram {
    fn default() -> Self {
        EquityHistogram { mass: [0.0; EQUITY_BINS] }
    }
}

/// EMD 1-D entre histogramas normalizados (soporte [0,1], Δ = 1/BINS).
#[inline]
pub fn emd_1d(p: &EquityHistogram, q: &EquityHistogram) -> f32 {
    let mut cdf_diff = 0.0f32;
    let mut acc = 0.0f32;
    for k in 0..EQUITY_BINS {
        cdf_diff += p.mass[k] - q.mass[k];
        acc += cdf_diff.abs();
    }
    acc / EQUITY_BINS as f32
}

/// Fuerza de mano "ahora" (equity vs rango) — la aporta la capa `eval`.
pub trait StrengthOracle: Send + Sync {
    /// Equity de `hole` en `board` (3-5 cartas) contra el rango de referencia.
    fn hand_strength(&self, hole: Combo, board: &[Card]) -> f32;
}

/// Histograma de equity sobre todas las cartas siguientes posibles (flop→turn o turn→river).
pub fn next_street_histogram<O: StrengthOracle + ?Sized>(oracle: &O, hole: Combo, board: &[Card]) -> EquityHistogram {
    debug_assert!((3..=4).contains(&board.len()));
    let dead = board.iter().fold(hole.mask(), |m, c| m | c.mask());
    let mut next = [Card(0); 5];
    next[..board.len()].copy_from_slice(board);
    let mut h = EquityHistogram::default();
    let mut count = 0u32;
    for c in 0..52u8 {
        if dead & (1u64 << c) != 0 {
            continue;
        }
        next[board.len()] = Card(c);
        let e = oracle.hand_strength(hole, &next[..=board.len()]).clamp(0.0, 1.0);
        let bin = ((e * EQUITY_BINS as f32) as usize).min(EQUITY_BINS - 1);
        h.mass[bin] += 1.0;
        count += 1;
    }
    let inv = 1.0 / count.max(1) as f32;
    h.mass.iter_mut().for_each(|m| *m *= inv);
    h
}

/// Asignación de bucket: contrato consumido por el MCCFR postflop.
pub trait Bucketer: Send + Sync {
    /// Bucket en `[0, num_buckets)` de `hole` en `board`. Debe ser barato (lookup):
    /// la versión de producción precomputa tablas indexadas por (board isomórfico, combo).
    fn bucket(&self, hole: Combo, board: &[Card]) -> u16;
    fn num_buckets(&self) -> u16;
}

/// Bucketer de centroides EMD (flop/turn). La asignación en caliente es
/// `argmin_c EMD(h, c)` — O(K·BINS); producción lo sustituye por la tabla precomputada,
/// este tipo se usa para *generar* esa tabla offline.
#[derive(Debug)]
pub struct EmdCentroids<O> {
    pub centroids: Vec<EquityHistogram>,
    pub oracle: O,
}

impl<O: StrengthOracle> EmdCentroids<O> {
    pub fn nearest(&self, h: &EquityHistogram) -> u16 {
        let mut best = (f32::INFINITY, 0u16);
        for (i, c) in self.centroids.iter().enumerate() {
            let d = emd_1d(h, c);
            if d < best.0 {
                best = (d, i as u16);
            }
        }
        best.1
    }
}

impl<O: StrengthOracle> Bucketer for EmdCentroids<O> {
    fn bucket(&self, hole: Combo, board: &[Card]) -> u16 {
        self.nearest(&next_street_histogram(&self.oracle, hole, board))
    }
    fn num_buckets(&self) -> u16 {
        self.centroids.len() as u16
    }
}

/// K-means con inicialización k-means++ bajo EMD (256 / 1k / 4k buckets según presupuesto).
pub fn kmeans_emd(samples: &[EquityHistogram], k: usize, iters: u32, seed: u64) -> Vec<EquityHistogram> {
    assert!(k >= 1 && samples.len() >= k);
    let mut rng = SplitMix64::new(seed);
    // --- k-means++: muestreo ∝ D(x)² ---
    let mut centroids = vec![samples[(rng.next_u64() % samples.len() as u64) as usize]];
    let mut d2: Vec<f32> = samples.iter().map(|s| emd_1d(s, &centroids[0]).powi(2)).collect();
    while centroids.len() < k {
        let total: f64 = d2.iter().map(|&x| x as f64).sum();
        let mut target = rng.next_f64_open0() * total;
        let mut pick = samples.len() - 1;
        for (i, &x) in d2.iter().enumerate() {
            target -= x as f64;
            if target <= 0.0 {
                pick = i;
                break;
            }
        }
        centroids.push(samples[pick]);
        let c = centroids.last().unwrap();
        for (i, s) in samples.iter().enumerate() {
            d2[i] = d2[i].min(emd_1d(s, c).powi(2));
        }
    }
    // --- Lloyd ---
    let mut assign = vec![0usize; samples.len()];
    for _ in 0..iters {
        let mut changed = false;
        for (i, s) in samples.iter().enumerate() {
            let mut best = (f32::INFINITY, 0);
            for (j, c) in centroids.iter().enumerate() {
                let d = emd_1d(s, c);
                if d < best.0 {
                    best = (d, j);
                }
            }
            changed |= assign[i] != best.1;
            assign[i] = best.1;
        }
        let mut sums = vec![[0.0f64; EQUITY_BINS]; k];
        let mut counts = vec![0u32; k];
        for (s, &a) in samples.iter().zip(&assign) {
            counts[a] += 1;
            for b in 0..EQUITY_BINS {
                sums[a][b] += s.mass[b] as f64;
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for b in 0..EQUITY_BINS {
                    centroids[j].mass[b] = (sums[j][b] / counts[j] as f64) as f32;
                }
            }
        }
        if !changed {
            break;
        }
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spike(bin: usize) -> EquityHistogram {
        let mut h = EquityHistogram::default();
        h.mass[bin] = 1.0;
        h
    }

    #[test]
    fn emd_is_ground_distance_between_spikes() {
        // Mover toda la masa 8 bins = 8/32 de equity.
        assert!((emd_1d(&spike(4), &spike(12)) - 0.25).abs() < 1e-6);
        assert_eq!(emd_1d(&spike(3), &spike(3)), 0.0);
    }

    #[test]
    fn emd_separates_draw_from_made_hand_with_same_mean() {
        // "Proyecto": bimodal 0/1 con media 0.5. "Mano hecha": pico en 0.5.
        let mut draw = EquityHistogram::default();
        draw.mass[0] = 0.5;
        draw.mass[EQUITY_BINS - 1] = 0.5;
        let made = spike(EQUITY_BINS / 2);
        assert!(emd_1d(&draw, &made) > 0.4);
    }

    #[test]
    fn kmeans_recovers_two_clusters() {
        let samples: Vec<_> = (0..50).map(|i| spike(if i % 2 == 0 { 2 } else { 29 })).collect();
        let c = kmeans_emd(&samples, 2, 10, 5);
        let mut modes: Vec<usize> = c
            .iter()
            .map(|h| h.mass.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0)
            .collect();
        modes.sort();
        assert_eq!(modes, vec![2, 29]);
    }
}
