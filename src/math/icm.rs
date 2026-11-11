//! # ICM Malmuth-Harville
//!
//! ## Modelo
//! Harville: la probabilidad de que el jugador `j` termine 1º es `s_j / T`. Condicionado
//! a que un conjunto `M` ya ocupe los puestos `1..|M|`, la probabilidad de que `j ∉ M`
//! ocupe el puesto `|M|+1` es `s_j / (T − Σ_{i∈M} s_i)`.
//!
//! Es exactamente el modelo **Plackett-Luce** con pesos `s_i`, lo que da dos algoritmos:
//!
//! 1. **DP sobre subconjuntos** (exacto). Sea `F(M)` = P(el conjunto de los `|M|` primeros
//!    es exactamente `M`, en cualquier orden). Entonces
//!    `F(∅) = 1`,  `F(M ∪ {j}) += F(M) · s_j / (T − S(M))`,
//!    y la contribución `F(M) · s_j / (T − S(M))` es P(j acaba en el puesto |M|+1 ∧ top = M).
//!    La recursión naive sobre permutaciones es O(N!); la memoización de `F` por máscara de
//!    bits colapsa todas las permutaciones de `M` en un único estado → O(2^N · N).
//!
//! 2. **Truncado por premios**: solo necesitamos los estados con `|M| < P` (P = puestos
//!    pagados). El coste real es `Σ_{k<P} C(N,k)·(N−k)`:
//!    * P ≤ 3 (sit&go 50/30/20, satélites de 3 plazas): **O(N³)** exacto.
//!    * Mesa final de 9 con 9 premios: 2⁹·9 ≈ 4.6k mult-adds (~2 µs).
//!    Los estados de la capa `P` nunca se escriben: el DP termina en la capa P−1.
//!
//! 3. **Monte Carlo Plackett-Luce** para N > [`MAX_EXACT_PLAYERS`] (MTT con cientos de
//!    jugadores): si `E_i ~ Exp(1)`, el orden ascendente de `E_i / s_i` se distribuye
//!    exactamente como Harville (carrera de exponenciales + falta de memoria). Estimador
//!    insesgado, O(S · N log P).
//!
//! ## Por qué esto no está en el hot path del CFR
//! En el árbol la *transición de stacks* de cada terminal está determinada por el árbol
//! (quién ganó/perdió qué bote), no por las cartas. Por tanto la utilidad $ de cada
//! *outcome* se precomputa una sola vez al construir el árbol ([`IcmMemo`] deduplica
//! estados repetidos) y el CFR solo mezcla `Σ_o P(o | manos) · U(o)` con la equity.

use super::rng::SplitMix64;
use std::cell::RefCell;
use std::collections::HashMap;

/// Límite del DP denso: 2²⁰ estados × (8 B de `F` + 8 B de `S`) = 16 MiB de scratch por
/// hilo. Por encima se usa Monte Carlo.
pub const MAX_EXACT_PLAYERS: usize = 20;

#[derive(Default)]
struct DpScratch {
    /// `F(M)`. Invariante entre llamadas: **todo a cero**. Cada estado visitado se pone a 0
    /// tras consumirse, así que no hay `memset` de 2^N por llamada — el coste es
    /// proporcional a los estados alcanzados, no al tamaño del buffer.
    f: Vec<f64>,
    /// `S(M)` = fichas acumuladas en `M`. Se escribe antes de leerse (al propagar), no
    /// necesita limpieza.
    mass: Vec<f64>,
}

thread_local! {
    static SCRATCH: RefCell<DpScratch> = RefCell::new(DpScratch::default());
}

/// Equity ICM exacta. `stacks` deben ser > 0 (los eliminados se resuelven fuera, ver
/// [`super::terminal`]). `payouts[k]` = premio del puesto `k+1` entre los jugadores vivos.
///
/// Campos grandes (ICM multimesa, N > [`MAX_EXACT_PLAYERS`]):
/// * si los estados del DP truncado (`Σ_{k<P} C(N,k)`) caben en [`MAX_SPARSE_STATES`],
///   DP **exacto** con las capas en tablas hash (sin el buffer denso de 2^N);
/// * si no, Monte Carlo Plackett-Luce con **números aleatorios comunes** (una matriz fija
///   de exponenciales por N): las diferencias entre terminales con stacks parecidos son
///   suaves, que es lo que importa al solver, aunque el nivel absoluto tenga ~0.5 % de
///   error.
pub fn icm_equity(stacks: &[f64], payouts: &[f64], out: &mut [f64]) {
    let n = stacks.len();
    assert_eq!(out.len(), n, "out.len() debe ser igual a stacks.len()");
    out.fill(0.0);
    let paid = payouts.len().min(n);
    if n == 0 || paid == 0 {
        return;
    }
    debug_assert!(stacks.iter().all(|&s| s > 0.0 && s.is_finite()));
    if n > MAX_EXACT_PLAYERS {
        if sparse_states(n, paid) <= MAX_SPARSE_STATES {
            icm_equity_sparse(stacks, payouts, out);
        } else {
            icm_equity_crn(stacks, payouts, CRN_SAMPLES, out);
        }
        return;
    }
    let total: f64 = stacks.iter().sum();
    SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let DpScratch { f, mass } = &mut *guard;
        let size = 1usize << n;
        if f.len() < size {
            f.resize(size, 0.0);
            mass.resize(size, 0.0);
        }
        dp_layers(stacks, total, &payouts[..paid], f, mass, out);
    });
}

/// Núcleo del DP: recorre las capas `k = |M|` en orden, enumerando las máscaras de
/// popcount `k` con el truco de Gosper (siguiente entero con el mismo número de bits),
/// lo que garantiza orden topológico: todo `F(M)` de la capa `k` está completo antes de
/// consumirse.
#[inline(never)]
fn dp_layers(stacks: &[f64], total: f64, payouts: &[f64], f: &mut [f64], mass: &mut [f64], out: &mut [f64]) {
    let n = stacks.len();
    let paid = payouts.len();
    let full: u32 = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
    f[0] = 1.0;
    mass[0] = 0.0;
    for k in 0..paid {
        let pay = payouts[k];
        let propagate = k + 1 < paid;
        let mut mask: u32 = if k == 0 { 0 } else { (1u32 << k) - 1 };
        loop {
            let fm = f[mask as usize];
            if fm != 0.0 {
                f[mask as usize] = 0.0; // restaura el invariante "scratch a cero"
                let base = mass[mask as usize];
                // Una división por estado, no por arista.
                let w = fm / (total - base);
                let mut free = !mask & full;
                while free != 0 {
                    let j = free.trailing_zeros() as usize;
                    free &= free - 1;
                    let p = w * stacks[j];
                    out[j] += p * pay;
                    if propagate {
                        let nm = (mask | (1u32 << j)) as usize;
                        f[nm] += p;
                        mass[nm] = base + stacks[j];
                    }
                }
            }
            if k == 0 {
                break;
            }
            // Gosper's hack (`isolate_lowest_one` no está en toolchains algo antiguos).
            #[allow(clippy::manual_isolate_lowest_one)]
            let c = mask & mask.wrapping_neg();
            let r = mask + c;
            let next = (((r ^ mask) >> 2) / c) | r;
            if next > full || next < mask {
                break;
            }
            mask = next;
        }
    }
}

/// Límite de estados del DP disperso (≈ 40 ms por terminal a este tamaño).
pub const MAX_SPARSE_STATES: usize = 400_000;
/// Muestras del Monte Carlo con números aleatorios comunes.
pub const CRN_SAMPLES: usize = 40_000;

/// `Σ_{k<paid} C(n,k)`, saturado.
pub fn sparse_states(n: usize, paid: usize) -> usize {
    let mut total = 0usize;
    let mut c = 1usize;
    for k in 0..paid.min(n) {
        total = total.saturating_add(c);
        c = c.saturating_mul(n - k) / (k + 1);
    }
    total
}

/// DP truncado exacto con capas dispersas (`HashMap<máscara, (F, masa)>`), para N ≤ 64.
pub fn icm_equity_sparse(stacks: &[f64], payouts: &[f64], out: &mut [f64]) {
    use std::collections::HashMap;
    let n = stacks.len();
    assert!(n <= 64);
    let paid = payouts.len().min(n);
    let total: f64 = stacks.iter().sum();
    let full: u64 = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let mut layer: HashMap<u64, (f64, f64)> = HashMap::new();
    layer.insert(0, (1.0, 0.0));
    for k in 0..paid {
        let pay = payouts[k];
        let propagate = k + 1 < paid;
        let mut next: HashMap<u64, (f64, f64)> = HashMap::with_capacity(if propagate { layer.len() * (n - k) / (k + 1) + 1 } else { 0 });
        for (&mask, &(fm, base)) in &layer {
            let w = fm / (total - base);
            let mut free = !mask & full;
            while free != 0 {
                let j = free.trailing_zeros() as usize;
                free &= free - 1;
                let p = w * stacks[j];
                out[j] += p * pay;
                if propagate {
                    let e = next.entry(mask | (1u64 << j)).or_insert((0.0, base + stacks[j]));
                    e.0 += p;
                }
            }
        }
        layer = next;
    }
}

thread_local! {
    /// Exponenciales Exp(1) fijas por (N, muestras): números aleatorios comunes.
    static CRN: RefCell<HashMap<(usize, usize), std::rc::Rc<Vec<f64>>>> = RefCell::new(HashMap::new());
}

/// Monte Carlo Plackett-Luce con una matriz fija de exponenciales (CRN).
pub fn icm_equity_crn(stacks: &[f64], payouts: &[f64], samples: usize, out: &mut [f64]) {
    let n = stacks.len();
    let paid = payouts.len().min(n);
    let exps = CRN.with(|c| {
        c.borrow_mut()
            .entry((n, samples))
            .or_insert_with(|| {
                let mut rng = SplitMix64::new(0xC0DE_5EED ^ n as u64);
                std::rc::Rc::new((0..n * samples).map(|_| -rng.next_f64_open0().ln()).collect())
            })
            .clone()
    });
    let inv: Vec<f64> = stacks.iter().map(|&s| 1.0 / s).collect();
    let mut keys: Vec<(f64, u32)> = vec![(0.0, 0); n];
    for s in 0..samples {
        let row = &exps[s * n..(s + 1) * n];
        for (i, k) in keys.iter_mut().enumerate() {
            *k = (row[i] * inv[i], i as u32);
        }
        if paid < n {
            keys.select_nth_unstable_by(paid - 1, |a, b| a.0.total_cmp(&b.0));
        }
        keys[..paid].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for (place, &(_, i)) in keys[..paid].iter().enumerate() {
            out[i as usize] += payouts[place];
        }
    }
    let norm = 1.0 / samples as f64;
    out.iter_mut().for_each(|x| *x *= norm);
    // Stacks iguales ⇒ equity igual (promedio del grupo): elimina el ruido entre
    // jugadores idénticos, típico de los "N jugadores de otras mesas con S bb".
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| stacks[a].total_cmp(&stacks[b]));
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && stacks[idx[j]] == stacks[idx[i]] {
            j += 1;
        }
        if j - i > 1 {
            let mean = idx[i..j].iter().map(|&k| out[k]).sum::<f64>() / (j - i) as f64;
            idx[i..j].iter().for_each(|&k| out[k] = mean);
        }
        i = j;
    }
}

/// Versión precisa para una única consulta (calculadora ICM): DP disperso hasta 50 M de
/// estados (segundos), y si no, CRN con 400k muestras.
pub fn icm_equity_precise(stacks: &[f64], payouts: &[f64], out: &mut [f64]) {
    let n = stacks.len();
    let paid = payouts.len().min(n);
    out.fill(0.0);
    if n == 0 || paid == 0 {
        return;
    }
    if n <= MAX_EXACT_PLAYERS {
        icm_equity(stacks, payouts, out);
    } else if sparse_states(n, paid) <= 50_000_000 {
        icm_equity_sparse(stacks, payouts, out);
    } else {
        icm_equity_crn(stacks, payouts, 400_000, out);
    }
}

/// Monte Carlo Plackett-Luce (exponential race). Insesgado para cualquier N.
pub fn icm_equity_monte_carlo(stacks: &[f64], payouts: &[f64], samples: u32, seed: u64, out: &mut [f64]) {
    let n = stacks.len();
    assert_eq!(out.len(), n);
    out.fill(0.0);
    let paid = payouts.len().min(n);
    if n == 0 || paid == 0 || samples == 0 {
        return;
    }
    let inv: Vec<f64> = stacks.iter().map(|&s| 1.0 / s).collect();
    let mut keys: Vec<(f64, u32)> = vec![(0.0, 0); n];
    let mut rng = SplitMix64::new(seed);
    for _ in 0..samples {
        for (i, k) in keys.iter_mut().enumerate() {
            *k = (-rng.next_f64_open0().ln() * inv[i], i as u32);
        }
        // Solo importan los `paid` primeros: selección parcial O(N) + orden O(P log P).
        if paid < n {
            keys.select_nth_unstable_by(paid - 1, |a, b| a.0.total_cmp(&b.0));
        }
        keys[..paid].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for (place, &(_, i)) in keys[..paid].iter().enumerate() {
            out[i as usize] += payouts[place];
        }
    }
    let norm = 1.0 / samples as f64;
    out.iter_mut().for_each(|x| *x *= norm);
}

/// Memoización de equities ICM para la fase de construcción del árbol.
///
/// Clave: stacks cuantizados a `1/quantum` fichas (p.ej. 0.01 bb) + longitud de premios.
/// Distintos terminales del árbol preflop (p.ej. BTN open-fold vs CO open-fold) producen a
/// menudo la misma distribución de stacks tras el bote; la memo evita recomputarla.
/// No es concurrente a propósito: la construcción del árbol se paraleliza por subárbol y
/// cada hilo tiene su propia memo (sin contención, sin cache-line ping-pong).
#[derive(Debug)]
pub struct IcmMemo {
    quantum: f64,
    table: HashMap<Box<[i64]>, Box<[f64]>>,
}

impl IcmMemo {
    pub fn new(quantum: f64) -> Self {
        IcmMemo { quantum, table: HashMap::new() }
    }

    pub fn equity(&mut self, stacks: &[f64], payouts: &[f64]) -> &[f64] {
        let key: Box<[i64]> = stacks
            .iter()
            .map(|&s| (s / self.quantum).round() as i64)
            .chain(std::iter::once(payouts.len() as i64))
            .collect();
        self.table.entry(key).or_insert_with(|| {
            let mut out = vec![0.0; stacks.len()];
            icm_equity(stacks, payouts, &mut out);
            out.into_boxed_slice()
        })
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Referencia O(N!) por enumeración de permutaciones.
    fn brute(stacks: &[f64], payouts: &[f64]) -> Vec<f64> {
        fn rec(stacks: &[f64], payouts: &[f64], used: &mut Vec<bool>, place: usize, p: f64, out: &mut [f64]) {
            if place >= payouts.len() || place >= stacks.len() {
                return;
            }
            let rem: f64 = stacks.iter().zip(used.iter()).filter(|(_, &u)| !u).map(|(s, _)| s).sum();
            for j in 0..stacks.len() {
                if used[j] {
                    continue;
                }
                let q = p * stacks[j] / rem;
                out[j] += q * payouts[place];
                used[j] = true;
                rec(stacks, payouts, used, place + 1, q, out);
                used[j] = false;
            }
        }
        let mut out = vec![0.0; stacks.len()];
        rec(stacks, payouts, &mut vec![false; stacks.len()], 0, 1.0, &mut out);
        out
    }

    #[test]
    fn heads_up_is_chip_proportional() {
        let mut out = [0.0; 2];
        icm_equity(&[3.0, 1.0], &[1.0, 0.0], &mut out);
        assert!((out[0] - 0.75).abs() < 1e-12 && (out[1] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn dp_matches_bruteforce_and_conserves_prizes() {
        let mut rng = SplitMix64::new(7);
        for n in 1..=7usize {
            for paid in 1..=n {
                let stacks: Vec<f64> = (0..n).map(|_| 1.0 + 100.0 * rng.next_f64_open0()).collect();
                let payouts: Vec<f64> = (0..paid).map(|k| 100.0 / (k as f64 + 1.0)).collect();
                let mut out = vec![0.0; n];
                icm_equity(&stacks, &payouts, &mut out);
                let reference = brute(&stacks, &payouts);
                for (a, b) in out.iter().zip(&reference) {
                    assert!((a - b).abs() < 1e-9, "n={n} paid={paid}: {a} vs {b}");
                }
                let total: f64 = out.iter().sum();
                assert!((total - payouts.iter().sum::<f64>()).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn sparse_dp_matches_dense() {
        let mut rng = SplitMix64::new(9);
        let stacks: Vec<f64> = (0..14).map(|_| 5.0 + 60.0 * rng.next_f64_open0()).collect();
        let payouts = [40.0, 25.0, 15.0, 10.0, 6.0, 4.0];
        let (mut a, mut b) = (vec![0.0; 14], vec![0.0; 14]);
        icm_equity(&stacks, &payouts, &mut a);
        icm_equity_sparse(&stacks, &payouts, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-9, "{x} vs {y}");
        }
        assert_eq!(sparse_states(24, 9), (0..9).map(|k| binom(24, k)).sum::<usize>());
    }

    fn binom(n: usize, k: usize) -> usize {
        (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
    }

    /// Campo grande: el DP disperso es exacto y el CRN es suave entre terminales vecinos
    /// (la diferencia de equity por mover 1 bb entre dos jugadores es casi igual a la exacta).
    #[test]
    fn large_field_exact_and_crn_smoothness() {
        let mut rng = SplitMix64::new(3);
        let stacks: Vec<f64> = (0..24).map(|_| 8.0 + 40.0 * rng.next_f64_open0()).collect();
        let payouts: Vec<f64> = vec![100.0, 60.0, 40.0, 30.0, 25.0, 20.0, 15.0, 12.0, 10.0];
        let mut exact = vec![0.0; 24];
        icm_equity(&stacks, &payouts, &mut exact);
        assert!((exact.iter().sum::<f64>() - 312.0).abs() < 1e-6);
        let mut moved = stacks.clone();
        moved[0] += 1.0;
        moved[1] -= 1.0;
        let mut exact2 = vec![0.0; 24];
        icm_equity(&moved, &payouts, &mut exact2);
        let (mut mc1, mut mc2) = (vec![0.0; 24], vec![0.0; 24]);
        icm_equity_crn(&stacks, &payouts, CRN_SAMPLES, &mut mc1);
        icm_equity_crn(&moved, &payouts, CRN_SAMPLES, &mut mc2);
        for i in 0..2 {
            let (d_exact, d_mc) = (exact2[i] - exact[i], mc2[i] - mc1[i]);
            assert!((d_exact - d_mc).abs() < 0.15 * d_exact.abs() + 0.02, "jugador {i}: Δ exacta {d_exact} vs CRN {d_mc}");
            assert!((mc1[i] - exact[i]).abs() < 0.01 * 312.0, "nivel absoluto");
        }
    }

    #[test]
    fn scratch_invariant_survives_repeated_calls() {
        let stacks = [10.0, 20.0, 30.0, 40.0, 50.0];
        let payouts = [50.0, 30.0, 20.0];
        let mut a = [0.0; 5];
        let mut b = [0.0; 5];
        icm_equity(&stacks, &payouts, &mut a);
        icm_equity(&[1.0, 2.0, 3.0], &[1.0], &mut [0.0; 3]);
        icm_equity(&stacks, &payouts, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn monte_carlo_converges_to_exact() {
        let stacks = [50.0, 30.0, 20.0, 15.0, 5.0];
        let payouts = [50.0, 30.0, 20.0];
        let mut exact = [0.0; 5];
        let mut mc = [0.0; 5];
        icm_equity(&stacks, &payouts, &mut exact);
        icm_equity_monte_carlo(&stacks, &payouts, 400_000, 42, &mut mc);
        for (e, m) in exact.iter().zip(&mc) {
            assert!((e - m).abs() < 0.25, "{e} vs {m}");
        }
    }
}
