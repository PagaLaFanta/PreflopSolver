//! # ICM Malmuth-Harville para campos grandes (MTT con miles de jugadores)
//!
//! El DP sobre subconjuntos es inviable por encima de ~20-25 jugadores. Aquí se usa una
//! representación **exacta** del modelo como carrera de exponenciales: si cada jugador
//! tiene un reloj `T_j ~ Exp(s_j)` (tasa = su stack), el orden ascendente de los relojes
//! se distribuye exactamente como Malmuth-Harville (el primero en sonar gana: P = s_j/S).
//! Por tanto
//!
//! ```text
//! P(i acaba k-ésimo) = ∫₀^∞ s_i e^{-s_i t} · [x^{k-1}] Π_{j≠i} (e^{-s_j t} + (1 − e^{-s_j t}) x) dt
//! equity_i           = Σ_{k ≤ K} premio_k · P(i acaba k-ésimo)
//! ```
//!
//! (el coeficiente `[x^{k-1}]` es la probabilidad de que exactamente `k−1` rivales hayan
//! sonado antes de `t`: una binomial de Poisson, truncada en `K` = puestos pagados).
//!
//! * La integral se hace con la regla del trapecio en `ln t`: el integrando es analítico y
//!   decae doble-exponencialmente por ambos extremos, así que converge geométricamente
//!   (error ~1e-7 relativo con el paso por defecto; ver tests contra el DP exacto).
//! * Los jugadores de **otras mesas** no cambian dentro de una mano: su polinomio en cada
//!   nodo de la integral se calcula **una vez** ([`FieldIcm::new`], O(nodos · N · K)) y
//!   cada evaluación de un terminal solo multiplica los factores de los jugadores de la
//!   mesa (O(nodos · T² · K), T ≤ 10). Así el árbol puede tener miles de rivales fuera de
//!   la mesa con el ICM prácticamente exacto (la alternativa, Monte Carlo, tenía ~0,5 % de
//!   error).

/// Paso por defecto en `ln t` (al principio de la integral; se estrecha como 1/√μ).
pub const DEFAULT_STEP: f64 = 1.0;

/// Precomputación de los jugadores de otras mesas para un reparto de premios.
#[derive(Debug, Clone)]
pub struct FieldIcm {
    /// Fichas totales del torneo (mesa + otras mesas): constantes dentro de una mano.
    pub total: f64,
    /// Número de jugadores de otras mesas.
    pub n_off: usize,
    /// Premios por puesto (1º, 2º…).
    pub payouts: Vec<f64>,
    /// Nodos de la integral: `t` (en unidades de 1/fichas totales normalizadas) y peso del
    /// trapecio en `ln t` (`h · t`).
    nodes: Vec<(f64, f64)>,
    /// Polinomio de otras mesas en cada nodo, truncado a `K` coeficientes.
    off_poly: Vec<Vec<f64>>,
    /// Hilos para repartir los nodos en cada evaluación.
    threads: usize,
}

impl FieldIcm {
    /// `off_stacks`: stacks de los jugadores de otras mesas; `table_total`: fichas de la mesa
    /// (constantes en la mano); `min_table_stack`: el stack más corto que puede quedar en la
    /// mesa (fija el final de la integral).
    pub fn new(off_stacks: &[f64], table_total: f64, min_table_stack: f64, payouts: &[f64], threads: usize) -> Self {
        Self::with_step(off_stacks, table_total, min_table_stack, payouts, threads, DEFAULT_STEP)
    }

    pub fn with_step(off_stacks: &[f64], table_total: f64, min_table_stack: f64, payouts: &[f64], threads: usize, step: f64) -> Self {
        let total = table_total + off_stacks.iter().sum::<f64>();
        let k = payouts.len().max(1);
        let rates: Vec<f64> = off_stacks.iter().map(|&s| s / total).collect();
        // Inicio: antes de t_lo casi nadie ha sonado (P ≈ t_lo con tasas que suman 1).
        let t_lo: f64 = 1e-10;
        // Final: o bien han sonado seguro más de K rivales (el integrando de cualquier
        // puesto pagado es 0), o bien el reloj del stack más corto ya ha sonado.
        let r_min = rates.iter().copied().fold((min_table_stack.max(1e-6)) / total, f64::min);
        let mut t_hi = 60.0 / r_min;
        let n_all = off_stacks.len() + 10;
        if k + 50 < n_all {
            let target = k as f64 + 12.0 * (k as f64).sqrt() + 60.0;
            let mu = |t: f64| rates.iter().map(|&r| 1.0 - (-r * t).exp()).sum::<f64>();
            let (mut lo, mut hi) = (1e-6f64, t_hi);
            if mu(hi) > target {
                for _ in 0..200 {
                    let mid = (lo * hi).sqrt();
                    if mu(mid) > target {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                t_hi = hi;
            }
        }
        let (y0, y1) = (t_lo.ln(), t_hi.ln());
        // Cambio de variable y = ln t → u con dy/du = step / √(μ(t) + 1): el coeficiente
        // "exactamente k rivales ya han sonado" es un pico de anchura ~1/√k en ln t, así
        // que el paso se estrecha donde muchos han sonado. La regla del trapecio en u
        // (aplicación suave) conserva la convergencia geométrica.
        let mu_of = |t: f64| rates.iter().map(|&r| -(-r * t).exp_m1()).sum::<f64>();
        let fine = 4096usize;
        let dy = (y1 - y0) / fine as f64;
        let dens: Vec<f64> = (0..=fine).map(|i| (mu_of((y0 + i as f64 * dy).exp()) + 1.0).sqrt() / step).collect();
        let mut cum = vec![0.0f64; fine + 1];
        for i in 1..=fine {
            cum[i] = cum[i - 1] + 0.5 * (dens[i - 1] + dens[i]) * dy;
        }
        let m = cum[fine].ceil() as usize + 1;
        let hu = cum[fine] / (m - 1) as f64;
        let mut nodes: Vec<(f64, f64)> = Vec::with_capacity(m);
        let mut j = 0usize;
        for i in 0..m {
            let u = i as f64 * hu;
            while j + 1 < fine && cum[j + 1] < u {
                j += 1;
            }
            let frac = if cum[j + 1] > cum[j] { ((u - cum[j]) / (cum[j + 1] - cum[j])).clamp(0.0, 1.0) } else { 0.0 };
            let y = y0 + (j as f64 + frac) * dy;
            let t = y.exp();
            // dy/du exacto de la aplicación en ese punto (1 / densidad).
            let dydu = step / (mu_of(t) + 1.0).sqrt();
            let w = if i == 0 || i == m - 1 { 0.5 } else { 1.0 };
            nodes.push((t, w * hu * dydu * t));
        }
        let threads = threads.max(1);
        let chunk = m.div_ceil(threads);
        let off_poly: Vec<Vec<f64>> = std::thread::scope(|sc| {
            let hs: Vec<_> = nodes
                .chunks(chunk)
                .map(|c| {
                    let rates = &rates;
                    sc.spawn(move || {
                        c.iter()
                            .map(|&(t, _)| {
                                let mut p = vec![0.0f64; k];
                                p[0] = 1.0;
                                for &r in rates.iter() {
                                    let a = (-r * t).exp();
                                    mul_linear(&mut p, a, 1.0 - a);
                                }
                                p
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().expect("hilo de ICM de campo")).collect()
        });
        FieldIcm { total, n_off: off_stacks.len(), payouts: payouts.to_vec(), nodes, off_poly, threads }
    }

    /// Equity de los jugadores de la mesa con stacks `table` (todos > 0; los eliminados
    /// se tratan fuera) cuando los premios en juego son `payouts[..k]` (`k ≤ K`). Los
    /// nodos de la integral se reparten entre hilos.
    pub fn equity(&self, table: &[f64], k: usize, out: &mut [f64]) {
        let k = k.min(self.payouts.len());
        out.fill(0.0);
        if k == 0 || table.is_empty() {
            return;
        }
        let rates: Vec<f64> = table.iter().map(|&s| s / self.total).collect();
        let work = self.nodes.len() * table.len() * k * 3;
        let threads = if work > 2_000_000 { self.threads.max(1) } else { 1 };
        let chunk = self.nodes.len().div_ceil(threads);
        // Por nodo, en O(T·K) en vez de O(T²·K): solo interesa `premios · (polinomio de los
        // demás)`. Prefijos hacia delante P_i = off · L_0 ⋯ L_{i−1} y, hacia atrás, el vector
        // de premios transformado por los transpuestos G_{i+1} = L_{i+1}ᵀ ⋯ L_{T−1}ᵀ · premios;
        // entonces premios · (off · Π_{j≠i} L_j) = P_i · G_{i+1}.
        let tn = table.len();
        let pay = &self.payouts[..k];
        let partial = |lo: usize, hi: usize| -> Vec<f64> {
            let mut eq = vec![0.0f64; tn];
            let mut pre = vec![0.0f64; (tn + 1) * k];
            let mut g = vec![0.0f64; k];
            let mut a = vec![0.0f64; tn];
            for node in lo..hi {
                let (t, w) = self.nodes[node];
                for (x, &r) in a.iter_mut().zip(&rates) {
                    *x = (-r * t).exp();
                }
                pre[..k].copy_from_slice(&self.off_poly[node][..k]);
                for i in 0..tn {
                    let (head, tail) = pre.split_at_mut((i + 1) * k);
                    let next = &mut tail[..k];
                    next.copy_from_slice(&head[i * k..]);
                    mul_linear(next, a[i], 1.0 - a[i]);
                }
                g.copy_from_slice(pay);
                for i in (0..tn).rev() {
                    // Integrando: s_i e^{-s_i t} · (premios · polinomio de los demás).
                    let dens = rates[i] * a[i] * w;
                    if dens > 1e-300 {
                        eq[i] += dens * pre[i * k..(i + 1) * k].iter().zip(&g).map(|(p, q)| p * q).sum::<f64>();
                    }
                    adj_linear(&mut g, a[i], 1.0 - a[i]);
                }
            }
            eq
        };
        let eq: Vec<f64> = if threads == 1 {
            partial(0, self.nodes.len())
        } else {
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..threads).map(|c| (c * chunk, ((c + 1) * chunk).min(self.nodes.len()))).filter(|(lo, hi)| lo < hi).map(|(lo, hi)| sc.spawn(move || partial(lo, hi))).collect();
                let mut acc = vec![0.0f64; tn];
                for h in hs {
                    for (x, y) in acc.iter_mut().zip(h.join().expect("hilo de ICM de campo")) {
                        *x += y;
                    }
                }
                acc
            })
        };
        out.copy_from_slice(&eq);
    }
    /// Número de nodos de la integral (diagnóstico).
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Transpuesto de [`mul_linear`]: `g ← Lᵀ g`, con `(Lᵀ g)[c] = a·g[c] + b·g[c+1]`.
#[inline]
fn adj_linear(g: &mut [f64], a: f64, b: f64) {
    let n = g.len();
    for c in 0..n - 1 {
        g[c] = a * g[c] + b * g[c + 1];
    }
    g[n - 1] *= a;
}

/// `p ← p · (a + b x)` truncado a `p.len()` coeficientes.
#[inline]
fn mul_linear(p: &mut [f64], a: f64, b: f64) {
    for c in (1..p.len()).rev() {
        p[c] = p[c] * a + p[c - 1] * b;
    }
    p[0] *= a;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::icm::{icm_equity, icm_equity_sparse};
    use crate::math::rng::SplitMix64;

    /// Frente al DP exacto (≤ 20 jugadores): error relativo < 1e-6 del premio total.
    #[test]
    fn matches_exact_dp() {
        let mut rng = SplitMix64::new(42);
        for (n_table, n_off, paid) in [(6, 10, 9), (9, 8, 3), (3, 15, 18), (2, 12, 5)] {
            let table: Vec<f64> = (0..n_table).map(|_| 5.0 + 60.0 * rng.next_f64_open0()).collect();
            let off: Vec<f64> = (0..n_off).map(|_| 5.0 + 60.0 * rng.next_f64_open0()).collect();
            let payouts: Vec<f64> = (0..paid).map(|k| 100.0 / (k as f64 + 1.5)).collect();
            let all: Vec<f64> = table.iter().chain(&off).copied().collect();
            let mut exact = vec![0.0; all.len()];
            icm_equity(&all, &payouts, &mut exact);
            let min_t = table.iter().copied().fold(f64::INFINITY, f64::min);
            let f = FieldIcm::new(&off, table.iter().sum(), min_t, &payouts, 2);
            let mut got = vec![0.0; n_table];
            f.equity(&table, paid, &mut got);
            let pool: f64 = payouts.iter().sum();
            for i in 0..n_table {
                assert!((got[i] - exact[i]).abs() < 1e-6 * pool, "({n_table},{n_off},{paid}) jugador {i}: {} vs {}", got[i], exact[i]);
            }
        }
    }

    /// Campo de 60 jugadores con 5 premios: frente al DP disperso exacto.
    #[test]
    fn matches_sparse_dp_on_60_players() {
        let mut rng = SplitMix64::new(7);
        let table: Vec<f64> = (0..8).map(|_| 10.0 + 40.0 * rng.next_f64_open0()).collect();
        let off: Vec<f64> = (0..52).map(|_| 10.0 + 40.0 * rng.next_f64_open0()).collect();
        let payouts = vec![40.0, 25.0, 15.0, 12.0, 8.0];
        let all: Vec<f64> = table.iter().chain(&off).copied().collect();
        let mut exact = vec![0.0; all.len()];
        icm_equity_sparse(&all, &payouts, &mut exact);
        let f = FieldIcm::new(&off, table.iter().sum(), 10.0, &payouts, 2);
        let mut got = vec![0.0; 8];
        f.equity(&table, payouts.len(), &mut got);
        for i in 0..8 {
            assert!((got[i] - exact[i]).abs() < 1e-6 * 100.0, "{i}: {} vs {}", got[i], exact[i]);
        }
    }

    /// Campo simétrico de 2000 stacks iguales y 300 premios: cada jugador vale exactamente
    /// el total de premios / 2000. Y la evaluación es rápida tras la precomputación.
    #[test]
    fn symmetric_field_of_thousands() {
        let payouts: Vec<f64> = (0..300).map(|k| 1000.0 / (k as f64 + 3.0).powf(1.1)).collect();
        let pool: f64 = payouts.iter().sum();
        let t0 = std::time::Instant::now();
        let f = FieldIcm::new(&vec![30.0; 1994], 180.0, 1.0, &payouts, 4);
        let pre = t0.elapsed();
        let t0 = std::time::Instant::now();
        let mut got = vec![0.0; 6];
        f.equity(&[30.0; 6], payouts.len(), &mut got);
        let eval = t0.elapsed();
        eprintln!("precomputación {pre:?}, evaluación {eval:?}");
        for g in &got {
            assert!((g - pool / 2000.0).abs() < 1e-5 * pool / 2000.0, "{g} vs {}", pool / 2000.0);
        }
        assert!(eval.as_millis() < 500, "evaluación lenta: {eval:?}");
    }

    /// Campo aleatorio de 2000 jugadores: la integral está convergida (con la mitad de
    /// paso cambia menos de 1e-5 relativo). Frente a Monte Carlo con 4 M de muestras la
    /// diferencia está dentro del ruido de este (±0,5 %).
    #[test]
    fn random_field_converged() {
        let mut rng = SplitMix64::new(9);
        let table: Vec<f64> = vec![8.0, 25.0, 40.0, 15.0, 60.0, 30.0];
        let off: Vec<f64> = (0..1994).map(|_| 5.0 + 70.0 * rng.next_f64_open0()).collect();
        let payouts: Vec<f64> = (0..300).map(|k| 1000.0 / (k as f64 + 3.0).powf(1.1)).collect();
        let (mut a, mut b) = (vec![0.0; 6], vec![0.0; 6]);
        FieldIcm::new(&off, table.iter().sum(), 1.0, &payouts, 4).equity(&table, 300, &mut a);
        FieldIcm::with_step(&off, table.iter().sum(), 1.0, &payouts, 4, DEFAULT_STEP / 2.0).equity(&table, 300, &mut b);
        for i in 0..6 {
            assert!((a[i] - b[i]).abs() < 1e-5 * b[i], "{i}: {} vs {}", a[i], b[i]);
        }
    }
}
