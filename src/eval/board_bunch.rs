//! # Board bunching preflop
//!
//! Los jugadores que foldearon se llevan cartas del mazo, y no al azar: foldean manos
//! flojas, así que se llevan sobre todo cartas bajas y el board sale con más cartas altas.
//! La matriz de equity [`EquityMatrix`] supone
//! un board uniforme. Sin esto, en 10/12/5/15/7 bb ICM el push de la SB tras tres folds
//! valía +0,086 $ de más con 72o.
//!
//! ## Modelo (primer orden)
//! Si los que foldearon tienen la carta `c` con probabilidad `Q(c)`, el board la ve con
//! peso `ω_c ∝ 1 − Q(c)`. Con `δ_c = ω_c/ω̄ − 1` pequeño, la equity de `h` contra `v`
//! cambia en `Cov_B(win(h,v,B), Σ_{c∈B} δ_c)`. Con `δ` por rango (los rangos son simétricos
//! en palos), eso es `Σ_r δ_r·D_r(h,v)` con
//!
//! `D_r(h,v) = 5/48 · n_r(h,v) · (E[win | una carta de rango r en el board] − E[win])`,
//!
//! donde `n_r` son las cartas de rango `r` que no tienen `h` ni `v`. Salen, exactas, de la
//! misma enumeración de todos los boards que la matriz base: `Σ_r b(r)·D_r` es la
//! covarianza entre `win` y `Σ_{c∈B} b(rango c)`.
//!
//! Para no multiplicar el coste de cada terminal por 13, `δ` se ajusta por mínimos
//! cuadrados sobre unas pocas funciones base de rango (más una constante, que no cambia
//! nada: `Σ_r D_r = 0`) y se guardan solo las combinaciones `Σ_r b_k(r)·D_r`.

use super::equity::{EquityMatrix, PairProducts};
use super::{HAND_STRIDE, NUM_COMBOS};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"PFBB0002";
const CELLS: usize = NUM_COMBOS * HAND_STRIDE;

/// Una combinación `Σ_r b(r)·D_r`, cuantizada a `i16` con escala propia.
#[derive(Debug)]
struct Mat {
    w: Vec<i16>,
    t: Vec<i16>,
    sw: f32,
    st: f32,
}

#[derive(Debug)]
pub struct BoardBunching {
    /// Funciones base sobre los 13 rangos (2 … A).
    pub basis: Vec<[f32; 13]>,
    mats: Vec<Mat>,
    pub boards: u32,
}

/// Base por defecto: polinomios de grado 1-3 en el rango más un indicador del as (los
/// rangos de push están cargados de ases: su `δ` va aparte del resto).
pub fn default_basis() -> Vec<[f32; 13]> {
    let mut b = Vec::new();
    for d in 1..=3 {
        let mut f = [0.0f32; 13];
        for (r, x) in f.iter_mut().enumerate() {
            *x = ((r as f32 - 6.0) / 6.0).powi(d);
        }
        b.push(f);
    }
    let mut ace = [0.0f32; 13];
    ace[12] = 1.0;
    b.push(ace);
    b
}

/// Base completa (exacta a primer orden): un indicador por rango salvo el as, que queda en
/// la constante.
pub fn full_basis() -> Vec<[f32; 13]> {
    (0..12)
        .map(|r| {
            let mut f = [0.0f32; 13];
            f[r] = 1.0;
            f
        })
        .collect()
}

fn quantize(v: &[f32]) -> (Vec<i16>, f32) {
    let m = v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let s = if m > 0.0 { m / 32767.0 } else { 1.0 };
    (v.iter().map(|&x| (x / s).round() as i16).collect(), s)
}

impl BoardBunching {
    /// Construye las combinaciones de la base enumerando todos los boards (exacto).
    pub fn build(basis: Vec<[f32; 13]>, threads: usize) -> Self {
        let (_, dm) = EquityMatrix::exact_with_basis(threads, &basis);
        let mats = dm
            .iter()
            .map(|(w, t)| {
                let ((w, sw), (t, st)) = (quantize(w), quantize(t));
                Mat { w, t, sw, st }
            })
            .collect();
        BoardBunching { basis, mats, boards: 0 }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&self.boards.to_le_bytes())?;
        f.write_all(&(self.basis.len() as u32).to_le_bytes())?;
        for b in &self.basis {
            for x in b {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        for m in &self.mats {
            f.write_all(&m.sw.to_le_bytes())?;
            f.write_all(&m.st.to_le_bytes())?;
            for x in m.w.iter().chain(&m.t) {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        f.flush()
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "caché de board bunching inválida");
        let mut b = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut b)?;
        if b.len() < 16 || &b[..8] != MAGIC {
            return Err(bad());
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let f32_at = |i: usize| f32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let (boards, k) = (u32_at(8), u32_at(12) as usize);
        let mut at = 16;
        if b.len() != at + k * 52 + k * (8 + 4 * CELLS) {
            return Err(bad());
        }
        let basis = (0..k)
            .map(|_| {
                let mut f = [0.0f32; 13];
                for x in f.iter_mut() {
                    *x = f32_at(at);
                    at += 4;
                }
                f
            })
            .collect();
        let mats = (0..k)
            .map(|_| {
                let (sw, st) = (f32_at(at), f32_at(at + 4));
                at += 8;
                let mut read = || {
                    let v: Vec<i16> = b[at..at + 2 * CELLS].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
                    at += 2 * CELLS;
                    v
                };
                let w = read();
                let t = read();
                Mat { w, t, sw, st }
            })
            .collect();
        Ok(BoardBunching { basis, mats, boards })
    }

    /// Carga la caché si coincide la base; si no, construye y guarda.
    pub fn load_or_build(path: &Path, basis: Vec<[f32; 13]>, threads: usize) -> Self {
        if let Ok(m) = Self::load(path) {
            if m.boards == 0 && m.basis == basis {
                return m;
            }
        }
        eprintln!("board bunching: calculando las matrices exactas (solo la primera vez, 1-3 min)…");
        let m = Self::build(basis, threads);
        if let Err(e) = m.save(path) {
            eprintln!("aviso: no se pudo guardar la caché de board bunching en {}: {e}", path.display());
        }
        m
    }

    /// Pesos del mazo por rango a partir de los rangos de los que no están en el bote:
    /// `δ_r = ω_r/ω̄ − 1` con `ω_r = 1 − Q_r`, `Q_r` la probabilidad media de que alguno
    /// tenga una carta de rango `r`. `None` si no hay nadie fuera.
    pub fn deck_deltas<'a>(outs: impl Iterator<Item = &'a [f32]>) -> Option<[f32; 13]> {
        let mut q = [0.0f64; 13];
        let mut any = false;
        for r in outs {
            let bs = super::equity::BlockerSums::new(r);
            if bs.total <= 0.0 {
                continue;
            }
            any = true;
            for c in 0..52 {
                q[c >> 2] += (bs.card[c] / bs.total) as f64 / 4.0;
            }
        }
        if !any {
            return None;
        }
        let mean = q.iter().map(|x| 1.0 - x).sum::<f64>() / 13.0;
        let mut d = [0.0f32; 13];
        for r in 0..13 {
            d[r] = ((1.0 - q[r]) / mean - 1.0) as f32;
        }
        Some(d)
    }

    /// Coeficientes de `δ` en la base (mínimos cuadrados con constante libre).
    pub fn coefs(&self, delta: &[f32; 13]) -> Vec<f32> {
        let k = self.basis.len() + 1;
        let col = |j: usize, r: usize| if j == 0 { 1.0 } else { self.basis[j - 1][r] as f64 };
        let mut a = vec![0.0f64; k * (k + 1)];
        for i in 0..k {
            for j in 0..k {
                a[i * (k + 1) + j] = (0..13).map(|r| col(i, r) * col(j, r)).sum();
            }
            a[i * (k + 1) + k] = (0..13).map(|r| col(i, r) * delta[r] as f64).sum();
        }
        // Gauss-Jordan con pivote parcial (sistema de 5-13 incógnitas).
        for c in 0..k {
            let p = (c..k).max_by(|&x, &y| a[x * (k + 1) + c].abs().total_cmp(&a[y * (k + 1) + c].abs())).unwrap();
            for j in 0..=k {
                a.swap(c * (k + 1) + j, p * (k + 1) + j);
            }
            let d = a[c * (k + 1) + c];
            if d.abs() < 1e-12 {
                continue;
            }
            for j in 0..=k {
                a[c * (k + 1) + j] /= d;
            }
            for i in (0..k).filter(|&i| i != c) {
                let f = a[i * (k + 1) + c];
                if f != 0.0 {
                    for j in 0..=k {
                        a[i * (k + 1) + j] -= f * a[c * (k + 1) + j];
                    }
                }
            }
        }
        (1..k).map(|j| a[j * (k + 1) + k] as f32).collect()
    }

    /// Suma a los productos `W·r`, `T·r` de `p` la corrección `Σ_k c_k·(D_k·r)`.
    pub fn apply(&self, p: &mut PairProducts, r: &[f32], coefs: &[f32]) {
        let nz = r[..NUM_COMBOS].iter().filter(|&&x| x != 0.0).count();
        let (mut cw, mut ct) = (vec![0.0f32; HAND_STRIDE], vec![0.0f32; HAND_STRIDE]);
        if nz * 2 < NUM_COMBOS {
            // `D^W(h,v) = −D^W(v,h) − D^T(v,h)` (win + lose + tie = 1 en cada matriz) y
            // `D^T` simétrica: basta recorrer las filas de los combos del rango.
            let mut aw = vec![0.0f32; HAND_STRIDE];
            for (m, &c) in self.mats.iter().zip(coefs) {
                if c == 0.0 {
                    continue;
                }
                for (v, &x) in r[..NUM_COMBOS].iter().enumerate() {
                    if x != 0.0 {
                        let row = v * HAND_STRIDE..(v + 1) * HAND_STRIDE;
                        axpy(c * x * m.sw, &m.w[row.clone()], &mut aw);
                        axpy(c * x * m.st, &m.t[row], &mut ct);
                    }
                }
            }
            for h in 0..NUM_COMBOS {
                cw[h] = -(aw[h] + ct[h]);
            }
        } else {
            for (m, &c) in self.mats.iter().zip(coefs) {
                if c == 0.0 {
                    continue;
                }
                for h in 0..NUM_COMBOS {
                    let row = h * HAND_STRIDE..(h + 1) * HAND_STRIDE;
                    cw[h] += c * m.sw * dot_q(&m.w[row.clone()], r);
                    ct[h] += c * m.st * dot_q(&m.t[row], r);
                }
            }
        }
        for h in 0..NUM_COMBOS {
            // `all` puede salir en −1e-9 por redondeo: sin `clamp`, que exige min ≤ max.
            let all = p.all[h].max(0.0);
            let t = (p.t[h] + ct[h]).max(0.0).min(all);
            p.w[h] = (p.w[h] + cw[h]).max(0.0).min(all - t);
            p.t[h] = t;
        }
    }
}

#[inline]
fn dot_q(a: &[i16], r: &[f32]) -> f32 {
    let mut x = [0.0f32; 16];
    for (ca, cr) in a.chunks_exact(16).zip(r.chunks_exact(16)) {
        for l in 0..16 {
            x[l] += ca[l] as f32 * cr[l];
        }
    }
    x.iter().sum()
}

#[inline]
fn axpy(x: f32, a: &[i16], out: &mut [f32]) {
    for (ca, co) in a.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        for l in 0..16 {
            co[l] += x * ca[l] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::COMBO_MASKS;
    use crate::math::rng::SplitMix64;

    /// Matrices aleatorias con la estructura de las reales: `D^T` simétrica y
    /// `D^W(h,v) + D^W(v,h) + D^T(h,v) = 0`, cero en pares que comparten carta.
    fn random_bb(k: usize) -> BoardBunching {
        let mut rng = SplitMix64::new(5);
        let mats = (0..k)
            .map(|_| {
                let (mut w, mut t) = (vec![0.0f32; CELLS], vec![0.0f32; CELLS]);
                for v in 1..NUM_COMBOS {
                    for h in 0..v {
                        if COMBO_MASKS[h] & COMBO_MASKS[v] != 0 {
                            continue;
                        }
                        let (a, b) = (rng.next_f32() - 0.5, (rng.next_f32() - 0.5) * 0.1);
                        w[h * HAND_STRIDE + v] = a;
                        w[v * HAND_STRIDE + h] = -a - b;
                        t[h * HAND_STRIDE + v] = b;
                        t[v * HAND_STRIDE + h] = b;
                    }
                }
                let ((w, sw), (t, st)) = (quantize(&w), quantize(&t));
                Mat { w, t, sw, st }
            })
            .collect();
        BoardBunching { basis: default_basis()[..k].to_vec(), mats, boards: 0 }
    }

    #[test]
    fn sparse_and_dense_corrections_agree() {
        let bb = random_bb(2);
        let coefs = [0.7f32, -0.3];
        let mut rng = SplitMix64::new(9);
        // Rango estrecho (ruta dispersa) y el mismo con un peso ínfimo en el resto (densa).
        let mut narrow = vec![0.0f32; HAND_STRIDE];
        for h in 0..NUM_COMBOS {
            if rng.next_f32() < 0.2 {
                narrow[h] = rng.next_f32();
            }
        }
        let mut wide = narrow.clone();
        for x in wide[..NUM_COMBOS].iter_mut().filter(|x| **x == 0.0) {
            *x = 1e-9;
        }
        let blank = || PairProducts { w: vec![0.5; HAND_STRIDE], t: vec![0.5; HAND_STRIDE], all: vec![1e6; HAND_STRIDE] };
        let (mut a, mut b) = (blank(), blank());
        bb.apply(&mut a, &narrow, &coefs);
        bb.apply(&mut b, &wide, &coefs);
        for h in 0..NUM_COMBOS {
            assert!((a.w[h] - b.w[h]).abs() < 1e-3 * (1.0 + a.w[h].abs()), "w[{h}] {} vs {}", a.w[h], b.w[h]);
            assert!((a.t[h] - b.t[h]).abs() < 1e-3 * (1.0 + a.t[h].abs()), "t[{h}] {} vs {}", a.t[h], b.t[h]);
        }
    }

    #[test]
    fn full_basis_fits_any_deck_exactly() {
        let bb = BoardBunching { basis: full_basis(), mats: Vec::new(), boards: 0 };
        let d = [0.03f32, -0.02, 0.0, 0.01, 0.05, -0.04, 0.02, 0.0, 0.01, -0.01, 0.02, 0.03, 0.09];
        let c = bb.coefs(&d);
        // δ_r − δ_A = c_r (el as queda en la constante).
        for r in 0..12 {
            assert!((c[r] - (d[r] - d[12])).abs() < 1e-5, "{r}: {} vs {}", c[r], d[r] - d[12]);
        }
    }
}
