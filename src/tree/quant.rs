//! # Arena cuantizada preflop: `u16` + exponente compartido por bloque (block floating point)
//!
//! ## Representación
//! `valor = q · 2^(e − EXP_BIAS)`, con `q: u16` por (acción, combo) y `e: u8` compartido
//! por **bloque** = 16 combos consecutivos × todas las acciones del nodo.
//!
//! * `u16` sin signo porque la arena solo almacena magnitudes no negativas: regrets
//!   **CFR+** (el suelo `max(R, 0)` es parte del algoritmo) y sumas de estrategia. El bit
//!   de signo que no gastamos es un bit más de mantisa.
//! * Exponente por bloque (no por nodo): un combo con regret enorme (AA frente a un
//!   shove) no destruye la precisión de los otros 1310 combos del nodo; solo comparte
//!   escala con sus 15 vecinos de bloque.
//! * Exponente compartido entre acciones del mismo combo ⇒ **regret matching no necesita
//!   decodificar**: `σ(a|h) = q_a / Σ_b q_b`, el factor `2^e` se cancela. Idem para la
//!   estrategia media. El exponente solo importa al *acumular*.
//!
//! ## Acumulación
//! Se re-escala el delta al exponente del bloque y se redondea **estocásticamente**
//! (`⌊x + U[0,1)⌋`): el redondeo es insesgado, `E[q̂] = x`, así que el error de
//! cuantización es ruido de media cero que CFR promedia (mismo argumento que la
//! convergencia de MCCFR con estimadores insesgados). Redondeo a nearest introduciría un
//! sesgo sistemático hacia 0 en regrets pequeños → estrategias puras espurias.
//!
//! Si algún carril del bloque desborda `u16`, se incrementa `e` en el mínimo `s` necesario
//! y se desplazan **todos** los carriles del bloque `>> s` (con redondeo). La relación
//! entre acciones de cada combo se preserva, luego σ no cambia más allá del redondeo.
//!
//! ## Concurrencia
//! El CFR+ preflop es *full-width* sobre rangos: en cada iteración cada nodo se visita
//! exactamente una vez por jugador que actualiza. El scheduler reparte subárboles
//! disjuntos entre hilos ⇒ un bloque tiene **un único escritor por iteración** y no hace
//! falta RMW atómico. Aun así los carriles son `AtomicU16` con `Relaxed`:
//! * en x86-64/AArch64 `load/store Relaxed` compila a `mov`/`ldrh` plano: coste cero;
//! * lectores concurrentes (snapshot de estrategia, cálculo de explotabilidad en otro hilo
//!   mientras se resuelve) son data-race-free por definición del modelo de memoria de Rust.
//! El punto de sincronización entre iteraciones es la barrera del scheduler (Release/Acquire).

use super::buf::AlignedBuf;
use super::nodes::{CFRNode, BLOCKS_PER_ROW, HAND_STRIDE, LANE_BLOCK, MAX_ACTIONS};
use crate::math::rng::SplitMix64;
use std::sync::atomic::{AtomicU16, AtomicU8, Ordering::Relaxed};

/// `e = 0` ⇒ resolución `2^-40 ≈ 9.1e-13` unidades de utilidad.
pub const EXP_BIAS: i32 = 40;
/// Máximo `e` que mantiene `2^(e−BIAS)` como `f32` normal (exponente IEEE ≤ 254).
pub const EXP_MAX: u8 = (254 - 127 + EXP_BIAS) as u8;

#[inline(always)]
fn pow2(e: i32) -> f64 {
    // Construcción directa del f64 (sin `powi`): exponente sesgado IEEE-754 doble.
    f64::from_bits(((e + 1023) as u64) << 52)
}

#[derive(Debug)]
pub struct QuantArena {
    lanes: AlignedBuf<AtomicU16>,
    exps: AlignedBuf<AtomicU8>,
}

impl QuantArena {
    /// Reserva `rows` filas de 1328 carriles y `blocks` exponentes (normalmente
    /// `num_cfr_nodes × 83`). Reserva perezosa: ver [`super::buf`].
    pub fn new(rows: usize, blocks: usize) -> Self {
        QuantArena { lanes: AlignedBuf::zeroed(rows * HAND_STRIDE), exps: AlignedBuf::zeroed(blocks) }
    }

    #[inline(always)]
    fn row(&self, row: usize) -> &[AtomicU16] {
        &self.lanes[row * HAND_STRIDE..(row + 1) * HAND_STRIDE]
    }

    /// Pone toda la arena a cero (p.ej. el reinicio de estrategias: descartar la estrategia
    /// media acumulada conservando los regrets).
    pub fn clear(&self) {
        for l in self.lanes.iter() {
            l.store(0, Relaxed);
        }
        for e in self.exps.iter() {
            e.store(0, Relaxed);
        }
    }

    /// Volcado binario (carriles `u16` LE + exponentes) para guardar soluciones.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.lanes.len() * 2 + self.exps.len() + 16);
        v.extend((self.lanes.len() as u64).to_le_bytes());
        v.extend((self.exps.len() as u64).to_le_bytes());
        for l in self.lanes.iter() {
            v.extend(l.load(Relaxed).to_le_bytes());
        }
        v.extend(self.exps.iter().map(|e| e.load(Relaxed)));
        v
    }

    /// Arena nueva con las dimensiones y el contenido de un volcado de [`Self::to_bytes`].
    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        let rd = |o: usize| -> Result<u64, String> { Ok(u64::from_le_bytes(b.get(o..o + 8).ok_or("volcado truncado")?.try_into().unwrap())) };
        let (nl, ne) = (rd(0)? as usize, rd(8)? as usize);
        if nl % HAND_STRIDE != 0 {
            return Err("volcado con filas incompletas".into());
        }
        let a = QuantArena::new(nl / HAND_STRIDE, ne);
        a.load_bytes(b)?;
        Ok(a)
    }

    /// Carga un volcado de [`Self::to_bytes`]; falla si las dimensiones no coinciden.
    pub fn load_bytes(&self, b: &[u8]) -> Result<usize, String> {
        let rd = |o: usize| -> Result<u64, String> { Ok(u64::from_le_bytes(b.get(o..o + 8).ok_or("volcado truncado")?.try_into().unwrap())) };
        let (nl, ne) = (rd(0)? as usize, rd(8)? as usize);
        if nl != self.lanes.len() || ne != self.exps.len() {
            return Err(format!("dimensiones distintas: {nl}/{ne} frente a {}/{}", self.lanes.len(), self.exps.len()));
        }
        let end = 16 + nl * 2 + ne;
        if b.len() < end {
            return Err("volcado truncado".into());
        }
        for (i, l) in self.lanes.iter().enumerate() {
            l.store(u16::from_le_bytes([b[16 + 2 * i], b[17 + 2 * i]]), Relaxed);
        }
        for (i, e) in self.exps.iter().enumerate() {
            e.store(b[16 + 2 * nl + i], Relaxed);
        }
        Ok(end)
    }

    /// Valor decodificado (para depuración/exportación, no para el hot path).
    pub fn value(&self, node: &CFRNode, action: usize, combo: usize) -> f64 {
        let q = self.row(node.row as usize + action)[combo].load(Relaxed) as f64;
        let e = self.exps[node.exp_base as usize + combo / LANE_BLOCK].load(Relaxed) as i32;
        q * pow2(e - EXP_BIAS)
    }

    /// Acumula `deltas[a * HAND_STRIDE + h]` en el nodo con suelo en 0 (CFR+ / sumas de
    /// estrategia). Contrato: único escritor del nodo durante la llamada (ver doc del módulo).
    pub fn accumulate(&self, node: &CFRNode, deltas: &[f32], rng: &mut SplitMix64) {
        let na = node.num_actions as usize;
        debug_assert!(na <= MAX_ACTIONS && deltas.len() >= na * HAND_STRIDE);
        let mut rows: [&[AtomicU16]; MAX_ACTIONS] = [&[]; MAX_ACTIONS];
        for (a, r) in rows.iter_mut().enumerate().take(na) {
            *r = self.row(node.row as usize + a);
        }
        // Buffer local del bloque: 8 acciones × 16 carriles × 8 B = 1 KB, vive en L1.
        let mut tmp = [0u64; MAX_ACTIONS * LANE_BLOCK];
        for blk in 0..BLOCKS_PER_ROW {
            let exp_cell = &self.exps[node.exp_base as usize + blk];
            let e = exp_cell.load(Relaxed);
            let inv_scale = pow2(EXP_BIAS - e as i32);
            let base = blk * LANE_BLOCK;
            let mut max = 0u64;
            for a in 0..na {
                let d = &deltas[a * HAND_STRIDE + base..a * HAND_STRIDE + base + LANE_BLOCK];
                for l in 0..LANE_BLOCK {
                    let old = rows[a][base + l].load(Relaxed) as f64;
                    let x = old + d[l] as f64 * inv_scale;
                    // Redondeo estocástico + suelo CFR+. `as u64` satura (NaN → 0).
                    let q = if x <= 0.0 { 0 } else { (x + rng.next_f32() as f64).floor() as u64 };
                    tmp[a * LANE_BLOCK + l] = q;
                    max = max.max(q);
                }
            }
            let shift = (64 - max.leading_zeros()).saturating_sub(16);
            if shift > 0 {
                let new_e = (e as u32 + shift).min(EXP_MAX as u32) as u8;
                exp_cell.store(new_e, Relaxed);
                let s = (new_e - e) as u32;
                let half = if s > 0 { 1u64 << (s - 1) } else { 0 };
                for v in tmp[..na * LANE_BLOCK].iter_mut() {
                    // Redondeo half-up; `min` protege el caso degenerado de EXP_MAX saturado.
                    *v = ((*v + half) >> s).min(u16::MAX as u64);
                }
            }
            for a in 0..na {
                for l in 0..LANE_BLOCK {
                    rows[a][base + l].store(tmp[a * LANE_BLOCK + l] as u16, Relaxed);
                }
            }
        }
    }

    /// Copia las filas y los exponentes del nodo `src_node` de `src` en `node` (mismo
    /// número de acciones): arranque en caliente desde otra solución.
    pub fn copy_node_from(&self, node: &CFRNode, src: &QuantArena, src_node: &CFRNode) {
        let na = node.num_actions as usize;
        debug_assert_eq!(na, src_node.num_actions as usize);
        for blk in 0..BLOCKS_PER_ROW {
            let e = src.exps[src_node.exp_base as usize + blk].load(Relaxed);
            self.exps[node.exp_base as usize + blk].store(e, Relaxed);
        }
        for a in 0..na {
            for (to, from) in self.row(node.row as usize + a).iter().zip(src.row(src_node.row as usize + a)) {
                to.store(from.load(Relaxed), Relaxed);
            }
        }
    }

    /// Regret matching (CFR+) o normalización de estrategia media, según la arena.
    /// Escribe `out[a * HAND_STRIDE + h]`. Combos sin masa positiva → uniforme.
    ///
    /// Estructura en dos pasadas fila-a-fila (no combo-a-combo) para que cada bucle
    /// interno sea un stream contiguo de 1328 elementos → autovectorizable.
    pub fn normalized(&self, node: &CFRNode, out: &mut [f32]) {
        let na = node.num_actions as usize;
        let mut sum = [0.0f32; HAND_STRIDE];
        for a in 0..na {
            let row = self.row(node.row as usize + a);
            let o = &mut out[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
            for h in 0..HAND_STRIDE {
                let q = row[h].load(Relaxed) as f32;
                o[h] = q;
                sum[h] += q;
            }
        }
        let uniform = 1.0 / na as f32;
        for a in 0..na {
            let o = &mut out[a * HAND_STRIDE..(a + 1) * HAND_STRIDE];
            for h in 0..HAND_STRIDE {
                o[h] = if sum[h] > 0.0 { o[h] / sum[h] } else { uniform };
            }
        }
    }

    /// Como [`Self::normalized`], pero las manos con suma nula toman su columna de
    /// `fallback` (mismo formato) en vez de la uniforme.
    pub fn normalized_or(&self, node: &CFRNode, out: &mut [f32], fallback: &[f32]) {
        let na = node.num_actions as usize;
        let mut sum = [0.0f32; HAND_STRIDE];
        for a in 0..na {
            let row = self.row(node.row as usize + a);
            for h in 0..HAND_STRIDE {
                sum[h] += row[h].load(Relaxed) as f32;
            }
        }
        self.normalized(node, out);
        for a in 0..na {
            for h in 0..HAND_STRIDE {
                if sum[h] <= 0.0 {
                    out[a * HAND_STRIDE + h] = fallback[a * HAND_STRIDE + h];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(na: u8) -> CFRNode {
        CFRNode { row: 0, exp_base: 0, num_actions: na, ..Default::default() }
    }

    #[test]
    fn stochastic_rounding_is_unbiased() {
        let arena = QuantArena::new(2, BLOCKS_PER_ROW);
        let n = node(2);
        let mut rng = SplitMix64::new(1);
        // Deltas de 0.3 unidades de resolución: redondeo a nearest daría siempre 0.
        let tiny = 0.3 * pow2(-EXP_BIAS) as f32;
        let deltas = vec![tiny; 2 * HAND_STRIDE];
        for _ in 0..1000 {
            arena.accumulate(&n, &deltas, &mut rng);
        }
        let mean: f64 = (0..1326).map(|h| arena.value(&n, 0, h)).sum::<f64>() / 1326.0;
        let expected = 300.0 * pow2(-EXP_BIAS);
        assert!((mean / expected - 1.0).abs() < 0.02, "{mean} vs {expected}");
    }

    #[test]
    fn overflow_rescales_block_and_preserves_ratios() {
        let arena = QuantArena::new(3, BLOCKS_PER_ROW);
        let n = node(3);
        let mut rng = SplitMix64::new(2);
        let mut deltas = vec![0.0f32; 3 * HAND_STRIDE];
        for h in 0..HAND_STRIDE {
            deltas[h] = 1.0;
            deltas[HAND_STRIDE + h] = 3.0;
            deltas[2 * HAND_STRIDE + h] = 0.0;
        }
        for _ in 0..10 {
            arena.accumulate(&n, &deltas, &mut rng);
        }
        assert!((arena.value(&n, 1, 7) - 30.0).abs() / 30.0 < 1e-3);
        let mut sigma = vec![0.0f32; 3 * HAND_STRIDE];
        arena.normalized(&n, &mut sigma);
        assert!((sigma[7] - 0.25).abs() < 1e-3);
        assert!((sigma[HAND_STRIDE + 7] - 0.75).abs() < 1e-3);
        assert_eq!(sigma[2 * HAND_STRIDE + 7], 0.0);
    }

    #[test]
    fn cfr_plus_floor_and_uniform_fallback() {
        let arena = QuantArena::new(2, BLOCKS_PER_ROW);
        let n = node(2);
        let mut rng = SplitMix64::new(3);
        arena.accumulate(&n, &vec![-5.0f32; 2 * HAND_STRIDE], &mut rng);
        let mut sigma = vec![0.0f32; 2 * HAND_STRIDE];
        arena.normalized(&n, &mut sigma);
        assert!(sigma.iter().all(|&p| p == 0.5));
    }
}
