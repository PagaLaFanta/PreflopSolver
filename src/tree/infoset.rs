//! # Infosets postflop: un `AtomicU64` por infoset + tabla hash lock-free
//!
//! El árbol postflop del Agente Nash tiene **exactamente 2 acciones por nodo**
//! (Check | GeoBet, o Fold | Call). Eso permite empaquetar el estado CFR completo de un
//! infoset en 64 bits con **block floating point por par**:
//!
//! ```text
//!  63      56 55     44 43     32 31      24 23     12 11      0
//! [ e_strat ][ s1:12  ][ s0:12  ][ e_regret ][ r1:±11  ][ r0:±11  ]
//! valor = m · 2^(e − 64)
//! ```
//!
//! * Actualización atómica completa del infoset con **un único CAS**: nunca se observa un
//!   estado a medio escribir, sin mutex ni seqlock.
//! * 8 B/infoset + 8 B de clave = 16 B/slot → 4 infosets por línea de caché.
//! * El exponente compartido por par hace que regret matching (`r0/(r0+r1)`) y la
//!   estrategia media no necesiten decodificar: el factor `2^e` se cancela.
//! * Rango dinámico `2^-64 … 2^191`: no hay desbordamiento ni reescalados que descuenten
//!   la historia. La precisión relativa es 1/4096 respecto a la mayor componente del par;
//!   el redondeo es **estocástico** (insesgado), así que incrementos por debajo de la
//!   resolución se acumulan correctamente en esperanza.
//!
//! (La versión de la iteración 1, `u16` sin exponente con división a la mitad al
//! desbordar, descontaba la historia casi en cada actualización con los valores reales del
//! MCCFR: el regret acumulado degeneraba en el instantáneo.)

use super::buf::{AlignedBuf, ZeroInit};
use crate::math::rng::SplitMix64;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
#[repr(transparent)]
pub struct PostflopInfoset(AtomicU64);

// SAFETY: repr(transparent) sobre AtomicU64; cero = regrets y sumas nulos (válido).
unsafe impl ZeroInit for PostflopInfoset {}

const MANT_BITS: u32 = 12;
const MANT_MAX: u64 = (1 << MANT_BITS) - 1;
const EXP_BIAS: i32 = 64;

#[inline(always)]
fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// Par `(m0, m1, e)` empaquetado en 32 bits.
#[inline(always)]
fn decode_pair(w: u32) -> (u64, u64, i32) {
    ((w as u64) & MANT_MAX, ((w as u64) >> MANT_BITS) & MANT_MAX, (w >> 24) as i32)
}

#[inline(always)]
fn pair_values(w: u32) -> [f64; 2] {
    let (m0, m1, e) = decode_pair(w);
    let s = pow2(e - EXP_BIAS);
    [m0 as f64 * s, m1 as f64 * s]
}

/// Codifica dos magnitudes ≥ 0 con el menor exponente en que ambas caben en 12 bits.
#[inline(always)]
fn encode_pair(v: [f64; 2], u: [f32; 2]) -> u32 {
    let max = v[0].max(v[1]);
    if max <= 0.0 {
        return 0;
    }
    // Exponente binario de `max` leído de los bits del f64 (sin log2).
    let k = ((max.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let mut e = (k - (MANT_BITS as i32 - 1) + EXP_BIAS).clamp(0, 255);
    let mut inv = pow2(EXP_BIAS - e);
    // Si el redondeo hacia arriba pudiera producir 4096, sube un exponente: la saturación
    // en 4095 congelaría el valor (bug detectado por el test de acumulación).
    if max * inv >= MANT_MAX as f64 && e < 255 {
        e += 1;
        inv *= 0.5;
    }
    let m = |x: f64, u: f32| -> u64 { ((x * inv + u as f64).floor() as u64).min(MANT_MAX) };
    (m(v[0], u[0]) | (m(v[1], u[1]) << MANT_BITS) | ((e as u64) << 24)) as u32
}

#[inline(always)]
fn ratio(w: u32) -> [f32; 2] {
    let (m0, m1, _) = decode_pair(w);
    let s = m0 + m1;
    if s == 0 {
        [0.5, 0.5]
    } else {
        let p = m0 as f32 / s as f32;
        [p, 1.0 - p]
    }
}

/// Regrets con signo: como [`encode_pair`] pero cada campo de 12 bits es signo + 11 bits
/// de magnitud (regret matching sin suelo necesita regrets negativos).
const SMAG_BITS: u32 = 11;
const SMAG_MAX: u64 = (1 << SMAG_BITS) - 1;

#[inline(always)]
fn signed_values(w: u32) -> [f64; 2] {
    let (f0, f1, e) = decode_pair(w);
    let s = pow2(e - EXP_BIAS);
    let v = |f: u64| -> f64 {
        let m = (f & SMAG_MAX) as f64 * s;
        if f >> SMAG_BITS != 0 { -m } else { m }
    };
    [v(f0), v(f1)]
}

#[inline(always)]
fn encode_signed(v: [f64; 2], u: [f32; 2]) -> u32 {
    let max = v[0].abs().max(v[1].abs());
    if max <= 0.0 {
        return 0;
    }
    let k = ((max.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let mut e = (k - (SMAG_BITS as i32 - 1) + EXP_BIAS).clamp(0, 255);
    let mut inv = pow2(EXP_BIAS - e);
    if max * inv >= SMAG_MAX as f64 && e < 255 {
        e += 1;
        inv *= 0.5;
    }
    let m = |x: f64, u: f32| -> u64 {
        let q = ((x.abs() * inv + u as f64).floor() as u64).min(SMAG_MAX);
        if x < 0.0 && q > 0 { q | (1 << SMAG_BITS) } else { q }
    };
    (m(v[0], u[0]) | (m(v[1], u[1]) << MANT_BITS) | ((e as u64) << 24)) as u32
}

/// Regret matching sobre la parte positiva de los regrets con signo.
#[inline(always)]
fn positive_ratio(w: u32) -> [f32; 2] {
    let [a, b] = signed_values(w);
    let (a, b) = (a.max(0.0), b.max(0.0));
    let s = a + b;
    if s <= 0.0 {
        [0.5, 0.5]
    } else {
        let p = (a / s) as f32;
        [p, 1.0 - p]
    }
}

impl PostflopInfoset {
    /// Estrategia actual por regret matching (parte positiva de los regrets).
    #[inline]
    pub fn strategy(&self) -> [f32; 2] {
        positive_ratio(self.0.load(Ordering::Relaxed) as u32)
    }

    /// Estrategia media. Un infoset al que el propio jugador nunca llega (suma de
    /// estrategia nula: manos fuera de su rango en esa línea) no tiene media; sus regrets
    /// sí se entrenan (van ponderados por el alcance de los rivales), así que se usa la
    /// estrategia actual en vez de 50/50: el EV de las manos que no juegan una acción
    /// (lo que se ve en la tabla de EV) sale de jugarla con sentido.
    #[inline]
    pub fn average_strategy(&self) -> [f32; 2] {
        let w = self.0.load(Ordering::Relaxed);
        if (w >> 32) as u32 & 0x00FF_FFFF == 0 {
            return positive_ratio(w as u32);
        }
        ratio((w >> 32) as u32)
    }

    /// `[r0, r1, s0, s1]` decodificados.
    pub fn values(&self) -> [f64; 4] {
        let w = self.0.load(Ordering::Relaxed);
        let [r0, r1] = signed_values(w as u32);
        let [s0, s1] = pair_values((w >> 32) as u32);
        [r0, r1, s0, s1]
    }

    /// Suma deltas de regret y de suma de estrategia. `plus`: suelo CFR+ en 0 tras cada
    /// actualización; sin él, regret matching clásico (regrets negativos acumulados). Con
    /// muestreo (un board por actualización) el suelo sobre ruido sesga: una acción mala
    /// pero ruidosa mantiene regret ~σ²/2μ y probabilidad positiva mucho tiempo.
    /// Lock-free: bucle CAS; la aleatoriedad del redondeo se fija antes del bucle para que
    /// un reintento no consuma estado del RNG.
    pub fn update(&self, regret: [f32; 2], strat: [f32; 2], plus: bool, rng: &mut SplitMix64) {
        let u = [rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32()];
        let mut cur = self.0.load(Ordering::Relaxed);
        loop {
            let [r0, r1] = signed_values(cur as u32);
            let [s0, s1] = pair_values((cur >> 32) as u32);
            let mut r = [r0 + regret[0] as f64, r1 + regret[1] as f64];
            if plus {
                r = [r[0].max(0.0), r[1].max(0.0)];
            }
            let st = [(s0 + strat[0] as f64).max(0.0), (s1 + strat[1] as f64).max(0.0)];
            let new = encode_signed(r, [u[0], u[1]]) as u64 | ((encode_pair(st, [u[2], u[3]]) as u64) << 32);
            // Relaxed basta: el valor es autocontenido (no publica otra memoria).
            match self.0.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }
}

/// Slot de 16 B: clave y datos en la misma línea → un único cache miss por lookup+update.
#[derive(Debug)]
#[repr(C, align(16))]
struct Slot {
    /// `0` = vacío; en otro caso `key | OCCUPIED`.
    key: AtomicU64,
    data: PostflopInfoset,
}
// SAFETY: todos los campos son ZeroInit.
unsafe impl ZeroInit for Slot {}

const OCCUPIED: u64 = 1 << 63;

/// Tabla hash de direccionamiento abierto con sondeo lineal, **insert-only** (MCCFR nunca
/// borra infosets), capacidad potencia de 2.
///
/// Inserción lock-free: el primer hilo que gana el CAS `0 → key` reclama el slot; los
/// perdedores releen la clave ganadora y o bien coinciden (mismo infoset, lo usan) o bien
/// siguen sondeando. Como las claves nunca cambian una vez escritas, no hay ABA.
/// Orden `Acquire` en la lectura de clave: empareja con el `AcqRel` del CAS ganador.
#[derive(Debug)]
pub struct InfosetTable {
    slots: AlignedBuf<Slot>,
    shift: u32,
    mask: usize,
}

impl InfosetTable {
    /// `log2_capacity`: dimensionar para factor de carga ≤ 0.7.
    pub fn with_log2_capacity(log2_capacity: u32) -> Self {
        assert!((4..=40).contains(&log2_capacity));
        let cap = 1usize << log2_capacity;
        InfosetTable { slots: AlignedBuf::zeroed(cap), shift: 64 - log2_capacity, mask: cap - 1 }
    }

    /// Hash multiplicativo de Fibonacci: toma los bits altos, que mezclan todos los de la clave.
    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// Devuelve el infoset de `key`, creándolo si no existe. `None` si la tabla está llena.
    pub fn get_or_insert(&self, key: u64) -> Option<&PostflopInfoset> {
        debug_assert!(key & OCCUPIED == 0, "las claves usan como máximo 63 bits");
        let tagged = key | OCCUPIED;
        let mut i = self.home(key);
        for _ in 0..=self.mask {
            let slot = &self.slots[i];
            let k = slot.key.load(Ordering::Acquire);
            if k == tagged {
                return Some(&slot.data);
            }
            if k == 0 {
                match slot.key.compare_exchange(0, tagged, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => return Some(&slot.data),
                    Err(winner) if winner == tagged => return Some(&slot.data),
                    Err(_) => {}
                }
            }
            i = (i + 1) & self.mask;
        }
        None
    }

    /// Pares `(clave, estado)` de los infosets ocupados (para guardar soluciones).
    pub fn export(&self) -> Vec<(u64, u64)> {
        self.slots
            .iter()
            .filter_map(|s| {
                let k = s.key.load(Ordering::Acquire);
                (k != 0).then(|| (k & !OCCUPIED, s.data.0.load(Ordering::Relaxed)))
            })
            .collect()
    }

    /// Restaura infosets exportados. Devuelve cuántos no cupieron.
    pub fn import(&self, items: &[(u64, u64)]) -> usize {
        let mut lost = 0;
        for &(k, v) in items {
            match self.get_or_insert(k) {
                Some(info) => info.0.store(v, Ordering::Relaxed),
                None => lost += 1,
            }
        }
        lost
    }

    pub fn get(&self, key: u64) -> Option<&PostflopInfoset> {
        let tagged = key | OCCUPIED;
        let mut i = self.home(key);
        for _ in 0..=self.mask {
            let slot = &self.slots[i];
            match slot.key.load(Ordering::Acquire) {
                0 => return None,
                k if k == tagged => return Some(&slot.data),
                _ => i = (i + 1) & self.mask,
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn block_float_accumulates_without_discounting() {
        let info = PostflopInfoset::default();
        let mut rng = SplitMix64::new(9);
        info.update([3e6, 1e6], [1.0, 3.0], true, &mut rng);
        info.update([6e6, 2e6], [0.0, 0.0], true, &mut rng);
        let s = info.strategy();
        assert!((s[0] - 0.75).abs() < 1e-3, "{s:?}");
        assert!((info.average_strategy()[1] - 0.75).abs() < 1e-6);
        let v = info.values();
        assert!((v[0] / 9e6 - 1.0).abs() < 1e-3, "sin descuento: {v:?}");
        // Incrementos muy por debajo de la resolución se acumulan en esperanza.
        let small = PostflopInfoset::default();
        small.update([1000.0, 1000.0], [0.0, 0.0], true, &mut rng);
        for _ in 0..10_000 {
            small.update([0.01, 0.0], [0.0, 0.0], true, &mut rng);
        }
        let v = small.values();
        assert!((v[0] - 1100.0).abs() < 15.0, "{v:?}");
        // Suelo CFR+.
        small.update([-1e9, 0.0], [0.0, 0.0], true, &mut rng);
        assert_eq!(small.strategy(), [0.0, 1.0]);
        // Sin suelo: el regret negativo se recuerda y hay que compensarlo antes de volver a
        // jugar la acción.
        let rm = PostflopInfoset::default();
        rm.update([-300.0, 100.0], [0.0, 0.0], false, &mut rng);
        rm.update([200.0, 0.0], [0.0, 0.0], false, &mut rng);
        assert_eq!(rm.strategy(), [0.0, 1.0]);
        let v = rm.values();
        assert!((v[0] + 100.0).abs() < 1.0 && (v[1] - 100.0).abs() < 1.0, "{v:?}");
        rm.update([300.0, 0.0], [0.0, 0.0], false, &mut rng);
        assert!((rm.strategy()[0] - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn concurrent_insert_and_update_is_consistent() {
        let table = Arc::new(InfosetTable::with_log2_capacity(12));
        let threads: Vec<_> = (0..8u64)
            .map(|t| {
                let table = Arc::clone(&table);
                std::thread::spawn(move || {
                    let mut rng = SplitMix64::new(t);
                    for k in 0..1000u64 {
                        let info = table.get_or_insert(k * 7919).unwrap();
                        info.update([0.0, 0.0], [1.0, 0.0], true, &mut rng);
                    }
                })
            })
            .collect();
        threads.into_iter().for_each(|h| h.join().unwrap());
        for k in 0..1000u64 {
            let info = table.get(k * 7919).expect("infoset perdido");
            assert_eq!(info.values()[2], 8.0, "clave {k}: actualizaciones perdidas bajo contención");
        }
    }
}
