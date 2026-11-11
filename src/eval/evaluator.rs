//! # Evaluador de 5-7 cartas por máscaras de bits
//!
//! Entrada: máscara `u64` de cartas (`bit = rank*4 + suit`). Salida: `u32` con
//! `categoría << 20 | payload`, donde el payload son hasta 5 rangos de 4 bits en orden
//! de desempate. Mayor valor = mano más fuerte; cualquier mano válida es > 0.
//!
//! Sin tablas: todo son máscaras de 13 bits (una por palo + por multiplicidad) y
//! `leading_zeros`. ~20-40 ns por mano de 7 cartas; suficiente porque los consumidores
//! calientes (matriz de equity, pool de boards) evalúan cada (combo, board) una sola vez
//! y trabajan después sobre los rangos numéricos.
//!
//! Escalera: con la máscara de rangos desplazada un bit y el As replicado en el bit 0
//! (`m14`), `m14 & m14>>1 & … & m14>>4` deja un bit por cada inicio de escalera; el más
//! alto da la escalera máxima (la rueda A-5 cae naturalmente en la posición 0).

use super::HandEvaluator;

pub const HIGH_CARD: u32 = 0;
pub const PAIR: u32 = 1;
pub const TWO_PAIR: u32 = 2;
pub const TRIPS: u32 = 3;
pub const STRAIGHT: u32 = 4;
pub const FLUSH: u32 = 5;
pub const FULL_HOUSE: u32 = 6;
pub const QUADS: u32 = 7;
pub const STRAIGHT_FLUSH: u32 = 8;

#[inline(always)]
fn top(m: u16) -> u32 {
    15 - m.leading_zeros()
}

#[inline(always)]
fn straight_high(m: u16) -> Option<u32> {
    let m14 = ((m as u32) << 1) | ((m as u32 >> 12) & 1);
    let s = m14 & (m14 >> 1) & (m14 >> 2) & (m14 >> 3) & (m14 >> 4);
    // bit p ⇒ bits p..p+4 de m14 ⇒ carta alta = rango p+3.
    (s != 0).then(|| 31 - s.leading_zeros() + 3)
}

/// Empaqueta los `k` rangos más altos de `m` como nibbles tras los `lead` ya empaquetados.
#[inline(always)]
fn kickers(mut m: u16, k: u32, mut payload: u32, lead: u32) -> u32 {
    for _ in 0..k {
        let r = top(m);
        m &= !(1 << r);
        payload = (payload << 4) | r;
    }
    payload << (4 * (5 - lead - k))
}

#[inline(always)]
fn make(cat: u32, payload: u32) -> u32 {
    (cat << 20) | payload
}

/// Rango de la mejor mano de 5 cartas contenida en `cards` (5 a 7 cartas).
#[inline]
pub fn rank_mask(cards: u64) -> u32 {
    let mut suits = [0u16; 4];
    let mut counts = [0u8; 13];
    let mut m = cards;
    while m != 0 {
        let c = m.trailing_zeros();
        m &= m - 1;
        suits[(c & 3) as usize] |= 1 << (c >> 2);
        counts[(c >> 2) as usize] += 1;
    }
    // Con ≤ 7 cartas, si hay color es la única categoría ≥ color posible salvo póker/full,
    // que requieren ≥ 4 cartas de otros palos: con 5 del mismo palo solo quedan 2.
    for &sm in &suits {
        if sm.count_ones() >= 5 {
            return match straight_high(sm) {
                Some(h) => make(STRAIGHT_FLUSH, h << 16),
                None => make(FLUSH, kickers(sm, 5, 0, 0)),
            };
        }
    }
    let (mut m2, mut m3, mut m4, mut any) = (0u16, 0u16, 0u16, 0u16);
    for (r, &c) in counts.iter().enumerate() {
        let b = 1u16 << r;
        match c {
            0 => continue,
            1 => {}
            2 => m2 |= b,
            3 => m3 |= b,
            _ => m4 |= b,
        }
        any |= b;
    }
    if m4 != 0 {
        let q = top(m4);
        return make(QUADS, kickers(any & !(1 << q), 1, q, 1));
    }
    if m3 != 0 {
        let t = top(m3);
        let pairs = (m3 & !(1 << t)) | m2;
        if pairs != 0 {
            return make(FULL_HOUSE, ((t << 4) | top(pairs)) << 12);
        }
    }
    if let Some(h) = straight_high(any) {
        return make(STRAIGHT, h << 16);
    }
    if m3 != 0 {
        let t = top(m3);
        return make(TRIPS, kickers(any & !(1 << t), 2, t, 1));
    }
    if m2.count_ones() >= 2 {
        let p1 = top(m2);
        let p2 = top(m2 & !(1 << p1));
        return make(TWO_PAIR, kickers(any & !(1 << p1) & !(1 << p2), 1, (p1 << 4) | p2, 2));
    }
    if m2 != 0 {
        let p = top(m2);
        return make(PAIR, kickers(any & !(1 << p), 3, p, 1));
    }
    // Carta alta: payload > 0 siempre (5 rangos distintos, al menos uno ≥ 4).
    make(HIGH_CARD, kickers(any, 5, 0, 0))
}

#[inline(always)]
pub fn category(rank: u32) -> u32 {
    rank >> 20
}

/// Implementación del trait de la capa.
#[derive(Clone, Copy, Debug, Default)]
pub struct BitEvaluator;

impl HandEvaluator for BitEvaluator {
    #[inline]
    fn rank_mask(&self, cards: u64) -> u32 {
        rank_mask(cards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Card;

    fn mask(cards: &[(u8, u8)]) -> u64 {
        cards.iter().fold(0, |m, &(r, s)| m | Card::new(r, s).mask())
    }

    /// Frecuencias exactas de categorías sobre las 2 598 960 manos de 5 cartas.
    #[test]
    fn five_card_category_counts_are_exact() {
        let mut counts = [0u32; 9];
        for a in 0..52u64 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    for d in c + 1..52 {
                        for e in d + 1..52 {
                            let m = (1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e);
                            counts[category(rank_mask(m)) as usize] += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(counts, [1_302_540, 1_098_240, 123_552, 54_912, 10_200, 5_108, 3_744, 624, 40]);
    }

    #[test]
    fn tie_breaks() {
        // Rueda < escalera al 6.
        let wheel = rank_mask(mask(&[(12, 0), (0, 1), (1, 2), (2, 3), (3, 0), (9, 1), (10, 2)]));
        let six = rank_mask(mask(&[(0, 1), (1, 2), (2, 3), (3, 0), (4, 1), (9, 1), (10, 2)]));
        assert_eq!(category(wheel), STRAIGHT);
        assert!(six > wheel);
        // Doble pareja con tercera pareja: el kicker sale de la tercera pareja.
        let tp = rank_mask(mask(&[(12, 0), (12, 1), (11, 0), (11, 1), (10, 0), (10, 1), (2, 2)]));
        let tp2 = rank_mask(mask(&[(12, 0), (12, 1), (11, 0), (11, 1), (9, 0), (9, 1), (2, 2)]));
        assert_eq!(category(tp), TWO_PAIR);
        assert!(tp > tp2);
        // Dos tríos = full con el trío alto.
        let fh = rank_mask(mask(&[(5, 0), (5, 1), (5, 2), (3, 0), (3, 1), (3, 2), (0, 0)]));
        assert_eq!(category(fh), FULL_HOUSE);
        // Board jugado: empate exacto.
        let board = [(12, 0), (11, 0), (10, 0), (9, 0), (8, 0)];
        let a = rank_mask(mask(&board) | mask(&[(0, 1), (1, 2)]));
        let b = rank_mask(mask(&board) | mask(&[(2, 1), (3, 2)]));
        assert_eq!(a, b);
    }
}
