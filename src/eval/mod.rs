//! Capa `eval`: representación de cartas, indexación canónica de los 1326 combos
//! y los contratos (traits) que deben cumplir evaluadores de manos y oráculos de equity.
//!
//! * [`evaluator`]: evaluador de 5-7 cartas por máscaras de bits (sin tablas).
//! * [`equity`]: matriz de equity preflop 1326×1326 (win/tie) por Monte Carlo
//!   simetrizado por palos, con caché en disco.
//! * [`showdown`]: valores de showdown rango-contra-rango en O(n) con bloqueadores.

pub mod board_bunch;
pub mod equity;
pub mod evaluator;
pub mod showdown;

/// Carta codificada como `rank * 4 + suit`, `rank ∈ [0,12]` (2..A), `suit ∈ [0,3]`.
/// Un `u8` permite representar el mazo completo en un `u64` (bitmask de cartas muertas).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Card(pub u8);

impl Card {
    #[inline(always)]
    pub const fn new(rank: u8, suit: u8) -> Self {
        Card(rank * 4 + suit)
    }
    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 2
    }
    #[inline(always)]
    pub const fn suit(self) -> u8 {
        self.0 & 3
    }
    #[inline(always)]
    pub const fn mask(self) -> u64 {
        1u64 << self.0
    }
}

/// Número de combos de dos cartas: C(52,2).
pub const NUM_COMBOS: usize = 1326;

/// `COMBO_CARDS[idx] = [lo, hi]`, generada en tiempo de compilación.
pub const COMBO_CARDS: [[u8; 2]; NUM_COMBOS] = build_combo_cards();
/// Máscara de 52 bits de cada combo (para test de bloqueo con un AND).
pub const COMBO_MASKS: [u64; NUM_COMBOS] = build_combo_masks();

const fn build_combo_cards() -> [[u8; 2]; NUM_COMBOS] {
    let mut t = [[0u8; 2]; NUM_COMBOS];
    let mut hi = 1;
    while hi < 52 {
        let mut lo = 0;
        while lo < hi {
            t[hi * (hi - 1) / 2 + lo] = [lo as u8, hi as u8];
            lo += 1;
        }
        hi += 1;
    }
    t
}

const fn build_combo_masks() -> [u64; NUM_COMBOS] {
    let cards = build_combo_cards();
    let mut m = [0u64; NUM_COMBOS];
    let mut i = 0;
    while i < NUM_COMBOS {
        m[i] = (1u64 << cards[i][0]) | (1u64 << cards[i][1]);
        i += 1;
    }
    m
}

/// Ancho de bloque SIMD en carriles `u16`: 16 × 16 bit = 256 bit = un registro AVX2
/// (o medio AVX-512, y `f32 ×16` tras decodificar = un registro AVX-512).
pub const LANE_BLOCK: usize = 16;
/// Stride de fila de todo vector de rango: 1326 combos redondeado al múltiplo de 16 →
/// 1328 = 83 bloques. Los 2 carriles de relleno (0.15 %) garantizan alineación y que el
/// bucle interno no tenga epílogo escalar.
pub const HAND_STRIDE: usize = NUM_COMBOS.div_ceil(LANE_BLOCK) * LANE_BLOCK;

/// Combos compatibles con uno dado: C(50,2).
pub const COMPATIBLE_COMBOS: f32 = 1225.0;

/// Índice de combo en `[0, 1326)`. Se usa como índice de carril (lane) en todos los
/// vectores de rango: `regrets[action * HAND_STRIDE + combo]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Combo(pub u16);

impl Combo {
    /// Sistema numérico combinatorio: para `lo < hi`, `idx = C(hi,2) + lo`.
    /// Es una biyección ordenada (colex) entre pares y `[0,1326)`, sin tablas.
    #[inline(always)]
    pub const fn from_cards(a: Card, b: Card) -> Self {
        let (lo, hi) = if a.0 < b.0 { (a.0 as u16, b.0 as u16) } else { (b.0 as u16, a.0 as u16) };
        Combo(hi * (hi - 1) / 2 + lo)
    }

    /// Inversa de `from_cards` por tabla.
    #[inline(always)]
    pub fn cards(self) -> (Card, Card) {
        let [lo, hi] = COMBO_CARDS[self.0 as usize];
        (Card(lo), Card(hi))
    }

    #[inline(always)]
    pub fn mask(self) -> u64 {
        COMBO_MASKS[self.0 as usize]
    }
}

/// Evaluador de manos de 5-7 cartas. Solo se exige que el orden sea total y
/// consistente (`a > b` ⇔ `a` gana) y que toda mano válida tenga rango > 0.
pub trait HandEvaluator: Send + Sync {
    fn rank_mask(&self, cards: u64) -> u32;
}

/// Oráculo de equity de showdown, usado en:
/// * nodos terminales preflop all-in (equity exacta combo-vs-combo, precomputada 1326×1326),
/// * rollouts de bucketing (EHS / histogramas de equity futura).
pub trait EquityOracle: Send + Sync {
    /// Probabilidades `(win, tie)` de `hero` contra `villain` corriendo el board completo.
    /// Debe devolver `(0,0)` si los combos comparten carta (bloqueo).
    fn heads_up_allin(&self, hero: Combo, villain: Combo) -> (f32, f32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_index_is_a_bijection() {
        let mut seen = vec![false; NUM_COMBOS];
        for hi in 0u8..52 {
            for lo in 0u8..hi {
                let c = Combo::from_cards(Card(lo), Card(hi));
                assert!((c.0 as usize) < NUM_COMBOS);
                assert!(!seen[c.0 as usize]);
                seen[c.0 as usize] = true;
                assert_eq!(c.cards(), (Card(lo), Card(hi)));
            }
        }
        assert!(seen.iter().all(|&s| s));
    }
}
