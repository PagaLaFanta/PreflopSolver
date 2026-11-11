//! # Imperfect Recall postflop
//!
//! Un infoset postflop no es la historia completa; es una **clave de 31 bits** que solo
//! retiene lo que el árbol geométrico necesita para ser un juego bien definido:
//!
//! ```text
//! bit  30..28  27    26..25  24      23..22     21..16    15..0
//!     [pot_cls][ip ][street][facing][bets_made][spr_bin ][ bucket ]
//! ```
//!
//! ## Qué se olvida al pasar de Flop a Turn (y de Turn a River)
//! * El **bucket del flop**: el turn se clasifica de nuevo desde cero con el histograma
//!   turn→river. Dos manos que estaban en buckets de flop distintos pero caen en el mismo
//!   bucket de turn comparten infoset (esa es la fusión).
//! * La **secuencia exacta** de acciones del flop (check-check vs bet-call). Solo
//!   sobreviven sus efectos: `bets_made` (define cuántas apuestas quedan → estructura del
//!   subárbol) y el nuevo SPR (define el sizing).
//! * Las **cartas exactas** del board.
//!
//! ## Qué NO se puede olvidar (y por qué)
//! Imperfect recall es legal en CFR solo si todos los nodos fusionados en un infoset
//! tienen **el mismo conjunto de acciones** (si no, σ(I) no está definida). En el árbol
//! geométrico el conjunto de acciones es función de `(street, facing, bets_made)` +
//! `max_bets`; por eso esos campos están en la clave. El *tamaño* exacto del bet sí puede
//! diferir entre nodos fusionados (SPR agrupado en bins de ¼ de octava): cada nodo usa su
//! propio tamaño exacto para el payoff, y el infoset comparte la frecuencia.
//!
//! `pot_class` + `ip` son el único "recuerdo" preflop: sin ellos, el turn de un bote
//! 4-beteado y el de un bote limpeado con igual SPR compartirían estrategia pese a tener
//! rangos opuestos. Es el compromiso clásico: rompemos las garantías de Nash del juego
//! completo (con imperfect recall CFR ya no garantiza convergencia a equilibrio del juego
//! original), a cambio de que las raíces preflop — lo que realmente se exporta — vean una
//! realización de equity postflop realista y barata.

use super::geometric::Street;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum PotClass {
    Limped = 0,
    SingleRaised = 1,
    ThreeBet = 2,
    FourBetPlus = 3,
    Multiway = 4,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct InfosetKey(pub u64);

const BUCKET_BITS: u32 = 16;
const SPR_SHIFT: u32 = 16;
const BETS_SHIFT: u32 = 22;
const FACING_SHIFT: u32 = 24;
const STREET_SHIFT: u32 = 25;
const IP_SHIFT: u32 = 27;
const POTCLS_SHIFT: u32 = 28;

/// SPR → bin logarítmico de ¼ de octava: `round(4·log2(SPR)) + 24`, saturado a [0, 63].
/// Cubre SPR de 1/64 a ~860; la resolución relativa constante (≈19%) es la adecuada
/// porque la `f` geométrica depende de `ln(1+2·SPR)`.
#[inline]
pub fn spr_bin(pot: f64, eff_stack: f64) -> u8 {
    let spr = (eff_stack / pot).max(1e-9);
    ((4.0 * spr.log2()).round() + 24.0).clamp(0.0, 63.0) as u8
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyFields {
    pub bucket: u16,
    pub spr_bin: u8,
    pub bets_made: u8,
    pub facing: bool,
    pub street: Street,
    pub in_position: bool,
    pub pot_class: PotClass,
}

impl InfosetKey {
    #[inline]
    pub fn pack(f: KeyFields) -> Self {
        debug_assert!(f.spr_bin < 64 && f.bets_made < 4);
        InfosetKey(
            f.bucket as u64
                | (f.spr_bin as u64) << SPR_SHIFT
                | (f.bets_made as u64) << BETS_SHIFT
                | (f.facing as u64) << FACING_SHIFT
                | (f.street as u64) << STREET_SHIFT
                | (f.in_position as u64) << IP_SHIFT
                | (f.pot_class as u64) << POTCLS_SHIFT,
        )
    }

    #[inline]
    pub fn unpack(self) -> KeyFields {
        let k = self.0;
        KeyFields {
            bucket: (k & ((1 << BUCKET_BITS) - 1)) as u16,
            spr_bin: ((k >> SPR_SHIFT) & 63) as u8,
            bets_made: ((k >> BETS_SHIFT) & 3) as u8,
            facing: (k >> FACING_SHIFT) & 1 == 1,
            street: match (k >> STREET_SHIFT) & 3 {
                1 => Street::Flop,
                2 => Street::Turn,
                _ => Street::River,
            },
            in_position: (k >> IP_SHIFT) & 1 == 1,
            pot_class: match (k >> POTCLS_SHIFT) & 7 {
                0 => PotClass::Limped,
                1 => PotClass::SingleRaised,
                2 => PotClass::ThreeBet,
                3 => PotClass::FourBetPlus,
                _ => PotClass::Multiway,
            },
        }
    }

    /// Mismo infoset, otra decisión dentro de la calle (p.ej. IP tras check, o ahora
    /// enfrentando bet). No cambia bucket ni SPR (el SPR solo se relee al cambiar de calle).
    #[inline]
    pub fn with_decision(self, in_position: bool, facing: bool, bets_made: u8) -> Self {
        let mut f = self.unpack();
        f.in_position = in_position;
        f.facing = facing;
        f.bets_made = bets_made;
        InfosetKey::pack(f)
    }
}

/// Transición de calle con **olvido**: construye la clave de la primera decisión de la
/// calle siguiente descartando el bucket y la secuencia de la calle anterior.
///
/// * `new_bucket`: bucket de la mano en la nueva calle (lo calcula el `Bucketer` con el
///   board ya extendido por el nodo de azar muestreado por MCCFR).
/// * `pot`, `eff_stack`: **exactos** tras cerrar la calle; el SPR se re-bin-ea aquí.
/// * El primero en actuar postflop es OOP ⇒ `in_position = false`, `facing = false`.
///
/// Devuelve `None` tras el river (la siguiente "calle" es el showdown terminal).
#[inline]
pub fn transition_to_next_street(prev: InfosetKey, new_bucket: u16, pot: f64, eff_stack: f64) -> Option<InfosetKey> {
    let f = prev.unpack();
    let street = f.street.next()?;
    Some(InfosetKey::pack(KeyFields {
        bucket: new_bucket,              // ← olvido del bucket anterior
        spr_bin: spr_bin(pot, eff_stack), // ← único resumen del historial de apuestas + bets_made
        bets_made: f.bets_made,
        facing: false,
        street,
        in_position: false,
        pot_class: f.pot_class,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flop_key(bucket: u16) -> InfosetKey {
        InfosetKey::pack(KeyFields {
            bucket,
            spr_bin: spr_bin(10.0, 40.0),
            bets_made: 0,
            facing: false,
            street: Street::Flop,
            in_position: false,
            pot_class: PotClass::SingleRaised,
        })
    }

    #[test]
    fn pack_roundtrip() {
        let k = flop_key(4095).with_decision(true, true, 1);
        let f = k.unpack();
        assert_eq!((f.bucket, f.in_position, f.facing, f.bets_made), (4095, true, true, 1));
        assert!(k.0 < 1 << 31);
    }

    #[test]
    fn distinct_flop_histories_merge_on_turn() {
        // Dos manos en buckets de flop distintos, mismo pot/stack al llegar al turn y
        // mismo bucket de turn ⇒ mismo infoset.
        let a = transition_to_next_street(flop_key(17), 300, 30.0, 30.0).unwrap();
        let b = transition_to_next_street(flop_key(912), 300, 30.0, 30.0).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.unpack().street, Street::Turn);
        // Pero no se fusiona con otro bucket de turn.
        assert_ne!(a, transition_to_next_street(flop_key(17), 301, 30.0, 30.0).unwrap());
    }

    #[test]
    fn river_has_no_next_street() {
        let river = InfosetKey::pack(KeyFields { street: Street::River, ..flop_key(1).unpack() });
        assert!(transition_to_next_street(river, 0, 1.0, 1.0).is_none());
    }
}
