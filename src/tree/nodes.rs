//! # Nodos del árbol: layout de memoria
//!
//! Principio rector: **separar lo inmutable-caliente de lo mutable**.
//!
//! | struct         | tamaño | mutabilidad            | quién lo toca                         |
//! |----------------|--------|------------------------|----------------------------------------|
//! | `ActionNode`   | 64 B   | inmutable tras build   | todos los hilos, solo lectura         |
//! | `TerminalNode` | 64 B   | inmutable tras build   | todos los hilos, solo lectura         |
//! | `CFRNode`      | 16 B   | descriptor inmutable   | apunta a carriles mutables en arenas  |
//!
//! Topología y estado de aprendizaje viven en arrays distintos. Consecuencias:
//! 1. Las líneas de caché de topología están siempre en estado *Shared* (MESI) en todos
//!    los núcleos: nunca se invalidan porque nadie las escribe. Si los regrets estuvieran
//!    embebidos en el nodo, cada escritura de regret invalidaría la topología en los demás
//!    núcleos (false sharing sistemático).
//! 2. Un recorrido del árbol toca **una** línea por nodo de decisión: hijos, acciones,
//!    bote e índice de infoset están en los mismos 64 B.
//! 3. Los arrays de nodos se construyen en orden DFS preorden, así que un descenso típico
//!    (primer hijo) es un acceso secuencial que el prefetcher hardware detecta.
//!
//! Índices `u32` en lugar de punteros: la mitad de tamaño, relocalizables (el árbol se
//! puede `mmap`-ear desde disco tal cual), y sin aliasing que impida vectorizar.

pub use crate::eval::{HAND_STRIDE, LANE_BLOCK};
use std::mem::{align_of, size_of};

/// Jugadores en la mesa.
pub const MAX_PLAYERS: usize = 10;
/// Acciones máximas por nodo preflop (fold, call, 2-3 tamaños de raise, all-in…).
pub const MAX_ACTIONS: usize = 8;
pub const BLOCKS_PER_ROW: usize = HAND_STRIDE / LANE_BLOCK;

/// Referencia etiquetada a un nodo: 2 bits de tipo + 30 bits de índice (1.07e9 nodos por
/// tipo). Un hijo puede ser de cualquier tipo sin necesidad de un enum de 8 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(transparent)]
pub struct NodeRef(u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum NodeKind {
    Action = 0,
    Terminal = 1,
    /// Nodo de azar postflop (reparto de flop/turn/river bajo MCCFR chance sampling).
    Chance = 2,
    Null = 3,
}

impl NodeRef {
    const SHIFT: u32 = 30;
    const MASK: u32 = (1 << Self::SHIFT) - 1;
    pub const NULL: NodeRef = NodeRef(u32::MAX);

    #[inline(always)]
    pub const fn new(kind: NodeKind, index: u32) -> Self {
        assert!(index <= Self::MASK);
        NodeRef(((kind as u32) << Self::SHIFT) | index)
    }
    #[inline(always)]
    pub const fn kind(self) -> NodeKind {
        match self.0 >> Self::SHIFT {
            0 => NodeKind::Action,
            1 => NodeKind::Terminal,
            2 => NodeKind::Chance,
            _ => NodeKind::Null,
        }
    }
    #[inline(always)]
    pub const fn index(self) -> usize {
        (self.0 & Self::MASK) as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ActionKind {
    Fold = 0,
    Check = 1,
    Call = 2,
    /// Raise preflop discreto; el parámetro indexa la tabla de tamaños del perfil
    /// (p.ej. 0 → 2.0x, 1 → 2.2x, 2 → 2.5x, 3 → 3-bet 3x IP …).
    Raise = 3,
    AllIn = 4,
    /// Bet geométrico postflop; tamaño derivado de (bote, stack efectivo, apuestas
    /// restantes), ver `abstraction::geometric`. El parámetro no se usa.
    GeoBet = 5,
}

/// Acción empaquetada en un byte: `[kind:3][param:5]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(transparent)]
pub struct PackedAction(u8);

impl PackedAction {
    #[inline(always)]
    pub const fn new(kind: ActionKind, param: u8) -> Self {
        assert!(param < 32);
        PackedAction(((kind as u8) << 5) | param)
    }
    #[inline(always)]
    pub const fn kind(self) -> ActionKind {
        match self.0 >> 5 {
            0 => ActionKind::Fold,
            1 => ActionKind::Check,
            2 => ActionKind::Call,
            3 => ActionKind::Raise,
            4 => ActionKind::AllIn,
            _ => ActionKind::GeoBet,
        }
    }
    #[inline(always)]
    pub const fn param(self) -> u8 {
        self.0 & 31
    }
}

/// Nodo de decisión. Exactamente una línea de caché.
///
/// Orden de campos por tamaño descendente → cero padding interno (verificado abajo en
/// tiempo de compilación). Fichas en `u32` de centésimas de ciega grande: 42.9M bb de
/// rango, precisión 0.01 bb — más que suficiente y la mitad que `f64`.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct ActionNode {
    /// Hijos inline: el descenso no necesita una segunda línea para una tabla de hijos.
    pub children: [NodeRef; MAX_ACTIONS], // 0..32
    pub actions: [PackedAction; MAX_ACTIONS], // 32..40
    /// Índice en el array de `CFRNode` (preflop) o semilla parcial de `InfosetKey` (postflop).
    pub infoset: u32, // 40..44
    pub pot: u32,       // 44..48
    pub to_call: u32,   // 48..52
    pub eff_stack: u32, // 52..56
    pub player: u8,     // 56
    pub num_actions: u8,
    pub street: u8,
    /// bit0: postflop (infoset en tabla hash), bit1: IP, bit2: multiway.
    pub flags: u8,
    pub bets_made: u8, // 60
    pub pot_class: u8,
    pub _reserved: [u8; 2], // 62..64
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TerminalKind {
    /// Todos foldean menos uno: outcome único, no depende de las cartas.
    Fold = 0,
    /// All-in preflop con runout completo: `P(o|manos)` = equity exacta precomputada.
    AllInRunout = 1,
    /// Check-down sin betting (se usa como terminal de showdown explícito).
    Showdown = 2,
    /// Entrada al flop: la resuelve un `PostflopEstimator` (MCCFR geométrico o check-down).
    FlopLeaf = 3,
    /// Bote a 3 (all-in con side pots o flop multiway en check-down): 6 outcomes, uno por
    /// orden de fuerza de manos (`PERMS3`).
    Multiway = 4,
    /// Bote de 4 a 6 jugadores: `k!` outcomes (orden de fuerza en código de Lehmer), con
    /// los jugadores en `PreflopTree::pots[leaf]`.
    MultiwayN = 5,
}

pub const fn factorial(k: usize) -> usize {
    let mut f = 1;
    let mut i = 2;
    while i <= k {
        f *= i;
        i += 1;
    }
    f
}

/// Índice lexicográfico (código de Lehmer) de una permutación de `0..k`. Para k = 3
/// coincide con el orden de `PERMS3`.
pub fn perm_index(order: &[u8]) -> usize {
    let k = order.len();
    let mut idx = 0;
    for i in 0..k {
        let smaller = order[i + 1..].iter().filter(|&&x| x < order[i]).count();
        idx += smaller * factorial(k - 1 - i);
    }
    idx
}

/// Inversa de [`perm_index`].
pub fn nth_perm(k: usize, mut idx: usize) -> Vec<u8> {
    let mut pool: Vec<u8> = (0..k as u8).collect();
    let mut out = Vec::with_capacity(k);
    for i in (0..k).rev() {
        let f = factorial(i);
        out.push(pool.remove(idx / f));
        idx %= f;
    }
    out
}

/// Órdenes de fuerza (mejor primero) de los 3 jugadores de un bote multiway, en índices
/// de `TerminalNode::players`. El outcome `k` de un terminal `Multiway` es `PERMS3[k]`.
pub const PERMS3: [[u8; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/// Resultados de un bote a 3 con empates: el puesto de cada posición (0 = mejor mano),
/// con puestos repetidos si empatan. Los 6 primeros son los órdenes estrictos de
/// [`PERMS3`] (en el mismo orden); luego los 6 empates dobles y el triple.
pub const PATTERNS3: [[u8; 3]; 13] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [2, 0, 1],
    [1, 2, 0],
    [2, 1, 0],
    [0, 0, 1], // 0 = 1 > 2
    [0, 1, 0], // 0 = 2 > 1
    [1, 0, 0], // 1 = 2 > 0
    [0, 1, 1], // 0 > 1 = 2
    [1, 0, 1], // 1 > 0 = 2
    [1, 1, 0], // 2 > 0 = 1
    [0, 0, 0], // triple
];

/// Índice en [`PATTERNS3`] de un vector de puestos.
#[inline]
pub fn pattern3_index(ranks: [u8; 3]) -> usize {
    PATTERNS3.iter().position(|p| *p == ranks).expect("3-player pattern")
}

/// Nodo terminal: una línea de caché. **No contiene utilidades**: contiene el índice a la
/// tabla `U[outcome][seat]` en `f32` precomputada por el Agente Quant
/// (`math::terminal_dollar_ev`). Los $EV dependen de la transición de stacks, que el árbol
/// fija; las cartas solo ponderan outcomes.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct TerminalNode {
    /// Fichas invertidas por asiento en la mano (para cEV y para side pots).
    pub contributions: [u32; MAX_PLAYERS], // 0..40
    pub pot: u32,             // 40..44
    pub utility_offset: u32,  // 44..48: base en la tabla de utilidades
    pub alive_mask: u16,      // 48..50: jugadores que no han foldeado
    pub kind: TerminalKind,   // 50
    pub num_outcomes: u8,     // 51: HU all-in → 3 (gana A / gana B / split)
    pub num_seats: u8,        // 52: stride de la tabla de utilidades
    /// Fold: `[ganador, 255, 255]`. HU (all-in/flop): los dos asientos en el bote
    /// (outcome 0 = gana `players[0]`, 1 = gana `players[1]`, 2 = empate).
    /// Multiway: los tres asientos (outcome `k` = orden `PERMS3[k]`).
    pub players: [u8; 3],     // 53..56
    /// Índice en `PreflopTree::leaves` si `kind == FlopLeaf`.
    pub leaf: u32,            // 56..60
    pub _reserved: [u8; 4],   // 60..64
}

impl TerminalNode {
    /// Slice `U[o][seat]` de este terminal dentro de la tabla global.
    #[inline(always)]
    pub fn utilities<'a>(&self, table: &'a [f32]) -> &'a [f32] {
        let base = self.utility_offset as usize;
        &table[base..base + self.num_outcomes as usize * self.num_seats as usize]
    }
}

/// Descriptor de infoset preflop (vectorizado sobre los 1326 combos del jugador activo).
///
/// No contiene los datos: apunta a filas de dos [`super::QuantArena`] con **idéntico
/// layout** (regrets y sumas de estrategia), para que un solo `row` direccione ambas:
///
/// ```text
/// regret arena  : ... | row+0: [u16; 1328] | row+1: [u16; 1328] | ... (num_actions filas)
/// strategy arena: ... | row+0: [u16; 1328] | row+1: [u16; 1328] | ...
/// exponentes    : ... | exp_base .. exp_base+83  (u8, uno por bloque de 16 combos)
/// ```
///
/// Las dos arenas están separadas a propósito: el paso de regret matching (cada
/// iteración, cada nodo) solo lee regrets, así el working set caliente es la mitad.
///
/// Presupuesto de memoria por nodo preflop de 3 acciones:
/// `2 arenas × 3 × 1328 × 2 B + 2 × 83 B ≈ 16 KB` (vs 32 KB con `f32`, 64 KB con `f64`).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct CFRNode {
    pub row: u32,
    pub exp_base: u32,
    pub num_actions: u8,
    pub player: u8,
    pub _reserved: [u8; 6],
}

// ---- Verificación del layout en tiempo de compilación -----------------------------------
const _: () = assert!(size_of::<ActionNode>() == 64 && align_of::<ActionNode>() == 64);
const _: () = assert!(size_of::<TerminalNode>() == 64 && align_of::<TerminalNode>() == 64);
const _: () = assert!(size_of::<CFRNode>() == 16);
const _: () = assert!(HAND_STRIDE == 1328 && BLOCKS_PER_ROW == 83);
const _: () = assert!(size_of::<NodeRef>() == 4 && size_of::<PackedAction>() == 1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_ref_roundtrip() {
        for kind in [NodeKind::Action, NodeKind::Terminal, NodeKind::Chance] {
            let r = NodeRef::new(kind, 123_456_789);
            assert_eq!(r.kind(), kind);
            assert_eq!(r.index(), 123_456_789);
        }
        assert_eq!(NodeRef::NULL.kind(), NodeKind::Null);
    }

    #[test]
    fn lehmer_roundtrip() {
        for k in 1..=6 {
            for i in 0..factorial(k) {
                assert_eq!(perm_index(&nth_perm(k, i)), i);
            }
        }
        for (i, p) in PERMS3.iter().enumerate() {
            assert_eq!(perm_index(p), i);
        }
    }

    #[test]
    fn packed_action_roundtrip() {
        let a = PackedAction::new(ActionKind::Raise, 17);
        assert_eq!((a.kind(), a.param()), (ActionKind::Raise, 17));
    }
}
