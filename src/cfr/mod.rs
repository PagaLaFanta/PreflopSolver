//! Capa `cfr` — solver y contratos entre módulos.
//!
//! ## Flujo de datos de una iteración
//!
//! ```text
//!             ┌──────────── CFR+ full-width preflop (vectorizado ×1326) ─────────────┐
//!  raíz ──►  ActionNode ──► ... ──► TerminalNode(Fold/AllInRunout)                    │
//!             │  σ = QuantArena::normalized            u(h) = Σ_o P(o|h)·U(o)  ◄── math (tabla $EV)
//!             │                                                ▲                     │
//!             │                                                └── eval (matriz equity)
//!             └──► FlopLeaf ──► PostflopEstimator
//!                                  ├─ CheckDown: realización de equity 100 % (matriz)
//!                                  └─ PostflopEngine: MCCFR public-chance-sampling
//!                                       key = abstraction::recall  (bucket, SPR, …)
//!                                       acciones = abstraction::geometric
//!                                       σ = tree::InfosetTable (u64 CAS lock-free)
//! ```
//!
//! ## Normalización de valores
//! Todos los vectores de valores devueltos a una hoja preflop para el jugador `i` son
//! `v_i[h] = E[u_i | h, se llega aquí] · P(los rivales llegan | h)`, con
//! `P(rival llega | h) = Σ_{v ∩ h = ∅} r[v] / 1225`. En la raíz (alcance 1) coincide con
//! la utilidad esperada condicionada a la mano: las regrets quedan en unidades de
//! utilidad ($ o bb), comparables entre nodos.

pub mod charts;
pub mod checkdown;
pub mod lanes;
pub mod multiway;
pub mod pool;
pub mod postflop;
pub mod preflop;
pub mod report;

pub use checkdown::CheckDown;
pub use postflop::PostflopEngine;
pub use preflop::{CiState, Solver, SolverConfig};

use crate::eval::{HAND_STRIDE, NUM_COMBOS};
use crate::tree::PreflopTree;

/// Rango vectorizado: probabilidad de alcance por combo, con el relleno SIMD a 1328.
pub type RangeVec = [f32; HAND_STRIDE];

pub fn zero_range() -> Box<RangeVec> {
    Box::new([0.0; HAND_STRIDE])
}

pub fn full_range() -> Box<RangeVec> {
    let mut r = zero_range();
    r[..NUM_COMBOS].fill(1.0);
    r
}

/// Qué estrategia usar al recorrer: la actual (regret matching) o la media (salida).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyMode {
    Current,
    Average,
}

/// Estimador del valor postflop de una hoja de flop HU. Implementado por
/// [`CheckDown`] (sin betting) y [`PostflopEngine`] (MCCFR geométrico).
pub trait PostflopEstimator: Send + Sync {
    /// Valores del jugador en el bote (`hero_is_ip` indica su rol) contra `r_opp`.
    /// Si `update`, además aplica una iteración de MCCFR a los infosets postflop del héroe.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn hero_values(
        &self,
        tree: &PreflopTree,
        leaf: usize,
        hero_is_ip: bool,
        r_hero: &RangeVec,
        r_opp: &RangeVec,
        mode: StrategyMode,
        update: bool,
        seed: u64,
        out: &mut RangeVec,
        pool: pool::Pool<'_>,
    );

    /// Como [`Self::hero_values`] pero con las utilidades de varios asientos a la vez
    /// (`seats[0]` = el propio jugador, cuyos infosets se actualizan; el resto son
    /// observadores). Devuelve un vector por asiento con la misma normalización.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn values(
        &self,
        tree: &PreflopTree,
        leaf: usize,
        hero_is_ip: bool,
        r_hero: &RangeVec,
        r_opp: &RangeVec,
        mode: StrategyMode,
        update: bool,
        seed: u64,
        seats: &[usize],
        pool: pool::Pool<'_>,
    ) -> Vec<Box<RangeVec>>;

    /// Hoja de flop a 3 jugadores: valores del jugador en la posición `pos` (orden de
    /// actuación postflop) con las utilidades de `seats` (`seats[0]` = él mismo).
    /// `reaches`: rangos de los tres por posición.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn values3(
        &self,
        _tree: &PreflopTree,
        _leaf: usize,
        _pos: usize,
        _reaches: [&RangeVec; 3],
        _mode: StrategyMode,
        _update: bool,
        _seed: u64,
        _seats: &[usize],
        _pool: pool::Pool<'_>,
    ) -> Vec<Box<RangeVec>> {
        panic!("este estimador no resuelve flops a 3 jugadores")
    }

    /// Valores de **todos** los jugadores de una hoja (2 o 3, rangos por posición) de una
    /// vez: `out[p][0]` = jugador en la posición `p`; `out[0][1..]` = observadores
    /// `observers` desde la perspectiva de la posición 0. `update_mask`: bit `p` ⇒ se
    /// actualizan los infosets de la posición `p`. `folded`: alcances de los jugadores que
    /// se retiraron antes del flop (bunching: los boards se ponderan por la probabilidad
    /// de salir dado que esas manos se foldearon). Por defecto, una llamada por jugador y
    /// sin bunching.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn values_all(&self, tree: &PreflopTree, leaf: usize, reaches: &[&RangeVec], mode: StrategyMode, update_mask: u8, seed: u64, observers: &[usize], _folded: &[&RangeVec], pool: pool::Pool<'_>) -> Vec<Vec<Box<RangeVec>>> {
        let l = &tree.leaves[leaf];
        if reaches.len() == 2 {
            let mut s0 = vec![l.oop as usize];
            s0.extend(observers);
            vec![
                self.values(tree, leaf, false, reaches[0], reaches[1], mode, update_mask & 1 != 0, seed, &s0, pool),
                self.values(tree, leaf, true, reaches[1], reaches[0], mode, update_mask & 2 != 0, seed ^ 1, &[l.ip as usize], pool),
            ]
        } else {
            let m = l.multi.as_ref().expect("hoja multiway");
            let r3 = [reaches[0], reaches[1], reaches[2]];
            (0..3)
                .map(|p| {
                    let mut seats = vec![m.seats[p] as usize];
                    if p == 0 {
                        seats.extend(observers);
                    }
                    self.values3(tree, leaf, p, r3, mode, update_mask >> p & 1 != 0, seed + p as u64, &seats, pool)
                })
                .collect()
        }
    }

    /// Iteración preflop en curso (para ponderar las sumas de estrategia postflop por
    /// `t^p` como las del preflop). Por defecto, se ignora.
    fn set_iteration(&self, _t: u32) {}

    /// Estado aprendido (para guardar soluciones). Por defecto, ninguno.
    fn save_state(&self) -> Vec<u8> {
        Vec::new()
    }

    /// Restaura el estado de [`Self::save_state`].
    fn load_state(&self, _bytes: &[u8]) -> Result<(), String> {
        Ok(())
    }

    /// Utilidad esperada de **cada asiento** de la mesa (incluidos los que ya no están en
    /// la mano: con ICM su $EV depende de quién gane) normalizada por `1326·1225`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn observer_values(&self, tree: &PreflopTree, leaf: usize, r_oop: &RangeVec, r_ip: &RangeVec, mode: StrategyMode, seed: u64, out: &mut [f64], pool: pool::Pool<'_>);
}

