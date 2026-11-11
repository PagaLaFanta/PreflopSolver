//! # preflop_engine
//!
//! Solver preflop de torneos con heurísticas postflop abstraídas.
//!
//! Capas (de abajo arriba; ninguna capa importa de una capa superior):
//!
//! ```text
//!  eval         cartas, índices de combos (1326), trait de evaluador/equity
//!  math         ICM Malmuth-Harville + PKO → $EV de nodos terminales      [Agente Quant]
//!  abstraction  geometric sizing, bucketing EHS/EMD, imperfect recall     [Agente Nash]
//!  tree         layout de nodos, arenas cuantizadas, tablas lock-free     [Agente C/R]
//!  cfr          CFR+ preflop, MCCFR postflop, multiway, FGS charts        [Agente Nash + C/R]
//!  io           historiales PokerStars, notación de rangos, informe HTML
//! ```
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]
// Los bucles indexados sobre carriles de 1328 son deliberados (patrón que LLVM vectoriza).
#![allow(clippy::needless_range_loop)]
// `chunks_exact(16)` en los productos de equity es deliberado: es el patrón que LLVM ya
// vectoriza, y `as_chunks` exige un toolchain muy reciente. Las funciones recursivas del
// recorrido llevan todo su estado por parámetros a propósito (sin structs intermedios).
#![allow(clippy::chunks_exact_to_as_chunks, clippy::too_many_arguments)]

pub mod abstraction;
pub mod cfr;
pub mod eval;
pub mod io;
pub mod math;
pub mod tree;

/// El recorrido CFR clona un `Box<RangeVec>` (~5.3 KB) por jugador en cada nodo/acción/
/// iteración (`walk_action`/`eval_all`, ver `cfr::preflop`): con rayon repartiendo bien el
/// trabajo entre hilos, esas asignaciones pasan a competir de verdad por el allocator
/// global. mimalloc está pensado justo para ráfagas de asignaciones pequeñas/medianas
/// concurrentes como esta.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
