//! Capa `tree` — dominio del **Agente C/R**.
//!
//! * [`nodes`]: `ActionNode`, `TerminalNode`, `CFRNode` con layout fijado a línea de caché.
//! * [`quant`]: arenas de carriles `u16` cuantizados con exponente compartido por bloque
//!   SIMD (block floating point) para la parte preflop vectorizada sobre 1326 combos.
//! * [`infoset`]: infosets postflop empaquetados en un único `AtomicU64` + tabla hash
//!   lock-free (open addressing, CAS de inserción) para MCCFR multihilo.
//! * [`builder`]: constructor del árbol preflop + tabla de utilidades $EV.
//! * [`postflop`]: subárboles geométricos de las hojas de flop.
//! * [`buf`]: buffers alineados a 64 B y asignados con `alloc_zeroed` (páginas cero
//!   perezosas del SO: una arena de 100 GB no toca RAM hasta que se escribe).

pub mod buf;
pub mod builder;
pub mod infoset;
pub mod nodes;
pub mod postflop;
pub mod quant;

pub use infoset::{InfosetTable, PostflopInfoset};
pub use nodes::{ActionKind, ActionNode, CFRNode, NodeKind, NodeRef, PackedAction, TerminalKind, TerminalNode};
pub use quant::QuantArena;
pub use builder::{build_tree, LevelSizes, Payoff, PreflopConfig, PreflopTree, RaiseSize, SizeBase};
pub use postflop::{FlopLeaf, MultiLeaf, PfNode, PfNode3};
