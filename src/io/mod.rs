//! Capa `io`: entrada/salida del solver.
//!
//! * [`hh`]: importación de historiales de mano de PokerStars.
//! * [`ranges`]: notación de rangos estándar (lectura y escritura).
//! * [`export`]: EV por clase, estrategia completa en JSON y rangos de flop para Pio.
//! * [`html`]: informe HTML autocontenido con rejillas 13×13.

pub mod export;
pub mod hh;
pub mod html;
pub mod ranges;
