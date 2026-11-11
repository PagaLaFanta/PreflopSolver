//! Capa `math` — dominio del **Agente Quant**.
//!
//! * [`icm`]: Malmuth-Harville exacto por programación dinámica sobre subconjuntos,
//!   truncado por profundidad de premios, + muestreo Plackett-Luce para campos grandes.
//! * [`terminal`]: fusión ICM + PKO en la utilidad $EV de un nodo terminal.
//! * [`fgs`]: Future Game Simulation (ICM tras simular las próximas manos en push/fold).
//! * [`rng`]: PRNG sin dependencias (SplitMix64) para Monte Carlo y redondeo estocástico.

pub mod fgs;
pub mod field_icm;
pub mod icm;
pub mod rng;
pub mod terminal;

pub use icm::{icm_equity, icm_equity_crn, icm_equity_monte_carlo, icm_equity_precise, icm_equity_sparse, IcmMemo, MAX_EXACT_PLAYERS};
pub use terminal::{bubble_factors, bubble_factors_detail, terminal_dollar_ev, terminal_dollar_ev_with, Elimination, MatchupDetail, PkoParams, PrizeModel, TerminalInput};
