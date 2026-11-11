//! Capa `abstraction` — dominio del **Agente Nash**.
//!
//! * [`geometric`]: abstracción de acción postflop (SPR → bet geométrico único).
//! * [`bucketing`]: abstracción de información (EHS, histogramas de equity, EMD, k-means).
//! * [`board_pool`]: runouts muestreados con buckets precalculados por calle.
//! * [`recall`]: claves de infoset con memoria imperfecta y transición entre calles.

pub mod board_pool;
pub mod bucketing;
pub mod geometric;
pub mod recall;

pub use bucketing::{emd_1d, kmeans_emd, Bucketer, EquityHistogram, StrengthOracle};
pub use geometric::{geometric_fraction, postflop_actions, GeometricConfig, PostflopActions, PostflopSpot, Street};
pub use recall::{spr_bin, transition_to_next_street, InfosetKey, KeyFields, PotClass};
pub use board_pool::{BoardPool, PoolConfig, SampledBoard};
