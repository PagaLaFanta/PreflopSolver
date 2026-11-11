//! # Abstracción de acción postflop: Geometric Sizing
//!
//! ## Derivación
//! Bote `P`, stack efectivo `S`, `n` apuestas restantes, cada una una fracción fija `f`
//! del bote y siempre pagada. Tras la apuesta `k` el bote es `P_k = P(1+2f)^k`, y lo
//! invertido por cada jugador es
//!
//! ```text
//! Σ_{k<n} f·P_k = f·P · ((1+2f)^n − 1) / (2f) = P · ((1+2f)^n − 1) / 2
//! ```
//!
//! Imponiendo que eso sea `S` (llegar exactamente all-in en la última apuesta):
//!
//! ```text
//!         (1 + 2·SPR)^(1/n) − 1
//!   f  =  ─────────────────────        con SPR = S / P
//!                   2
//! ```
//!
//! Propiedades que el solver explota:
//! * `f` depende solo de (SPR, n): el tamaño **no es una elección** → el árbol postflop
//!   tiene factor de ramificación 2 y su forma queda fijada por el SPR al llegar al flop.
//! * Tras un bet-call geométrico el SPR evoluciona de forma cerrada:
//!   `SPR' = (SPR − f) / (1 + 2f)`, y con `n−1` apuestas restantes la fórmula devuelve
//!   **la misma** `f` (auto-consistencia), así que un check-check que "gasta" una calle
//!   reajusta el tamaño de forma natural en la siguiente.
//! * El sizing geométrico es el que maximiza la fold equity acumulada de un rango
//!   polarizado que quiere llegar all-in (resultado clásico de juegos [0,1] multi-calle),
//!   así que es la mejor apuesta única posible si solo se permite una.

/// Calle postflop. `streets_left` incluye la calle actual.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Street {
    Flop = 1,
    Turn = 2,
    River = 3,
}

impl Street {
    #[inline(always)]
    pub const fn streets_left(self) -> u32 {
        4 - self as u32
    }
    #[inline(always)]
    pub const fn next(self) -> Option<Street> {
        match self {
            Street::Flop => Some(Street::Turn),
            Street::Turn => Some(Street::River),
            Street::River => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeometricConfig {
    /// Máximo de apuestas en toda la mano postflop (2 o 3 según el spec).
    pub max_bets: u8,
    /// Si la `f` resultante es menor, se reduce `n` (menos apuestas, más grandes).
    /// Evita "bets" de 5% de bote con SPR bajo, irrelevantes estratégicamente.
    pub min_fraction: f64,
    /// Tope de overbet. Con SPR enorme la `f` geométrica sería absurda (p.ej. 3× bote);
    /// se capa y la línea deja de llegar all-in (se marca en `GeoBet::reaches_all_in`).
    pub max_fraction: f64,
    /// Si la apuesta ≥ `jam_threshold · S`, se convierte en all-in (evita dejar
    /// "stack residual" de 1-2 bb que generaría un nodo extra degenerado).
    pub jam_threshold: f64,
    /// Raise frente a una apuesta: `Some(m)` ⇒ subir a `m ×` el tamaño de la apuesta
    /// (o all-in si no queda margen). Uno por calle, sin re-raise. `None` ⇒ sin raises.
    pub raise_mult: Option<f64>,
    /// Raises también en botes a 3 (el árbol pasa de ~200 a ~2 000 nodos por hoja).
    pub raise_multiway: bool,
    /// Máximo de apuestas en botes a 3 (el árbol multiway crece mucho más rápido).
    pub max_bets_multiway: u8,
    /// Tamaño geométrico %: si es `Some(g)`, el número de apuestas es el menor `n` (hasta
    /// `max_bets` y una por calle) cuya fracción geométrica no supera `g`; si ninguna lo
    /// consigue, se usan todas las permitidas.
    pub geo_size: Option<f64>,
    /// Raises con tamaño geométrico (el que reparte el stack restante en las apuestas que
    /// quedan) en vez de `raise_mult ×` la apuesta. `raise_mult` sigue activando los raises.
    pub raise_geometric: bool,
    /// Re-raises: se puede subir un raise mientras queden apuestas (`max_bets` cuenta
    /// apuestas y raises de toda la mano, como el check-down forzado).
    pub reraise: bool,
    /// Con SPR ≤ `allin_spr`, el all-in es una opción más además de la apuesta geométrica.
    /// 0 = no.
    pub allin_spr: f64,
    /// El que no tiene la iniciativa puede apostar primero en la calle siguiente (donk).
    /// Por defecto no: fuera de posición se pasa al que subió en la calle anterior.
    pub donk: bool,
    /// Sin `donk`: aun así puede apostar primero quien apostó o subió en alguna calle
    /// anterior, preflop incluido (donk "limitado", el comportamiento por defecto).
    pub donk_limited: bool,
    /// Tamaños de apuesta en el flop (fracción del bote). Si hay alguno, **sustituyen**
    /// al geométrico en las apuestas del flop.
    pub flop_bets: Vec<f64>,
    /// Tamaños de la c-bet del flop (solo quien subió el último preflop). Si los hay,
    /// mandan sobre `flop_bets` para él.
    pub flop_cbets: Vec<f64>,
    /// Tamaños del donk en el flop.
    pub flop_donks: Vec<f64>,
    /// Tamaños de raise en el flop (fracción del bote tras pagar).
    /// Si los hay, sustituyen al raise geométrico en el flop.
    pub flop_raises: Vec<f64>,
    /// Tamaño geométrico % de los botes a 3; `None` ⇒
    /// el de heads-up.
    pub geo_size_multiway: Option<f64>,
    /// Última calle con apuestas según los jugadores en la mano al empezar la calle
    /// (índice = jugadores − 2; 1 = flop, 2 = turn, 3 = river, 0 = check-down desde el
    /// flop); 9 valores para 2..10 jugadores. Por defecto: a 2 y 3 hasta el river, a 4
    /// hasta el turn, a 5 solo el flop.
    pub checkdown_after: Vec<u8>,
}

/// Check-down forzado por defecto, por jugadores en la mano (2..10).
pub const DEFAULT_CHECKDOWN_AFTER: [u8; 9] = [3, 3, 2, 1, 0, 0, 0, 0, 0];

impl Default for GeometricConfig {
    fn default() -> Self {
        GeometricConfig { max_bets: 3, min_fraction: 0.25, max_fraction: 2.0, jam_threshold: 0.85, raise_mult: None, raise_multiway: false, max_bets_multiway: 2, geo_size: None, raise_geometric: false, reraise: false, allin_spr: 0.0, donk: true, donk_limited: false, flop_bets: Vec::new(), flop_cbets: Vec::new(), flop_donks: Vec::new(), flop_raises: Vec::new(), geo_size_multiway: None, checkdown_after: DEFAULT_CHECKDOWN_AFTER.to_vec() }
    }
}

impl GeometricConfig {
    /// Última calle con apuestas (1-3; 0 = ninguna) con `players` en la mano.
    /// Tamaños del flop sin apuesta previa: c-bet (quien
    /// subió el último preflop), si no donk, si no los generales; vacío ⇒ geométrico.
    pub fn flop_sizes(&self, cbettor: bool, donk: bool) -> &[f64] {
        if cbettor && !self.flop_cbets.is_empty() {
            &self.flop_cbets
        } else if donk && !self.flop_donks.is_empty() {
            &self.flop_donks
        } else {
            &self.flop_bets
        }
    }

    /// El all-in se añade como opción cuando el SPR **de la decisión** (stack efectivo tras
    /// pagar entre el bote tras pagar) no pasa de `allin_spr`.
    pub fn low_spr(&self, eff_after_call: f64, pot_after_call: f64) -> bool {
        self.allin_spr > 0.0 && eff_after_call <= self.allin_spr * pot_after_call + 1e-9
    }

    pub fn last_street(&self, players: usize) -> u8 {
        self.checkdown_after.get(players.saturating_sub(2)).copied().unwrap_or(0)
    }

    /// Postflop por defecto: tamaño geométrico con Geo 66 % (el menor
    /// número de apuestas, entre calles y raises, cuyo tamaño no supera el 66 % del bote),
    /// all-in como opción con SPR ≤ 5, raises geométricos y re-raises sin límite, apuestas
    /// hasta el river heads-up (check-down forzado por jugadores), y sin donk.
    pub fn classic() -> Self {
        // Sin tope de apuestas por mano;
        // el número de apuestas lo fija el Geo a partir del SPR.
        // Las apuestas grandes no se convierten en all-in (el all-in se añade con SPR bajo):
        // `jam_threshold` 1.
        GeometricConfig { max_bets: 12, max_bets_multiway: 12, raise_multiway: true, geo_size: Some(0.66), raise_mult: Some(3.0), raise_geometric: true, reraise: true, allin_spr: 5.0, donk: false, donk_limited: true, jam_threshold: 1.0, ..Self::default() }
    }

    /// Raise frente a una apuesta `facing` con el bote `pot` (que ya la incluye), `behind`
    /// fichas detrás del que sube y `bets_left` apuestas por hacer: lo que pone en total
    /// (call incluido). `None` si no hay raise posible.
    pub fn raise_total(&self, facing: f64, pot: f64, behind: f64, bets_left: u32) -> Option<f64> {
        self.raise_mult?;
        if behind <= facing + 1e-9 || bets_left == 0 {
            return None;
        }
        if !self.raise_geometric {
            return self.raise_amount(facing, behind);
        }
        // Tras pagar, el resto del stack se reparte geométricamente en las que quedan.
        let pot2 = pot + facing;
        let rest = behind - facing;
        let spr = rest / pot2;
        // Igual que las apuestas: con Geo, el número de apuestas cuya fracción está más cerca
        // de g.
        let mut n = bets_left;
        if let Some(g) = self.geo_size {
            n = closest_count(spr, bets_left, g);
        }
        let f = geometric_fraction(spr, n).min(self.max_fraction);
        let add = facing + f * pot2;
        Some(if add - facing >= self.jam_threshold * rest { behind } else { add })
    }

    /// Cantidad total que pone el que sube (sobre lo que ya tenía en la calle) frente a
    /// una apuesta `bet`, con `behind` fichas detrás. `None` si no hay raise posible.
    pub fn raise_amount(&self, bet: f64, behind: f64) -> Option<f64> {
        let m = self.raise_mult?;
        if behind <= bet + 1e-9 {
            return None; // pagar ya es all-in
        }
        let add = m * bet;
        Some(if add >= self.jam_threshold * behind { behind } else { add })
    }
}

/// Fracción geométrica pura. `bets ≥ 1`, `pot > 0`.
#[inline]
pub fn geometric_fraction(spr: f64, bets: u32) -> f64 {
    debug_assert!(bets >= 1 && spr >= 0.0);
    if bets == 1 {
        return spr; // una sola apuesta: jam
    }
    ((1.0 + 2.0 * spr).powf(1.0 / bets as f64) - 1.0) * 0.5
}

/// Estado postflop mínimo que determina el conjunto de acciones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostflopSpot {
    pub street: Street,
    pub pot: f64,
    pub eff_stack: f64,
    /// Apuestas (pagadas) ya realizadas en calles anteriores + esta.
    pub bets_made: u8,
    /// Apuesta pendiente de responder en esta calle (0 = nadie ha apostado).
    pub facing: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoBet {
    pub amount: f64,
    pub fraction: f64,
    pub is_all_in: bool,
    pub reaches_all_in: bool,
}

/// Conjunto de acciones postflop: siempre exactamente dos (por eso el infoset cabe en un
/// `u64`, ver `tree::infoset`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PostflopActions {
    /// Primero en actuar (o tras check): Check | Bet geométrico.
    CheckOrBet(GeoBet),
    /// Sin apuestas restantes o sin stack: solo Check (nodo trivial, se colapsa en el árbol).
    CheckOnly,
    /// Enfrentando apuesta: Fold | Call. **No hay raises** en la abstracción.
    FoldOrCall { call: f64 },
}

/// Núcleo de la abstracción de acción postflop: lee el SPR y fuerza el sizing.
pub fn postflop_actions(spot: &PostflopSpot, cfg: &GeometricConfig) -> PostflopActions {
    if spot.facing > 0.0 {
        return PostflopActions::FoldOrCall { call: spot.facing.min(spot.eff_stack) };
    }
    // Sin re-raises, una apuesta como máximo por calle ⇒ n ≤ calles restantes. Con ellos,
    // las apuestas del plan geométrico pueden caer en la misma calle.
    let budget = (cfg.max_bets.saturating_sub(spot.bets_made)) as u32;
    let mut n = if cfg.reraise { budget } else { budget.min(spot.street.streets_left()) };
    if n == 0 || spot.eff_stack <= 0.0 {
        return PostflopActions::CheckOnly;
    }
    let spr = spot.eff_stack / spot.pot;
    if let Some(g) = cfg.geo_size {
        // El tamaño geométrico más cercano al porcentaje pedido (no el menor número de
        // apuestas que no pasa de g: con 66 % daba 50 % en vez de 75 %).
        n = closest_count(spr, n, g);
    }
    let mut f = geometric_fraction(spr, n);
    while n > 1 && f < cfg.min_fraction {
        n -= 1;
        f = geometric_fraction(spr, n);
    }
    let capped = f > cfg.max_fraction;
    let f = f.min(cfg.max_fraction);
    let mut amount = f * spot.pot;
    let is_all_in = amount >= cfg.jam_threshold * spot.eff_stack;
    if is_all_in {
        amount = spot.eff_stack;
    }
    PostflopActions::CheckOrBet(GeoBet { amount, fraction: amount / spot.pot, is_all_in, reaches_all_in: !capped })
}

/// Número de apuestas (1..=`max`) cuyo tamaño geométrico para llevar `spr` al all-in está
/// más cerca de la fracción `g` (con empate, el de menos apuestas).
fn closest_count(spr: f64, max: u32, g: f64) -> u32 {
    let mut best = 1;
    let mut err = f64::INFINITY;
    for k in 1..=max.max(1) {
        let e = (geometric_fraction(spr, k) - g).abs();
        if e < err - 1e-12 {
            err = e;
            best = k;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tres apuestas geométricas pagadas deben dejar exactamente all-in.
    #[test]
    fn three_geometric_bets_end_all_in() {
        for spr in [0.8, 2.0, 4.5, 9.0] {
            let mut pot = 1.0;
            let mut stack = spr;
            let f = geometric_fraction(spr, 3);
            for _ in 0..3 {
                let b = f * pot;
                stack -= b;
                pot += 2.0 * b;
            }
            assert!(stack.abs() < 1e-9, "spr={spr}: resto {stack}");
        }
    }

    #[test]
    fn fraction_is_self_consistent_across_streets() {
        let spr = 6.0;
        let f = geometric_fraction(spr, 3);
        let spr_next = (spr - f) / (1.0 + 2.0 * f);
        assert!((geometric_fraction(spr_next, 2) - f).abs() < 1e-12);
    }

    #[test]
    fn low_spr_collapses_to_fewer_bets() {
        let cfg = GeometricConfig::default();
        let spot = PostflopSpot { street: Street::Flop, pot: 10.0, eff_stack: 4.0, bets_made: 0, facing: 0.0 };
        match postflop_actions(&spot, &cfg) {
            PostflopActions::CheckOrBet(b) => assert!(b.fraction >= cfg.min_fraction || b.is_all_in),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn geo_size_picks_bet_count() {
        // SPR 4: 1 apuesta = 400 %, 2 = 100 %, 3 = 54 %. Con geo 66 % ⇒ 3 apuestas; con
        // geo 150 % ⇒ 2 apuestas (100 %).
        let spot = PostflopSpot { street: Street::Flop, pot: 1.0, eff_stack: 4.0, bets_made: 0, facing: 0.0 };
        for (g, expect) in [(0.66, 0.5405), (1.5, 1.0)] {
            let cfg = GeometricConfig { geo_size: Some(g), ..GeometricConfig::default() };
            match postflop_actions(&spot, &cfg) {
                PostflopActions::CheckOrBet(b) => assert!((b.fraction - expect).abs() < 0.01, "geo {g}: {}", b.fraction),
                other => panic!("{other:?}"),
            }
        }
        let cfg = GeometricConfig { geo_size: Some(0.66), ..GeometricConfig::default() };
        // Con SPR bajo (1.0), una sola apuesta (all-in) ya cumple.
        let spot = PostflopSpot { street: Street::Flop, pot: 10.0, eff_stack: 5.0, bets_made: 0, facing: 0.0 };
        match postflop_actions(&spot, &cfg) {
            PostflopActions::CheckOrBet(b) => assert!(b.is_all_in),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn facing_bet_has_no_raise() {
        let spot = PostflopSpot { street: Street::Turn, pot: 20.0, eff_stack: 50.0, bets_made: 1, facing: 12.0 };
        assert_eq!(postflop_actions(&spot, &GeometricConfig::default()), PostflopActions::FoldOrCall { call: 12.0 });
    }
}
