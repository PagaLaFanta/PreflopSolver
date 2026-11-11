//! # Constructor del árbol preflop
//!
//! Genera en orden DFS preorden los arrays `ActionNode` / `CFRNode` / `TerminalNode`,
//! las hojas de flop y la tabla global de utilidades `U[outcome][seat]` (en $ con
//! ICM/PKO o en fichas con cEV), calculada una sola vez por outcome con
//! `math::terminal_dollar_ev`.
//!
//! ## Abstracción de acción preflop
//! * Opciones discretas: tamaños de open (distintos para SB), re-raises como múltiplo de
//!   la apuesta actual, all-in; límite de raises; limp solo desde SB (opcional).
//! * Los raises que superan `allin_threshold × stack` se convierten en all-in.
//! * **Máximo de jugadores en el bote** (`max_players_in_pot`, 2 a 6): un
//!   jugador solo puede pagar o subir si hay menos de ese número
//!   de jugadores comprometidos (dinero voluntario o ciega all-in). Si no, su única acción
//!   es foldear (sin nodo de decisión). Con 3 hay overcalls de all-in, cold calls y
//!   squeezes; los botes a 3 se resuelven con side pots (`Multiway`), los de 4-6 con
//!   `MultiwayN` (k! órdenes).
//! * Flops HU → hoja postflop (MCCFR geométrico o check-down); flops a 3 → MCCFR
//!   multiway (o check-down sin `postflop`); flops a 4+ → check-down.
//!
//! ## Asientos
//! Orden de acción preflop: `[UTG, …, BTN, SB, BB]`; en HU `[SB(BTN), BB]`.
//! Postflop actúa primero SB, luego BB, luego el resto en orden (en HU, BB es OOP).

use super::nodes::{ActionNode, CFRNode, NodeKind, NodeRef, PackedAction, TerminalKind, TerminalNode, BLOCKS_PER_ROW, MAX_ACTIONS, MAX_PLAYERS, PATTERNS3, factorial, nth_perm};
use super::postflop::{build_leaf_tree_multi, build_leaf_tree_with, FlopLeaf, LeafPayoff, LeafPayoffM, MultiLeaf, MAX_MULTI};
use crate::abstraction::geometric::GeometricConfig;
use crate::abstraction::recall::PotClass;
use crate::math::fgs::Fgs;
use crate::math::field_icm::FieldIcm;
use crate::math::{terminal_dollar_ev, terminal_dollar_ev_with, Elimination, PkoParams, TerminalInput};
use std::sync::Arc;
use crate::tree::nodes::ActionKind;

const EPS: f64 = 1e-9;

#[derive(Clone, Debug)]
pub enum Payoff {
    /// Utilidad = stack final en ciegas grandes.
    ChipEv,
    /// Utilidad = equity ICM en unidades de `payouts`.
    Icm { payouts: Vec<f64> },
    /// ICM + bounties progresivos. `bounties`: mesa y luego fuera de mesa.
    Pko { payouts: Vec<f64>, bounties: Vec<f64>, params: PkoParams },
    /// Bounties sin ICM: utilidad = fichas finales + fichas de cabeza cobradas, en la
    /// misma unidad que `stacks` — no `$`. (los bounties normalmente van
    /// con ICM), pero la fórmula de cash/crecimiento de cabeza de `math::terminal` no usa
    /// más que "cuánto vale tu cabeza y la de los demás" y es indiferente a si ese valor
    /// es ICM$ o fichas — útil en early-game de torneos PKO, donde la presión de ICM
    /// todavía es despreciable pero el incentivo de los bounties no. No soporta campo
    /// (ICM multimesa) ni FGS: ninguno de los dos tiene un equivalente sin ICM.
    PkoChipEv { bounties: Vec<f64>, params: PkoParams },
}

/// Tamaño de apuesta preflop con la gramática: `2.5bb` / `2.5` (ciegas), `3x`
/// (múltiplo de la apuesta actual), `75%` (del bote tras igualar), `ai` (all-in); con
/// incremento opcional por limper/caller: `2.5bb+1bb`, `3x+1x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SizeBase {
    Bb(f64),
    Mult(f64),
    PotPct(f64),
    AllIn,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaiseSize {
    pub base: SizeBase,
    /// Incremento por cada limper (opens) o caller (re-raises), en la unidad de `base`.
    pub per_caller: f64,
}

impl RaiseSize {
    pub fn bb(x: f64) -> Self {
        RaiseSize { base: SizeBase::Bb(x), per_caller: 0.0 }
    }
    pub fn mult(x: f64) -> Self {
        RaiseSize { base: SizeBase::Mult(x), per_caller: 0.0 }
    }

    /// Analiza `"2.5bb+1bb"`, `"3x+1x"`, `"75%"`, `"ai"`, `"2.5"`. Un número sin unidad
    /// se interpreta con `default_mult` (`true` ⇒ múltiplo, `false` ⇒ bb).
    pub fn parse(text: &str, default_mult: bool) -> Result<Self, String> {
        let t: String = text.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_ascii_lowercase();
        let mut parts = t.splitn(2, '+');
        let head = parts.next().unwrap_or("");
        let tail = parts.next();
        let unit = |s: &str| -> Result<(f64, Option<char>), String> {
            let s = s.trim();
            let (num, u) = if let Some(n) = s.strip_suffix("bb") {
                (n, Some('b'))
            } else if let Some(n) = s.strip_suffix('x') {
                (n, Some('x'))
            } else if let Some(n) = s.strip_suffix('%') {
                (n, Some('%'))
            } else {
                (s, None)
            };
            num.parse::<f64>().map(|v| (v, u)).map_err(|_| format!("invalid size: {text}"))
        };
        if head == "ai" || head == "all-in" || head == "allin" {
            return Ok(RaiseSize { base: SizeBase::AllIn, per_caller: 0.0 });
        }
        let (v, u) = unit(head)?;
        let base = match u {
            Some('b') => SizeBase::Bb(v),
            Some('x') => SizeBase::Mult(v),
            Some('%') => SizeBase::PotPct(v / 100.0),
            _ => {
                if default_mult {
                    SizeBase::Mult(v)
                } else {
                    SizeBase::Bb(v)
                }
            }
        };
        let per_caller = match tail {
            Some(t) => unit(t)?.0,
            None => 0.0,
        };
        Ok(RaiseSize { base, per_caller })
    }

    /// Cantidad total a la que se sube. `bet`: apuesta actual; `pot`: bote antes de
    /// actuar; `to_call`: lo que falta por igualar; `callers`: limpers/callers presentes.
    pub fn amount(&self, bb: f64, bet: f64, pot: f64, to_call: f64, callers: usize) -> Option<f64> {
        let c = callers as f64;
        Some(match self.base {
            SizeBase::Bb(x) => (x + self.per_caller * c) * bb,
            SizeBase::Mult(x) => (x + self.per_caller * c) * bet,
            SizeBase::PotPct(x) => bet + (x + self.per_caller * c) * (pot + to_call),
            SizeBase::AllIn => return None,
        })
    }
}

/// Tamaños de un nivel de re-raise (3-bet, 4-bet…): generales y en / fuera de posición
/// respecto al último agresor. `None` ⇒ se hereda.
#[derive(Clone, Debug, Default)]
pub struct LevelSizes {
    pub all: Option<Vec<RaiseSize>>,
    pub ip: Option<Vec<RaiseSize>>,
    pub oop: Option<Vec<RaiseSize>>,
}

#[derive(Clone, Debug)]
pub struct PreflopConfig {
    /// Stacks en bb por asiento, en orden de acción preflop.
    pub stacks: Vec<f64>,
    pub sb: f64,
    pub bb: f64,
    pub ante: f64,
    /// Si es true, la BB paga el ante de toda la mesa.
    pub bb_ante: bool,
    /// Tamaños de open desde cualquier posición salvo SB y BB (gramática de [`RaiseSize`]).
    pub open_sizes: Vec<RaiseSize>,
    pub sb_open_sizes: Vec<RaiseSize>,
    /// Tamaños de open de la BB (aislar un limp: nadie ha subido, solo pagado). Vacío ⇒
    /// `open_sizes`. Es habitual darle a la BB más de un tamaño ahí (p.ej. 2bb y 3.5bb) donde
    /// a CO/BTN solo uno: con `open_sizes_by_seat`/`--opens-by` ya se puede fijar por
    /// asiento, pero BB (como SB) tiene su propio hueco por ser el caso más común.
    pub bb_open_sizes: Vec<RaiseSize>,
    /// Re-raises (y raise sobre limp).
    pub reraise_mults: Vec<RaiseSize>,
    /// Tamaños de open por asiento (orden de acción); `None` o vector vacío ⇒ los generales.
    pub open_sizes_by_seat: Vec<Option<Vec<RaiseSize>>>,
    /// Poda: quita una acción concreta de un nodo concreto, por su
    /// ruta exacta (`"" ` = raíz, como en `paths`) y la etiqueta que tendría (`"R2.2"`,
    /// `"A"`, `"F"`…) — a diferencia de `no_call`/`no_allin` (que quitan un tipo de acción
    /// en TODO un nivel), esto apunta a un nodo. Si deja el nodo sin ninguna acción, es un
    /// error de configuración (se construye igual con pánico claro, no en silencio).
    pub pruned: Vec<(String, String)>,
    /// Re-raise en posición / fuera de posición respecto al último agresor (se permite
    /// 3-bets IP más pequeños que OOP). `None` ⇒ `reraise_mults`.
    pub reraise_mults_ip: Option<Vec<RaiseSize>>,
    pub reraise_mults_oop: Option<Vec<RaiseSize>>,
    /// Tamaños por nivel de re-raise: `level_sizes[0]` = 3-bet, `[1]` = 4-bet,
    /// `[2]` = 5-bet… (cada uno con variantes IP/OOP). Un nivel sin tamaños usa
    /// `reraise_mults_ip/oop` y, si no, `reraise_mults`.
    pub level_sizes: Vec<LevelSizes>,
    /// Pagar permitido según lo que se enfrenta: `[0]` = un open (cold call / flat; la BB
    /// siempre puede defender), `[1]` = un 3-bet, `[2]` = un 4-bet, `[3]` = un 5-bet.
    /// Índice ausente ⇒ permitido. Pagar un all-in siempre está permitido.
    pub call_allowed: Vec<bool>,
    /// La BB puede pagar un open aunque `call_allowed[0]` lo prohíba (lo normal). Con
    /// "flats por raise = 0" tampoco se lo permite: entonces `false`.
    pub bb_defends: bool,
    /// "Closing flats": quien cierra la acción (nadie más queda por responder a la
    /// subida vigente) siempre puede pagar, salten o no `max_flats`/`call_allowed` — igual
    /// que `bb_defends`, pero para cualquier nivel, no solo la BB frente al open (de hecho
    /// `bb_defends` es el caso particular de esto en el open, porque la BB siempre actúa
    /// última preflop). Sin esto, un `max_flats` agotado puede forzar a quien cierra a
    /// jugar raise-or-fold de forma nada realista -- causa real de confusión recurrente en
    pub allow_closing_flats: bool,
    /// Cold calls frente a 3-bet o más: con `false`,
    /// frente a un 3-bet o más quien todavía no ha actuado en la mano no puede pagar (salvo
    /// que cierre la acción). `true` por defecto.
    pub allow_cold_calls: bool,
    /// Flats permitidos: cuántos jugadores pueden pagar (sin subir) en cada nivel:
    /// `[0]` = limpear/overlimpear (antes de que nadie suba), `[1]` = el open, `[2]` = el
    /// 3-bet… `limp_seats` sigue decidiendo *quién* puede limpear; esto decide *cuántos*
    /// como máximo. La BB cuenta como uno más. Índice ausente ⇒ sin límite. Pagar un
    /// all-in no cuenta.
    pub max_flats: Vec<u8>,
    /// All-in permitido como opción propia por nivel: `[0]` = open-shove, `[1]` = 3-bet
    /// all-in, `[2]` = 4-bet all-in… Índice ausente ⇒ `allow_allin`. (Un tamaño que acaba en
    /// all-in por umbral o SPR sigue existiendo.)
    pub allin_allowed: Vec<bool>,
    /// Limp permitido por asiento (orden de acción); vacío ⇒ `allow_limp` / `allow_sb_limp`.
    pub limp_seats: Vec<bool>,
    /// Raises no-all-in máximos (1 = solo open; 3 = hasta 4-bet).
    pub max_raises: u8,
    pub allow_allin: bool,
    /// Sin flats: frente a un all-in que cubres, si detrás queda alguien que
    /// podría poner más que la apuesta actual, no se puede pagar (sería un flat): se
    /// resube all-in (hasta el stack efectivo). Si nadie detrás puede poner más, pagar.
    pub shove_over_allin: bool,
    pub allow_sb_limp: bool,
    /// Open-limp desde cualquier posición.
    pub allow_limp: bool,
    /// Máximo de jugadores que pueden llegar a showdown/flop (2 a 6). Con 3+ se permiten
    /// overcalls de all-in, cold calls y squeezes multiway.
    pub max_players_in_pot: u8,
    pub allin_threshold: f64,
    pub payoff: Payoff,
    /// Jugadores del torneo que no están en esta mesa (cuentan para el ICM).
    pub offtable_stacks: Vec<f64>,
    /// `Some` ⇒ hojas de flop con subárbol geométrico (MCCFR); `None` ⇒ check-down.
    pub postflop: Option<GeometricConfig>,
    /// `Some` ⇒ la equity de premios usa FGS en vez de ICM puro (solo con ICM/PKO).
    pub fgs: Option<Arc<Fgs>>,
    /// Infosets postflop propios de cada hoja de flop (recuerdo perfecto de la línea
    /// preflop) en vez de compartidos entre hojas con igual tipo de bote, SPR y posición.
    /// `None` ⇒ automático: propios con hasta 64 hojas.
    pub leaf_infosets: Option<bool>,
}

impl PreflopConfig {
    /// Push/fold puro: open-shove o fold; ante un shove, call o fold.
    pub fn push_fold(stacks: Vec<f64>, payoff: Payoff) -> Self {
        PreflopConfig {
            stacks,
            sb: 0.5,
            bb: 1.0,
            ante: 0.0,
            bb_ante: false,
            open_sizes: vec![],
            sb_open_sizes: vec![],
            bb_open_sizes: vec![],
            reraise_mults: vec![],
            open_sizes_by_seat: vec![],
            pruned: vec![],
            reraise_mults_ip: None,
            reraise_mults_oop: None,
            level_sizes: vec![],
            call_allowed: vec![],
            bb_defends: true,
            allow_closing_flats: true,
            allow_cold_calls: true,
            max_flats: vec![],
            allin_allowed: vec![],
            limp_seats: vec![],
            max_raises: 0,
            allow_allin: true,
            shove_over_allin: true,
            allow_sb_limp: false,
            allow_limp: false,
            max_players_in_pot: 3,
            allin_threshold: 0.5,
            payoff,
            offtable_stacks: vec![],
            postflop: None,
            fgs: None,
            leaf_infosets: None,
        }
    }

    /// Perfil estándar de MTT: opens 2.0/2.5 (SB 3.0), 3-bet 3×, 4-bet 2.2×, all-in.
    pub fn standard(stacks: Vec<f64>, payoff: Payoff) -> Self {
        PreflopConfig {
            open_sizes: vec![RaiseSize::bb(2.0), RaiseSize::bb(2.5)],
            sb_open_sizes: vec![RaiseSize::bb(3.0)],
            bb_open_sizes: vec![],
            reraise_mults: vec![RaiseSize::mult(3.0)],
            max_raises: 2,
            allow_sb_limp: true,
            allin_threshold: 0.4,
            postflop: Some(GeometricConfig::classic()),
            shove_over_allin: false,
            // Sin límite de limps ni de cold calls en el
            // open, pero ninguno extra en 3-bet+ -- ahí solo paga quien cierra la acción
            // (allow_closing_flats, activado por defecto). Más flats en 3-bet+ apenas
            // cambia el EV y hace el árbol bastante más grande.
            max_flats: vec![255, 255, 0, 0, 0],
            ..Self::push_fold(stacks, payoff)
        }
    }
}

/// Tabla de utilidades en fichas (stack final en bb) del mismo árbol que `cfg`: el árbol
/// (acciones, terminales, hojas postflop) no depende del modelo de premios, así que los
/// desplazamientos coinciden y la tabla puede sustituir a la de ICM/PKO para medir el EV
/// en fichas de la misma estrategia.
pub fn chip_utils(cfg: &PreflopConfig) -> Vec<f32> {
    let chip = PreflopConfig { payoff: Payoff::ChipEv, fgs: None, ..cfg.clone() };
    build_tree(chip).utils
}

pub fn seat_names(n: usize) -> Vec<&'static str> {
    match n {
        2 => vec!["SB", "BB"],
        3 => vec!["BTN", "SB", "BB"],
        4 => vec!["CO", "BTN", "SB", "BB"],
        5 => vec!["HJ", "CO", "BTN", "SB", "BB"],
        6 => vec!["LJ", "HJ", "CO", "BTN", "SB", "BB"],
        7 => vec!["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
        8 => vec!["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
        9 => vec!["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
        _ => vec!["UTG", "UTG1", "UTG2", "UTG3", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    }
}

#[derive(Debug)]
pub struct PreflopTree {
    pub cfg: PreflopConfig,
    pub n: usize,
    pub root: NodeRef,
    pub actions: Vec<ActionNode>,
    pub cfr: Vec<CFRNode>,
    pub terminals: Vec<TerminalNode>,
    pub leaves: Vec<FlopLeaf>,
    /// Jugadores de cada terminal `MultiwayN` (índice en `TerminalNode::leaf`).
    pub pots: Vec<Vec<u8>>,
    pub utils: Vec<f32>,
    /// Historial legible de cada nodo de acción (p.ej. `"BTN:R2.5 SB:F"`).
    pub paths: Vec<String>,
    pub labels: Vec<Vec<String>>,
    pub total_rows: usize,
}

impl PreflopTree {
    /// Jugadores del bote de un terminal (vacío en folds).
    pub fn pot_players(&self, t: &TerminalNode) -> Vec<usize> {
        match t.kind {
            TerminalKind::MultiwayN => self.pots[t.leaf as usize].iter().map(|&p| p as usize).collect(),
            TerminalKind::FlopLeaf if t.players[2] != 255 => self.leaves[t.leaf as usize].multi.as_ref().expect("multiway leaf").seats.iter().map(|&p| p as usize).collect(),
            TerminalKind::Fold => vec![],
            _ => t.players.iter().filter(|&&p| p != 255).map(|&p| p as usize).collect(),
        }
    }

    /// Tabla `U[outcome][seat]` de un terminal (incluye `MultiwayN` con `k!` outcomes).
    pub fn terminal_utils(&self, t: &TerminalNode) -> &[f32] {
        match t.kind {
            TerminalKind::MultiwayN => {
                let k = self.pots[t.leaf as usize].len();
                let base = t.utility_offset as usize;
                &self.utils[base..base + factorial(k) * self.n]
            }
            _ => t.utilities(&self.utils),
        }
    }

    pub fn find(&self, path: &str) -> Option<usize> {
        self.paths.iter().position(|p| p == path)
    }
    pub fn num_blocks(&self) -> usize {
        self.actions.len() * BLOCKS_PER_ROW
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Act {
    Fold,
    Check,
    Call,
    RaiseTo(f64),
    AllIn,
}

#[derive(Clone)]
struct St {
    contrib: [f64; MAX_PLAYERS],
    folded: u16,
    committed: u16,
    needs: u16,
    bet: f64,
    min_inc: f64,
    raises: u8,
    /// Último jugador que subió (la BB mientras nadie haya subido).
    aggressor: usize,
    /// Pagos (calls, incluidos los all-in por menos) del raise vigente: los "flats".
    calls: u8,
    /// Jugadores que ya han actuado en la mano.
    acted: u16,
    next: usize,
    path: String,
}

struct B {
    cfg: PreflopConfig,
    n: usize,
    eff: Vec<f64>,
    ante_paid: Vec<f64>,
    sb_seat: usize,
    bb_seat: usize,
    names: Vec<&'static str>,
    actions: Vec<ActionNode>,
    cfr: Vec<CFRNode>,
    terminals: Vec<TerminalNode>,
    leaves: Vec<FlopLeaf>,
    pots: Vec<Vec<u8>>,
    utils: Vec<f32>,
    paths: Vec<String>,
    labels: Vec<Vec<String>>,
    rows: usize,
    /// ICM de campo grande (otras mesas precomputadas) cuando el campo supera el DP exacto.
    field: Option<Arc<FieldIcm>>,
}

/// Campo a partir del cual el ICM de los terminales usa [`FieldIcm`] (el DP exacto y los
/// arrays de `terminal` llegan hasta aquí).
pub const FIELD_ICM_MIN_PLAYERS: usize = 21;

fn fmt_bb(x: f64) -> String {
    let s = format!("{x:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

impl B {
    /// Orden de actuación postflop (menor = OOP): SB, BB y luego el resto en orden; en
    /// HU la BB actúa primero.
    fn postflop_order(&self, k: usize) -> usize {
        if self.n == 2 {
            if k == self.bb_seat { 0 } else { 1 }
        } else if k == self.sb_seat {
            0
        } else if k == self.bb_seat {
            1
        } else {
            2 + k
        }
    }

    fn behind(&self, st: &St, p: usize) -> f64 {
        self.eff[p] - st.contrib[p]
    }

    fn options(&self, st: &St, p: usize) -> Vec<Act> {
        let bit = 1u16 << p;
        let behind = self.behind(st, p);
        let facing = st.bet - st.contrib[p];
        let committed_others = (st.committed & !bit).count_ones();
        let mut v = Vec::new();
        let pot_now: f64 = st.contrib[..self.n].iter().sum::<f64>() + self.dead_money();
        // Limpers (open) o callers (re-raise): comprometidos que igualan la apuesta actual
        // sin ser el agresor ni la BB obligada.
        let callers = (0..self.n)
            .filter(|&k| k != p && k != st.aggressor && st.folded & (1 << k) == 0 && st.committed & (1 << k) != 0 && (st.contrib[k] - st.bet).abs() <= EPS)
            .count();
        let mut force_allin = false;
        let mut raise_targets: Vec<f64> = Vec::new();
        // Sin raises previos (open, o la BB frente a limps): tamaños de open.
        let sizes: Option<&Vec<RaiseSize>> = if st.raises == 0 {
            let by_seat = self.cfg.open_sizes_by_seat.get(p).and_then(|o| o.as_ref()).filter(|v| !v.is_empty());
            Some(match by_seat {
                Some(v) => v,
                None if p == self.sb_seat => &self.cfg.sb_open_sizes,
                None if p == self.bb_seat && !self.cfg.bb_open_sizes.is_empty() => &self.cfg.bb_open_sizes,
                None => &self.cfg.open_sizes,
            })
        } else if st.raises < self.cfg.max_raises {
            let ip = self.postflop_order(p) > self.postflop_order(st.aggressor);
            let level = self.cfg.level_sizes.get(st.raises as usize - 1);
            let by_level = level.and_then(|l| if ip { l.ip.as_ref() } else { l.oop.as_ref() }.or(l.all.as_ref()));
            Some(match by_level {
                Some(v) => v,
                None => match (ip, &self.cfg.reraise_mults_ip, &self.cfg.reraise_mults_oop) {
                    (true, Some(v), _) | (false, _, Some(v)) => v,
                    _ => &self.cfg.reraise_mults,
                },
            })
        } else {
            None
        };
        if let Some(sizes) = sizes {
            for sz in sizes {
                match sz.amount(self.cfg.bb, st.bet, pot_now, facing, callers) {
                    Some(t) => raise_targets.push(t),
                    None => force_allin = true,
                }
            }
        }
        // Subir solo tiene sentido si alguien puede responder: sin bote abierto, o si algún
        // rival comprometido tiene fichas detrás (subir contra un all-in ≡ pagar).
        let others_can_respond = (0..self.n).any(|k| k != p && st.committed & (1 << k) != 0 && st.folded & (1 << k) == 0 && self.eff[k] - st.contrib[k] > EPS);
        // Limitar a `max_players_in_pot`: pagar o subir añade a `p` a los comprometidos.
        let room = committed_others < self.cfg.max_players_in_pot as u32;
        let can_raise = behind > facing + EPS && room && (committed_others == 0 || others_can_respond);
        if facing > EPS {
            if !room {
                return vec![Act::Fold];
            }
            v.push(Act::Fold);
            let limp_ok = match self.cfg.limp_seats.get(p) {
                Some(&ok) if !self.cfg.limp_seats.is_empty() => ok,
                _ => self.cfg.allow_limp || (p == self.sb_seat && self.cfg.allow_sb_limp),
            };
            // Cuántos pueden limpear/overlimpear como máximo ("Limps", el primer
            // hueco de los flats permitidos, antes de que nadie suba) -- `st.calls` ya cuenta
            // los limps de la misma forma que cuenta los flats de un raise, porque un
            // limp es justo eso: pagar con `st.raises == 0`.
            // El complete de la SB sin limpers delante no cuenta: es una opción
            // aparte (Allow SB complete), y un export con Limps = 0 sigue dejándola.
            let sb_complete = p == self.sb_seat && st.calls == 0;
            let limp_left = sb_complete || self.cfg.max_flats.first().is_none_or(|&m| st.calls < m);
            // Pagar un raise: según el nivel enfrentado, salvo la BB frente al open y
            // cualquiera frente a un all-in (debe poder pagarlo).
            let aggressor_allin = st.raises > 0 && self.eff[st.aggressor] - st.contrib[st.aggressor] <= EPS;
            let flats_cfg = !self.cfg.max_flats.is_empty();
            let paid = if flats_cfg { st.calls } else { callers as u8 };
            let flats_left = self.cfg.max_flats.get(st.raises as usize).is_none_or(|&m| paid < m);
            // "Closing flats": si nadie más queda por responder a la subida vigente
            // (todos los demás ya plegaron, están cubiertos, o ya igualaron), pagar aquí
            // cierra la acción preflop -- siempre se permite, salte o no `max_flats`, igual
            // que ya hace `bb_defends` para la BB frente al open en particular (la BB
            // siempre cierra ahí, por eso ese caso ya estaba cubierto).
            let closing = st.raises > 0 && st.needs & !bit == 0;
            // Cold call: frente a 3-bet o más, sin haber actuado antes.
            let cold_blocked = !self.cfg.allow_cold_calls && st.raises >= 2 && st.acted & bit == 0;
            let call_ok = st.raises == 0
                || aggressor_allin
                || (closing && self.cfg.allow_closing_flats)
                || (flats_left && !cold_blocked && ((st.raises == 1 && p == self.bb_seat && self.cfg.bb_defends) || self.cfg.call_allowed.get(st.raises as usize - 1).copied().unwrap_or(true)));
            // Sin flats frente a un all-in cubierto: resubir all-in en vez de pagar.
            let can_reshove = aggressor_allin
                && behind > facing + EPS
                && (0..self.n).any(|q| q != p && st.folded & (1 << q) == 0 && st.committed & (1 << q) == 0 && self.eff[q] > st.bet + EPS);
            // Con flats por raise, frente a un all-in: si alguien (en el bote o por
            // hablar) puede poner más que la apuesta, pagar cuenta como flat (solo si quedan)
            // y se puede resubir; si nadie puede, pagar cierra la acción y subir no tiene
            // sentido.
            let responders = (0..self.n).any(|q| q != p && st.folded & (1 << q) == 0 && self.eff[q] > st.bet + EPS && self.eff[q] - st.contrib[q] > EPS);
            if flats_cfg && aggressor_allin {
                if !responders {
                    v.push(Act::Call);
                    return v;
                }
                // Pagar all-in por menos (el stack no llega a la apuesta) no cuenta como flat.
                if flats_left || behind <= facing + EPS {
                    v.push(Act::Call);
                }
                if behind > facing + EPS {
                    v.push(Act::AllIn);
                }
                return v;
            } else if can_reshove && self.cfg.shove_over_allin {
                v.push(Act::AllIn);
            } else if (st.raises > 0 && call_ok) || (st.raises == 0 && limp_left && (limp_ok || committed_others > 0)) {
                v.push(Act::Call);
            }
        } else {
            v.push(Act::Check);
        }
        if can_raise {
            let stack_total = st.contrib[p] + behind;
            // Stack efectivo frente a los que siguen en la mano: el umbral de
            // all-in se mide contra él, no contra el stack propio.
            let live_eff = (0..self.n).filter(|&q| q != p && st.folded & (1 << q) == 0).map(|q| self.eff[q]).fold(0.0f64, f64::max).min(self.eff[p]);
            for t in raise_targets {
                // (El tamaño se lleva al mínimo legal y luego se aplica el
                // umbral de all-in): un tamaño por debajo del raise mínimo sube al mínimo en
                // vez de desaparecer.
                let t = t.max(st.bet + st.min_inc);
                if t >= self.cfg.allin_threshold * live_eff || t >= stack_total - EPS {
                    if !v.contains(&Act::AllIn) {
                        v.push(Act::AllIn);
                    }
                } else if !v.iter().any(|a| matches!(a, Act::RaiseTo(x) if (x - t).abs() <= EPS)) {
                    v.push(Act::RaiseTo(t));
                }
            }
            let allin_ok = self.cfg.allin_allowed.get(st.raises as usize).copied().unwrap_or(self.cfg.allow_allin);
            if (allin_ok || force_allin) && !v.contains(&Act::AllIn) {
                v.push(Act::AllIn);
            }
        }
        // AllIn al final para un orden de acciones estable (F, X/C, R…, A).
        if let Some(i) = v.iter().position(|a| *a == Act::AllIn) {
            let a = v.remove(i);
            v.push(a);
        }
        assert!(v.len() <= MAX_ACTIONS);
        v
    }

    fn apply(&self, st: &St, p: usize, act: Act) -> St {
        let mut s = st.clone();
        let bit = 1u16 << p;
        let behind = self.behind(st, p);
        let label = match act {
            Act::Fold => "F".to_string(),
            Act::Check => "X".to_string(),
            Act::Call => "C".to_string(),
            Act::RaiseTo(t) => format!("R{}", fmt_bb(t)),
            Act::AllIn => "A".to_string(),
        };
        match act {
            Act::Fold => s.folded |= bit,
            Act::Check => {}
            Act::Call => {
                s.contrib[p] += (st.bet - st.contrib[p]).min(behind);
                s.committed |= bit;
                s.calls = s.calls.saturating_add(1);
            }
            Act::RaiseTo(_) | Act::AllIn => {
                let t = match act {
                    Act::RaiseTo(t) => t,
                    _ => st.contrib[p] + behind,
                };
                if t > st.bet + EPS {
                    s.min_inc = s.min_inc.max(t - st.bet);
                    s.bet = t;
                    s.raises += 1;
                    s.aggressor = p;
                    s.calls = 0;
                    s.needs = 0;
                    for k in 0..self.n {
                        let kb = 1u16 << k;
                        if k != p && s.folded & kb == 0 && self.eff[k] - s.contrib[k] > EPS {
                            s.needs |= kb;
                        }
                    }
                }
                if t <= st.bet + EPS {
                    s.calls = s.calls.saturating_add(1); // all-in por menos: un pago
                }
                s.contrib[p] = t;
                s.committed |= bit;
            }
        }
        s.needs &= !bit;
        s.acted |= bit;
        s.next = (p + 1) % self.n;
        if !s.path.is_empty() {
            s.path.push(' ');
        }
        s.path.push_str(&format!("{}:{}", self.names[p], label));
        s
    }

    fn utility(&self, post: &[f64], elims: &[Elimination]) -> Vec<f32> {
        let mut v: Vec<f32> = match &self.cfg.payoff {
            Payoff::ChipEv => post.iter().map(|&x| x as f32).collect(),
            Payoff::Icm { payouts } => self.dollar(post, elims, payouts, &[], false, PkoParams::default()),
            Payoff::Pko { payouts, bounties, params } => self.dollar(post, elims, payouts, bounties, true, *params),
            Payoff::PkoChipEv { bounties, params } => self.dollar_chip(post, elims, bounties, *params),
        };
        v.truncate(self.n);
        v
    }

    /// Como `dollar` con PKO, pero la base no es ICM$ sino el stack en fichas directamente
    /// (modelo de premios = identidad). Sin campo (ICM multimesa) ni FGS: ninguno de los
    /// dos tiene un equivalente sin ICM, así que se avisa claro en vez de ignorarlos en
    /// silencio.
    fn dollar_chip(&self, post: &[f64], elims: &[Elimination], bounties: &[f64], params: PkoParams) -> Vec<f32> {
        debug_assert!(self.field.is_none(), "field_for() must return None for PkoChipEv (no ICM field without ICM)");
        // Como la rama sin campo de `dollar`: las otras mesas entran en pre/post (cuentan
        // para repartir bounties) aunque no actúen ni tengan ICM que calcular.
        let pre: Vec<f64> = self.cfg.stacks.iter().chain(&self.cfg.offtable_stacks).copied().collect();
        let post_all: Vec<f64> = post.iter().chain(&self.cfg.offtable_stacks).copied().collect();
        let mut out = vec![0.0; pre.len()];
        let inp = TerminalInput { pre_stacks: &pre, post_stacks: &post_all, payouts: &[], bounties, eliminations: elims, is_pko: true, pko: params, off_alive: 0, off_heads: 0.0, off_weight: 0.0 };
        terminal_dollar_ev_with(&inp, &|_, s, _, o| o.copy_from_slice(s), &mut out);
        out.iter().map(|&x| x as f32).collect()
    }

    fn dollar(&self, post: &[f64], elims: &[Elimination], payouts: &[f64], bounties: &[f64], pko: bool, params: PkoParams) -> Vec<f32> {
        // Campo grande: la mesa en los arrays y las otras mesas precomputadas (FieldIcm).
        if let Some(field) = &self.field {
            let n = self.n;
            let off = &self.cfg.offtable_stacks;
            let mut out = vec![0.0; n];
            let (off_heads, off_weight) = if pko {
                (bounties.get(n..).map_or(0.0, |b| b.iter().sum()), off.iter().map(|&s| if params.bounty_power == 1.0 { s } else { s.powf(params.bounty_power) }).sum())
            } else {
                (0.0, 0.0)
            };
            let inp = TerminalInput {
                pre_stacks: &self.cfg.stacks,
                post_stacks: post,
                payouts,
                bounties: if pko { &bounties[..n] } else { &[] },
                eliminations: elims,
                is_pko: pko,
                pko: params,
                off_alive: off.len(),
                off_heads,
                off_weight,
            };
            match &self.cfg.fgs {
                None => terminal_dollar_ev_with(&inp, &|_, s, p, o| field.equity(s, p.len(), o), &mut out),
                Some(fgs) => {
                    // FGS sobre la mesa (los vivos, en orden de acción) con el final de cada
                    // rama valorado por el ICM de campo.
                    let fgs = Fgs { field: Some(field.clone()), ..(**fgs).clone() };
                    let model = |idx: &[usize], stacks: &[f64], prizes: &[f64], eq: &mut [f64]| {
                        let table: Vec<usize> = (0..idx.len()).collect();
                        fgs.equity(&table, stacks, prizes, eq);
                    };
                    terminal_dollar_ev_with(&inp, &model, &mut out);
                }
            }
            return out.iter().map(|&x| x as f32).collect();
        }
        let pre: Vec<f64> = self.cfg.stacks.iter().chain(&self.cfg.offtable_stacks).copied().collect();
        let post_all: Vec<f64> = post.iter().chain(&self.cfg.offtable_stacks).copied().collect();
        let mut out = vec![0.0; pre.len()];
        let inp = TerminalInput { pre_stacks: &pre, post_stacks: &post_all, payouts, bounties, eliminations: elims, is_pko: pko, pko: params, off_alive: 0, off_heads: 0.0, off_weight: 0.0 };
        match &self.cfg.fgs {
            None => terminal_dollar_ev(&inp, &mut out),
            Some(fgs) => {
                // Los asientos de mesa vivos (en orden de acción de esta mano) juegan las
                // manos futuras; los de otras mesas solo cuentan en el ICM final.
                let n = self.n;
                let model = |idx: &[usize], stacks: &[f64], prizes: &[f64], eq: &mut [f64]| {
                    let table: Vec<usize> = (0..idx.len()).filter(|&k| idx[k] < n).collect();
                    fgs.equity(&table, stacks, prizes, eq);
                };
                terminal_dollar_ev_with(&inp, &model, &mut out);
            }
        }
        out.iter().map(|&x| x as f32).collect()
    }

    /// Stacks tras la mano sin repartir el bote (cada uno ha pagado su ante y su aportación).
    fn base_post(&self, st: &St) -> Vec<f64> {
        (0..self.n).map(|k| self.cfg.stacks[k] - self.ante_paid[k] - st.contrib[k]).collect()
    }

    fn dead_money(&self) -> f64 {
        self.ante_paid.iter().sum()
    }

    fn push_utils(&mut self, u: Vec<f32>) {
        self.utils.extend(u);
    }

    fn terminal(&mut self, st: &St, kind: TerminalKind, players: [u8; 3], num_outcomes: u8, utility_offset: u32, leaf: u32) -> NodeRef {
        let mut contributions = [0u32; MAX_PLAYERS];
        for k in 0..self.n {
            contributions[k] = (st.contrib[k] * 100.0).round() as u32;
        }
        let pot = st.contrib[..self.n].iter().sum::<f64>() + self.dead_money();
        let alive_mask = (!st.folded) & ((1u16 << self.n) - 1);
        self.terminals.push(TerminalNode {
            contributions,
            pot: (pot * 100.0).round() as u32,
            utility_offset,
            alive_mask,
            kind,
            num_outcomes,
            num_seats: self.n as u8,
            players,
            leaf,
            _reserved: [0; 4],
        });
        NodeRef::new(NodeKind::Terminal, (self.terminals.len() - 1) as u32)
    }

    fn round_end(&mut self, st: &St) -> NodeRef {
        let n = self.n;
        let active: Vec<usize> = (0..n).filter(|&k| st.folded & (1 << k) == 0).collect();
        let total_pot = st.contrib[..n].iter().sum::<f64>() + self.dead_money();
        if active.len() == 1 {
            let w = active[0];
            let mut post = self.base_post(st);
            post[w] += total_pot;
            let off = self.utils.len() as u32;
            let u = self.utility(&post, &[]);
            self.push_utils(u);
            return self.terminal(st, TerminalKind::Fold, [w as u8, 255, 255], 1, off, u32::MAX);
        }
        if active.len() >= 3 {
            // Postflop con apuestas si al menos dos tienen fichas, hasta 4 jugadores y si el
            // check-down forzado deja apostar en el flop a ese número.
            let with_chips = active.iter().filter(|&&k| self.eff[k] - st.contrib[k] > EPS).count();
            if let Some(geo) = &self.cfg.postflop {
                if active.len() <= MAX_MULTI && with_chips >= 2 && geo.last_street(active.len()) >= 1 {
                    return self.multi_leaf(st, &active);
                }
            }
            if active.len() == 3 {
                return self.multiway(st, [active[0], active[1], active[2]]);
            }
            return self.multiway_n(st, &active);
        }
        let (a, b) = (active[0], active[1]);
        let matched = st.contrib[a].min(st.contrib[b]);
        let mut base = self.base_post(st);
        base[a] += st.contrib[a] - matched; // devolución del exceso no pagado
        base[b] += st.contrib[b] - matched;
        let pot = total_pot - (st.contrib[a] - matched) - (st.contrib[b] - matched);
        let behind = [self.eff[a] - matched, self.eff[b] - matched];
        let pot_class = match st.raises {
            0 => PotClass::Limped,
            1 => PotClass::SingleRaised,
            2 => PotClass::ThreeBet,
            _ => PotClass::FourBetPlus,
        };
        if behind[0].min(behind[1]) <= EPS {
            // All-in: 3 outcomes (gana a, gana b, empate).
            let off = self.utils.len() as u32;
            for w in [Some(a), Some(b), None] {
                let mut post = base.clone();
                let mut elims = Vec::new();
                match w {
                    Some(w) => {
                        let l = if w == a { b } else { a };
                        post[w] += pot;
                        if post[l] <= EPS {
                            post[l] = 0.0;
                            elims.push(Elimination { victim: l as u8, winner_mask: 1 << w });
                        }
                    }
                    None => {
                        post[a] += pot / 2.0;
                        post[b] += pot / 2.0;
                    }
                }
                let u = self.utility(&post, &elims);
                self.push_utils(u);
            }
            return self.terminal(st, TerminalKind::AllInRunout, [a as u8, b as u8, 255], 3, off, u32::MAX);
        }
        // Flop HU.
        let (oop, ip) = if self.postflop_order(a) < self.postflop_order(b) { (a, b) } else { (b, a) };
        let eff = behind[0].min(behind[1]);
        let mut utils = std::mem::take(&mut self.utils);
        let (checkdown, nodes) = {
            let this = &*self;
            let payoff = |post: &[f64], el: &[Elimination]| this.utility(post, el);
            let pay = LeafPayoff { base_post: base.clone(), oop, ip, pot, utility: &payoff };
            let checkdown = pay.push(&mut utils, [0.0, 0.0], None);
            let nodes = match &this.cfg.postflop {
                Some(geo) => {
                    // Agresor preflop (para el donk): el último que subió, si llegó al flop.
                    let aggr = (st.raises > 0).then_some(if st.aggressor == oop { super::postflop::OOP } else { super::postflop::IP });
                    build_leaf_tree_with(pot, eff, pot_class, geo, &pay, &mut utils, aggr)
                }
                None => Vec::new(),
            };
            (checkdown, nodes)
        };
        self.utils = utils;
        self.leaves.push(FlopLeaf { oop: oop as u8, ip: ip as u8, pot, eff_stack: eff, pot_class, checkdown_util: checkdown, nodes, multi: None });
        let leaf = (self.leaves.len() - 1) as u32;
        self.terminal(st, TerminalKind::FlopLeaf, [oop as u8, ip as u8, 255], 3, checkdown, leaf)
    }

    /// Reparto de botes (principal + laterales) para un orden de fuerza dado, de mejor a
    /// peor en grupos de manos empatadas (cada bote se reparte a partes iguales entre los
    /// del mejor grupo con derecho a él). Devuelve stacks finales y eliminaciones (el
    /// eliminador es quien gana el bote más alto en el que participaba la víctima: el que
    /// contiene sus últimas fichas; con empate, todos los que lo reparten).
    fn resolve_pots(&self, st: &St, active: &[usize], order: &[Vec<usize>]) -> (Vec<f64>, Vec<Elimination>) {
        self.resolve_pots_with(&st.contrib[..self.n], active, order)
    }

    /// [`Self::resolve_pots`] con aportaciones dadas (preflop + postflop en las hojas).
    fn resolve_pots_with(&self, contrib: &[f64], active: &[usize], order: &[Vec<usize>]) -> (Vec<f64>, Vec<Elimination>) {
        let mut post: Vec<f64> = (0..self.n).map(|k| self.cfg.stacks[k] - self.ante_paid[k] - contrib[k]).collect();
        let mut levels: Vec<f64> = active.iter().map(|&k| contrib[k]).collect();
        levels.sort_by(|a, b| a.total_cmp(b));
        levels.dedup_by(|a, b| (*a - *b).abs() <= EPS);
        let mut prev = 0.0;
        let mut top_winner = vec![0u64; self.n];
        for (li, &level) in levels.iter().enumerate() {
            let mut slice: f64 = (0..self.n).map(|k| contrib[k].min(level) - contrib[k].min(prev)).sum();
            if li == 0 {
                slice += self.dead_money();
            }
            if li + 1 == levels.len() {
                // Fichas de jugadores retirados por encima del último nivel activo.
                slice += (0..self.n).filter(|k| !active.contains(k)).map(|k| (contrib[k] - level).max(0.0)).sum::<f64>();
            }
            let group = order.iter().find(|g| g.iter().any(|&k| contrib[k] >= level - EPS)).expect("elegible");
            let winners: Vec<usize> = group.iter().copied().filter(|&k| contrib[k] >= level - EPS).collect();
            let mask: u64 = winners.iter().map(|&w| 1u64 << w).sum();
            for &w in &winners {
                post[w] += slice / winners.len() as f64;
            }
            for &k in active {
                if contrib[k] >= level - EPS {
                    top_winner[k] = mask;
                }
            }
            prev = level;
        }
        let mut elims = Vec::new();
        for &k in active {
            if post[k] <= EPS {
                post[k] = 0.0;
                let w = top_winner[k] & !(1u64 << k);
                if w != 0 {
                    elims.push(Elimination { victim: k as u8, winner_mask: w });
                }
            }
        }
        (post, elims)
    }

    /// Bote a 3: all-in (con side pots) o flop multiway resuelto en check-down (las fichas
    /// detrás de los jugadores con stack no se arriesgan: aproximación del modelo).
    ///
    /// 13 resultados ([`PATTERNS3`]): los 6 órdenes estrictos y los empates, cuyo reparto
    /// no es la media de los órdenes que separa (con ICM, repartir un bote vale más que
    /// jugárselo a cara o cruz).
    fn multiway(&mut self, st: &St, players: [usize; 3]) -> NodeRef {
        let off = self.utils.len() as u32;
        for ranks in PATTERNS3 {
            let order: Vec<Vec<usize>> = (0..3u8).map(|r| (0..3).filter(|&i| ranks[i] == r).map(|i| players[i]).collect()).filter(|g: &Vec<usize>| !g.is_empty()).collect();
            let (post, elims) = self.resolve_pots(st, &players, &order);
            let u = self.utility(&post, &elims);
            self.push_utils(u);
        }
        self.terminal(st, TerminalKind::Multiway, [players[0] as u8, players[1] as u8, players[2] as u8], PATTERNS3.len() as u8, off, u32::MAX)
    }

    /// Bote de 4 a 6 jugadores: `k!` outcomes (orden de fuerza en código de Lehmer).
    fn multiway_n(&mut self, st: &St, players: &[usize]) -> NodeRef {
        let k = players.len();
        let off = self.utils.len() as u32;
        for idx in 0..factorial(k) {
            let order: Vec<Vec<usize>> = nth_perm(k, idx).iter().map(|&i| vec![players[i as usize]]).collect();
            let (post, elims) = self.resolve_pots(st, players, &order);
            let u = self.utility(&post, &elims);
            self.push_utils(u);
        }
        self.pots.push(players.iter().map(|&p| p as u8).collect());
        let pot_idx = (self.pots.len() - 1) as u32;
        self.terminal(st, TerminalKind::MultiwayN, [255; 3], factorial(k).min(255) as u8, off, pot_idx)
    }

    /// Flop a 3 o 4 con al menos dos jugadores con fichas: hoja postflop multiway (MCCFR
    /// geométrico). Cada uno juega con su stack: cuando el corto se queda sin fichas, los
    /// demás siguen apostando en el bote lateral, y cada final reparte principal y laterales
    /// según el orden de fuerza de los que llegan.
    fn multi_leaf(&mut self, st: &St, players: &[usize]) -> NodeRef {
        let mut seats = players.to_vec();
        seats.sort_by_key(|&k| self.postflop_order(k));
        let k = seats.len();
        let pot = st.contrib[..self.n].iter().sum::<f64>() + self.dead_money();
        let cap: Vec<f64> = seats.iter().map(|&s| (self.eff[s] - st.contrib[s]).max(0.0)).collect();
        let mut utils = std::mem::take(&mut self.utils);
        let nodes = {
            let this = &*self;
            let seats = &seats;
            let payoff = |inv: &[f64], alive: u8, order: &[u8]| -> Vec<f32> {
                let mut contrib = st.contrib;
                for (p, &s) in seats.iter().enumerate() {
                    contrib[s] += inv[p];
                }
                let active: Vec<usize> = (0..k).filter(|&p| alive & (1 << p) != 0).map(|p| seats[p]).collect();
                let ord: Vec<Vec<usize>> = order.iter().map(|&p| vec![seats[p as usize]]).collect();
                let (post, elims) = this.resolve_pots_with(&contrib[..this.n], &active, &ord);
                this.utility(&post, &elims)
            };
            let pay = LeafPayoffM { k, utility: &payoff };
            // Agresor preflop en orden postflop (para el donk), si llegó al flop.
            let aggr = (st.raises > 0).then(|| seats.iter().position(|&s| s == st.aggressor).map(|i| i as u8)).flatten();
            build_leaf_tree_multi(pot, &cap, this.cfg.postflop.as_ref().unwrap(), &pay, &mut utils, aggr)
        };
        self.utils = utils;
        let eff = {
            let mut c = cap.clone();
            c.sort_by(|a, b| b.total_cmp(a));
            c[1]
        };
        self.leaves.push(FlopLeaf {
            oop: seats[0] as u8,
            ip: seats[k - 1] as u8,
            pot,
            eff_stack: eff,
            pot_class: PotClass::Multiway,
            checkdown_util: 0,
            nodes: Vec::new(),
            multi: Some(MultiLeaf { seats: seats.iter().map(|&s| s as u8).collect(), nodes }),
        });
        let leaf = (self.leaves.len() - 1) as u32;
        self.terminal(st, TerminalKind::FlopLeaf, [seats[0] as u8, seats[1] as u8, seats[2] as u8], 0, 0, leaf)
    }

    fn build(&mut self, st: St) -> NodeRef {
        let alive = (!st.folded & ((1u16 << self.n) - 1)).count_ones();
        if st.needs == 0 || alive == 1 {
            return self.round_end(&st);
        }
        let mut p = st.next % self.n;
        while st.needs & (1 << p) == 0 {
            p = (p + 1) % self.n;
        }
        let mut opts = self.options(&st, p);
        if !self.cfg.pruned.is_empty() {
            opts.retain(|&a| {
                let label = match a {
                    Act::Fold => "F".to_string(),
                    Act::Check => "X".to_string(),
                    Act::Call => "C".to_string(),
                    Act::RaiseTo(t) => format!("R{}", fmt_bb(t)),
                    Act::AllIn => "A".to_string(),
                };
                !self.cfg.pruned.iter().any(|(path, lbl)| *path == st.path && *lbl == label)
            });
            assert!(!opts.is_empty(), "--prune leaves node {:?} without any action", if st.path.is_empty() { "(root)" } else { &st.path });
        }
        if opts.len() == 1 {
            let s2 = self.apply(&st, p, opts[0]);
            return self.build(s2);
        }
        let idx = self.actions.len();
        let pot: f64 = st.contrib[..self.n].iter().sum::<f64>() + self.dead_money();
        let facing = st.bet - st.contrib[p];
        let mut node = ActionNode {
            children: [NodeRef::NULL; MAX_ACTIONS],
            actions: [PackedAction::default(); MAX_ACTIONS],
            infoset: idx as u32,
            pot: (pot * 100.0).round() as u32,
            to_call: (facing.max(0.0) * 100.0).round() as u32,
            eff_stack: (self.behind(&st, p) * 100.0).round() as u32,
            player: p as u8,
            num_actions: opts.len() as u8,
            street: 0,
            flags: 0,
            bets_made: st.raises,
            pot_class: 0,
            _reserved: [0; 2],
        };
        self.actions.push(node);
        self.cfr.push(CFRNode { row: self.rows as u32, exp_base: (idx * BLOCKS_PER_ROW) as u32, num_actions: opts.len() as u8, player: p as u8, _reserved: [0; 6] });
        self.rows += opts.len();
        self.paths.push(st.path.clone());
        self.labels.push(Vec::new());
        let mut labels = Vec::new();
        for (i, &a) in opts.iter().enumerate() {
            let (kind, param) = match a {
                Act::Fold => (ActionKind::Fold, 0),
                Act::Check => (ActionKind::Check, 0),
                Act::Call => (ActionKind::Call, 0),
                Act::RaiseTo(_) => (ActionKind::Raise, i as u8),
                Act::AllIn => (ActionKind::AllIn, 0),
            };
            node.actions[i] = PackedAction::new(kind, param);
            let s2 = self.apply(&st, p, a);
            labels.push(s2.path.rsplit(' ').next().unwrap().split(':').nth(1).unwrap().to_string());
            node.children[i] = self.build(s2);
        }
        self.actions[idx] = node;
        self.labels[idx] = labels;
        NodeRef::new(NodeKind::Action, idx as u32)
    }
}

/// ICM de campo para la configuración si hace falta: ICM/PKO (con o sin FGS) con más de
/// [`FIELD_ICM_MIN_PLAYERS`] jugadores en total (mesa + otras mesas).
pub fn field_for(cfg: &PreflopConfig) -> Option<Arc<FieldIcm>> {
    let payouts = match &cfg.payoff {
        Payoff::ChipEv | Payoff::PkoChipEv { .. } => return None,
        Payoff::Icm { payouts } | Payoff::Pko { payouts, .. } => payouts,
    };
    let total = cfg.stacks.len() + cfg.offtable_stacks.len();
    if total < FIELD_ICM_MIN_PLAYERS {
        return None;
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    Some(Arc::new(FieldIcm::new(&cfg.offtable_stacks, cfg.stacks.iter().sum(), 0.05, payouts, threads)))
}

/// Recorre un árbol binario postflop desde `idx` y asigna a cada decisión el hash del
/// camino (secuencia de hijos elegidos) que lleva a ella.
fn path_keys(len: usize, idx: usize, h: u64, kids: &mut dyn FnMut(usize) -> Option<[u32; 2]>, set: &mut dyn FnMut(usize, u64)) {
    debug_assert!(idx < len);
    if let Some(ch) = kids(idx) {
        set(idx, h);
        for (k, &c) in ch.iter().enumerate() {
            path_keys(len, c as usize, h.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17) ^ (k as u64 + 1), kids, set);
        }
    }
}

pub fn build_tree(cfg: PreflopConfig) -> PreflopTree {
    let n = cfg.stacks.len();
    assert!((2..=MAX_PLAYERS).contains(&n), "between 2 and 10 players");
    assert!((2..=6).contains(&cfg.max_players_in_pot), "max_players_in_pot entre 2 y 6");
    let sb_seat = if n == 2 { 0 } else { n - 2 };
    let bb_seat = n - 1;
    let mut ante_paid = vec![0.0; n];
    if cfg.bb_ante {
        ante_paid[bb_seat] = (cfg.ante * n as f64).min(cfg.stacks[bb_seat]);
    } else {
        for k in 0..n {
            ante_paid[k] = cfg.ante.min(cfg.stacks[k]);
        }
    }
    let eff: Vec<f64> = (0..n).map(|k| cfg.stacks[k] - ante_paid[k]).collect();
    let mut b = B {
        n,
        eff,
        ante_paid,
        sb_seat,
        bb_seat,
        names: seat_names(n),
        actions: Vec::new(),
        cfr: Vec::new(),
        terminals: Vec::new(),
        leaves: Vec::new(),
        pots: Vec::new(),
        utils: Vec::new(),
        paths: Vec::new(),
        labels: Vec::new(),
        rows: 0,
        field: field_for(&cfg),
        cfg,
    };
    let mut contrib = [0.0; MAX_PLAYERS];
    contrib[sb_seat] = b.cfg.sb.min(b.eff[sb_seat]);
    contrib[bb_seat] = b.cfg.bb.min(b.eff[bb_seat]);
    let mut needs = 0u16;
    let mut committed = 0u16;
    for k in 0..n {
        if b.eff[k] - contrib[k] > EPS {
            needs |= 1 << k;
        } else {
            committed |= 1 << k; // ciega all-in: ya está en el bote
        }
    }
    let st = St { contrib, folded: 0, committed, needs, bet: b.cfg.bb, min_inc: b.cfg.bb, raises: 0, aggressor: bb_seat, calls: 0, acted: 0, next: 0, path: String::new() };
    let root = b.build(st);
    // Recuerdo perfecto de la secuencia de acciones postflop (cada nodo del árbol
    // es su infoset y solo las cartas se agrupan en buckets por calle). La clave es un grupo
    // (bits 40+; el 63 es de la tabla hash) y el hash del camino de decisiones desde el
    // flop (bits 16-39) sobre el bucket (bits 0-15). Fusionar historias distintas con igual
    // número de apuestas (en el river, "pasó el flop y apostó el turn" con "apostó el flop y
    // pasó el turn") castigaba a las manos marginales de quien paga: la BB defendía un 23 %
    // frente al open del CO en vez del 37 %. Con pocas hojas, cada una es su grupo;
    // con muchas, se agrupan por tipo de bote, SPR y posición (el árbol es el mismo).
    let per_leaf = b.cfg.leaf_infosets.unwrap_or(b.leaves.len() <= 64);
    for (i, leaf) in b.leaves.iter_mut().enumerate() {
        // `players`: 2 en HU, 3 o 4 en multiway (árboles distintos no comparten grupo).
        let group = |root_key: u64, players: usize| -> u64 {
            if per_leaf {
                i as u64 + 1
            } else {
                // Campos de la plantilla de la raíz: SPR, calle, posición, tipo de bote.
                (1 << 20) | ((players as u64 & 7) << 15) | ((root_key >> 16) & 0x7FFF)
            }
        };
        if let Some(m) = leaf.multi.as_mut() {
            let Some(super::postflop::PfNode3::Decision { key: root, .. }) = m.nodes.first().copied() else { continue };
            let g = group(root, m.seats.len()) << 40;
            let kids: Vec<Option<[u32; 2]>> = m.nodes.iter().map(|n| match n {
                super::postflop::PfNode3::Decision { children, .. } => Some(*children),
                _ => None,
            }).collect();
            let nodes = &mut m.nodes;
            path_keys(nodes.len(), 0, 17, &mut |idx| kids[idx], &mut |idx, h| {
                if let super::postflop::PfNode3::Decision { key, .. } = &mut nodes[idx] {
                    *key = g | (h & 0xFF_FFFF) << 16;
                }
            });
        } else {
            let Some(super::postflop::PfNode::Decision { key: root, .. }) = leaf.nodes.first().copied() else { continue };
            let g = group(root, 2) << 40;
            let nodes = &mut leaf.nodes;
            let kids: Vec<Option<[u32; 2]>> = nodes.iter().map(|n| match n {
                super::postflop::PfNode::Decision { children, .. } => Some(*children),
                _ => None,
            }).collect();
            path_keys(nodes.len(), 0, 17, &mut |idx| kids[idx], &mut |idx, h| {
                if let super::postflop::PfNode::Decision { key, .. } = &mut nodes[idx] {
                    *key = g | (h & 0xFF_FFFF) << 16;
                }
            });
        }
    }
    PreflopTree {
        n,
        root,
        actions: b.actions,
        cfr: b.cfr,
        terminals: b.terminals,
        leaves: b.leaves,
        pots: b.pots,
        utils: b.utils,
        paths: b.paths,
        labels: b.labels,
        total_rows: b.rows,
        cfg: b.cfg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opciones por nivel: tamaños de 3-bet IP/OOP, 4-bet y 5-bet propios,
    /// sin pagar frente al 4-bet (pero sí frente al 5-bet all-in), sin open-shove ni
    /// 3-bet all-in, y limp solo desde las posiciones elegidas.
    #[test]
    fn per_level_sizes_calls_allins_and_limps() {
        let mut cfg = PreflopConfig::standard(vec![100.0; 3], Payoff::ChipEv);
        cfg.postflop = None;
        cfg.level_sizes = vec![
            LevelSizes { all: Some(vec![RaiseSize::mult(3.0)]), ip: None, oop: Some(vec![RaiseSize::mult(3.8)]) },
            LevelSizes { all: Some(vec![RaiseSize::mult(2.2)]), ..Default::default() },
            LevelSizes { all: Some(vec![RaiseSize::parse("ai", true).unwrap()]), ..Default::default() },
        ];
        cfg.max_raises = 4;
        cfg.call_allowed = vec![true, true, false];
        cfg.allin_allowed = vec![false, false];
        let t = build_tree(cfg.clone());
        let labels = |p: &str| t.labels[t.find(p).unwrap_or_else(|| panic!("falta {p}"))].clone();
        assert_eq!(labels(""), vec!["F", "R2", "R2.5"], "sin open-shove");
        assert_eq!(labels("BTN:R2.5"), vec!["F", "C", "R9.5"], "3-bet OOP 3.8x, sin all-in");
        assert_eq!(labels("BTN:R2.5 SB:F BB:R9.5"), vec!["F", "C", "R20.9", "A"], "4-bet 2.2x");
        // BB cierra la acción (SB ya folded): puede pagar el 4-bet aunque call_allowed en
        // ese nivel sea false -- "closing flats", quien cierra siempre puede pagar.
        assert_eq!(labels("BTN:R2.5 SB:F BB:R9.5 BTN:R20.9"), vec!["F", "C", "A"], "BB cierra: paga pese a call_allowed[4-bet]=false; 5-bet = all-in");
        assert_eq!(labels("BTN:R2.5 SB:F BB:R9.5 BTN:R20.9 BB:A"), vec!["F", "C"], "el all-in siempre se puede pagar");
        // Limp solo desde el BTN: la SB ya no puede completar.
        cfg.limp_seats = vec![true, false, false];
        let t = build_tree(cfg);
        assert_eq!(t.labels[t.find("").unwrap()], vec!["F", "C", "R2", "R2.5"]);
        assert!(!t.labels[t.find("BTN:F").unwrap()].contains(&"C".to_string()));
    }

    /// "Closing flats": con call_allowed[3-bet]=false nadie puede pagar un 3-bet en
    /// general, salvo quien cierra la acción (nadie más queda por responder) -- y solo si
    /// allow_closing_flats está activo (lo está por defecto). Usa el nivel de 3-bet (no el
    /// open) para no solaparse con bb_defends, que ya cubre la BB frente al open.
    #[test]
    fn closing_flats_lets_the_last_player_call_despite_call_allowed() {
        let mut cfg = PreflopConfig::standard(vec![100.0; 4], Payoff::ChipEv);
        cfg.postflop = None;
        cfg.call_allowed = vec![true, false];
        let t = build_tree(cfg.clone());
        let labels = |p: &str| t.labels[t.find(p).unwrap_or_else(|| panic!("falta {p}"))].clone();
        let threebet = labels("CO:R2.5").iter().find(|l| l.starts_with('R')).unwrap_or_else(|| panic!("sin 3-bet")).clone();
        let after_3bet = format!("CO:R2.5 BTN:{threebet}");
        // SB frente al 3-bet: CO (el que abrió) todavía tiene que hablar, SB no cierra.
        assert!(!labels(&after_3bet).contains(&"C".to_string()), "SB no cierra: sin call pese a closing flats");
        // CO frente al 3-bet (SB y BB ya foldearon): nadie queda por hablar, CO cierra --
        // puede pagar aunque call_allowed[3-bet] sea false.
        let closing_path = format!("{after_3bet} SB:F BB:F");
        assert!(labels(&closing_path).contains(&"C".to_string()), "CO cierra: paga pese a call_allowed[3-bet]=false");
        // Con closing flats desactivado, ni siquiera quien cierra puede pagar.
        cfg.allow_closing_flats = false;
        let t2 = build_tree(cfg);
        let labels2 = |p: &str| t2.labels[t2.find(p).unwrap_or_else(|| panic!("falta {p}"))].clone();
        assert!(!labels2(&closing_path).contains(&"C".to_string()), "sin closing flats, ni quien cierra paga");
    }

    /// Sin cold calls: frente al 3-bet, quien no ha actuado todavía (la
    /// SB, con el CO que abrió aún por hablar) no puede pagar; el que abrió sí (no es cold
    /// call) mientras queden flats en ese nivel.
    #[test]
    fn no_cold_calls_option() {
        let mut cfg = PreflopConfig::standard(vec![100.0; 4], Payoff::ChipEv);
        cfg.postflop = None;
        cfg.max_flats = vec![255, 255, 2, 0, 0];
        let t = build_tree(cfg.clone());
        let labels = |t: &PreflopTree, p: &str| t.labels[t.find(p).unwrap_or_else(|| panic!("falta {p}"))].clone();
        let threebet = labels(&t, "CO:R2.5").iter().find(|l| l.starts_with('R')).unwrap_or_else(|| panic!("sin 3-bet")).clone();
        let after = format!("CO:R2.5 BTN:{threebet}");
        assert!(labels(&t, &after).contains(&"C".to_string()), "con cold calls la SB puede pagar el 3-bet");
        cfg.allow_cold_calls = false;
        let t2 = build_tree(cfg);
        assert!(!labels(&t2, &after).contains(&"C".to_string()), "sin cold calls la SB no paga");
        let opener = format!("{after} SB:F BB:F");
        assert!(labels(&t2, &opener).contains(&"C".to_string()), "el que abrió sí puede pagar");
    }

    /// `max_flats[0]` limita cuántos
    /// pueden limpear/overlimpear -- no solo quién puede, que ya cubría `allow_limp`.
    #[test]
    fn max_flats_limps_caps_overlimping() {
        let mut cfg = PreflopConfig::standard(vec![100.0; 4], Payoff::ChipEv);
        cfg.postflop = None;
        cfg.allow_limp = true; // cualquiera puede limpear (si no, solo la SB por defecto)
        cfg.max_flats = vec![1, 255, 0, 0, 0]; // como máximo 1 limp, el resto sin tocar
        let t = build_tree(cfg);
        let labels = |p: &str| t.labels[t.find(p).unwrap_or_else(|| panic!("falta {p}"))].clone();
        assert!(labels("").contains(&"C".to_string()), "CO (primero en hablar) puede limpear");
        assert!(!labels("CO:C").contains(&"C".to_string()), "BTN no puede overlimpear: el límite de 1 ya está gastado");
    }

    /// Con Limps = 0 la SB sigue
    /// pudiendo completar cuando le llega foldeado: es la opción del complete de la SB.
    #[test]
    fn sb_complete_ignores_zero_limps() {
        let mut cfg = PreflopConfig::standard(vec![100.0; 4], Payoff::ChipEv);
        cfg.postflop = None;
        cfg.max_flats = vec![0, 1, 0, 0, 0];
        let t = build_tree(cfg);
        let labels = |p: &str| t.labels[t.find(p).unwrap_or_else(|| panic!("falta {p}"))].clone();
        assert!(!labels("").contains(&"C".to_string()), "CO no limpea con Limps = 0");
        assert!(labels("CO:F BTN:F").contains(&"C".to_string()), "la SB completa");
    }

    /// La tabla en fichas tiene la misma disposición que la de ICM (mismo árbol, mismos
    /// desplazamientos), y en ella el valor de foldear es el stack.
    #[test]
    fn chip_utils_share_layout_with_icm_tree() {
        let mut cfg = PreflopConfig::standard(vec![20.0, 30.0, 25.0], Payoff::Icm { payouts: vec![50.0, 30.0, 20.0] });
        cfg.postflop = Some(crate::abstraction::geometric::GeometricConfig::default());
        let icm = build_tree(cfg.clone());
        let chip = build_tree(PreflopConfig { payoff: Payoff::ChipEv, ..cfg.clone() });
        assert_eq!(icm.utils.len(), chip.utils.len());
        assert_eq!(icm.paths, chip.paths);
        for (a, b) in icm.terminals.iter().zip(&chip.terminals) {
            assert_eq!((a.utility_offset, a.num_outcomes, a.num_seats, a.kind), (b.utility_offset, b.num_outcomes, b.num_seats, b.kind));
        }
        assert_eq!(chip_utils(&cfg), chip.utils);
    }

    /// Campo de 22 jugadores (3 en la mesa): las utilidades del árbol con el ICM de campo
    /// coinciden con el ICM exacto (DP disperso) de todo el campo, terminal a terminal.
    #[test]
    fn field_icm_utilities_match_exact() {
        let off: Vec<f64> = (0..19).map(|k| 8.0 + 3.0 * k as f64).collect();
        let payouts = vec![40.0, 25.0, 15.0, 10.0, 6.0, 4.0];
        let cfg = PreflopConfig { offtable_stacks: off.clone(), ..PreflopConfig::push_fold(vec![12.0, 30.0, 7.0], Payoff::Icm { payouts: payouts.clone() }) };
        assert!(field_for(&cfg).is_some());
        let t = build_tree(cfg.clone());
        let b = B {
            n: 3,
            eff: vec![],
            ante_paid: vec![0.0; 3],
            sb_seat: 1,
            bb_seat: 2,
            names: seat_names(3),
            actions: vec![],
            cfr: vec![],
            terminals: vec![],
            leaves: vec![],
            pots: vec![],
            utils: vec![],
            paths: vec![],
            labels: vec![],
            rows: 0,
            field: None,
            cfg: cfg.clone(),
        };
        // Fold de BTN y SB: la BB gana las ciegas.
        let post = [12.0, 29.5, 7.5];
        let exact = b.dollar(&post, &[], &payouts, &[], false, PkoParams::default());
        let k = t.find("BTN:F").unwrap();
        let term = &t.terminals[t.actions[k].children[0].index()];
        let got = &t.utils[term.utility_offset as usize..term.utility_offset as usize + 3];
        for s in 0..3 {
            assert!((got[s] - exact[s]).abs() < 2e-4, "asiento {s}: {} vs {}", got[s], exact[s]);
        }
        // Un all-in con eliminación: el eliminado cobra el puesto 22.
        let bust = b.dollar(&[24.0, 30.0, 0.0], &[], &payouts, &[], false, PkoParams::default());
        let field = B { field: field_for(&cfg), ..b };
        let bust_f = field.dollar(&[24.0, 30.0, 0.0], &[], &payouts, &[], false, PkoParams::default());
        for s in 0..3 {
            assert!((bust[s] - bust_f[s]).abs() < 2e-4, "eliminación, asiento {s}: {} vs {}", bust[s], bust_f[s]);
        }
    }

    /// Sin flat frente al open: solo la BB puede pagarlo.
    #[test]
    fn no_cold_call_keeps_bb_defense() {
        let mut cfg = PreflopConfig::standard(vec![50.0; 4], Payoff::ChipEv);
        cfg.postflop = None;
        cfg.call_allowed = vec![false];
        let t = build_tree(cfg);
        assert!(!t.labels[t.find("CO:R2").unwrap()].contains(&"C".to_string()), "BTN no hace cold call");
        assert!(!t.labels[t.find("CO:R2 BTN:F").unwrap()].contains(&"C".to_string()), "SB no hace flat");
        assert!(t.labels[t.find("CO:R2 BTN:F SB:F").unwrap()].contains(&"C".to_string()), "la BB defiende");
    }

    #[test]
    fn heads_up_push_fold_structure() {
        let t = build_tree(PreflopConfig::push_fold(vec![10.0, 10.0], Payoff::ChipEv));
        assert_eq!(t.actions.len(), 2);
        assert_eq!(t.labels[0], vec!["F", "A"]);
        assert_eq!(t.paths[1], "SB:A");
        assert_eq!(t.labels[1], vec!["F", "C"]);
        // 3 terminales: SB fold, BB fold, all-in.
        assert_eq!(t.terminals.len(), 3);
        let allin = t.terminals.iter().find(|x| x.kind == TerminalKind::AllInRunout).unwrap();
        let u = allin.utilities(&t.utils);
        assert_eq!(&u[..2], &[20.0, 0.0]); // gana SB
        assert_eq!(&u[4..6], &[10.0, 10.0]); // empate
    }

    #[test]
    fn pots_respect_max_players() {
        for max in [2u8, 3] {
            let mut cfg = PreflopConfig::standard(vec![25.0, 30.0, 18.0, 40.0], Payoff::Icm { payouts: vec![50.0, 30.0, 20.0] });
            cfg.max_players_in_pot = max;
            let t = build_tree(cfg);
            assert!(t.terminals.iter().all(|x| x.alive_mask.count_ones() <= max as u32));
            assert_eq!(t.terminals.iter().any(|x| x.kind == TerminalKind::Multiway), max == 3);
        }
        let mut cfg = PreflopConfig::standard(vec![25.0, 30.0, 18.0, 40.0], Payoff::Icm { payouts: vec![50.0, 30.0, 20.0] });
        cfg.max_players_in_pot = 2;
        let t = build_tree(cfg);
        assert!(t.actions.len() > 20, "{}", t.actions.len());
        // ICM: cada outcome reparte exactamente el prize pool.
        for chunk in t.utils.chunks(4) {
            let s: f32 = chunk.iter().sum();
            assert!((s - 100.0).abs() < 1e-3, "{s}");
        }
        assert!(!t.leaves.is_empty() && t.leaves.iter().all(|l| !l.nodes.is_empty()));
    }

    #[test]
    fn no_raise_against_all_in() {
        let pf = |st: Vec<f64>| PreflopConfig { shove_over_allin: false, ..PreflopConfig::push_fold(st, Payoff::ChipEv) };
        let t = build_tree(pf(vec![12.0, 15.0, 20.0]));
        let k = t.find("BTN:F SB:A").unwrap();
        assert_eq!(t.labels[k], vec!["F", "C"]);
        // Overcall a 3 bandas; la BB también puede resubir all-in para jugar el lateral
        // contra la SB, que aún tiene fichas.
        let k = t.find("BTN:A SB:C").unwrap();
        assert_eq!(t.labels[k], vec!["F", "C", "A"]);
        // Si nadie puede responder, no hay resubida.
        let t = build_tree(pf(vec![12.0, 10.0, 20.0]));
        assert_eq!(t.labels[t.find("BTN:A SB:C").unwrap()], vec!["F", "C"]);
    }

    /// Sin flats: frente a un all-in que cubres, si detrás queda alguien que podría
    /// poner más, se resube all-in en vez de pagar; si no, se paga. Spot 10/12/5/15/7 bb con
    /// máximo 4 activos (29 nodos).
    #[test]
    fn shove_over_covered_allin() {
        let cfg = PreflopConfig { max_players_in_pot: 4, ..PreflopConfig::push_fold(vec![10.0, 12.0, 5.0, 15.0, 7.0], Payoff::ChipEv) };
        let t = build_tree(cfg);
        assert_eq!(t.actions.len(), 29);
        assert_eq!(t.labels[t.find("HJ:A").unwrap()], vec!["F", "A"], "el CO cubre y la SB detrás tiene más");
        assert_eq!(t.labels[t.find("HJ:F CO:F BTN:A").unwrap()], vec!["F", "A"], "la SB cubre y la BB puede poner más");
        assert_eq!(t.labels[t.find("HJ:F CO:A BTN:C").unwrap()], vec!["F", "C"], "la SB cubre pero la BB (7) no llega a 12");
    }

    #[test]
    fn per_seat_opens_and_ip_oop_reraises() {
        let mut cfg = PreflopConfig::standard(vec![40.0; 4], Payoff::ChipEv);
        cfg.max_players_in_pot = 2;
        cfg.open_sizes_by_seat = vec![Some(vec![RaiseSize::bb(2.2)]), None, None, None];
        cfg.reraise_mults_ip = Some(vec![RaiseSize::mult(3.0)]);
        cfg.reraise_mults_oop = Some(vec![RaiseSize::mult(4.0)]);
        let t = build_tree(cfg);
        assert_eq!(t.labels[0], vec!["F", "R2.2", "A"]); // CO con su tamaño propio
        // BTN (IP frente al CO) 3-betea 3×; la SB (OOP) 4×.
        assert!(t.labels[t.find("CO:R2.2").unwrap()].contains(&"R6.6".to_string()));
        assert!(t.labels[t.find("CO:R2.2 BTN:F").unwrap()].contains(&"R8.8".to_string()));
    }

    #[test]
    fn prune_removes_one_action_at_one_node() {
        let mut cfg = PreflopConfig::standard(vec![40.0; 3], Payoff::ChipEv);
        cfg.open_sizes = vec![RaiseSize::bb(2.0), RaiseSize::bb(2.5)];
        let t = build_tree(cfg.clone());
        assert_eq!(t.labels[0], vec!["F", "R2", "R2.5", "A"]);
        // Quita solo el R2.5 de la raíz: el resto del árbol (incluidos otros nodos con la
        // misma ruta de etiqueta pero en otro sitio) no se entera.
        cfg.pruned = vec![("".to_string(), "R2.5".to_string())];
        let t = build_tree(cfg.clone());
        assert_eq!(t.labels[0], vec!["F", "R2", "A"]);
        // También a un nodo que no es la raíz, sin tocar los hermanos de otras rutas.
        let bb = t.labels[t.find("BTN:R2").unwrap()].clone();
        let mut cfg2 = cfg.clone();
        cfg2.pruned.push(("BTN:R2".to_string(), "A".to_string()));
        let t2 = build_tree(cfg2);
        assert_eq!(t2.labels[t2.find("BTN:R2").unwrap()].len(), bb.len() - 1);
        assert!(!t2.labels[t2.find("BTN:R2").unwrap()].contains(&"A".to_string()));
        assert!(t2.labels[t2.find("BTN:F").unwrap()].contains(&"A".to_string()), "no debe afectar a otras rutas");
    }

    #[test]
    #[should_panic(expected = "--prune leaves node")]
    fn prune_all_actions_at_a_node_panics() {
        let mut cfg = PreflopConfig::standard(vec![40.0; 3], Payoff::ChipEv);
        cfg.pruned = vec![("".to_string(), "F".to_string()), ("".to_string(), "R2".to_string()), ("".to_string(), "R2.5".to_string()), ("".to_string(), "A".to_string())];
        build_tree(cfg);
    }

    #[test]
    fn size_grammar() {
        assert_eq!(RaiseSize::parse("2.5bb", true).unwrap(), RaiseSize::bb(2.5));
        assert_eq!(RaiseSize::parse("3x", false).unwrap(), RaiseSize::mult(3.0));
        assert_eq!(RaiseSize::parse("2.5bb + 1bb", false).unwrap(), RaiseSize { base: SizeBase::Bb(2.5), per_caller: 1.0 });
        assert_eq!(RaiseSize::parse("ai", false).unwrap().base, SizeBase::AllIn);
        assert_eq!(RaiseSize::parse("75%", false).unwrap().base, SizeBase::PotPct(0.75));
        assert!(RaiseSize::parse("abc", false).is_err());
        // "+1bb por limper": con un limper delante, el open es 3.5 en vez de 2.5.
        let mut cfg = PreflopConfig::standard(vec![40.0; 4], Payoff::ChipEv);
        cfg.allow_limp = true;
        cfg.open_sizes = vec![RaiseSize::parse("2.5bb+1bb", false).unwrap()];
        let t = build_tree(cfg);
        assert!(t.labels[0].contains(&"R2.5".to_string()));
        assert!(t.labels[t.find("CO:C").unwrap()].contains(&"R3.5".to_string()), "{:?}", t.labels[t.find("CO:C").unwrap()]);
        // "3x + 1x por caller": 3-bet de 7.5 sin callers, 10 con un caller.
        let mut cfg = PreflopConfig::standard(vec![40.0; 4], Payoff::ChipEv);
        cfg.reraise_mults = vec![RaiseSize::parse("3x+1x", true).unwrap()];
        let t = build_tree(cfg);
        assert!(t.labels[t.find("CO:R2.5").unwrap()].contains(&"R7.5".to_string()));
        assert!(t.labels[t.find("CO:R2.5 BTN:C").unwrap()].contains(&"R10".to_string()), "{:?}", t.labels[t.find("CO:R2.5 BTN:C").unwrap()]);
        // Pot %: 3-bet al 100 % del bote frente a un open de 2.5 de la SB: bote 3.5, call
        // 1.5 ⇒ subir a 2.5 + (3.5 + 1.5) = 7.5.
        let mut cfg = PreflopConfig::standard(vec![40.0; 2], Payoff::ChipEv);
        cfg.reraise_mults = vec![RaiseSize::parse("100%", true).unwrap()];
        cfg.sb_open_sizes = vec![RaiseSize::bb(2.5)];
        let t = build_tree(cfg);
        assert!(t.labels[t.find("SB:R2.5").unwrap()].contains(&"R7.5".to_string()), "{:?}", t.labels[t.find("SB:R2.5").unwrap()]);
    }

    #[test]
    fn fgs_utilities_conserve_prize_pool() {
        use crate::eval::equity::{EquityMatrix, RangeEquity};
        use crate::math::fgs::{Fgs, FgsMode, PushFoldCharts, CHART_MAX_PLAYERS};
        let charts = PushFoldCharts {
            x_grid: vec![2.0, 50.0],
            push: (2..=CHART_MAX_PLAYERS).map(|n| vec![vec![0.3f32; n]; 2]).collect(),
            call: (2..=CHART_MAX_PLAYERS).map(|n| vec![vec![0.2f32; n * n]; 2]).collect(),
        };
        let m = EquityMatrix::monte_carlo(200, 1, 4);
        let fgs = Fgs { depth: 1, sb: 0.5, bb: 1.0, ante: 0.1, bb_ante: false, prune: 1e-6, charts: Arc::new(charts), ranges: Arc::new(RangeEquity::from_matrix(&m, 20)), field: None, mode: FgsMode::Charts, nash_iters: (200, 50), cache: Arc::default(), three: Arc::default(), next_level: None };
        let cfg = PreflopConfig {
            ante: 0.1,
            offtable_stacks: vec![20.0],
            fgs: Some(Arc::new(fgs)),
            ..PreflopConfig::push_fold(vec![8.0, 12.0, 5.0], Payoff::Icm { payouts: vec![50.0, 30.0, 20.0] })
        };
        let t = build_tree(cfg);
        // Sin el jugador de otra mesa en la tabla de utilidades, la suma de la mesa es el
        // prize pool menos su equity: basta con comprobar que nunca lo excede y que varía.
        for o in t.utils.chunks(3) {
            let s: f32 = o.iter().sum();
            assert!(s > 0.0 && s < 100.0 + 1e-3, "{o:?}");
        }
    }

    #[test]
    fn four_way_pots_and_three_way_flops() {
        // 4 jugadores all-in: 24 órdenes, fichas conservadas en cada uno.
        let cfg = PreflopConfig { max_players_in_pot: 4, ..PreflopConfig::push_fold(vec![5.0, 10.0, 15.0, 20.0], Payoff::ChipEv) };
        let t = build_tree(cfg);
        for term in t.terminals.iter().filter(|x| x.kind == TerminalKind::MultiwayN) {
            assert_eq!(t.pot_players(term).len(), 4);
            let u = t.terminal_utils(term);
            assert_eq!(u.len(), 24 * 4);
            for o in u.chunks(4) {
                assert!((o.iter().sum::<f32>() - 50.0).abs() < 1e-3);
            }
        }
        // CO 5, BTN 10, SB 15 all-in y la BB paga 15: orden [0,1,2,3] ⇒ cada uno gana su
        // bote lateral y la BB conserva los 5 que no arriesgó.
        let term = t.terminals.iter().find(|x| x.kind == TerminalKind::MultiwayN && x.contributions[..4] == [500, 1000, 1500, 1500]).unwrap();
        let u = t.terminal_utils(term);
        assert_eq!(&u[..4], &[20.0, 15.0, 10.0, 5.0]);
        // Orden [3,2,1,0] (índice 23): la BB lo gana todo.
        assert_eq!(&u[23 * 4..24 * 4], &[0.0, 0.0, 0.0, 50.0]);
        // Con postflop, los flops a 3 son hojas multiway.
        let cfg = PreflopConfig { allow_limp: true, ..PreflopConfig::standard(vec![40.0; 4], Payoff::ChipEv) };
        let t = build_tree(cfg);
        assert!(t.leaves.iter().any(|l| l.multi.is_some()));
    }

    #[test]
    fn three_way_side_pots() {
        // BTN 5, SB 10, BB 20 all-in; outcome "BTN > SB > BB".
        let t = build_tree(PreflopConfig { shove_over_allin: false, ..PreflopConfig::push_fold(vec![5.0, 10.0, 20.0], Payoff::ChipEv) });
        let term = t.terminals.iter().find(|x| x.kind == TerminalKind::Multiway && x.contributions[..3] == [500, 1000, 2000]).expect("all-in a 3");
        let u = term.utilities(&t.utils);
        assert_eq!(term.players, [0, 1, 2]);
        // Orden 0 = [0,1,2]: BTN gana el principal (15), SB el lateral (10), BB recupera 10.
        assert_eq!(&u[..3], &[15.0, 10.0, 10.0]);
        // Orden 5 = [2,1,0]: BB lo gana todo.
        assert_eq!(&u[15..18], &[0.0, 0.0, 35.0]);
        for o in u.chunks(3) {
            assert!((o.iter().sum::<f32>() - 35.0).abs() < 1e-4);
        }
    }

    #[test]
    fn pko_side_pot_bounty_goes_to_top_pot_winner() {
        let payoff = Payoff::Pko { payouts: vec![50.0, 30.0, 20.0, 0.0], bounties: vec![10.0; 4], params: PkoParams::default() };
        let cfg = PreflopConfig { offtable_stacks: vec![40.0], shove_over_allin: false, ..PreflopConfig::push_fold(vec![5.0, 10.0, 20.0], payoff) };
        let t = build_tree(cfg);
        let term = t.terminals.iter().find(|x| x.kind == TerminalKind::Multiway && x.contributions[..3] == [500, 1000, 2000]).unwrap();
        let u = term.utilities(&t.utils);
        // Orden [0,1,2]: nadie cae (BTN gana principal, SB lateral). Orden [1,0,2] = índice 2:
        // SB gana el principal y elimina a BTN → cobra 5 $ de cabeza.
        let (none, sb_ko) = (&u[0..3], &u[6..9]);
        assert!(sb_ko[1] > none[1] + 4.0, "{sb_ko:?} vs {none:?}");
        let total: f32 = u[..3].iter().sum();
        assert!(total < 100.0 + 40.0);
    }

    #[test]
    fn short_blind_all_in_keeps_pot_heads_up() {
        let cfg = PreflopConfig { max_players_in_pot: 2, ..PreflopConfig::push_fold(vec![10.0, 12.0, 0.8], Payoff::ChipEv) };
        let t = build_tree(cfg);
        for term in &t.terminals {
            assert!(term.alive_mask.count_ones() <= 2);
        }
    }
}
