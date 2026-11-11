//! # Subárbol postflop geométrico de cada hoja de flop
//!
//! Cada línea preflop que ve flop heads-up produce un [`FlopLeaf`] con un árbol rígido
//! construido con `abstraction::geometric`: en cada nodo solo hay 2 acciones
//! (Check|Bet o Fold|Call), el tamaño lo fija el SPR y, como mucho, hay una apuesta por
//! calle. El árbol típico tiene 30-120 nodos.
//!
//! Los nodos de decisión llevan la **plantilla** de su `InfosetKey` (bucket = 0): el MCCFR
//! hace `key | bucket(combo)` para obtener el infoset con memoria imperfecta. Como la clave
//! no incluye el índice de la hoja, hojas distintas con igual (pot class, SPR bin, calle,
//! apuestas, posición) comparten infosets: esa es la fusión.
//!
//! Utilidades: cada terminal guarda un offset en la tabla global `U[outcome][seat]`
//! (fold: 1 outcome; showdown: 3 outcomes = gana OOP, gana IP, empate), calculadas por el
//! callback de payoff (ICM/PKO/cEV) al construir.

use crate::abstraction::geometric::{postflop_actions, GeometricConfig, PostflopActions, PostflopSpot, Street};
use crate::abstraction::recall::{spr_bin, InfosetKey, KeyFields, PotClass};
use crate::math::Elimination;
use super::nodes::{factorial, nth_perm};

pub const OOP: u8 = 0;
pub const IP: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PfNode {
    /// `children[0]` = Check o Fold, `children[1]` = Bet o Call.
    Decision { role: u8, street: Street, key: u64, facing: bool, children: [u32; 2], amount: f32 },
    Fold { winner: u8, util: u32 },
    Showdown { util: u32 },
}

#[derive(Clone, Debug)]
pub struct FlopLeaf {
    pub oop: u8,
    pub ip: u8,
    /// Bote al llegar al flop (incluye dinero muerto) y stack efectivo detrás.
    pub pot: f64,
    pub eff_stack: f64,
    pub pot_class: PotClass,
    /// Offset del bloque de 3 outcomes del check-down (gana OOP, gana IP, empate).
    pub checkdown_util: u32,
    /// Subárbol geométrico (vacío en modo check-down). Raíz = índice 0.
    pub nodes: Vec<PfNode>,
    /// Hoja a 3 jugadores (flop multiway con MCCFR). Si es `Some`, `oop`/`ip`/`nodes`
    /// no se usan.
    pub multi: Option<MultiLeaf>,
}

/// Nodo del subárbol postflop multiway (3 o 4 jugadores). Posiciones `0..k` en orden de
/// actuación postflop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PfNode3 {
    /// `children[0]` = Check o Fold, `children[1]` = Bet o Call.
    Decision { player: u8, street: Street, key: u64, facing: bool, children: [u32; 2] },
    /// Fin de la mano con los jugadores de `alive` (máscara de posiciones, incluidos los que
    /// están all-in). Con `m` vivos, `m!` outcomes: uno por orden de fuerza de mejor a peor,
    /// en código de Lehmer ([`super::nodes::nth_perm`]) sobre la lista de vivos en orden de
    /// posición. Con botes laterales el reparto depende del orden entero, no solo del
    /// ganador. Los empates se tratan como desempate al azar.
    End { alive: u8, util: u32 },
}

#[derive(Clone, Debug)]
pub struct MultiLeaf {
    /// Asientos en orden de actuación postflop (3 o 4).
    pub seats: Vec<u8>,
    pub nodes: Vec<PfNode3>,
}

/// Jugadores máximos de una hoja multiway con apuestas.
pub const MAX_MULTI: usize = 4;

/// Bits extra de las claves postflop para los raises (decisiones binarias encadenadas):
/// `STAGE_CONTINUE` = segunda decisión (call o raise) tras elegir "continuar" frente a una
/// apuesta; `FACING_RAISE` = se responde a un raise.
pub const STAGE_CONTINUE: u64 = 1 << 36;
pub const FACING_RAISE: u64 = 1 << 37;
/// Tercera etapa: la apuesta o el raise geométrico frente al all-in (con SPR bajo).
pub const STAGE_ALLIN: u64 = 1 << 38;

/// Clave de infoset multiway: la de `recall` con `PotClass::Multiway` más posición
/// (bits 31-32), jugadores vivos − 2 (bits 33 y 35) y si ya hay un call delante (bit 34).
/// (Al construir el árbol se sustituye por la clave de camino, ver `tree::builder`.)
pub fn multi_key(street: Street, spr: u8, bets: u8, facing: bool, pos: u8, alive: u8, callers: u8) -> u64 {
    let base = InfosetKey::pack(KeyFields { bucket: 0, spr_bin: spr, bets_made: bets.min(3), facing, street, in_position: pos == 2, pot_class: PotClass::Multiway }).0;
    let a = alive.saturating_sub(2) as u64;
    base | (pos as u64 & 3) << 31 | (a & 1) << 33 | ((callers.min(1)) as u64) << 34 | (a >> 1 & 1) << 35
}

/// Payoff de una hoja multiway: utilidades de un final dado lo invertido postflop por
/// posición, la máscara de vivos y el orden de fuerza (posiciones, de mejor a peor; solo
/// los vivos). Quien la construye resuelve botes laterales y eliminaciones.
#[allow(missing_debug_implementations)]
pub struct LeafPayoffM<'a> {
    pub k: usize,
    pub utility: &'a dyn Fn(&[f64], u8, &[u8]) -> Vec<f32>,
}

impl LeafPayoffM<'_> {
    /// Bloque de outcomes de un final: `m!` utilidades (una por orden de los `m` vivos).
    pub fn push(&self, utils: &mut Vec<f32>, inv: &[f64], alive: u8) -> u32 {
        let off = utils.len() as u32;
        let list: Vec<u8> = (0..self.k as u8).filter(|&p| alive & (1 << p) != 0).collect();
        let m = list.len();
        for idx in 0..factorial(m) {
            let order: Vec<u8> = nth_perm(m, idx).iter().map(|&i| list[i as usize]).collect();
            utils.extend((self.utility)(inv, alive, &order));
        }
        off
    }
}

type Inv = [f64; MAX_MULTI];

/// Constructor del árbol multiway: cada posición con su stack (`cap`, lo que tiene detrás al
/// llegar al flop). Quien se queda sin fichas deja de actuar y los demás siguen apostando
/// en el bote lateral; una apuesta nunca supera lo que puede pagar el rival más largo.
struct BuilderM<'a> {
    nodes: Vec<PfNode3>,
    geo: &'a GeometricConfig,
    k: usize,
    cap: Inv,
    pay: &'a LeafPayoffM<'a>,
    utils: &'a mut Vec<f32>,
}

const EPS: f64 = 1e-9;

impl BuilderM<'_> {
    fn reserve(&mut self) -> u32 {
        self.nodes.push(PfNode3::End { alive: 0, util: 0 });
        (self.nodes.len() - 1) as u32
    }

    fn end(&mut self, inv: Inv, alive: u8) -> u32 {
        let util = self.pay.push(self.utils, &inv[..self.k], alive);
        let i = self.reserve();
        self.nodes[i as usize] = PfNode3::End { alive, util };
        i
    }

    fn rem(&self, inv: &Inv, p: usize) -> f64 {
        self.cap[p] - inv[p]
    }

    /// Vivos que aún tienen fichas (los que pueden actuar).
    fn actors(&self, inv: &Inv, alive: u8) -> u8 {
        (0..self.k).filter(|&p| alive & (1 << p) != 0 && self.rem(inv, p) > EPS).map(|p| 1u8 << p).sum()
    }

    /// Lo que `p` puede poner de más con sentido: su stack, sin pasar de lo que puede
    /// llegar a poner en total el rival vivo más largo.
    fn reach(&self, inv: &Inv, alive: u8, p: usize) -> f64 {
        let top = (0..self.k).filter(|&q| q != p && alive & (1 << q) != 0).map(|q| inv[q] + self.rem(inv, q)).fold(0.0, f64::max);
        self.rem(inv, p).min(top - inv[p]).max(0.0)
    }

    /// Stack efectivo de la calle: el segundo mayor de los que pueden actuar.
    fn eff(&self, inv: &Inv, act: u8) -> f64 {
        let mut r: Vec<f64> = (0..self.k).filter(|&p| act & (1 << p) != 0).map(|p| self.rem(inv, p)).collect();
        r.sort_by(|a, b| b.total_cmp(a));
        r.get(1).copied().unwrap_or(0.0)
    }

    fn next_after(&self, from: usize, mask: u8) -> Option<usize> {
        (1..=self.k).map(|d| (from + d) % self.k).find(|&q| mask & (1 << q) != 0)
    }

    /// `prev`: posición del último agresor de la calle anterior (o del preflop): sin donk,
    /// nadie abre la calle antes de que él haya actuado.
    fn street_start(&mut self, street: Street, pot: f64, inv: Inv, alive: u8, bets: u8, prev: Option<u8>, agg: u8) -> u32 {
        let act = self.actors(&inv, alive);
        // Sin dos jugadores con fichas no hay apuestas posibles; tampoco pasada la última
        // calle con apuestas para este número de jugadores.
        // Cuentan los jugadores con fichas (no los all-in) para el check-down forzado.
        if act.count_ones() < 2 || street as u8 > self.geo.last_street(act.count_ones() as usize) {
            return self.end(inv, alive);
        }
        let spr = spr_bin(pot, self.eff(&inv, act).max(EPS));
        // All-in extra: con el SPR del inicio de la calle, antes de
        // ninguna apuesta, para toda la calle (también para los raises).
        let first = (0..self.k).find(|&p| act & (1 << p) != 0).unwrap() as u8;
        self.checks(street, pot, inv, alive, bets, spr, 0, first, prev, agg)
    }

    fn street_end(&mut self, street: Street, pot: f64, inv: Inv, alive: u8, bets: u8, aggr: Option<u8>, agg: u8) -> u32 {
        if alive.count_ones() == 1 {
            return self.end(inv, alive);
        }
        match street.next() {
            Some(next) => self.street_start(next, pot, inv, alive, bets, aggr, agg),
            None => self.end(inv, alive),
        }
    }

    /// Cadena binaria de tamaños (el menor frente al resto).
    fn size_chain(&mut self, player: u8, street: Street, key: u64, facing: bool, kids: &[u32]) -> u32 {
        match kids {
            [] => unreachable!("sin tamaños"),
            [x] => *x,
            [x, rest @ ..] => {
                let idx = self.reserve();
                let r = self.size_chain(player, street, key, facing, rest);
                self.nodes[idx as usize] = PfNode3::Decision { player, street, key: key | STAGE_ALLIN, facing, children: [*x, r] };
                idx
            }
        }
    }

    /// Tamaños (lo que pone el jugador, sin redondear el all-in) ordenados y sin repetidos.
    fn sizes(&self, mut v: Vec<f64>, pot: f64, stack: f64) -> Vec<f64> {
        for x in v.iter_mut() {
            if *x >= self.geo.jam_threshold * stack {
                *x = stack;
            }
            *x = x.min(stack);
        }
        v.retain(|&x| x > EPS);
        v.sort_by(|a, b| a.total_cmp(b));
        v.dedup_by(|a, b| (*a - *b).abs() <= 0.02 * pot);
        v
    }

    /// Ronda sin apuesta: cada jugador con fichas, en orden, pasa o apuesta (geométrica,
    /// tamaños extra del flop, all-in con SPR bajo).
    #[allow(clippy::too_many_arguments)]
    fn checks(&mut self, street: Street, pot: f64, inv: Inv, alive: u8, bets: u8, spr: u8, acted: u8, pos: u8, prev: Option<u8>, agg: u8) -> u32 {
        let act = self.actors(&inv, alive);
        if acted & act == act {
            return self.street_end(street, pot, inv, alive, bets, None, agg);
        }
        let p = pos as usize;
        let next_pos = self.next_after(p, act & !acted & !(1 << p)).unwrap_or(p) as u8;
        let stack = self.reach(&inv, alive, p);
        // Donk limitado: sí puede abrir la calle quien apostó o subió en alguna
        // calle anterior (preflop incluido).
        let is_donk = prev.is_some_and(|a| a != pos && act & (1 << a) != 0 && acted & (1 << a) == 0);
        let donk_blocked = !self.geo.donk && is_donk && !(self.geo.donk_limited && agg & (1 << pos) != 0);
        let spot = PostflopSpot { street, pot, eff_stack: stack, bets_made: bets, facing: 0.0 };
        let mut amounts: Vec<f64> = Vec::new();
        if !donk_blocked && stack > EPS && bets < self.geo.max_bets {
            // Flop: tamaños de c-bet / donk / generales en vez del geométrico.
            let flop = if street == Street::Flop { self.geo.flop_sizes(prev == Some(pos), is_donk) } else { &[] };
            if flop.is_empty() {
                if let PostflopActions::CheckOrBet(bet) = postflop_actions(&spot, self.geo) {
                    amounts.push(bet.amount);
                }
            } else {
                amounts.extend(flop.iter().map(|f| f * pot));
            }
            // All-in extra con el SPR de esta decisión.
            if self.geo.low_spr(stack, pot) {
                amounts.push(stack);
            }
        }
        let amounts = self.sizes(amounts, pot, stack);
        let check = |this: &mut Self| this.checks(street, pot, inv, alive, bets, spr, acted | (1 << p), next_pos, prev, agg);
        if amounts.is_empty() {
            return check(self);
        }
        let idx = self.reserve();
        let key = multi_key(street, spr, bets, false, pos, alive.count_ones() as u8, 0);
        let check = check(self);
        let kids: Vec<u32> = amounts
            .iter()
            .map(|&amount| {
                let mut inv2 = inv;
                inv2[p] += amount;
                let pending = self.actors(&inv2, alive) & !(1 << p);
                self.responses(street, pot + amount, inv2, alive, bets + 1, spr, pos, pending, 0, false, agg | (1 << pos))
            })
            .collect();
        let bet = self.size_chain(pos, street, key, false, &kids);
        self.nodes[idx as usize] = PfNode3::Decision { player: pos, street, key, facing: false, children: [check, bet] };
        idx
    }

    /// Respuestas a una apuesta: cada jugador con fichas tras el apostador (en orden
    /// circular) foldea, paga (all-in por menos si no le llega) o sube; tras un raise, todos
    /// los demás con fichas vuelven a responder.
    #[allow(clippy::too_many_arguments)]
    fn responses(&mut self, street: Street, pot: f64, inv: Inv, alive: u8, bets: u8, spr: u8, bettor: u8, pending: u8, callers: u8, raised: bool, agg: u8) -> u32 {
        if pending == 0 {
            return self.street_end(street, pot, inv, alive, bets, Some(bettor), agg);
        }
        let p = self.next_after(bettor as usize, pending).unwrap();
        let target = inv[bettor as usize];
        let idx = self.reserve();
        let flags = if raised { FACING_RAISE } else { 0 };
        let key = multi_key(street, spr, bets, true, p as u8, alive.count_ones() as u8, callers) | flags;
        let alive_f = alive & !(1 << p);
        let fold = if alive_f.count_ones() == 1 { self.end(inv, alive_f) } else { self.responses(street, pot, inv, alive_f, bets, spr, bettor, pending & !(1 << p), callers, raised, agg) };
        let call_amt = (target - inv[p]).min(self.rem(&inv, p)).max(0.0);
        let mut inv2 = inv;
        inv2[p] += call_amt;
        let call = self.responses(street, pot + call_amt, inv2, alive, bets, spr, bettor, pending & !(1 << p), callers + 1, raised, agg);
        let behind = self.reach(&inv, alive, p);
        let can_raise = self.geo.raise_multiway && (!raised || self.geo.reraise) && bets < self.geo.max_bets && behind > call_amt + EPS;
        let mut amounts: Vec<f64> = Vec::new();
        if can_raise {
            let bets_left = (self.geo.max_bets as u32).saturating_sub(bets as u32);
            if street == Street::Flop && !self.geo.flop_raises.is_empty() {
                // Raise del flop en % del bote tras pagar.
                amounts.extend(self.geo.flop_raises.iter().map(|f| call_amt + f * (pot + call_amt)));
            } else if let Some(add) = self.geo.raise_total(call_amt, pot, behind, bets_left) {
                amounts.push(add);
            }
            if self.geo.low_spr(behind - call_amt, pot + call_amt) {
                amounts.push(behind);
            }
        }
        let amounts: Vec<f64> = self.sizes(amounts, pot, behind).into_iter().filter(|&a| a > call_amt + EPS).collect();
        let cont = if amounts.is_empty() {
            call
        } else {
            let kids: Vec<u32> = amounts
                .iter()
                .map(|&add| {
                    let mut inv3 = inv;
                    inv3[p] += add;
                    let others = self.actors(&inv3, alive) & !(1 << p);
                    self.responses(street, pot + add, inv3, alive, bets + 1, spr, p as u8, others, 0, true, agg | (1 << p))
                })
                .collect();
            let raise = self.size_chain(p as u8, street, key | STAGE_CONTINUE, true, &kids);
            let stage = self.reserve();
            self.nodes[stage as usize] = PfNode3::Decision { player: p as u8, street, key: key | STAGE_CONTINUE, facing: true, children: [call, raise] };
            stage
        };
        self.nodes[idx as usize] = PfNode3::Decision { player: p as u8, street, key, facing: true, children: [fold, cont] };
        idx
    }
}

/// Construye el subárbol de una hoja multiway (3 o 4 jugadores) con los stacks detrás de
/// cada posición (`cap`, en orden postflop) y la posición del agresor preflop (donk).
pub fn build_leaf_tree_multi(pot: f64, cap: &[f64], geo: &GeometricConfig, pay: &LeafPayoffM<'_>, utils: &mut Vec<f32>, aggressor: Option<u8>) -> Vec<PfNode3> {
    let k = cap.len();
    assert!((3..=MAX_MULTI).contains(&k) && pay.k == k, "hoja multiway de 3 o 4 jugadores");
    let geo = &GeometricConfig { max_bets: geo.max_bets.min(geo.max_bets_multiway), geo_size: geo.geo_size_multiway.or(geo.geo_size), ..geo.clone() };
    let mut c = [0.0; MAX_MULTI];
    c[..k].copy_from_slice(cap);
    let mut b = BuilderM { nodes: Vec::new(), geo, k, cap: c, pay, utils };
    b.street_start(Street::Flop, pot, [0.0; MAX_MULTI], ((1u16 << k) - 1) as u8, 0, aggressor, aggressor.map_or(0, |a| 1 << a));
    b.nodes
}

/// Cómo convertir un final de mano en utilidades: stacks tras el preflop (sin el bote) y
/// la función de payoff del torneo.
#[allow(missing_debug_implementations)]
pub struct LeafPayoff<'a> {
    pub base_post: Vec<f64>,
    pub oop: usize,
    pub ip: usize,
    pub pot: f64,
    pub utility: &'a dyn Fn(&[f64], &[Elimination]) -> Vec<f32>,
}

impl LeafPayoff<'_> {
    /// Añade a `utils` el bloque de outcomes de un final con inversiones postflop `inv`.
    /// `winner`: `Some(rol)` para fold (1 outcome), `None` para showdown (3 outcomes).
    pub fn push(&self, utils: &mut Vec<f32>, inv: [f64; 2], winner: Option<u8>) -> u32 {
        let off = utils.len() as u32;
        let pot = self.pot + inv[0] + inv[1];
        let seats = [self.oop, self.ip];
        let mut base = self.base_post.clone();
        base[self.oop] -= inv[0];
        base[self.ip] -= inv[1];
        let mut outcome = |w: Option<u8>| {
            let mut post = base.clone();
            let mut elims = Vec::new();
            match w {
                Some(r) => {
                    post[seats[r as usize]] += pot;
                    let loser = seats[1 - r as usize];
                    if post[loser] <= 1e-9 {
                        post[loser] = 0.0;
                        elims.push(Elimination { victim: loser as u8, winner_mask: 1 << seats[r as usize] });
                    }
                }
                None => {
                    post[self.oop] += pot * 0.5;
                    post[self.ip] += pot * 0.5;
                }
            }
            utils.extend((self.utility)(&post, &elims));
        };
        match winner {
            Some(r) => outcome(Some(r)),
            None => {
                outcome(Some(OOP));
                outcome(Some(IP));
                outcome(None);
            }
        }
        off
    }
}

struct Builder<'a> {
    nodes: Vec<PfNode>,
    geo: &'a GeometricConfig,
    pot_class: PotClass,
    eff: f64,
    pay: &'a LeafPayoff<'a>,
    utils: &'a mut Vec<f32>,
}

impl Builder<'_> {
    fn reserve(&mut self) -> u32 {
        self.nodes.push(PfNode::Showdown { util: 0 });
        (self.nodes.len() - 1) as u32
    }

    fn terminal(&mut self, inv: [f64; 2], winner: Option<u8>) -> u32 {
        let util = self.pay.push(self.utils, inv, winner);
        let idx = self.reserve();
        self.nodes[idx as usize] = match winner {
            Some(w) => PfNode::Fold { winner: w, util },
            None => PfNode::Showdown { util },
        };
        idx
    }

    fn key(&self, street: Street, spr: u8, bets: u8, facing: bool, role: u8) -> u64 {
        InfosetKey::pack(KeyFields { bucket: 0, spr_bin: spr, bets_made: bets, facing, street, in_position: role == IP, pot_class: self.pot_class }).0
    }

    /// `prev`: quién hizo la última agresión de la calle anterior (o del preflop, para el
    /// flop); sin donk, fuera de posición no puede apostar primero contra él.
    fn street_start(&mut self, street: Street, pot: f64, inv: [f64; 2], bets: u8, prev: Option<u8>, agg: u8) -> u32 {
        let behind = self.eff - inv[0].max(inv[1]);
        let spr = spr_bin(pot, behind.max(1e-9));
        self.node(street, OOP, pot, inv, bets, 0.0, spr, false, prev, None, agg)
    }

    fn after_check(&mut self, street: Street, role: u8, pot: f64, inv: [f64; 2], bets: u8, spr: u8, prev: Option<u8>, agg: u8) -> u32 {
        if role == OOP {
            return self.node(street, IP, pot, inv, bets, 0.0, spr, false, prev, None, agg);
        }
        match street.next() {
            // Calle pasada: nadie tiene la iniciativa en la siguiente.
            Some(next) => self.street_start(next, pot, inv, bets, None, agg),
            None => self.terminal(inv, None),
        }
    }

    /// Decisión binaria entre la acción agresiva "normal" `a` y el all-in `b` (all-in
    /// como opción extra con SPR bajo); con una sola, esa.
    fn aggressive(&mut self, role: u8, street: Street, key: u64, facing: bool, a: Option<(u32, f64)>, b: Option<(u32, f64)>) -> Option<u32> {
        match (a, b) {
            (Some((x, ax)), Some((y, _))) => {
                let idx = self.reserve();
                self.nodes[idx as usize] = PfNode::Decision { role, street, key: key | STAGE_ALLIN, facing, children: [x, y], amount: ax as f32 };
                Some(idx)
            }
            (Some((x, _)), None) | (None, Some((x, _))) => Some(x),
            (None, None) => None,
        }
    }

    /// Cadena binaria de tamaños frente a una apuesta (raises del flop configurados + all-in):
    /// el menor frente al resto.
    fn chain(&mut self, role: u8, street: Street, key: u64, facing: bool, kids: &[(u32, f64)]) -> u32 {
        match kids {
            [] => unreachable!("sin tamaños"),
            [(x, _)] => *x,
            [(x, ax), rest @ ..] => {
                let idx = self.reserve();
                let r = self.chain(role, street, key, facing, rest);
                self.nodes[idx as usize] = PfNode::Decision { role, street, key, facing, children: [*x, r], amount: *ax as f32 };
                idx
            }
        }
    }

    /// Elección entre varios tamaños como cadena de decisiones binarias: el menor frente al
    /// resto, y así sucesivamente.
    fn size_chain(&mut self, role: u8, street: Street, key: u64, kids: &[(u32, f64)]) -> u32 {
        match kids {
            [] => unreachable!("sin tamaños"),
            [(x, _)] => *x,
            [(x, ax), rest @ ..] => {
                let idx = self.reserve();
                let r = self.size_chain(role, street, key, rest);
                self.nodes[idx as usize] = PfNode::Decision { role, street, key: key | STAGE_ALLIN, facing: false, children: [*x, r], amount: *ax as f32 };
                idx
            }
        }
    }

    /// `raised`: la apuesta que se afronta ya es un raise. `prev`/`cur`: última agresión de
    /// la calle anterior y de esta (rol).
    #[allow(clippy::too_many_arguments)]
    fn node(&mut self, street: Street, role: u8, pot: f64, inv: [f64; 2], bets: u8, facing: f64, spr: u8, raised: bool, prev: Option<u8>, cur: Option<u8>, agg: u8) -> u32 {
        let r = role as usize;
        let behind = [self.eff - inv[0], self.eff - inv[1]];
        if facing > 0.0 {
            let idx = self.reserve();
            let flags = if raised { FACING_RAISE } else { 0 };
            let key = self.key(street, spr, bets.min(3), true, role) | flags;
            let fold = self.terminal(inv, Some(1 - role));
            let call = facing.min(behind[r]);
            let mut inv2 = inv;
            inv2[r] += call;
            let pot2 = pot + call;
            let all_in = self.eff - inv2[0].max(inv2[1]) <= 1e-9;
            let call_child = match street.next() {
                Some(next) if !all_in => self.street_start(next, pot2, inv2, bets, Some(1 - role), agg),
                _ => self.terminal(inv2, None),
            };
            // Fold/call/raise(/all-in) en decisiones binarias encadenadas: fold vs
            // continuar, call vs subir y, si hay dos tamaños, raise vs all-in.
            let bets_left = (self.geo.max_bets as u32).saturating_sub(bets as u32);
            let other_behind = behind[1 - r];
            let can_raise = (!raised || self.geo.reraise) && other_behind > 1e-9 && behind[r] > facing + 1e-9;
            // Raises: en el flop, los tamaños configurados (%
            // del bote tras pagar) en vez del geométrico; más el all-in si el SPR de esta
            // decisión, tras pagar, es bajo.
            let mut raises: Vec<f64> = Vec::new();
            if can_raise && bets_left > 0 {
                if street == Street::Flop && !self.geo.flop_raises.is_empty() {
                    raises.extend(self.geo.flop_raises.iter().map(|f| (facing + f * (pot + facing)).min(behind[r])));
                } else if let Some(a) = self.geo.raise_total(facing, pot, behind[r], bets_left) {
                    raises.push(a);
                }
                if self.geo.low_spr((behind[r] - facing).min(other_behind), pot + facing) {
                    raises.push(behind[r]);
                }
            }
            raises.retain(|&a| a > facing + 1e-9);
            raises.sort_by(|a, b| a.total_cmp(b));
            raises.dedup_by(|a, b| (*a - *b).abs() <= 0.02 * pot);
            let kids: Vec<(u32, f64)> = raises
                .iter()
                .map(|&add| {
                    let mut inv3 = inv;
                    inv3[r] += add;
                    (self.node(street, 1 - role, pot + add, inv3, bets + 1, add - facing, spr, true, prev, Some(role), agg | (1 << role)), add)
                })
                .collect();
            let (agg_a, agg_b) = match kids.len() {
                0 => (None, None),
                1 => (Some(kids[0]), None),
                _ => (Some(kids[0]), Some((self.chain(role, street, key | STAGE_CONTINUE | STAGE_ALLIN, true, &kids[1..]), kids[1].1))),
            };
            let cont = match self.aggressive(role, street, key | STAGE_CONTINUE, true, agg_a, agg_b) {
                Some(agg) => {
                    let stage = self.reserve();
                    self.nodes[stage as usize] = PfNode::Decision { role, street, key: key | STAGE_CONTINUE, facing: true, children: [call_child, agg], amount: call as f32 };
                    stage
                }
                None => call_child,
            };
            self.nodes[idx as usize] = PfNode::Decision { role, street, key, facing: true, children: [fold, cont], amount: call as f32 };
            return idx;
        }
        // Sin donk: fuera de posición no abre la calle contra el que tuvo la iniciativa.
        // Donk limitado: puede abrir la calle si apostó o subió en alguna calle
        // anterior (preflop incluido).
        let is_donk = role == OOP && cur.is_none() && prev == Some(IP);
        let donk_blocked = !self.geo.donk && is_donk && !(self.geo.donk_limited && agg & (1 << OOP) != 0);
        let spot = PostflopSpot { street, pot, eff_stack: behind[0].min(behind[1]), bets_made: bets, facing: 0.0 };
        let geo_bet = match postflop_actions(&spot, self.geo) {
            PostflopActions::CheckOrBet(bet) if !donk_blocked => Some(bet.amount.min(behind[r])),
            _ => None,
        };
        let stack = behind[0].min(behind[1]);
        let can_bet = !donk_blocked && bets < self.geo.max_bets && stack > 1e-9;
        // Flop: si hay tamaños de c-bet / donk / generales, sustituyen al geométrico.
        let flop: &[f64] = if street == Street::Flop && can_bet { self.geo.flop_sizes(prev == Some(role), is_donk) } else { &[] };
        let geo_bet = if flop.is_empty() { geo_bet } else { None };
        // All-in extra con el SPR de esta decisión (sin apuesta delante: el de la calle).
        let jam = (can_bet && self.geo.low_spr(stack, pot) && geo_bet.is_none_or(|a| a < stack - 1e-9)).then_some(stack);
        let flop_sizes: Vec<f64> = flop.iter().map(|f| (f * pot).min(stack)).collect();
        if geo_bet.is_none() && jam.is_none() && flop_sizes.is_empty() {
            return self.after_check(street, role, pot, inv, bets, spr, prev, agg);
        }
        let idx = self.reserve();
        let key = self.key(street, spr, bets.min(3), false, role);
        let check = self.after_check(street, role, pot, inv, bets, spr, prev, agg);
        let child = |this: &mut Self, amount: f64| {
            let mut inv2 = inv;
            inv2[r] += amount;
            (this.node(street, 1 - role, pot + amount, inv2, bets + 1, amount, spr, false, prev, Some(role), agg | (1 << role)), amount)
        };
        // Tamaños: el geométrico, los extra del flop y el all-in,
        // de menor a mayor y sin repetidos; los que casi vacían el stack son el all-in.
        let mut sizes: Vec<f64> = geo_bet.into_iter().collect();
        for x in flop_sizes {
            sizes.push(if x >= self.geo.jam_threshold * stack { stack } else { x });
        }
        sizes.extend(jam);
        sizes.sort_by(|a, b| a.total_cmp(b));
        sizes.dedup_by(|a, b| (*a - *b).abs() <= 0.02 * pot);
        let kids: Vec<(u32, f64)> = sizes.iter().map(|&x| child(self, x)).collect();
        let amount = kids.first().map_or(0.0, |x| x.1);
        let agg = self.size_chain(role, street, key, &kids);
        self.nodes[idx as usize] = PfNode::Decision { role, street, key, facing: false, children: [check, agg], amount: amount as f32 };
        idx
    }
}

/// Construye el subárbol geométrico de una hoja.
///
/// `aggressor`: quién subió el último preflop (`OOP`/`IP`), para la regla del donk.
pub fn build_leaf_tree(pot: f64, eff: f64, pot_class: PotClass, geo: &GeometricConfig, pay: &LeafPayoff<'_>, utils: &mut Vec<f32>) -> Vec<PfNode> {
    build_leaf_tree_with(pot, eff, pot_class, geo, pay, utils, None)
}

/// [`build_leaf_tree`] con el agresor preflop.
pub fn build_leaf_tree_with(pot: f64, eff: f64, pot_class: PotClass, geo: &GeometricConfig, pay: &LeafPayoff<'_>, utils: &mut Vec<f32>, aggressor: Option<u8>) -> Vec<PfNode> {
    let mut b = Builder { nodes: Vec::new(), geo, pot_class, eff, pay, utils };
    b.street_start(Street::Flop, pot, [0.0, 0.0], 0, aggressor, aggressor.map_or(0, |a| 1 << a));
    b.nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chip_utility(post: &[f64], _: &[Elimination]) -> Vec<f32> {
        post.iter().map(|&x| x as f32).collect()
    }

    #[test]
    fn raises_as_chained_binary_decisions() {
        let geo = GeometricConfig { raise_mult: Some(3.0), raise_multiway: true, ..GeometricConfig::default() };
        let pay = LeafPayoff { base_post: vec![40.0, 40.0], oop: 1, ip: 0, pot: 5.0, utility: &chip_utility };
        let mut u1 = Vec::new();
        let plain = build_leaf_tree(5.0, 40.0, PotClass::SingleRaised, &GeometricConfig::default(), &pay, &mut u1);
        let mut u2 = Vec::new();
        let with_raise = build_leaf_tree(5.0, 40.0, PotClass::SingleRaised, &geo, &pay, &mut u2);
        assert!(with_raise.len() > plain.len());
        for chunk in u2.chunks(2) {
            assert!((chunk[0] + chunk[1] - 85.0).abs() < 1e-3);
        }
        let stages = with_raise.iter().filter(|n| matches!(n, PfNode::Decision { key, .. } if key & STAGE_CONTINUE != 0)).count();
        let vs_raise = with_raise.iter().filter(|n| matches!(n, PfNode::Decision { key, .. } if key & FACING_RAISE != 0)).count();
        assert!(stages > 0 && vs_raise > 0);
        // No hay re-raise: frente a un raise no existe segunda etapa.
        assert!(!with_raise.iter().any(|n| matches!(n, PfNode::Decision { key, .. } if key & FACING_RAISE != 0 && key & STAGE_CONTINUE != 0)));
        // A 3 jugadores.
        let res = side_pots(&[30.0; 3], 6.0);
        let pay3 = LeafPayoffM { k: 3, utility: &res };
        let mut u3 = Vec::new();
        let n3 = build_leaf_tree_multi(6.0, &[30.0; 3], &geo, &pay3, &mut u3, None);
        for chunk in u3.chunks(3) {
            assert!((chunk.iter().sum::<f32>() - 96.0).abs() < 1e-3);
        }
        assert!(n3.iter().any(|n| matches!(n, PfNode3::Decision { key, .. } if key & FACING_RAISE != 0)));
    }

    /// Postflop por defecto: fuera de posición no abre contra el agresor preflop,
    /// con SPR ≤ 5 el all-in es una opción aparte, hay re-raises y las fichas se conservan.
    #[test]
    fn default_postflop_rules() {
        let geo = GeometricConfig::classic();
        let pay = LeafPayoff { base_post: vec![20.0, 20.0], oop: 1, ip: 0, pot: 5.0, utility: &chip_utility };
        let mut u = Vec::new();
        let nodes = build_leaf_tree_with(5.0, 20.0, PotClass::SingleRaised, &geo, &pay, &mut u, Some(IP));
        for chunk in u.chunks(2) {
            assert!((chunk[0] + chunk[1] - 45.0).abs() < 1e-3);
        }
        // La raíz es del IP: el OOP solo puede pasar.
        assert!(matches!(nodes[0], PfNode::Decision { role: IP, .. }), "{:?}", nodes[0]);
        assert!(nodes.iter().any(|n| matches!(n, PfNode::Decision { key, .. } if key & STAGE_ALLIN != 0)));
        assert!(nodes.iter().any(|n| matches!(n, PfNode::Decision { key, facing: true, .. } if key & FACING_RAISE != 0 && key & STAGE_CONTINUE != 0)));
        // Con el agresor fuera de posición (o sin agresor), el OOP abre.
        let mut u2 = Vec::new();
        let n2 = build_leaf_tree_with(5.0, 20.0, PotClass::SingleRaised, &geo, &pay, &mut u2, Some(OOP));
        assert!(matches!(n2[0], PfNode::Decision { role: OOP, .. }));
    }

    /// Reparto de referencia para los tests (chip EV): fichas tras la mano de cada
    /// posición con botes laterales, dados los stacks detrás al llegar al flop (`cap`) y el
    /// bote del preflop repartido a partes iguales.
    fn side_pots(cap: &[f64], pot: f64) -> impl Fn(&[f64], u8, &[u8]) -> Vec<f32> {
        let cap = cap.to_vec();
        move |inv: &[f64], alive: u8, order: &[u8]| {
            let k = cap.len();
            let mut post: Vec<f64> = (0..k).map(|p| cap[p] - inv[p]).collect();
            let mut left: Vec<f64> = inv.to_vec();
            let mut dead = pot;
            // Por niveles: el mejor vivo con fichas en el nivel se lleva su porción.
            loop {
                let elig: Vec<usize> = (0..k).filter(|&p| alive & (1 << p) != 0 && left[p] > 1e-12).collect();
                let lvl = elig.iter().map(|&p| left[p]).fold(f64::INFINITY, f64::min);
                if elig.is_empty() {
                    break;
                }
                let mut slice = dead;
                dead = 0.0;
                for p in 0..k {
                    let x = left[p].min(lvl);
                    slice += x;
                    left[p] -= x;
                }
                let w = *order.iter().find(|&&p| elig.contains(&(p as usize))).unwrap() as usize;
                post[w] += slice;
            }
            // Lo que pusieron los retirados por encima del último nivel, y el bote si no hay
            // nadie con fichas en juego (todos pasan).
            let w = order[0] as usize;
            post[w] += dead + left.iter().sum::<f64>();
            post.iter().map(|&x| x as f32).collect()
        }
    }

    #[test]
    fn three_way_leaf_conserves_chips() {
        let res = side_pots(&[30.0; 3], 6.0);
        let pay = LeafPayoffM { k: 3, utility: &res };
        let mut utils = Vec::new();
        let nodes = build_leaf_tree_multi(6.0, &[30.0; 3], &GeometricConfig::default(), &pay, &mut utils, None);
        assert!(nodes.len() > 20, "{}", nodes.len());
        for chunk in utils.chunks(3) {
            assert!((chunk.iter().sum::<f32>() - 96.0).abs() < 1e-3);
        }
        let mut keys = std::collections::HashSet::new();
        for n in &nodes {
            match n {
                PfNode3::Decision { children, key, .. } => {
                    assert!(children.iter().all(|&c| (c as usize) < nodes.len()));
                    keys.insert(*key);
                }
                PfNode3::End { alive, .. } => assert!(*alive != 0),
            }
        }
        assert!(keys.len() > 5);
    }

    /// Stacks desiguales: cuando el corto se queda sin fichas los otros siguen apostando
    /// (más líneas que con todos al stack del corto), las fichas se conservan en cada
    /// orden de manos y cada final tiene `m!` outcomes.
    #[test]
    fn side_pot_betting_with_unequal_stacks() {
        let geo = GeometricConfig::classic();
        for cap in [vec![8.0, 25.0, 30.0], vec![30.0, 8.0, 25.0], vec![8.0, 25.0, 30.0, 18.0]] {
            let k = cap.len();
            let res = side_pots(&cap, 6.0);
            let pay = LeafPayoffM { k, utility: &res };
            let mut utils = Vec::new();
            let nodes = build_leaf_tree_multi(6.0, &cap, &geo, &pay, &mut utils, None);
            let total = cap.iter().sum::<f64>() as f32 + 6.0;
            for chunk in utils.chunks(k) {
                assert!((chunk.iter().sum::<f32>() - total).abs() < 1e-3, "{cap:?} {chunk:?}");
            }
            for n in &nodes {
                if let PfNode3::End { alive, util } = n {
                    let m = alive.count_ones() as usize;
                    assert!(*util as usize + factorial(m) * k <= utils.len());
                }
            }
            let short = vec![8.0; k];
            let res2 = side_pots(&short, 6.0);
            let pay2 = LeafPayoffM { k, utility: &res2 };
            let mut u2 = Vec::new();
            let flat = build_leaf_tree_multi(6.0, &short, &geo, &pay2, &mut u2, None);
            assert!(nodes.len() > flat.len(), "{cap:?}: {} vs {}", nodes.len(), flat.len());
            // Algún final donde el corto pierde todo y otro se lleva más que el bote principal.
            assert!(utils.chunks(k).any(|c| c.iter().any(|&x| x.abs() < 1e-4)));
        }
    }

    /// Check-down forzado por número de jugadores: a 4 con "1" solo se apuesta en el
    /// flop; la línea de checks no tiene decisiones fuera del flop. Con "0", sin apuestas.
    #[test]
    fn checkdown_after_by_players() {
        let cap = [20.0; 4];
        let res = side_pots(&cap, 8.0);
        let pay = LeafPayoffM { k: 4, utility: &res };
        let geo = GeometricConfig { checkdown_after: vec![3, 3, 1], ..GeometricConfig::classic() };
        let mut utils = Vec::new();
        let nodes = build_leaf_tree_multi(8.0, &cap, &geo, &pay, &mut utils, None);
        let mut ni = 0usize;
        while let PfNode3::Decision { street, children, .. } = nodes[ni] {
            assert_eq!(street, Street::Flop);
            ni = children[0] as usize;
        }
        assert!(nodes.iter().any(|n| matches!(n, PfNode3::Decision { street: Street::Turn, .. })), "a 3 tras un fold sí se sigue");
        let geo0 = GeometricConfig { checkdown_after: vec![3, 3, 0], ..GeometricConfig::classic() };
        let mut u0 = Vec::new();
        let n0 = build_leaf_tree_multi(8.0, &cap, &geo0, &pay, &mut u0, None);
        assert_eq!(n0.len(), 1);
        assert_eq!(u0.len(), 24 * 4);
    }

    /// Apuestas del flop que puede hacer el que abre la calle: los nodos de respuesta (frente
    /// a apuesta) a los que lleva su opción agresiva, recorriendo la cadena de tamaños.
    fn opening_bets(nodes: &[PfNode], root: u32) -> Vec<f64> {
        let PfNode::Decision { role, children, .. } = nodes[root as usize] else { panic!("raíz sin decisión") };
        let mut out = Vec::new();
        let mut stack = vec![children[1]];
        while let Some(i) = stack.pop() {
            match nodes[i as usize] {
                PfNode::Decision { role: r, facing: false, children, .. } if r == role => stack.extend(children),
                PfNode::Decision { facing: true, amount, .. } => out.push(amount as f64),
                _ => {}
            }
        }
        out.sort_by(|a, b| a.total_cmp(b));
        out
    }

    /// Con tamaños de flop configurados, el flop usa solo esos
    /// (sin el geométrico); en el turn vuelve el geométrico.
    #[test]
    fn flop_sizes_replace_geometric() {
        let pay = LeafPayoff { base_post: vec![40.0, 40.0], oop: 1, ip: 0, pot: 5.0, utility: &chip_utility };
        let geo = GeometricConfig { flop_bets: vec![0.33, 0.66], ..GeometricConfig::classic() };
        let mut u = Vec::new();
        let nodes = build_leaf_tree(5.0, 40.0, PotClass::SingleRaised, &geo, &pay, &mut u);
        let bets = opening_bets(&nodes, 0);
        assert_eq!(bets.len(), 2, "{bets:?}");
        assert!((bets[0] - 1.65).abs() < 1e-3 && (bets[1] - 3.3).abs() < 1e-3, "{bets:?}");
        // Sin tamaños configurados: un único tamaño, el geométrico.
        let mut u2 = Vec::new();
        let plain = build_leaf_tree(5.0, 40.0, PotClass::SingleRaised, &GeometricConfig::classic(), &pay, &mut u2);
        let geo_bets = opening_bets(&plain, 0);
        assert_eq!(geo_bets.len(), 1, "{geo_bets:?}");
        assert!((geo_bets[0] - 1.65).abs() > 0.1 && (geo_bets[0] - 3.3).abs() > 0.1);
    }

    /// El all-in se añade con el SPR de cada decisión tras pagar: con
    /// SPR 6 al empezar la calle no hay all-in al apostar, pero frente a la apuesta (SPR tras
    /// pagar < 5) el que responde sí puede ir all-in.
    #[test]
    fn allin_by_spr_of_each_decision() {
        let pay = LeafPayoff { base_post: vec![24.0, 24.0], oop: 1, ip: 0, pot: 4.0, utility: &chip_utility };
        let mut u = Vec::new();
        let nodes = build_leaf_tree(4.0, 24.0, PotClass::SingleRaised, &GeometricConfig::classic(), &pay, &mut u);
        let bets = opening_bets(&nodes, 0);
        assert_eq!(bets.len(), 1, "sin all-in al apostar con SPR 6: {bets:?}");
        let b = bets[0];
        let shove_call = 24.0 - b;
        let faces_shove = nodes.iter().any(|n| matches!(n, PfNode::Decision { role, street: Street::Flop, facing: true, amount, .. } if *role == OOP && ((*amount as f64) - shove_call).abs() < 1e-3));
        assert!(faces_shove, "el que responde a la apuesta debería poder ir all-in (apuesta {b})");
    }

    #[test]
    fn geometric_leaf_is_zero_sum_and_reaches_all_in() {
        let pay = LeafPayoff { base_post: vec![40.0, 40.0], oop: 1, ip: 0, pot: 5.0, utility: &chip_utility };
        let mut utils = Vec::new();
        let nodes = build_leaf_tree(5.0, 40.0, PotClass::SingleRaised, &GeometricConfig::default(), &pay, &mut utils);
        assert!(nodes.len() > 10);
        // Conservación de fichas en cada outcome.
        for chunk in utils.chunks(2) {
            assert!((chunk[0] + chunk[1] - 85.0).abs() < 1e-3);
        }
        // Existe un showdown en el que alguien queda a 0 (línea bet-call ×3 hasta all-in).
        assert!(utils.chunks(2).any(|c| c[0].abs() < 1e-3 || c[1].abs() < 1e-3));
        // Todas las decisiones son binarias y referencian nodos válidos.
        for n in &nodes {
            if let PfNode::Decision { children, .. } = n {
                assert!(children.iter().all(|&c| (c as usize) < nodes.len()));
            }
        }
    }
}
