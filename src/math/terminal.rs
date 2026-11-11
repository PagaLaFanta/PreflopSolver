//! # Utilidad $EV de un nodo terminal: fusión ICM + PKO
//!
//! ## Descomposición
//! En un Progressive Knockout el dinero del torneo vive en dos "bolsas" disjuntas:
//! el *prize pool* regular `Π` (pagado por puesto) y las cabezas `H_i` (bounties).
//! La utilidad de un estado post-mano `σ' = (s', H')` para el jugador `i` es
//!
//! ```text
//! U_i(σ') =  ICM_i(s', Π)                         (1) premios por puesto, Malmuth-Harville
//!          + C_i                                   (2) cash inmediato cobrado en esta mano
//!          + w_i(s') · Σ_{j vivo} H'_j             (3) equity de bounties futura
//! ```
//!
//! (2) **Cash inmediato**. Por cada eliminación `v` ganada por el conjunto `W` (bote
//! partido ⇒ |W|>1), cada ganador cobra `φ·H_v/|W|` en efectivo y su cabeza crece
//! `(1−φ)·H_v/|W|` (φ = 0.5 en PKO estándar, φ = 1 en KO clásico no progresivo). La
//! cabeza de `v` desaparece. Esta es la parte "progresiva": la cabeza del eliminador
//! engorda y ese incremento solo se realiza en (3).
//!
//! (3) **Bounty Power Model**. La cabeza `H_j` de cada superviviente acabará en manos de
//! quien lo elimine — o del propio `j` si gana el torneo (se cobra su propia cabeza).
//! Suponemos que la fracción de *todas* las cabezas vivas que captura `i` es
//! `w_i = s_i^α / Σ_k s_k^α`:
//! * α = 1 → modelo proporcional a fichas (el estándar de facto, el "Proportional KO").
//!   Justificación: en un modelo de paseo aleatorio de fichas sin sesgo, la fracción
//!   esperada de fichas que un jugador termina acumulando es proporcional a su stack, y
//!   las cabezas se capturan con las fichas.
//! * α > 1 → *bounty power* superlineal: el stack grande cubre a todos y puede ganar
//!   cualquier cabeza, el corto no (solo gana cabezas de quien cubre). Valores empíricos
//!   típicos 1.0-1.2 según la fase del torneo.
//!
//! ## Propiedad de conservación
//! Como `Σ_i w_i = 1`, `Σ_i ICM_i = Σ Π_restante` y el cash + crecimiento de cabezas
//! redistribuye exactamente `H_v`, se cumple `Σ_i U_i = Π_restante + Σ_i H_i` (juego de
//! suma constante en dólares). Lo verificamos en tests: si se rompe, el CFR converge a
//! basura silenciosamente.
//!
//! ## Fusión en un único nodo terminal
//! El *estado* post-mano de cada *outcome* `o` (quién gana cada bote principal/lateral)
//! es independiente de las cartas; solo `P(o | h_1..h_n)` depende de ellas (equity de
//! showdown). Por eso el nodo terminal almacena la tabla `U_i(o)` precomputada con esta
//! función, y el CFR evalúa `u_i(h) = Σ_o P(o|h) · U_i(o)`: la no-linealidad del ICM y
//! los saltos discretos del bounty quedan dentro de `U`, el acoplamiento con la mano es
//! lineal → vectorizable sobre los 1326 combos.

use super::icm::icm_equity;

/// Número máximo de asientos que resuelve el terminal (el ICM puede ver más jugadores
/// del torneo que asientos tiene la mesa, pero los ejemplos de solver son de mesa final).
pub const MAX_TERMINAL_PLAYERS: usize = 64;

/// Eliminación ocurrida en la mano. `winner_mask`: bit `i` activo si el jugador `i`
/// gana (una parte de) el bote que elimina a `victim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Elimination {
    pub victim: u8,
    pub winner_mask: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PkoParams {
    /// φ: fracción de la cabeza pagada en efectivo al eliminador (0.5 PKO, 1.0 KO fijo).
    pub cash_fraction: f64,
    /// α del Bounty Power Model.
    pub bounty_power: f64,
}

impl Default for PkoParams {
    fn default() -> Self {
        PkoParams { cash_fraction: 0.5, bounty_power: 1.0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TerminalInput<'a> {
    /// Stacks al inicio de la mano (desempate de puestos entre eliminados simultáneos).
    pub pre_stacks: &'a [f64],
    /// Stacks tras resolver el bote. `0.0` ⇒ eliminado en esta mano.
    pub post_stacks: &'a [f64],
    /// Premios restantes; `payouts[0]` = 1º puesto entre los jugadores de `pre_stacks`.
    pub payouts: &'a [f64],
    /// Cabezas actuales `H_i` en la misma unidad monetaria que `payouts`.
    pub bounties: &'a [f64],
    pub eliminations: &'a [Elimination],
    pub is_pko: bool,
    pub pko: PkoParams,
    /// Jugadores de otras mesas que **no** están en los arrays (campos grandes, ver
    /// `math::field_icm`): siguen vivos, cuentan para los puestos de los eliminados y, con
    /// PKO, sus cabezas (`off_heads`) y pesos `stack^α` (`off_weight`) entran en la bolsa de
    /// bounties futura. El modelo de premios debe incluirlos por su cuenta.
    pub off_alive: usize,
    pub off_heads: f64,
    pub off_weight: f64,
}

/// Modelo de equity de premios para los supervivientes: recibe los índices originales de
/// los vivos, sus stacks y los premios restantes, y escribe la equity de cada uno.
/// Por defecto es ICM Malmuth-Harville; FGS (`math::fgs`) lo sustituye.
pub type PrizeModel<'a> = &'a dyn Fn(&[usize], &[f64], &[f64], &mut [f64]);

/// Calcula `U_i(σ')` para todos los jugadores con ICM Malmuth-Harville.
pub fn terminal_dollar_ev(inp: &TerminalInput<'_>, out: &mut [f64]) {
    terminal_dollar_ev_with(inp, &|_, s, p, o| icm_equity(s, p, o), out)
}

/// Igual que [`terminal_dollar_ev`] con un modelo de premios arbitrario.
pub fn terminal_dollar_ev_with(inp: &TerminalInput<'_>, prize_model: PrizeModel<'_>, out: &mut [f64]) {
    let n = inp.pre_stacks.len();
    assert!(n <= MAX_TERMINAL_PLAYERS);
    assert!(inp.post_stacks.len() == n && out.len() == n);
    assert!(!inp.is_pko || inp.bounties.len() == n);
    out.fill(0.0);

    // --- 1. Particionar vivos / eliminados -------------------------------------------
    let mut alive_idx = [0u8; MAX_TERMINAL_PLAYERS];
    let mut alive_stacks = [0.0f64; MAX_TERMINAL_PLAYERS];
    let mut busted = [0u8; MAX_TERMINAL_PLAYERS];
    let (mut m, mut b) = (0usize, 0usize);
    for i in 0..n {
        debug_assert!(inp.pre_stacks[i] > 0.0, "player {i} was already eliminated before the hand");
        if inp.post_stacks[i] > 0.0 {
            alive_idx[m] = i as u8;
            alive_stacks[m] = inp.post_stacks[i];
            m += 1;
        } else {
            busted[b] = i as u8;
            b += 1;
        }
    }

    // --- 2. Premios de los eliminados ------------------------------------------------
    // Ocupan los puestos m+1..m+b. Regla estándar: entre eliminados en la misma mano, el
    // de mayor stack inicial queda mejor clasificado; empates exactos se reparten los
    // puestos implicados a partes iguales.
    let busted = &mut busted[..b];
    busted.sort_unstable_by(|&x, &y| inp.pre_stacks[y as usize].total_cmp(&inp.pre_stacks[x as usize]));
    let prize_at = |place: usize| inp.payouts.get(place).copied().unwrap_or(0.0);
    let mut r = 0;
    while r < b {
        let s0 = inp.pre_stacks[busted[r] as usize];
        let mut e = r + 1;
        while e < b && inp.pre_stacks[busted[e] as usize] == s0 {
            e += 1;
        }
        let share = (r..e).map(|k| prize_at(m + inp.off_alive + k)).sum::<f64>() / (e - r) as f64;
        for &v in &busted[r..e] {
            out[v as usize] = share;
        }
        r = e;
    }

    // --- 3. ICM sobre los supervivientes (término (1)) --------------------------------
    let mut icm = [0.0f64; MAX_TERMINAL_PLAYERS];
    let prize_slice = &inp.payouts[..inp.payouts.len().min(m + inp.off_alive)];
    let idx: Vec<usize> = alive_idx[..m].iter().map(|&x| x as usize).collect();
    prize_model(&idx, &alive_stacks[..m], prize_slice, &mut icm[..m]);
    for k in 0..m {
        out[alive_idx[k] as usize] += icm[k];
    }

    if !inp.is_pko {
        return;
    }

    // --- 4. Cash inmediato + crecimiento de cabezas (término (2)) --------------------
    let phi = inp.pko.cash_fraction;
    let mut heads = [0.0f64; MAX_TERMINAL_PLAYERS];
    heads[..n].copy_from_slice(inp.bounties);
    for el in inp.eliminations {
        let v = el.victim as usize;
        debug_assert!(inp.post_stacks[v] == 0.0, "victim {v} is still alive");
        debug_assert!(el.winner_mask != 0 && el.winner_mask & (1 << v) == 0);
        let head = heads[v];
        heads[v] = 0.0;
        let k = el.winner_mask.count_ones() as f64;
        let (cash, grow) = (phi * head / k, (1.0 - phi) * head / k);
        let mut w = el.winner_mask;
        while w != 0 {
            let i = w.trailing_zeros() as usize;
            w &= w - 1;
            out[i] += cash;
            heads[i] += grow;
        }
    }

    // --- 5. Bounty equity futura (término (3)) ----------------------------------------
    let alpha = inp.pko.bounty_power;
    let mut pool = inp.off_heads;
    let mut norm = inp.off_weight;
    let mut weight = [0.0f64; MAX_TERMINAL_PLAYERS];
    for k in 0..m {
        let i = alive_idx[k] as usize;
        pool += heads[i];
        weight[k] = if alpha == 1.0 { alive_stacks[k] } else { alive_stacks[k].powf(alpha) };
        norm += weight[k];
    }
    if norm > 0.0 {
        let scale = pool / norm;
        for k in 0..m {
            out[alive_idx[k] as usize] += weight[k] * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pre: &[f64], post: &[f64], pay: &[f64], bty: &[f64], el: &[Elimination], pko: bool, alpha: f64) -> Vec<f64> {
        let mut out = vec![0.0; pre.len()];
        let inp = TerminalInput {
            pre_stacks: pre,
            post_stacks: post,
            payouts: pay,
            bounties: bty,
            eliminations: el,
            is_pko: pko,
            pko: PkoParams { cash_fraction: 0.5, bounty_power: alpha },
            off_alive: 0,
            off_heads: 0.0,
            off_weight: 0.0,
        };
        terminal_dollar_ev(&inp, &mut out);
        out
    }

    #[test]
    fn dollar_mass_is_conserved_with_bounties() {
        let pre = [40.0, 25.0, 20.0, 15.0];
        // Jugador 0 elimina al 3 (15 fichas) y el 2 pierde 5 contra el 1.
        let post = [55.0, 30.0, 15.0, 0.0];
        let pay = [50.0, 30.0, 20.0];
        let bty = [12.0, 10.0, 10.0, 8.0];
        let el = [Elimination { victim: 3, winner_mask: 1 }];
        for alpha in [1.0, 1.15, 0.8] {
            let out = run(&pre, &post, &pay, &bty, &el, true, alpha);
            let total: f64 = out.iter().sum();
            assert!((total - (100.0 + 40.0)).abs() < 1e-9, "alpha={alpha}: {total}");
            assert_eq!(out[3], 0.0, "4º puesto no paga y su cabeza ya no es suya");
        }
    }

    #[test]
    fn bounty_without_icm_uses_chips_as_the_baseline() {
        // Igual que eliminator_collects_half_bounty_in_cash (HU, elimina al rival), pero sin
        // payouts: la base no es ICM$ (no hay), es el stack final en fichas directamente —
        // así es como Payoff::PkoChipEv llama a esta misma función (ver tree/builder.rs).
        let mut out = vec![0.0; 2];
        let inp = TerminalInput {
            pre_stacks: &[10.0, 10.0],
            post_stacks: &[20.0, 0.0],
            payouts: &[],
            bounties: &[5.0, 5.0],
            eliminations: &[Elimination { victim: 1, winner_mask: 1 }],
            is_pko: true,
            pko: PkoParams { cash_fraction: 0.5, bounty_power: 1.0 },
            off_alive: 0,
            off_heads: 0.0,
            off_weight: 0.0,
        };
        terminal_dollar_ev_with(&inp, &|_, s, _, o| o.copy_from_slice(s), &mut out);
        // Base (fichas) 20 + cash inmediato 2.5 + equity futura de la cabeza ya crecida 7.5:
        // único jugador vivo, se la queda entera.
        assert!((out[0] - (20.0 + 2.5 + 7.5)).abs() < 1e-12, "{out:?}");
        assert_eq!(out[1], 0.0, "el eliminado no tiene ICM$ que cobrar (no hay payouts)");
    }

    #[test]
    fn eliminator_collects_half_bounty_in_cash() {
        // HU → heads-up terminado: el ganador cobra 1º, su cabeza + la mitad de la ajena en
        // cash + la otra mitad (que creció en su cabeza) como ganador del torneo.
        let out = run(&[10.0, 10.0], &[20.0, 0.0], &[70.0, 30.0], &[5.0, 5.0], &[Elimination { victim: 1, winner_mask: 1 }], true, 1.0);
        assert!((out[0] - (70.0 + 2.5 + 7.5)).abs() < 1e-12);
        assert!((out[1] - 30.0).abs() < 1e-12);
    }

    #[test]
    fn split_pot_elimination_splits_bounty() {
        let out = run(
            &[10.0, 10.0, 4.0],
            &[12.0, 12.0, 0.0],
            &[60.0, 40.0],
            &[4.0, 4.0, 4.0],
            &[Elimination { victim: 2, winner_mask: 0b011 }],
            true,
            1.0,
        );
        assert!((out[0] - out[1]).abs() < 1e-12);
        // ICM 50 + cash 4·½/2 = 1 + (cabeza propia 4 + crecimiento 1) = 56.
        assert!((out[0] - 56.0).abs() < 1e-12);
    }

    #[test]
    fn simultaneous_bust_ranks_by_starting_stack() {
        // Los jugadores 1 y 2 pierden all-in contra 0 en la misma mano.
        let out = run(&[30.0, 20.0, 10.0], &[60.0, 0.0, 0.0], &[50.0, 30.0, 20.0], &[], &[], false, 1.0);
        assert_eq!(out, vec![50.0, 30.0, 20.0]);
    }
}

/// Bubble factor y risk premium entre parejas de jugadores.
///
/// Para el jugador `i` frente a `j`, en un all-in por `min(s_i, s_j)`:
/// * `gain = U_i(gana) − U_i(ahora)`, `loss = U_i(ahora) − U_i(pierde)`;
/// * **bubble factor** `= loss / gain` (1.0 = chip EV; 2.0 = perder una ficha cuesta el
///   doble de lo que vale ganarla);
/// * **equity requerida** para pagar un all-in a bote sin dinero muerto:
///   `loss / (loss + gain)` (chip EV: 50 %);
/// * **risk premium** `= equity requerida − 50 %`.
///
/// `prize_model`: ICM o FGS. `stacks` incluye a los de otras mesas; solo se calcula para
/// `table` (índices de mesa).
pub fn bubble_factors(stacks: &[f64], payouts: &[f64], table: &[usize], prize_model: PrizeModel<'_>) -> Vec<Vec<Option<(f64, f64)>>> {
    table
        .iter()
        .map(|&i| table.iter().map(|&j| matchup(stacks, payouts, prize_model, i, j).map(|d| (d.bubble_factor, d.risk_premium))).collect())
        .collect()
}

/// Desglose completo de un emparejamiento `i` contra `j` (mismos números que
/// [`bubble_factors`], pero con el EV base/ganando/perdiendo de por medio para mostrarlo en un
/// panel de detalle al hacer clic en una celda de la matriz).
#[derive(Debug)]
pub struct MatchupDetail {
    pub ev_base: f64,
    pub ev_win: f64,
    pub ev_lose: f64,
    pub bubble_factor: f64,
    pub risk_premium: f64,
}

/// Igual que [`bubble_factors`] pero devolviendo el desglose completo de cada pareja.
pub fn bubble_factors_detail(stacks: &[f64], payouts: &[f64], table: &[usize], prize_model: PrizeModel<'_>) -> Vec<Vec<Option<MatchupDetail>>> {
    table.iter().map(|&i| table.iter().map(|&j| matchup(stacks, payouts, prize_model, i, j)).collect()).collect()
}

fn matchup(stacks: &[f64], payouts: &[f64], prize_model: PrizeModel<'_>, i: usize, j: usize) -> Option<MatchupDetail> {
    if i == j {
        return None;
    }
    let n = stacks.len();
    let all: Vec<usize> = (0..n).collect();
    let mut now = vec![0.0; n];
    prize_model(&all, stacks, payouts, &mut now);
    let eq = |s: &[f64]| -> Vec<f64> {
        let idx: Vec<usize> = (0..n).filter(|&k| s[k] > 0.0).collect();
        let alive: Vec<f64> = idx.iter().map(|&k| s[k]).collect();
        let mut e = vec![0.0; idx.len()];
        prize_model(&idx, &alive, &payouts[..payouts.len().min(alive.len())], &mut e);
        let mut out = vec![0.0; n];
        for (k, &i) in idx.iter().enumerate() {
            out[i] = e[k];
        }
        // El eliminado cobra el siguiente puesto.
        if let Some(&dead) = (0..n).find(|&k| s[k] <= 0.0 && stacks[k] > 0.0).as_ref() {
            out[dead] = payouts.get(idx.len()).copied().unwrap_or(0.0);
        }
        out
    };
    let m = stacks[i].min(stacks[j]);
    let mut win = stacks.to_vec();
    win[i] += m;
    win[j] -= m;
    let mut lose = stacks.to_vec();
    lose[i] -= m;
    lose[j] += m;
    let ev_base = now[i];
    let ev_win = eq(&win)[i];
    let ev_lose = eq(&lose)[i];
    let gain = ev_win - ev_base;
    if gain <= 1e-12 {
        return None;
    }
    let loss = ev_base - ev_lose;
    let req = loss / (loss + gain);
    Some(MatchupDetail { ev_base, ev_win, ev_lose, bubble_factor: loss / gain, risk_premium: req - 0.5 })
}

#[cfg(test)]
mod bubble_tests {
    use super::*;
    use crate::math::icm::icm_equity;

    #[test]
    fn bubble_factor_is_one_in_chip_ev_and_higher_on_bubble() {
        let model = |_: &[usize], s: &[f64], p: &[f64], o: &mut [f64]| icm_equity(s, p, o);
        // Winner-take-all: ICM es lineal ⇒ bubble factor 1.
        let bf = bubble_factors(&[10.0, 20.0, 30.0], &[100.0], &[0, 1, 2], &model);
        let (f, rp) = bf[0][1].unwrap();
        assert!((f - 1.0).abs() < 1e-6 && rp.abs() < 1e-6, "{f} {rp}");
        // Satélite 2 de 3: el chip leader vs el medio tiene bubble factor alto.
        let bf = bubble_factors(&[10.0, 20.0, 30.0], &[100.0, 100.0], &[0, 1, 2], &model);
        let (f, rp) = bf[1][2].unwrap();
        assert!(f > 2.0 && rp > 0.1, "{f} {rp}");
    }
}
