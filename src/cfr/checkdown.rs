//! Estimador postflop trivial: se pasa el board completo sin apuestas (realización de
//! equity del 100 %). Es el modelo de referencia de los calculadores push/fold y sirve de
//! cota/fallback del MCCFR.

use super::{pool::Pool, PostflopEstimator, RangeVec, StrategyMode};
use crate::eval::equity::EquityMatrix;
use crate::tree::PreflopTree;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CheckDown {
    pub eq: Arc<EquityMatrix>,
}

impl PostflopEstimator for CheckDown {
    fn values(&self, tree: &PreflopTree, leaf: usize, hero_is_ip: bool, _r_hero: &RangeVec, r_opp: &RangeVec, _m: StrategyMode, _u: bool, _s: u64, seats: &[usize], _pool: Pool<'_>) -> Vec<Box<RangeVec>> {
        let l = &tree.leaves[leaf];
        let n = tree.n;
        let u = &tree.utils[l.checkdown_util as usize..];
        let prod = self.eq.products(&r_opp[..]);
        seats
            .iter()
            .map(|&seat| {
                let (u_oop, u_ip, u_tie) = (u[seat], u[n + seat], u[2 * n + seat]);
                let (uw, ul) = if hero_is_ip { (u_ip, u_oop) } else { (u_oop, u_ip) };
                let mut out = super::zero_range();
                prod.hero_values(uw, u_tie, ul, &mut out[..]);
                out
            })
            .collect()
    }

    fn hero_values(&self, tree: &PreflopTree, leaf: usize, hero_is_ip: bool, _r_hero: &RangeVec, r_opp: &RangeVec, _m: StrategyMode, _u: bool, _s: u64, out: &mut RangeVec, _pool: Pool<'_>) {
        let l = &tree.leaves[leaf];
        let n = tree.n;
        let seat = if hero_is_ip { l.ip } else { l.oop } as usize;
        let u = &tree.utils[l.checkdown_util as usize..];
        let (u_oop, u_ip, u_tie) = (u[seat], u[n + seat], u[2 * n + seat]);
        let (uw, ul) = if hero_is_ip { (u_ip, u_oop) } else { (u_oop, u_ip) };
        self.eq.hero_values(r_opp, uw, u_tie, ul, out);
    }

    fn observer_values(&self, tree: &PreflopTree, leaf: usize, r_oop: &RangeVec, r_ip: &RangeVec, _m: StrategyMode, _s: u64, out: &mut [f64], _pool: Pool<'_>) {
        let l = &tree.leaves[leaf];
        let n = tree.n;
        let u = &tree.utils[l.checkdown_util as usize..];
        let (p_oop, p_ip, p_tie) = self.eq.pair_outcomes(r_oop, r_ip);
        for (s, o) in out.iter_mut().enumerate().take(n) {
            *o = p_oop * u[s] as f64 + p_ip * u[n + s] as f64 + p_tie * u[2 * n + s] as f64;
        }
    }
}
