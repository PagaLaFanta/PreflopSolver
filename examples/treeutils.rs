//! Utilidades de cada terminal de un árbol push/fold (depuración): `treeutils 10,10,10 50,30,20`.
use preflop_engine::tree::{build_tree, Payoff, PreflopConfig};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let stacks: Vec<f64> = a[0].split(',').map(|x| x.parse().unwrap()).collect();
    let payoff = match a.get(1) {
        Some(p) => Payoff::Icm { payouts: p.split(',').map(|x| x.parse().unwrap()).collect() },
        None => Payoff::ChipEv,
    };
    let mut cfg0 = PreflopConfig::push_fold(stacks.clone(), payoff.clone());
    if let Ok(m) = std::env::var("MAXPOT") { cfg0.max_players_in_pot = m.parse().unwrap(); }
    let t = if std::env::var("PF").is_ok() { build_tree(cfg0) } else if std::env::var("STD").is_ok() { build_tree(PreflopConfig { postflop: None, ..PreflopConfig::standard(stacks, payoff) }) } else { build_tree(PreflopConfig::push_fold(stacks, payoff)) };
    let mw = t.terminals.iter().filter(|x| x.kind == preflop_engine::tree::TerminalKind::Multiway).count();
    eprintln!("terminales {} · a 3: {mw}", t.terminals.len());
    for (k, term) in t.terminals.iter().enumerate() {
        let u = term.utilities(&t.utils);
        if std::env::var("COUNT").is_ok() { continue; }
        println!("{k} {:?} players {:?} contrib {:?} u {:?}", term.kind, term.players, &term.contributions[..t.n], u.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    }
}
