//! Pasa el importador de historiales por ficheros reales: cuántas manos se leen por sala,
//! cuáles fallan (con el motivo y el encabezado) y comprobaciones básicas de cada mano.
//!
//! `cargo run --release --example hhcheck -- fichero1.txt fichero2.txt …` (`-v` lista cada mano)
use preflop_engine::io::hh::{parse_hand, split_hands};
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v");
    let mut ok: BTreeMap<String, usize> = BTreeMap::new();
    let mut bad = 0usize;
    for path in args.iter().filter(|a| *a != "-v") {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let text = String::from_utf8_lossy(&bytes);
        for (k, hand) in split_hands(&text).iter().enumerate() {
            let header = hand.lines().next().unwrap_or("").trim();
            match parse_hand(hand) {
                Ok(h) => {
                    let mut issues = Vec::new();
                    if !(h.sb > 0.0 && h.bb >= h.sb) {
                        issues.push(format!("ciegas {}/{}", h.sb, h.bb));
                    }
                    if h.players.iter().any(|p| p.1 <= 0.0) {
                        issues.push("stack ≤ 0".into());
                    }
                    if h.hero.is_some() && h.hero_index().is_none() {
                        issues.push("héroe sin asiento".into());
                    }
                    if !issues.is_empty() {
                        bad += 1;
                        println!("RARA  {path} #{}: {} — {header}", k + 1, issues.join(", "));
                    } else {
                        *ok.entry(format!("{:?}", h.site)).or_default() += 1;
                    }
                    if verbose {
                        let st: Vec<String> = h.stacks_bb().iter().map(|s| format!("{s:.2}")).collect();
                        println!(
                            "{:?} #{} sb {} bb {} ante {}{} | {} | bounties {:?} | héroe {:?} {:?}",
                            h.site,
                            h.hand_id,
                            h.sb,
                            h.bb,
                            h.ante,
                            if h.bb_ante { " (BB)" } else { "" },
                            st.join(" "),
                            h.bounties,
                            h.hero_index(),
                            h.hero_cards
                        );
                    }
                }
                Err(e) => {
                    bad += 1;
                    println!("FALLA {path} #{}: {e} — {header}", k + 1);
                }
            }
        }
    }
    println!("leídas: {ok:?}; con problemas: {bad}");
}
