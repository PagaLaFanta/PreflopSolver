//! Importación de historiales de mano de torneo en texto: PokerStars, GGPoker y Winamax
//!.
//!
//! Se extraen: nivel de ciegas, antes (también el ante de la BB), asientos con stacks y
//! cabezas, botón, quién pone cada ciega y las cartas del héroe. El resultado se traduce a
//! stacks en bb en el orden de acción preflop que usa el solver (`[UTG…BTN, SB, BB]`).
//! Los premios no figuran en el historial y se pasan aparte.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    PokerStars,
    GgPoker,
    Winamax,
}

/// Encabezados con que empieza cada mano.
const HEADERS: [(&str, Site); 3] = [("PokerStars Hand #", Site::PokerStars), ("Poker Hand #", Site::GgPoker), ("Winamax Poker - ", Site::Winamax)];

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedHand {
    pub site: Site,
    pub hand_id: String,
    pub sb: f64,
    pub bb: f64,
    /// Ante por jugador (con `bb_ante`, la BB paga `ante · jugadores`).
    pub ante: f64,
    /// Solo la BB pone el ante, por toda la mesa.
    pub bb_ante: bool,
    /// Jugadores en orden de acción preflop: `(nombre, fichas)`.
    pub players: Vec<(String, f64)>,
    /// Cabezas PKO en el mismo orden, si el historial las da para todos.
    pub bounties: Option<Vec<f64>>,
    pub hero: Option<String>,
    pub hero_cards: Option<String>,
}

impl ParsedHand {
    /// Stacks en ciegas grandes, en orden de acción.
    pub fn stacks_bb(&self) -> Vec<f64> {
        self.players.iter().map(|(_, c)| c / self.bb).collect()
    }

    /// Índice del héroe en el orden de acción.
    pub fn hero_index(&self) -> Option<usize> {
        let h = self.hero.as_ref()?;
        self.players.iter().position(|(n, _)| n == h)
    }
}

fn num(s: &str) -> Option<f64> {
    s.trim().trim_matches(['$', '€', '£']).replace(',', "").parse().ok()
}

/// Primer número de un texto ("big blind 1,200", "8000 in chips", "2.50€ bounty").
fn first_num(s: &str) -> Option<f64> {
    s.split_whitespace().find_map(num)
}

fn site_of(header: &str) -> Option<Site> {
    HEADERS.iter().find(|(h, _)| header.starts_with(h)).map(|&(_, s)| s)
}

/// Separa un fichero en manos: bloques que empiezan, a principio de línea, por el
/// encabezado de PokerStars, GGPoker o Winamax.
pub fn split_hands(text: &str) -> Vec<&str> {
    let mut starts: Vec<usize> = text
        .char_indices()
        .filter(|&(i, _)| (i == 0 || text[..i].ends_with(['\n', '\u{feff}'])) && site_of(&text[i..]).is_some())
        .map(|(i, _)| i)
        .collect();
    starts.push(text.len());
    starts.windows(2).map(|w| &text[w[0]..w[1]]).collect()
}

/// Nivel de ciegas del encabezado. Se prueba, por orden, el paréntesis que sigue al último
/// "Level" (PokerStars, GGPoker; en Winamax "level: n" va antes de las ciegas), el que sigue a
/// "No Limit" y el último paréntesis; vale el primero que se lea como ciegas (el nombre del
/// torneo puede llevar "No Limit Hold'em" o paréntesis). Admite `sb/bb` (PokerStars),
/// `sb/bb(ante)` (GGPoker) y `ante/sb/bb` (Winamax). Devuelve `(sb, bb, ante)`.
fn blind_level(header: &str) -> Option<(f64, f64, f64)> {
    let lower = header.to_ascii_lowercase();
    let after = |i: usize| header[i..].find('(').map(|j| i + j);
    [lower.rfind("level").and_then(after), lower.find("no limit").and_then(after), header.rfind('(')].into_iter().flatten().find_map(|open| parens_blinds(header, open))
}

/// Ciegas del paréntesis que abre en `open` (con paréntesis anidados para el ante de GG).
fn parens_blinds(header: &str, open: usize) -> Option<(f64, f64, f64)> {
    let mut depth = 0;
    let mut end = None;
    for (j, c) in header[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(open + j);
                    break;
                }
            }
            _ => {}
        }
    }
    let inner = &header[open + 1..end?];
    let (main, ante) = match inner.split_once('(') {
        Some((m, a)) => (m, num(a.trim_end_matches(')'))?),
        None => (inner, 0.0),
    };
    let parts: Vec<f64> = main.split('/').map(first_num).collect::<Option<_>>()?;
    match parts[..] {
        [sb, bb] => Some((sb, bb, ante)),
        [a, sb, bb] => Some((sb, bb, a)),
        _ => None,
    }
}

fn hand_id(site: Site, header: &str) -> String {
    match site {
        Site::Winamax => header.split("HandId: #").nth(1).and_then(|r| r.split_whitespace().next()).unwrap_or("").to_string(),
        _ => {
            let h = HEADERS.iter().find(|(_, s)| *s == site).map_or(0, |h| h.0.len());
            header[h..].split(':').next().unwrap_or("").to_string()
        }
    }
}

pub fn parse_hand(text: &str) -> Result<ParsedHand, String> {
    let text = text.trim_start_matches('\u{feff}');
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty hand history")?.trim();
    let site = site_of(header).ok_or("not a PokerStars, GGPoker or Winamax hand history")?;
    let hand_id = hand_id(site, header);
    let (mut sb, mut bb, header_ante) = blind_level(header).ok_or("blind level not found")?;

    let mut button: Option<u32> = None;
    let mut seats: Vec<(u32, String, f64, Option<f64>)> = Vec::new();
    // Antes puestos: (jugador, cantidad).
    let mut antes: Vec<(String, f64)> = Vec::new();
    let mut bb_player: Option<String> = None;
    let mut hero = None;
    let mut hero_cards = None;
    for line in lines {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Dealt to ") {
            // GGPoker también lista a los rivales, sin cartas.
            if let Some((name, cards)) = rest.rsplit_once(" [") {
                hero = Some(name.to_string());
                hero_cards = Some(cards.trim_end_matches(']').to_string());
            }
            continue;
        }
        if line.starts_with("*** ") {
            if (line.contains("FLOP") && !line.contains("PRE-FLOP")) || line.contains("SUMMARY") {
                break;
            }
            continue;
        }
        if line.contains("is the button") {
            if let Some(i) = line.find("Seat #") {
                button = line[i + 6..].split_whitespace().next().and_then(|x| x.parse().ok());
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("Seat ") {
            // "Seat 3: Hero (8000 in chips)", "… (8000 in chips, $5 bounty)" o, en Winamax,
            // "Seat 3: Hero (8000, 2.50€ bounty)"; excluye "is sitting out".
            if let Some((seat, rest)) = rest.split_once(": ") {
                if let (Ok(seat), Some((name, chips))) = (seat.parse::<u32>(), rest.rsplit_once(" (")) {
                    if rest.contains("sitting out") {
                        continue;
                    }
                    let inside = chips.split(')').next().unwrap_or("");
                    let bounty = inside.split(", ").find(|p| p.contains("bounty")).and_then(first_num);
                    if let Some(c) = inside.split(", ").next().and_then(first_num) {
                        seats.push((seat, name.to_string(), c, bounty));
                    }
                }
            }
            continue;
        }
        // "name: posts …" (PokerStars, GGPoker) o "name posts …" (Winamax).
        if let Some((name, act)) = line.split_once(": posts ").or_else(|| line.split_once(" posts ")) {
            let act = act.trim_start_matches("the ");
            let amount = first_num(act).unwrap_or(0.0);
            if act.starts_with("ante") {
                antes.push((name.to_string(), amount));
            } else if act.starts_with("small blind") {
                sb = sb.max(amount);
            } else if act.starts_with("big blind") {
                bb_player = Some(name.to_string());
                bb = bb.max(amount);
            }
        }
    }
    if seats.len() < 2 {
        return Err("at least 2 players are needed".into());
    }
    seats.sort_by_key(|s| s.0);
    let button = button.ok_or("button not found")?;
    // Orden horario empezando por el primer asiento tras el botón.
    let start = seats.iter().position(|s| s.0 > button).unwrap_or(0);
    let clockwise: Vec<_> = seats[start..].iter().chain(&seats[..start]).cloned().collect();
    // clockwise = [SB, BB, UTG, …, BTN] (en HU: [BB, BTN=SB]).
    let n = clockwise.len();
    let bb_pos = match &bb_player {
        Some(p) => clockwise.iter().position(|s| &s.1 == p).ok_or("big blind player not seated")?,
        None => if n == 2 { 0 } else { 1 },
    };
    // Orden de acción preflop: tras la BB, en sentido horario, terminando en la BB.
    let order: Vec<_> = (1..=n).map(|k| clockwise[(bb_pos + k) % n].clone()).collect();
    let players: Vec<(String, f64)> = order.iter().map(|s| (s.1.clone(), s.2)).collect();
    let bounties: Option<Vec<f64>> = order.iter().map(|s| s.3).collect();
    // Ante de la BB: solo la BB pone ante y lo pone por toda la mesa; se pasa a por jugador.
    let bb_ante = antes.len() == 1 && bb_player.as_ref() == Some(&antes[0].0);
    let ante = if bb_ante { antes[0].1 / n as f64 } else { antes.iter().map(|a| a.1).fold(header_ante, f64::max) };
    Ok(ParsedHand { site, hand_id, sb, bb, ante, bb_ante, players, bounties, hero, hero_cards })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HH: &str = "PokerStars Hand #231234567890: Tournament #3141592653, $10+$1 USD Hold'em No Limit - Level XV (400/800) - 2024/01/01 12:00:00 ET
Table '3141592653 12' 9-max Seat #3 is the button
Seat 1: villain1 (15000 in chips)
Seat 2: villain2 (23,000 in chips)
Seat 3: Hero (8000 in chips)
Seat 5: sb_guy (12000 in chips)
Seat 7: bb_guy (9600 in chips)
Seat 8: away (5000 in chips) is sitting out
villain1: posts the ante 100
villain2: posts the ante 100
Hero: posts the ante 100
sb_guy: posts the ante 100
bb_guy: posts the ante 100
sb_guy: posts small blind 400
bb_guy: posts big blind 800
*** HOLE CARDS ***
Dealt to Hero [Ah Kd]
villain1: folds
*** SUMMARY ***
";

    const GG: &str = "Poker Hand #TM2345678901: Tournament #123456789, Bounty Hunters $10.80 Hold'em No Limit - Level12(500/1,000(125)) - 2024/01/01 12:00:00
Table '25' 8-max Seat #4 is the button
Seat 1: a1b2c3 (31,250 in chips)
Seat 2: Hero (18,000 in chips)
Seat 4: d4e5f6 (24,600 in chips)
Seat 6: g7h8i9 (12,000 in chips)
Seat 8: j0k1l2 (9,450 in chips)
a1b2c3: posts the ante 125
Hero: posts the ante 125
d4e5f6: posts the ante 125
g7h8i9: posts the ante 125
j0k1l2: posts the ante 125
g7h8i9: posts small blind 500
j0k1l2: posts big blind 1,000
*** HOLE CARDS ***
Dealt to a1b2c3
Dealt to Hero [Qs Qd]
Dealt to d4e5f6
Dealt to g7h8i9
Dealt to j0k1l2
a1b2c3: folds
*** SUMMARY ***
";

    const WINAMAX: &str = "Winamax Poker - Tournament \"Monster Stack\" buyIn: 4.50€ + 0.50€ level: 9 - HandId: #2345678-123-1700000000 - Holdem no limit (50/200/400) - 2024/01/01 12:00:00 UTC
Table: 'Monster Stack(612345678)#004' 6-max (real money) Seat #2 is the button
Seat 1: Lucas (12000, 2.50€ bounty)
Seat 2: Hero (8400, 2.50€ bounty)
Seat 3: Marie (20000, 5€ bounty)
Seat 5: Paul (6000, 2.50€ bounty)
*** ANTE/BLINDS ***
Lucas posts ante 50
Hero posts ante 50
Marie posts ante 50
Paul posts ante 50
Marie posts small blind 200
Paul posts big blind 400
Dealt to Hero [7c 7d]
*** PRE-FLOP ***
Lucas folds
*** FLOP *** [2c 5d Th]
*** SUMMARY ***
";

    fn order(h: &ParsedHand) -> Vec<&str> {
        h.players.iter().map(|p| p.0.as_str()).collect()
    }

    #[test]
    fn parses_pokerstars_tournament_hand() {
        let h = parse_hand(HH).unwrap();
        assert_eq!(h.site, Site::PokerStars);
        assert_eq!((h.sb, h.bb, h.ante, h.bb_ante), (400.0, 800.0, 100.0, false));
        assert_eq!(order(&h), vec!["villain1", "villain2", "Hero", "sb_guy", "bb_guy"]);
        assert_eq!(h.stacks_bb(), vec![18.75, 28.75, 10.0, 15.0, 12.0]);
        assert_eq!(h.hero_index(), Some(2));
        assert_eq!(h.hero_cards.as_deref(), Some("Ah Kd"));
        assert_eq!(h.bounties, None);
        assert_eq!(split_hands(&format!("{HH}\n\n{HH}")).len(), 2);
    }

    /// Ante de la BB: solo la BB lo pone, por los 5 jugadores; cabezas de PokerStars.
    #[test]
    fn parses_big_blind_ante_and_bounties() {
        let mut t = String::new();
        for line in HH.lines() {
            if line.contains("posts the ante") && !line.starts_with("bb_guy") {
                continue;
            }
            let line = line.replace("posts the ante 100", "posts the ante 800").replace(" in chips)", " in chips, $5.50 bounty)");
            t.push_str(&line);
            t.push('\n');
        }
        let h = parse_hand(&t).unwrap();
        assert!(h.bb_ante);
        assert_eq!(h.ante, 160.0);
        assert_eq!(h.bounties, Some(vec![5.5; 5]));
        assert_eq!(h.stacks_bb(), vec![18.75, 28.75, 10.0, 15.0, 12.0]);
    }

    #[test]
    fn parses_ggpoker_tournament_hand() {
        let h = parse_hand(GG).unwrap();
        assert_eq!(h.site, Site::GgPoker);
        assert_eq!(h.hand_id, "TM2345678901");
        assert_eq!((h.sb, h.bb, h.ante, h.bb_ante), (500.0, 1000.0, 125.0, false));
        assert_eq!(order(&h), vec!["a1b2c3", "Hero", "d4e5f6", "g7h8i9", "j0k1l2"]);
        assert_eq!(h.stacks_bb(), vec![31.25, 18.0, 24.6, 12.0, 9.45]);
        assert_eq!(h.hero_index(), Some(1));
        assert_eq!(h.hero_cards.as_deref(), Some("Qs Qd"));
    }

    #[test]
    fn parses_winamax_tournament_hand() {
        let h = parse_hand(WINAMAX).unwrap();
        assert_eq!(h.site, Site::Winamax);
        assert_eq!(h.hand_id, "2345678-123-1700000000");
        assert_eq!((h.sb, h.bb, h.ante, h.bb_ante), (200.0, 400.0, 50.0, false));
        assert_eq!(order(&h), vec!["Lucas", "Hero", "Marie", "Paul"]);
        assert_eq!(h.stacks_bb(), vec![30.0, 21.0, 50.0, 15.0]);
        assert_eq!(h.bounties, Some(vec![2.5, 2.5, 5.0, 2.5]));
        assert_eq!(h.hero_cards.as_deref(), Some("7c 7d"));
    }

    #[test]
    fn splits_mixed_sites_and_reads_header_levels() {
        let all = format!("\u{feff}{GG}\n\n{WINAMAX}\n\n{HH}");
        let hands = split_hands(&all);
        assert_eq!(hands.len(), 3);
        let sites: Vec<Site> = hands.iter().map(|t| parse_hand(t).unwrap().site).collect();
        assert_eq!(sites, vec![Site::GgPoker, Site::Winamax, Site::PokerStars]);
        // Sin ante en el encabezado de Winamax ni ciegas con divisa.
        assert_eq!(blind_level("Winamax Poker - Tournament \"X\" - HandId: #1-2-3 - Holdem no limit (200/400) - x UTC"), Some((200.0, 400.0, 0.0)));
        assert_eq!(blind_level("PokerStars Hand #1: Hold'em No Limit ($0.01/$0.02 USD) - x"), Some((0.01, 0.02, 0.0)));
        // GGPoker real: "Level10 (1,000/2,000)" con espacio y sin ante, y "No Limit" y
        // paréntesis en el nombre del torneo.
        let gg = "Poker Hand #TM1: Tournament #2, WSOP #77: $5,000 No Limit Hold'em (6-Max) Main Event, $25M GTD Hold'em No Limit - Level10 (1,000/2,000) - 2020/08/30 15:49:08";
        assert_eq!(blind_level(gg), Some((1000.0, 2000.0, 0.0)));
        assert_eq!(blind_level("Winamax Poker - Tournament \"A (B)\" buyIn: 1€ level: 3 - HandId: #1 - Holdem no limit (10/25/50) - x UTC"), Some((25.0, 50.0, 10.0)));
    }
}
