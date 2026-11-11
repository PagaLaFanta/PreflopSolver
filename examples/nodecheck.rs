//! Árbitro de un all-in a 3 con reparto real de las manos de todos (también las de los que
//! foldearon) y board del mazo que queda: EV de pagar menos EV de foldear para el héroe.
//! Las utilidades salen del propio árbol (`DUMP_TERM=ruta solve … > term.txt`): el terminal
//! `F` (all-in HU de los otros dos, 3 outcomes) y el `C` (bote a 3, 13 outcomes de
//! `PATTERNS3`). `nodecheck term.txt héroe "rango asiento 0" … "rango asiento n-1" -- KTo 98s …`
//! (el rango del héroe se ignora). Entorno: `N` muestras por clase (400 000), `SEED`.
use preflop_engine::eval::evaluator::rank_mask;
use preflop_engine::eval::{COMBO_MASKS, NUM_COMBOS};
use preflop_engine::io::ranges::parse_range;
use preflop_engine::math::rng::SplitMix64;
use preflop_engine::tree::nodes::PATTERNS3;

struct Term {
    players: Vec<usize>,
    rows: Vec<Vec<f64>>,
}

fn read_terms(path: &str) -> (Term, Term) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut terms: Vec<(String, Term)> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("  ") {
            terms.last_mut().unwrap().1.rows.push(rest.split_whitespace().map(|x| x.parse().unwrap()).collect());
        } else if let Some((label, rest)) = line.split_once(": ") {
            let p = &rest[rest.find('[').unwrap() + 1..rest.find(']').unwrap()];
            let players = p.split(',').map(|x| x.trim().parse::<usize>().unwrap()).filter(|&x| x != 255).collect();
            terms.push((label.to_string(), Term { players, rows: Vec::new() }));
        }
    }
    let take = |l: &str| terms.iter().position(|t| t.0 == l).map(|i| Term { players: terms[i].1.players.clone(), rows: terms[i].1.rows.clone() }).unwrap();
    (take("F"), take("C"))
}

struct Sampler {
    cum: Vec<f64>,
}

impl Sampler {
    fn new(r: &[f32]) -> Self {
        let mut acc = 0.0f64;
        Sampler { cum: r[..NUM_COMBOS].iter().map(|&x| { acc += x as f64; acc }).collect() }
    }
    fn draw(&self, rng: &mut SplitMix64) -> usize {
        let x = rng.next_f64_open0() * *self.cum.last().unwrap();
        self.cum.partition_point(|&c| c < x).min(NUM_COMBOS - 1)
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let sep = a.iter().position(|x| x == "--").expect("falta --");
    let (fold, call) = read_terms(&a[0]);
    let hero: usize = a[1].parse().unwrap();
    let seats: Vec<Option<Sampler>> = a[2..sep].iter().enumerate().map(|(i, s)| (i != hero).then(|| Sampler::new(&parse_range(s).unwrap()))).collect();
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(400_000);
    let mut rng = SplitMix64::new(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(7));
    println!("{:6} {:>9} {:>9} {:>9} {:>7}", "mano", "call", "fold", "call-fold", "±");
    for class in &a[sep + 1..] {
        let hr = parse_range(class).unwrap();
        let heroes: Vec<usize> = (0..NUM_COMBOS).filter(|&h| hr[h] > 0.0).collect();
        let (mut sc, mut sf, mut sd2, mut cnt) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        let mut hands = vec![0usize; seats.len()];
        while cnt < n {
            let h = heroes[(rng.next_u64() % heroes.len() as u64) as usize];
            let mut dead = COMBO_MASKS[h];
            hands[hero] = h;
            let mut ok = true;
            for (i, s) in seats.iter().enumerate() {
                if let Some(s) = s {
                    let x = s.draw(&mut rng);
                    if COMBO_MASKS[x] & dead != 0 {
                        ok = false;
                        break;
                    }
                    dead |= COMBO_MASKS[x];
                    hands[i] = x;
                }
            }
            if !ok {
                continue;
            }
            let mut board = 0u64;
            let mut k = 0;
            while k < 5 {
                let m = 1u64 << (rng.next_u64() % 52);
                if (dead | board) & m == 0 {
                    board |= m;
                    k += 1;
                }
            }
            let val = |p: usize| rank_mask(board | COMBO_MASKS[hands[p]]);
            // Bote a 3: puesto de cada jugador (0 = mejor).
            let v: Vec<_> = call.players.iter().map(|&p| val(p)).collect();
            let ranks: Vec<u8> = (0..3).map(|i| (0..3).filter(|&j| v[j] > v[i]).map(|j| v[j]).collect::<std::collections::BTreeSet<_>>().len() as u8).collect();
            let pi = PATTERNS3.iter().position(|p| p[..] == ranks[..]).unwrap();
            let uc = call.rows[pi][hero];
            let (f0, f1) = (val(fold.players[0]), val(fold.players[1]));
            let uf = fold.rows[if f0 > f1 { 0 } else if f1 > f0 { 1 } else { 2 }][hero];
            sc += uc;
            sf += uf;
            sd2 += (uc - uf) * (uc - uf);
            cnt += 1;
        }
        let (c, f) = (sc / n as f64, sf / n as f64);
        let sd = (sd2 / n as f64 - (c - f) * (c - f)).max(0.0).sqrt() / (n as f64).sqrt();
        println!("{:6} {:>9.3} {:>9.3} {:>9.3} {:>7.3}", class, c, f, c - f, sd);
    }
}
