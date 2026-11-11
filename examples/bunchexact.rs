//! Equity exacta con card bunching (Monte Carlo con reparto real) de un héroe contra el rango
//! de un rival, dados los rangos con los que foldearon los demás. Separa el efecto en dos:
//! - `sin`: manos independientes y board uniforme;
//! - `manos`: las manos de los que foldearon chocan con la del rival (y entre sí), pero el
//!   board no ve sus cartas;
//! - `exacto`: además el board sale del mazo que queda;
//! - `modelo`: el del solver sin linealizar: cada carta del board pesa `1 − Q(c)`, con `Q(c)`
//!   la probabilidad de que la tenga alguno de los que foldearon (suma de sus marginales).
//! `bunchexact rival fold1 fold2 … -- 32o 44 AKs …` (rangos con pesos). Entorno: `N`
//! repartos aceptados por clase (200 000), `SEED`.
use preflop_engine::eval::evaluator::rank_mask;
use preflop_engine::eval::{COMBO_CARDS, COMBO_MASKS, NUM_COMBOS};
use preflop_engine::io::ranges::parse_range;
use preflop_engine::math::rng::SplitMix64;

/// Muestreador de combos según un rango (pesos por combo).
struct Sampler {
    cum: Vec<f64>,
}

impl Sampler {
    fn new(r: &[f32]) -> Self {
        let mut acc = 0.0f64;
        let cum = r[..NUM_COMBOS].iter().map(|&x| {
            acc += x as f64;
            acc
        }).collect();
        Sampler { cum }
    }
    fn draw(&self, rng: &mut SplitMix64) -> usize {
        let t = *self.cum.last().unwrap();
        let x = rng.next_f64_open0() * t;
        self.cum.partition_point(|&c| c < x).min(NUM_COMBOS - 1)
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let sep = a.iter().position(|x| x == "--").expect("falta --");
    let ranges: Vec<Vec<f32>> = a[..sep].iter().map(|s| parse_range(s).unwrap()).collect();
    let (villain, folds) = (Sampler::new(&ranges[0]), ranges[1..].iter().map(|r| Sampler::new(r)).collect::<Vec<_>>());
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let mut rng = SplitMix64::new(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(7));
    // Q(c): suma de las marginales de carta de los rangos foldeados (como `deck_deltas`, por carta).
    let mut q = [0.0f64; 52];
    for r in &ranges[1..] {
        let tot: f64 = r[..NUM_COMBOS].iter().map(|&x| x as f64).sum();
        for h in 0..NUM_COMBOS {
            let [c1, c2] = COMBO_CARDS[h];
            q[c1 as usize] += r[h] as f64 / tot;
            q[c2 as usize] += r[h] as f64 / tot;
        }
    }
    let omega: Vec<f64> = q.iter().map(|x| (1.0 - x).max(0.0)).collect();
    // δ por rango, como `deck_deltas` (media de las 4 cartas del rango, relativo a la media).
    let qr: Vec<f64> = (0..13).map(|r| (0..4).map(|s| q[r * 4 + s]).sum::<f64>() / 4.0).collect();
    let mean = qr.iter().map(|x| 1.0 - x).sum::<f64>() / 13.0;
    let delta: Vec<f64> = (0..52).map(|c| (1.0 - qr[c / 4]) / mean - 1.0).collect();
    println!("{:6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}   (equity del héroe; empates a medias)", "mano", "sin", "manos", "exacto", "modelo", "lineal", "pesoB", "pesoB|hv", "B+v", "B+v+h");
    for class in &a[sep + 1..] {
        let hr = parse_range(class).unwrap();
        let heroes: Vec<usize> = (0..NUM_COMBOS).filter(|&h| hr[h] > 0.0).collect();
        let mut res = [0.0f64; 8];
        let mut lin = 0.0f64;
        for (mode, out) in res.iter_mut().enumerate() {
            let (mut sum, mut cnt) = (0.0f64, 0usize);
            let (mut ss, mut sws) = (0.0f64, 0.0f64);
            let mut wsum = 0.0f64;
            while cnt < n {
                let h = heroes[(rng.next_u64() % heroes.len() as u64) as usize];
                let mut dead = COMBO_MASKS[h];
                let v = villain.draw(&mut rng);
                if COMBO_MASKS[v] & dead != 0 {
                    continue;
                }
                dead |= COMBO_MASKS[v];
                let mut board_dead = dead;
                if mode == 1 || mode == 2 {
                    let mut ok = true;
                    for f in &folds {
                        let x = f.draw(&mut rng);
                        if COMBO_MASKS[x] & dead != 0 {
                            ok = false;
                            break;
                        }
                        dead |= COMBO_MASKS[x];
                    }
                    if !ok {
                        continue;
                    }
                    if mode == 2 {
                        board_dead = dead;
                    }
                }
                // Board de 5 cartas del mazo que queda.
                let mut board = 0u64;
                let mut k = 0;
                while k < 5 {
                    let c = rng.next_u64() % 52;
                    let m = 1u64 << c;
                    if (board_dead | board) & m == 0 && (mode != 3 || rng.next_f64_open0() <= omega[c as usize]) {
                        board |= m;
                        k += 1;
                    }
                }
                let (rh, rv) = (rank_mask(board | COMBO_MASKS[h]), rank_mask(board | COMBO_MASKS[v]));
                let w = if rh > rv { 1.0 } else if rh == rv { 0.5 } else { 0.0 };
                let iw = if mode >= 4 {
                    let cond = if mode == 5 { COMBO_MASKS[h] | COMBO_MASKS[v] } else { 0 };
                    // 6: los que foldearon no tienen cartas del board ni del rival; 7: ni del héroe.
                    let avoid = match mode { 6 => board | COMBO_MASKS[v], 7 => board | COMBO_MASKS[v] | COMBO_MASKS[h], _ => board };
                    ranges[1..].iter().map(|r| {
                        let (mut tot, mut ok) = (0.0f64, 0.0f64);
                        for x in 0..NUM_COMBOS {
                            if COMBO_MASKS[x] & cond != 0 { continue; }
                            tot += r[x] as f64;
                            if COMBO_MASKS[x] & avoid == 0 { ok += r[x] as f64; }
                        }
                        if tot > 0.0 { ok / tot } else { 1.0 }
                    }).product::<f64>()
                } else { 1.0 };
                sum += w * iw;
                wsum += iw;
                if mode == 0 {
                    let sd: f64 = (0..52).filter(|&c| board & (1u64 << c) != 0).map(|c| delta[c]).sum();
                    ss += sd;
                    sws += w * sd;
                }
                cnt += 1;
                let _ = COMBO_CARDS[h];
            }
            *out = sum / wsum;
            if mode == 0 {
                lin = *out + (sws / cnt as f64 - *out * ss / cnt as f64);
            }
        }
        println!("{:6} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}", class, res[0], res[1], res[2], res[3], lin, res[4], res[5], res[6], res[7]);
    }
}
