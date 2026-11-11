//! Notación de rangos estándar.
//!
//! Tokens separados por comas, con peso opcional `:w` (0..1):
//! `AA`, `AKs`, `AKo`, `AK` (ambos), `22+`, `A2s+` (kicker hasta uno por debajo de la
//! carta alta), `K9o+`, `99-55`, `A5s-A2s`, `T9s-65s` (ambas cartas bajan), `random`.
//! Notación: `Ax` (toda mano no pareja con carta alta A; `Axs`/`Axo` solo
//! suited/offsuit), `Kx+` (carta alta K o más), separadores con espacios y el porcentaje
//! inicial que algunas herramientas ponen delante (`32.6%, 22+ Ax K5s+`), que se ignora.

use crate::cfr::report::{class_name, grid_cell};
use crate::eval::{HAND_STRIDE, NUM_COMBOS};

const RANKS: &str = "23456789TJQKA";

fn rank(c: char) -> Option<usize> {
    RANKS.find(c.to_ascii_uppercase())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Pair,
    Suited,
    Offsuit,
    Both,
}

fn parse_hand(t: &str) -> Result<(usize, usize, Kind), String> {
    let cs: Vec<char> = t.chars().collect();
    if cs.len() < 2 || cs.len() > 3 {
        return Err(format!("mano inválida: {t}"));
    }
    let (a, b) = (rank(cs[0]).ok_or(format!("rango inválido en {t}"))?, rank(cs[1]).ok_or(format!("rango inválido en {t}"))?);
    let (hi, lo) = (a.max(b), a.min(b));
    let kind = match (hi == lo, cs.get(2).map(|c| c.to_ascii_lowercase())) {
        (true, None) => Kind::Pair,
        (true, Some(_)) => return Err(format!("un par no lleva s/o: {t}")),
        (false, Some('s')) => Kind::Suited,
        (false, Some('o')) => Kind::Offsuit,
        (false, None) => Kind::Both,
        _ => return Err(format!("sufijo inválido en {t}")),
    };
    Ok((hi, lo, kind))
}

fn names(hi: usize, lo: usize, kind: Kind) -> Vec<String> {
    let (h, l) = (RANKS.as_bytes()[hi] as char, RANKS.as_bytes()[lo] as char);
    match kind {
        Kind::Pair => vec![format!("{h}{h}")],
        Kind::Suited => vec![format!("{h}{l}s")],
        Kind::Offsuit => vec![format!("{h}{l}o")],
        Kind::Both => vec![format!("{h}{l}s"), format!("{h}{l}o")],
    }
}

/// Expande un token (sin peso) a nombres de clase.
fn expand(tok: &str) -> Result<Vec<String>, String> {
    let tok = tok.trim();
    if tok.eq_ignore_ascii_case("random") || tok == "*" || tok == "100%" {
        let mut v = Vec::new();
        for i in 0..13 {
            for j in 0..13 {
                v.push(class_name(i, j));
            }
        }
        return Ok(v);
    }
    // `Ax`, `Axs`, `Axo` y con `+`: carta alta fija (o esa o más), cualquier kicker.
    let cs: Vec<char> = tok.chars().collect();
    if cs.len() >= 2 && cs[1].eq_ignore_ascii_case(&'x') {
        let plus = tok.ends_with('+');
        let core: String = cs.iter().filter(|&&c| c != '+').collect();
        let hi = rank(cs[0]).ok_or(format!("rango inválido en {tok}"))?;
        let kind = match core.chars().nth(2).map(|c| c.to_ascii_lowercase()) {
            None => Kind::Both,
            Some('s') => Kind::Suited,
            Some('o') => Kind::Offsuit,
            _ => return Err(format!("sufijo inválido en {tok}")),
        };
        let mut v = Vec::new();
        for h in hi..if plus { 13 } else { hi + 1 } {
            for k in 0..h {
                v.extend(names(h, k, kind));
            }
        }
        return Ok(v);
    }
    if let Some(base) = tok.strip_suffix('+') {
        let (hi, lo, kind) = parse_hand(base)?;
        let mut v = Vec::new();
        if kind == Kind::Pair {
            for r in hi..13 {
                v.extend(names(r, r, kind));
            }
        } else {
            for k in lo..hi {
                v.extend(names(hi, k, kind));
            }
        }
        return Ok(v);
    }
    if let Some((a, b)) = tok.split_once('-') {
        let (h1, l1, k1) = parse_hand(a)?;
        let (h2, l2, k2) = parse_hand(b)?;
        if k1 != k2 {
            return Err(format!("extremos de distinto tipo: {tok}"));
        }
        let mut v = Vec::new();
        if k1 == Kind::Pair {
            for r in h1.min(h2)..=h1.max(h2) {
                v.extend(names(r, r, k1));
            }
        } else if h1 == h2 {
            for k in l1.min(l2)..=l1.max(l2) {
                v.extend(names(h1, k, k1));
            }
        } else if h1 - l1 == h2 - l2 {
            let (top, bottom) = (h1.max(h2), h1.min(h2));
            let gap = h1 - l1;
            for h in bottom..=top {
                v.extend(names(h, h - gap, k1));
            }
        } else {
            return Err(format!("rango con guion no soportado: {tok}"));
        }
        return Ok(v);
    }
    let (hi, lo, kind) = parse_hand(tok)?;
    Ok(names(hi, lo, kind))
}

/// Rango como peso por combo (vector de 1328, carriles de relleno a 0).
pub fn parse_range(s: &str) -> Result<Vec<f32>, String> {
    let mut class_w: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for raw in s.split([',', ' ', '\t', ';']).map(str::trim).filter(|t| !t.is_empty() && (!t.ends_with('%') || *t == "100%")) {
        let (tok, w) = match raw.split_once(':') {
            Some((t, w)) => (t, w.trim().parse::<f32>().map_err(|_| format!("peso inválido en {raw}"))?),
            None => (raw, 1.0),
        };
        if !(0.0..=1.0).contains(&w) {
            return Err(format!("peso fuera de [0,1] en {raw}"));
        }
        for name in expand(tok)? {
            class_w.insert(name, w);
        }
    }
    let mut out = vec![0.0f32; HAND_STRIDE];
    for (h, o) in out.iter_mut().enumerate().take(NUM_COMBOS) {
        let (i, j) = grid_cell(h);
        *o = class_w.get(&class_name(i, j)).copied().unwrap_or(0.0);
    }
    Ok(out)
}

/// Formato de salida: clases con frecuencia > 0.5 %, peso explícito si no es ~100 %.
pub fn format_range(freq_by_combo: &[f32]) -> String {
    let mut sum = [[0.0f32; 13]; 13];
    let mut cnt = [[0u32; 13]; 13];
    for (h, &p) in freq_by_combo[..NUM_COMBOS].iter().enumerate() {
        let (i, j) = grid_cell(h);
        sum[i][j] += p;
        cnt[i][j] += 1;
    }
    let mut parts = Vec::new();
    for i in 0..13 {
        for j in 0..13 {
            let f = sum[i][j] / cnt[i][j] as f32;
            if f >= 0.995 {
                parts.push(class_name(i, j));
            } else if f > 0.005 {
                parts.push(format!("{}:{:.3}", class_name(i, j), f));
            }
        }
    }
    parts.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::report::class_frequency;

    fn count(s: &str) -> f32 {
        parse_range(s).unwrap().iter().sum()
    }

    #[test]
    fn notation() {
        assert_eq!(count("AA"), 6.0);
        assert_eq!(count("AKs"), 4.0);
        assert_eq!(count("AK"), 16.0);
        assert_eq!(count("22+"), 78.0);
        assert_eq!(count("A2s+"), 48.0);
        assert_eq!(count("K9o+"), 48.0);
        assert_eq!(count("T9s-65s"), 20.0);
        assert_eq!(count("99-66,A5s-A2s"), 24.0 + 16.0);
        assert_eq!(count("random"), 1326.0);
        assert_eq!(count("AKo:0.5"), 6.0);
        assert!(parse_range("AKx").is_err());
        // Notación.
        assert_eq!(count("Ax"), 12.0 * 16.0);
        assert_eq!(count("Kx+"), 23.0 * 16.0);
        assert_eq!(count("Qxs"), 10.0 * 4.0);
        assert_eq!(count("32.6%, 22+ Ax K5s+"), 78.0 + 192.0 + 32.0);
    }

    #[test]
    fn roundtrip() {
        let r = parse_range("22+,A2s+,KTo+,T9s-65s,AJo:0.5").unwrap();
        let s = format_range(&r);
        let back = parse_range(&s).unwrap();
        assert_eq!(r, back);
        assert_eq!(class_frequency(&r, 0, "AJo"), Some(0.5));
    }
}
