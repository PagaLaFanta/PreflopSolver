//! Presentación de estrategias: rejilla 13×13 (pares en la diagonal, suited arriba,
//! offsuit abajo) y frecuencias por clase de mano.

use crate::eval::{COMBO_CARDS, HAND_STRIDE, NUM_COMBOS};

const RANKS: &[u8; 13] = b"23456789TJQKA";

/// Celda (fila, columna) de la rejilla, con el As en la esquina superior izquierda.
pub fn grid_cell(combo: usize) -> (usize, usize) {
    let [x, y] = COMBO_CARDS[combo];
    let (r1, r2) = ((x >> 2) as usize, (y >> 2) as usize);
    let (hi, lo) = (r1.max(r2), r1.min(r2));
    let (i, j) = (12 - hi, 12 - lo);
    if x & 3 == y & 3 {
        (i, j) // suited: arriba de la diagonal
    } else {
        (j, i) // offsuit/par: abajo (el par cae en la diagonal)
    }
}

/// Combos de cada una de las 169 clases de mano, en el orden de la rejilla
/// (`fila·13 + columna` de [`grid_cell`]: AA = 0, suited arriba, offsuit abajo).
pub static CLASS_COMBOS: std::sync::LazyLock<Vec<Vec<u16>>> = std::sync::LazyLock::new(|| {
    let mut out = vec![Vec::new(); 169];
    for h in 0..NUM_COMBOS {
        let (i, j) = grid_cell(h);
        out[i * 13 + j].push(h as u16);
    }
    out
});

pub fn class_name(row: usize, col: usize) -> String {
    let (a, b) = (RANKS[12 - row.min(col)] as char, RANKS[12 - row.max(col)] as char);
    match row.cmp(&col) {
        std::cmp::Ordering::Equal => format!("{a}{a}"),
        std::cmp::Ordering::Less => format!("{a}{b}s"),
        std::cmp::Ordering::Greater => format!("{a}{b}o"),
    }
}

/// Frecuencia media de `action` por celda.
pub fn class_grid(strategy: &[f32], action: usize) -> [[f32; 13]; 13] {
    let row = &strategy[action * HAND_STRIDE..(action + 1) * HAND_STRIDE];
    let mut sum = [[0.0f32; 13]; 13];
    let mut cnt = [[0u32; 13]; 13];
    for (h, &p) in row[..NUM_COMBOS].iter().enumerate() {
        let (i, j) = grid_cell(h);
        sum[i][j] += p;
        cnt[i][j] += 1;
    }
    for i in 0..13 {
        for j in 0..13 {
            sum[i][j] /= cnt[i][j] as f32;
        }
    }
    sum
}

/// Frecuencia de `action` para una clase como `"AKs"`, `"72o"`, `"TT"`.
pub fn class_frequency(strategy: &[f32], action: usize, class: &str) -> Option<f32> {
    let g = class_grid(strategy, action);
    for i in 0..13 {
        for j in 0..13 {
            if class_name(i, j) == class {
                return Some(g[i][j]);
            }
        }
    }
    None
}

/// Fracción de combos (ponderada por combos, no por clases) que toman `action`.
pub fn overall_frequency(strategy: &[f32], action: usize) -> f32 {
    strategy[action * HAND_STRIDE..action * HAND_STRIDE + NUM_COMBOS].iter().sum::<f32>() / NUM_COMBOS as f32
}

pub fn format_grid(strategy: &[f32], action: usize) -> String {
    let g = class_grid(strategy, action);
    let mut s = String::from("     ");
    for &r in RANKS.iter().rev() {
        s.push_str(&format!("  {}  ", r as char));
    }
    s.push('\n');
    for (i, row) in g.iter().enumerate() {
        s.push_str(&format!("  {}  ", RANKS[12 - i] as char));
        for &p in row {
            s.push_str(&format!(" {:>3.0} ", p * 100.0));
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Card, Combo};

    #[test]
    fn grid_layout() {
        let aks = Combo::from_cards(Card::new(12, 0), Card::new(11, 0)).0 as usize;
        let ako = Combo::from_cards(Card::new(12, 0), Card::new(11, 1)).0 as usize;
        let tt = Combo::from_cards(Card::new(8, 0), Card::new(8, 1)).0 as usize;
        assert_eq!(grid_cell(aks), (0, 1));
        assert_eq!(grid_cell(ako), (1, 0));
        assert_eq!(grid_cell(tt), (4, 4));
        assert_eq!(class_name(0, 1), "AKs");
        assert_eq!(class_name(1, 0), "AKo");
        assert_eq!(class_name(12, 7), "72o");
    }
}
