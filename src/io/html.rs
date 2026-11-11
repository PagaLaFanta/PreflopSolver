//! Informe HTML autocontenido (sin dependencias externas) con rejillas coloreadas:
//! por cada nodo, una rejilla 13×13 donde cada celda muestra la mezcla de acciones como
//! barras horizontales proporcionales, más frecuencias por acción, EV por asiento y la
//! NashConv del perfil.

use crate::cfr::report::{class_grid, class_name, overall_frequency};
use crate::cfr::{PostflopEstimator, Solver};
use crate::tree::builder::seat_names;

fn color(label: &str, i: usize, n_raises: usize) -> String {
    match label {
        "F" => "#5b8def".into(),
        "X" | "C" => "#3fae6a".into(),
        "A" => "#8e1f2f".into(),
        _ => {
            // Raises: rojos de más claro a más oscuro según el tamaño.
            let t = if n_raises <= 1 { 0.5 } else { i as f64 / (n_raises - 1) as f64 };
            let l = 62.0 - 18.0 * t;
            format!("hsl(8, 72%, {l:.0}%)")
        }
    }
}

fn label_name(l: &str) -> String {
    match l {
        "F" => "Fold".into(),
        "X" => "Check".into(),
        "C" => "Call".into(),
        "A" => "All-in".into(),
        r => format!("Raise {}", r.trim_start_matches('R')),
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Genera el informe. `max_depth`: acciones previas máximas de los nodos incluidos.
pub fn render<E: PostflopEstimator>(s: &Solver<E>, title: &str, max_depth: usize) -> String {
    let names = seat_names(s.tree.n);
    let eval = s.evaluate();
    let nash_conv: f64 = eval.iter().map(|(e, b)| b - e).sum();
    let mut h = String::new();
    h.push_str(&format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{}</title><style>
:root{{--bg:#fafaf7;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ecebe6;--muted:#9a9993;--card:#1f1f1d;--line:#34332f}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:980px;margin:auto}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:0 0 8px}}
.muted{{color:var(--muted)}} .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}
table.ev{{border-collapse:collapse}} table.ev td,table.ev th{{padding:4px 10px;border-bottom:1px solid var(--line);text-align:right}}
.grid{{display:grid;grid-template-columns:repeat(13,1fr);gap:2px;max-width:640px}}
.cell{{position:relative;aspect-ratio:1.35;border-radius:3px;overflow:hidden;display:flex;background:var(--line)}}
.cell span{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#fff;text-shadow:0 0 2px #000a}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}} .legend i{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
nav a{{margin-right:10px;white-space:nowrap}} a{{color:inherit}}
</style></head><body><main>
<h1>{}</h1><p class="muted">{} players · iterations: {} · NashConv preflop: {:.5}</p>"#,
        escape(title),
        escape(title),
        s.tree.n,
        s.iteration,
        nash_conv
    ));
    h.push_str(r#"<div class="card"><h2>EV per seat</h2><table class="ev"><tr><th style="text-align:left">Seat</th><th>Stack</th><th>EV</th><th>Gain from deviating</th></tr>"#);
    for (k, (ev, br)) in eval.iter().enumerate() {
        h.push_str(&format!("<tr><td style=\"text-align:left\">{}</td><td>{:.2}</td><td>{:.4}</td><td>{:.5}</td></tr>", names[k], s.tree.cfg.stacks[k], ev, br - ev));
    }
    h.push_str("</table></div>");

    let nodes: Vec<usize> = (0..s.tree.actions.len()).filter(|&k| s.tree.paths[k].split(' ').filter(|x| !x.is_empty()).count() <= max_depth).collect();
    h.push_str("<div class=\"card\"><h2>Nodes</h2><nav>");
    for &k in &nodes {
        let p = if s.tree.paths[k].is_empty() { "(root)".to_string() } else { s.tree.paths[k].clone() };
        h.push_str(&format!("<a href=\"#n{k}\">{}</a> ", escape(&p)));
    }
    h.push_str("</nav></div>");

    for &k in &nodes {
        let strat = s.average_strategy(k);
        let labels = &s.tree.labels[k];
        let n_raises = labels.iter().filter(|l| l.starts_with('R')).count();
        let mut ri = 0;
        let colors: Vec<String> = labels
            .iter()
            .map(|l| {
                let c = color(l, ri, n_raises);
                if l.starts_with('R') {
                    ri += 1;
                }
                c
            })
            .collect();
        let who = names[s.tree.actions[k].player as usize];
        let path = if s.tree.paths[k].is_empty() { "(root)".to_string() } else { s.tree.paths[k].clone() };
        h.push_str(&format!("<div class=\"card\" id=\"n{k}\"><h2>{} → {who} to act</h2><div class=\"legend\">", escape(&path)));
        for (a, l) in labels.iter().enumerate() {
            h.push_str(&format!("<span><i style=\"background:{}\"></i>{} {:.1}%</span>", colors[a], label_name(l), overall_frequency(&strat, a) * 100.0));
        }
        h.push_str("</div><div class=\"grid\">");
        let grids: Vec<[[f32; 13]; 13]> = (0..labels.len()).map(|a| class_grid(&strat, a)).collect();
        for i in 0..13 {
            for j in 0..13 {
                let name = class_name(i, j);
                let mut bars = String::new();
                let mut tip = name.clone();
                for (a, g) in grids.iter().enumerate() {
                    let f = g[i][j];
                    if f > 0.001 {
                        bars.push_str(&format!("<div style=\"width:{:.2}%;background:{}\"></div>", f * 100.0, colors[a]));
                        tip.push_str(&format!(" · {} {:.0}%", label_name(&labels[a]), f * 100.0));
                    }
                }
                h.push_str(&format!("<div class=\"cell\" title=\"{tip}\">{bars}<span>{name}</span></div>"));
            }
        }
        h.push_str("</div></div>");
    }
    h.push_str("</main></body></html>\n");
    h
}
