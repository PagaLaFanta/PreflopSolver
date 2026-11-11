//! CLI del solver preflop.
//!
//! ```text
//! solve --stacks 10,10 --iters 500
//! solve --stacks 12,20,15 --payouts 50,30,20 --iters 300
//! solve --stacks 25,25 --profile standard --boards 128 --iters 200 --show "SB:R3"
//! solve --stacks 15,30 --payouts 50,30,20 --bounties 10,10,10 --offtable 25
//! ```

use preflop_engine::abstraction::board_pool::{BoardPool, PoolConfig};
use preflop_engine::cfr::report::{format_grid, overall_frequency};
use preflop_engine::cfr::{CheckDown, PostflopEngine, PostflopEstimator, Solver, SolverConfig};
use preflop_engine::cfr::charts::{load_or_build_charts, load_or_build_nash_ranking, ChartSpec};
use preflop_engine::eval::board_bunch;
use preflop_engine::eval::equity::{EquityMatrix, RangeEquity};
use preflop_engine::io::export;
use preflop_engine::io::hh::{parse_hand, split_hands};
use preflop_engine::io::html::render as render_html;
use preflop_engine::io::ranges::{format_range, parse_range};
use preflop_engine::math::fgs::{auto_depth, Fgs, FgsMode, NextLevel};
use preflop_engine::math::PkoParams;
use preflop_engine::tree::{build_tree, Payoff, PreflopConfig, RaiseSize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// RAM física total y libre del sistema, en MB; `None` si
/// no se puede consultar (fuera de Windows, o si falla la llamada). Sin dependencias
/// externas: FFI directo a `GlobalMemoryStatusEx` de `kernel32.dll`.
#[cfg(windows)]
fn system_memory_mb() -> Option<(u64, u64)> {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut m = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) };
    (ok != 0).then_some((m.ull_total_phys / 1_000_000, m.ull_avail_phys / 1_000_000))
}
#[cfg(not(windows))]
fn system_memory_mb() -> Option<(u64, u64)> {
    None
}

/// Saca a `solve` del modo de eficiencia (EcoQoS) de Windows en CPUs híbridas: lanzado en
/// segundo plano (sin consola, desde un script…) Windows lo manda a los núcleos E y no pasa
/// de ~4 núcleos. `serve` ya lo hacía con el hijo; aquí lo hace el propio proceso,
/// se lance como se lance. Sin efecto en CPUs sin núcleos E ni fuera de Windows.
#[cfg(windows)]
fn disable_power_throttling() {
    #[repr(C)]
    struct ProcessPowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn SetProcessInformation(h: *mut std::ffi::c_void, class: i32, info: *mut ProcessPowerThrottlingState, size: u32) -> i32;
    }
    let mut state = ProcessPowerThrottlingState { version: 1, control_mask: 0x1, state_mask: 0 };
    unsafe {
        SetProcessInformation(GetCurrentProcess(), 4, &mut state, std::mem::size_of::<ProcessPowerThrottlingState>() as u32);
    }
}
#[cfg(not(windows))]
fn disable_power_throttling() {}

const HELP: &str = "uso: solve [icm] [opciones]
  icm                    calculadora ICM (y FGS con --fgs) de los stacks, sin resolver;
                         con --json, salida en JSON
  --spot fichero         lee opciones de un fichero (clave = valor por línea, # comenta)
  --compare fichero      compara la solución con rangos de referencia (TSV como --export)
  --target x             para al bajar la NashConv de x (con --iters como máximo)
  --target-ci x          para al bajar el CI de x (misma fórmula y escala: < 10
                         estable, ≤ 3 calidad de referencia; --ci-legacy para el anterior)
  --stop-file fichero    para de forma ordenada (como --target) si el fichero existe; se
                         comprueba en cada iteración
  --reset-strategies n   descarta la estrategia media en la iteración n (conserva regrets)
  --hh fichero --hand k  importa la mano k (1ª por defecto) de un historial de
                         PokerStars, GGPoker o Winamax (ante de BB y cabezas incluidos)
  --lock ruta@ACC=rango  fija la estrategia de un nodo (p.ej. \"@A=22+,A2s+\", \"SB:A@C=QQ+\")
  --freq ruta@ACC=f      bloquea la frecuencia de una acción (0-1); CFR elige los combos
  --edge ruta@ACC=x      edge manual (+ encarece, − subsidia), en unidades de utilidad
  --html fichero         informe HTML con rejillas coloreadas (--html-depth d, --title t)
  --export fichero       rangos de todos los nodos en notación estándar (TSV)
  --export-ev fichero    EV por acción y clase de mano de cada nodo (TSV, 169 valores
                         en el orden del grid; vacío si los rivales no llegan): en las
                         unidades del modelo ($ con ICM) y en fichas (bb netas)
  --export-json fichero  solución completa en JSON: configuración, EV por asiento y, por
                         nodo y acción, frecuencia y EV por clase; flops HU con rangos
  --export-pio fichero   flops heads-up: ruta, asientos, bote, stack efectivo y rangos
                         de llegada de ambos jugadores (TSV; rangos en formato Pio)
  --stacks a,b,..        stacks en bb en orden de acción (UTG..SB,BB; HU: SB,BB)  [10,10]
  --payouts p1,p2,..     premios restantes ⇒ ICM (sin ⇒ chip EV)
  --bounties b1,..       cabezas PKO (mesa + fuera de mesa); requiere --payouts
  --ko pko|sko|p         con --bounties: progresivo (`pko` = 50 %), estándar
                         (`sko` = 100 %, se cobra toda la cabeza) o un % propio de la sala
                         (p.ej. `--ko 100%` o `--ko 0.5`): esa fracción de la cabeza del
                         eliminado se cobra en efectivo ya; el resto crece en la cabeza
                         del eliminador (0 con `sko`)                             [pko]
  --bounty-power a       α del Bounty Power Model                                 [1.0]
  --offtable s1,..       stacks de jugadores de otras mesas (ICM multimesa); admite
                         \"NxS\" para N jugadores con S bb (p.ej. 16x25,2x40)
  --fgs d|auto           FGS: simula d manos futuras en push/fold antes del ICM (auto =
                         reglas: 2-4 jugadores 3 manos, 5-7 dos, resto una)
  --river-buckets ochs|percentile
                         river: ochs = k-means global de la fuerza contra 8
                         grupos de rivales, ~20 buckets por board [defecto]; percentile =
                         percentil de fuerza por board (mucho más fino)
  --fgs-ranking nash|equity
                         ranking de los rangos del FGS: nash (defecto, push + call del Nash
                         HU) o equity contra una mano al azar
  --ci-legacy            CI anterior (variación media por combo) en vez del actual
  --fgs-mode nash|charts nash: push/fold resuelto con ICM en cada
                         estado futuro; charts: tablas chip EV fijas (más rápido)
  --fgs-next-level d@sb/bb[/ante]  subida de ciegas en las manos simuladas del FGS desde
                         la profundidad d (1 = primera mano simulada), en bb del
                         nivel actual
  --ante x  --bb-ante    ante por jugador / ante de BB
  --sb x                 ciega pequeña en bb                                        [0.5]
  --profile p            pushfold | standard                                      [pushfold]
  --opens a,b --sb-opens a --bb-opens a --reraise m --max-raises k --no-limp
                         tamaños: 2.5bb | 3x | 75% | ai | 2.5bb+1bb (por limper) | 3x+1x (por caller)
  --allin-threshold p    un raise a más de p del stack efectivo es all-in (10% o 0.1;
                         [pushfold 50%, standard 40%])
  --no-bb-flat           con --no-call open, la BB tampoco puede pagar el open
  --no-closing-flats     quien cierra la acción no puede pagar si --flats/--no-call ya no
                         dejan (por defecto, cerrar la acción siempre
                         permite pagar, salte o no --flats)
  --no-cold-calls        frente a 3-bet o más, quien no ha actuado aún no paga salvo que
                         cierre la acción
  --flats l,o,3,4,5      pagos permitidos por raise (limps, open, 3-bet…; la BB
                         cuenta): 0,1,1,0,0 = un solo call del open y del 3-bet
  --limp                 open-limp desde cualquier posición
  --opens-by \"CO:2.2;BTN:2.5,3\"  tamaños de open por posición (grupos con ;, tamaños con ,)
  --reraise-ip m,..  --reraise-oop m,..   3-bet/4-bet en posición / fuera de posición
  --3bet s  --4bet s  --5bet s   tamaños por nivel (y --3bet-ip, --3bet-oop, --4bet-ip…);
                         amplían --max-raises hasta ese nivel. Ej.: --3bet 3x --3bet-oop 3.8x
                         --4bet 2.2x --5bet ai
  --no-call niveles      no se permite pagar frente a esos niveles: open (cold call/flat;
                         la BB siempre defiende), 3bet, 4bet, 5bet. Pagar un all-in, siempre
  --no-allin niveles     sin all-in como opción propia en esos niveles (open, 3bet, 4bet…)
  --prune \"ruta@etiqueta;...\"  quita una acción de un nodo concreto por su ruta exacta
                         (\"\" = raíz, como en --show/--lock), no todo un nivel (Prune
                         Action/Children); deja el nodo sin acciones es un error
  --limp-by \"SB,BTN\"   open-limp solo desde esas posiciones
  --save fichero  --load fichero      guardar / cargar (y continuar) una solución
  --subtree ruta         muestreo de subárbol: cada iteración recorre solo el
                         subárbol de ese nodo (p. ej. «CO:R2 BTN:F»), con los rangos de
                         entrada de la estrategia media; úsalo con --load para afinar un spot
  --warm fichero         arranque en caliente desde la solución de otro spot parecido (p.
                         ej. stacks vecinos): copia los nodos preflop con la misma ruta y
                         las mismas acciones y sigue desde ahí (el postflop empieza de cero)
  --max-in-pot k         máximo de jugadores activos en el bote, de 2 a 6
                                                                          [3]
  --postflop             push/fold: los botes limpeados se juegan postflop (MCCFR, Geo 66 %)
  --pf-regrets rm|plus   regrets postflop: regret matching clásico (defecto) o CFR+ (suelo
                         en 0, sesgado con muestreo)
  --pf-regret-power q     regrets postflop de la iteración t pesan t^q (1 = Linear CFR) [0]
  --pf-batch on|off      suma las actualizaciones de los boards de cada visita        [off]
  --leaf-infosets on|off infosets postflop propios de cada hoja de flop (auto: con ≤ 64)
  --checkdown            hojas de flop sin betting (en vez de MCCFR geométrico)
  Postflop por defecto: Geo 66 %, all-in como opción con SPR ≤ 5,
  raises geométricos y re-raises hasta 3 apuestas por mano (2 a tres), sin donk.
  --pf-raise geo|m|0     raises geométricos (defecto), a m× la apuesta, o sin raises
  --allin-spr x          all-in como opción extra con SPR ≤ x (0 = no)               [5]
  --donk                 permite apostar primero fuera de posición contra el agresor
  --pf-jam x             apuestas postflop ≥ x·stack pasan a all-in (1 = nunca)     [0.85]
  --donk-never           sin donk ni siquiera para quien agredió en una calle anterior
  --no-reraise           un solo raise por calle
  --flop-bets 33,66      tamaños de apuesta en el flop, en % del bote; sustituyen al
                         geométrico en el flop
  --vr on|off            reducción de varianza (VR-MCCFR) en las hojas postflop heads-up:
                         línea base por bucket del river + su media en el pool        [on]
  --vr-boards n          boards del pool para la media de la línea base               [1024]
  --flop-cbets / --flop-donks / --flop-raises a,b
                         tamaños del flop para la c-bet (quien subió el último preflop), el
                         donk y los raises (raise: % del bote tras pagar)
  --geo-size g           tamaño geométrico % (p.ej. 0.66): el geométrico más cercano
  --pf-max-bets n        apuestas máximas por mano postflop                         [3]
  --mw-boards n          boards por visita a un all-in a 3 o más en entrenamiento
                         [32 con ≤ 64 terminales a 3, si no 256]
  --mw-adaptive on|off   esos boards, proporcionales a la masa de los rivales      [on]
  --mw-pool n            runouts precalculados para los all-in a 3      [8192 / 2048]
  --mw-eval-boards n     runouts al evaluar EV / NashConv los all-in a 3 o más    [2048]
  --no-pf-raise-multiway sin raises en los botes a 3 o 4
  --geo-size-mw g        tamaño geométrico de los botes a 3 o 4        [= heads-up]
  --checkdown-after l    última calle con apuestas según los jugadores en la mano, para 2,
                         3, 4… jugadores (1 flop, 2 turn, 3 river, 0 ninguna)
                                                                   [3,3,2,1,0]
                         Los botes a 3 y 4 se juegan con el stack de cada uno: con stacks
                         distintos, los largos siguen apostando en el bote lateral.
  --flat-allin           push/fold: permite pagar un all-in que cubres aunque detrás quede
                         alguien que podría poner más (por defecto hay que resubir all-in)
  --no-preflop-bunching  sin card bunching preflop (las manos foldeadas por los demás no
                         cambian las de los que siguen)
  --no-board-bunching    sin board bunching preflop (el board de los all-in no ve las
                         cartas que se llevaron los que foldearon)
  --board-basis b        base del board bunching: poly (4 funciones) | full (13 rangos)
  --no-bunching          sin card removal de los rangos foldeados en los boards postflop
  --eval-boards n        boards postflop al evaluar EV / BR  [2^24 / nodos postflop, mín. 256]
  --boards n             runouts del pool postflop (se guarda en .cache/): con 128 la
                         equity de cada mano llevaba ±0,15 bb de error fijo         [4096]
  --pool stratified|random  flops del pool: estratificados por las 1755 clases
                         isomórficas ponderadas, o al azar                 [stratified]
  --adaptive-boards on|off  boards por visita proporcionales a la probabilidad de
                         llegar a cada flop (hasta --boards-per-visit)               [on]
  --prune-mass x         en entrenamiento, las hojas con probabilidad de llegar < x se
                         evalúan en check-down (dinámico; 0 = sin poda)         [1e-3]
  --avg-power p          peso t^p de la estrategia media preflop (0 = uniforme; 2 reduce
                         la NashConv a la mitad por iteración frente a 1; 3, un 13-20 %
                         más)                                                     [3]
  --pf-avg-power p       peso t^p de la media postflop MCCFR                      [3]
  --regret-power q       descuento de regrets: el regret de la iteración t pesa t^q
                         (1.5 en HU sin MCCFR, 0 en el resto)                  [auto]
  --no-pool-cache        construye el pool sin leer ni escribir la caché
  --buckets f,t,r        buckets de flop/turn/river                    [256,256,256]
  --boards-per-visit n   boards mínimos por visita a una hoja (más boards = menos ruido por
                         iteración; a igual tiempo, 64 deja la NashConv en ~0,6× la
                         de 16 en un 4-max 25 bb y en ~0,4× en un 6-max PKO)        [64]
  --updates alt|sim      updates alternados o simultáneos        [sim con 3+ jugadores]
  --iters n              iteraciones CFR+                                         [500]
  --report-every n       informe (CI) cada n iteraciones                          [100]
  --nashconv-every k     NashConv cada k informes (0 = solo al final); siempre en el
                         último                                  [1; 0 con MCCFR postflop]
  --equity-boards n      boards Monte Carlo de la matriz de equity (exact = todos) [exact]
  --equity-cache ruta    caché de la matriz                     [.cache/equity_<n>.bin]
  --show ruta            nodo a mostrar (p.ej. \"SB:A\"); repetible. Por defecto: profundidad ≤ 1
  --threads n";

/// `--offtable`: stacks sueltos o grupos `NxS`.
fn offtable_list(v: &str) -> Vec<f64> {
    let mut out = Vec::new();
    for tok in v.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        match tok.split_once(['x', 'X']) {
            Some((n, s)) => {
                let n: usize = n.trim().parse().unwrap_or_else(|_| panic!("--offtable: grupo inválido {tok}"));
                let s: f64 = s.trim().parse().unwrap_or_else(|_| panic!("--offtable: grupo inválido {tok}"));
                out.extend(std::iter::repeat_n(s, n));
            }
            None => out.push(tok.parse().unwrap_or_else(|_| panic!("--offtable: número inválido {tok}"))),
        }
    }
    out
}

/// Tamaños con la gramática, separados por `,` o `;`: `2.5bb`, `3x`, `75%`, `ai`,
/// `2.5bb+1bb` (por limper), `3x+1x` (por caller).
fn sizes(v: &str, default_mult: bool) -> Vec<RaiseSize> {
    v.split([',', ';']).map(str::trim).filter(|s| !s.is_empty()).map(|s| RaiseSize::parse(s, default_mult).unwrap_or_else(|e| panic!("{e}"))).collect()
}

fn list(v: &str) -> Vec<f64> {
    v.split(',').filter(|s| !s.is_empty()).map(|s| s.trim().parse().unwrap_or_else(|_| panic!("número inválido «{s}» en la lista «{v}» (las listas se separan con comas; los decimales van con punto)"))).collect()
}

fn main() {
    disable_power_throttling();
    // Errores de configuración legibles (sin traza): "error: <mensaje>". La interfaz lo
    // muestra tal cual.
    std::panic::set_hook(Box::new(|info| {
        let msg = info.payload().downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| info.payload().downcast_ref::<String>().cloned()).unwrap_or_else(|| "error interno".into());
        eprintln!("error: {msg}");
    }));
    let mut args: HashMap<String, String> = HashMap::new();
    let mut shows: Vec<String> = Vec::new();
    let mut locks: Vec<String> = Vec::new();
    let mut freqs: Vec<String> = Vec::new();
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    let icm_mode = argv.first().map(|a| a == "icm").unwrap_or(false);
    if icm_mode {
        argv.remove(0);
    }
    // Vista previa del árbol: solo
    // construye el árbol (barato, sin CFR ni matriz de equity) y lo imprime en JSON.
    let preview_mode = argv.first().map(|a| a == "preview").unwrap_or(false);
    if preview_mode {
        argv.remove(0);
    }
    // Fichero de spot (`--spot f`): líneas `clave = valor` con las mismas claves que las
    // opciones (sin `--`); `#` comenta; `show` y `lock` pueden repetirse. Las opciones de
    // la línea de comandos tienen prioridad.
    if let Some(pos) = argv.iter().position(|a| a == "--spot") {
        let path = argv.get(pos + 1).cloned().unwrap_or_else(|| panic!("falta fichero para --spot"));
        argv.drain(pos..pos + 2);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("no se pudo leer {path}: {e}"));
        let mut extra: Vec<String> = Vec::new();
        for line in text.lines() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let (k, v) = match line.split_once('=') {
                Some((k, v)) => (k.trim(), v.trim()),
                None => (line, ""),
            };
            extra.push(format!("--{k}"));
            if !v.is_empty() {
                extra.push(v.trim_matches('"').to_string());
            }
        }
        extra.append(&mut argv);
        argv = extra;
    }
    let mut i = 0;
    while i < argv.len() {
        let k = argv[i].trim_start_matches("--").to_string();
        if k == "help" || k == "h" {
            println!("{HELP}");
            return;
        }
        let flag = matches!(k.as_str(), "bb-ante" | "flat-allin" | "no-preflop-bunching" | "no-board-bunching" | "no-bb-flat" | "no-closing-flats" | "no-cold-calls" | "postflop" | "donk" | "donk-never" | "no-pf-raise-multiway" | "no-reraise" | "no-limp" | "checkdown" | "limp" | "pf-raise-multiway" | "no-bunching" | "json");
        let v = if flag { String::new() } else { i += 1; argv.get(i).cloned().unwrap_or_else(|| panic!("falta valor para --{k}")) };
        match k.as_str() {
            "show" => shows.push(v),
            "lock" => locks.push(v),
            "freq" => freqs.push(format!("f{v}")),
            "edge" => freqs.push(format!("e{v}")),
            _ => {
                args.insert(k, v);
            }
        }
        i += 1;
    }
    // Una opción mal escrita o retirada no debe ignorarse en silencio: quien la pasa cree que
    // se aplica. Vale lo que documenta la ayuda (más las variantes por nivel que la ayuda
    // abrevia con «…»).
    for k in args.keys() {
        let documented = HELP.match_indices(&format!("--{k}")).any(|(i, m)| !HELP[i + m.len()..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '-'));
        if !documented && !matches!(k.as_str(), "pf-raise-multiway" | "4bet-oop" | "5bet-ip" | "5bet-oop") {
            eprintln!("aviso: opción desconocida --{k}: se ignora (solve --help para la lista)");
        }
    }
    let get = |k: &str, d: &str| args.get(k).cloned().unwrap_or_else(|| d.to_string());
    let threads: usize = get("threads", &std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).to_string()).trim().parse().unwrap_or_else(|_| panic!("--threads: número inválido"));

    // Historial (PokerStars, GGPoker, Winamax): stacks, ciegas y antes en bb, en orden de acción.
    let mut hand: Option<preflop_engine::io::hh::ParsedHand> = None;
    if let Some(path) = args.get("hh") {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("no se pudo leer {path}: {e}"));
        let hands = split_hands(&text);
        let idx: usize = num::<usize>("hand", &get("hand", "1")).max(1) - 1;
        let h = parse_hand(hands.get(idx).unwrap_or_else(|| panic!("el fichero tiene {} manos", hands.len()))).unwrap_or_else(|e| panic!("{e}"));
        let names = preflop_engine::tree::builder::seat_names(h.players.len());
        eprintln!("mano {:?} #{} — ciegas {}/{} ante {}{}", h.site, h.hand_id, h.sb, h.bb, h.ante, if h.bb_ante { " (ante de BB)" } else { "" });
        for (k, (name, chips)) in h.players.iter().enumerate() {
            let hero = if h.hero_index() == Some(k) { format!("  ← héroe {}", h.hero_cards.clone().unwrap_or_default()) } else { String::new() };
            eprintln!("  {:<5} {:<20} {:>9} ({:.2} bb){hero}", names[k], name, chips, chips / h.bb);
        }
        hand = Some(h);
    }
    let stacks = match &hand {
        Some(h) => h.stacks_bb(),
        None => list(&get("stacks", "10,10")),
    };
    // Cabezas del historial (en la divisa de los premios) si no se pasan con --bounties. Con
    // --offtable, los de otras mesas llevan la cabeza media de la mesa (la misma estimación
    // que la interfaz cuando no se sabe).
    let hh_bounties = match (&hand, args.get("payouts"), args.get("bounties")) {
        (Some(h), Some(_), None) => h.bounties.as_ref().map(|b| {
            let off = args.get("offtable").map_or(0, |s| offtable_list(s).len());
            let mean = b.iter().sum::<f64>() / b.len() as f64;
            let all: Vec<f64> = b.iter().copied().chain(std::iter::repeat_n(mean, off)).collect();
            eprintln!("cabezas del historial: {b:?}{}", if off > 0 { format!(" + {off} de otras mesas a {mean:.2}") } else { String::new() });
            all.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
        }),
        _ => None,
    };
    let payoff = match (args.get("payouts"), args.get("bounties").or(hh_bounties.as_ref())) {
        (None, None) => Payoff::ChipEv,
        (Some(p), None) => Payoff::Icm { payouts: list(p) },
        (payouts, Some(b)) => {
            let ko = get("ko", "pko");
            let cash_fraction = match ko.as_str() {
                "sko" => 1.0,
                "pko" => 0.5,
                k => {
                    let x: f64 = num("ko", k.trim_end_matches('%'));
                    if x > 1.0 || k.ends_with('%') { x / 100.0 } else { x }
                }
            };
            let params = PkoParams { cash_fraction, bounty_power: num("bounty-power", &get("bounty-power", "1.0")) };
            // Sin --payouts: bounties sin ICM, en la misma unidad que --stacks (bb), no $ —
            // (ver nota de Payoff::PkoChipEv): la fórmula de cash +
            // crecimiento de cabeza no necesita ICM para tener sentido.
            match payouts {
                Some(p) => Payoff::Pko { payouts: list(p), bounties: list(b), params },
                None => Payoff::PkoChipEv { bounties: list(b), params },
            }
        }
    };
    let mut cfg = match get("profile", "pushfold").as_str() {
        "standard" => PreflopConfig::standard(stacks, payoff),
        _ => PreflopConfig::push_fold(stacks, payoff),
    };
    cfg.offtable_stacks = args.get("offtable").map(|s| offtable_list(s)).unwrap_or_default();
    cfg.ante = num("ante", &get("ante", "0"));
    if let Some(v) = args.get("sb") {
        cfg.sb = num("sb", v);
    }
    if let Some(h) = &hand {
        cfg.sb = h.sb / h.bb;
        cfg.ante = h.ante / h.bb;
    }
    cfg.bb_ante = args.contains_key("bb-ante") || hand.as_ref().is_some_and(|h| h.bb_ante);
    if args.contains_key("flat-allin") {
        cfg.shove_over_allin = false;
    }
    if let Some(v) = args.get("opens") { cfg.open_sizes = sizes(v, false); }
    if let Some(v) = args.get("sb-opens") { cfg.sb_open_sizes = sizes(v, false); }
    if let Some(v) = args.get("bb-opens") { cfg.bb_open_sizes = sizes(v, false); }
    if let Some(v) = args.get("reraise") { cfg.reraise_mults = sizes(v, true); }
    if let Some(v) = args.get("reraise-ip") { cfg.reraise_mults_ip = Some(sizes(v, true)); }
    if let Some(v) = args.get("reraise-oop") { cfg.reraise_mults_oop = Some(sizes(v, true)); }
    // Umbral de all-in: un raise a más de ese % del stack efectivo es all-in.
    if let Some(v) = args.get("allin-threshold") {
        let x: f64 = num("allin-threshold", v.trim_end_matches('%'));
        cfg.allin_threshold = if x > 1.0 || v.ends_with('%') { x / 100.0 } else { x };
    }
    if args.contains_key("no-bb-flat") { cfg.bb_defends = false; }
    if args.contains_key("no-closing-flats") { cfg.allow_closing_flats = false; }
    if args.contains_key("no-cold-calls") { cfg.allow_cold_calls = false; }
    // Flats por raise: "0,1,1,0,0" = limps, open, 3-bet, 4-bet, 5-bet.
    if let Some(v) = args.get("flats") { cfg.max_flats = list(v).iter().map(|&x| x as u8).collect(); }
    if let Some(v) = args.get("opens-by") {
        let names = preflop_engine::tree::builder::seat_names(cfg.stacks.len());
        cfg.open_sizes_by_seat = vec![None; cfg.stacks.len()];
        for part in v.split(';').filter(|p| !p.is_empty()) {
            let (pos, sizes) = part.split_once(':').unwrap_or_else(|| panic!("--opens-by: falta ':' en {part}"));
            let k = names.iter().position(|n| n.eq_ignore_ascii_case(pos.trim())).unwrap_or_else(|| panic!("--opens-by: posición desconocida {pos} ({names:?})"));
            cfg.open_sizes_by_seat[k] = Some(self::sizes(&sizes.replace(',', ";"), false));
        }
    }
    if let Some(v) = args.get("max-raises") { cfg.max_raises = num("max-raises", v); }
    // Tamaños por nivel: --3bet, --4bet, --5bet… con variantes -ip / -oop. Dar
    // tamaños para un nivel amplía el límite de raises hasta él (5-bet ⇒ 4 raises).
    for n in 3..=7usize {
        for (suffix, which) in [("", 0), ("-ip", 1), ("-oop", 2)] {
            if let Some(v) = args.get(&format!("{n}bet{suffix}")) {
                let level = n - 3;
                if cfg.level_sizes.len() <= level {
                    cfg.level_sizes.resize(level + 1, Default::default());
                }
                let sz = Some(sizes(v, true));
                let l = &mut cfg.level_sizes[level];
                match which {
                    0 => l.all = sz,
                    1 => l.ip = sz,
                    _ => l.oop = sz,
                }
                cfg.max_raises = cfg.max_raises.max(n as u8 - 1);
            }
        }
    }
    // Niveles: open, 3bet, 4bet, 5bet… → índice 0, 1, 2, 3…
    let levels = |flag: &str, v: &str| -> Vec<bool> {
        let mut out: Vec<bool> = Vec::new();
        for tok in v.split(',').map(|t| t.trim().to_ascii_lowercase()).filter(|t| !t.is_empty()) {
            let k = match tok.as_str() {
                "open" | "2bet" => 0,
                t => t.strip_suffix("bet").and_then(|n| n.parse::<usize>().ok()).filter(|&n| n >= 3).map(|n| n - 2).unwrap_or_else(|| panic!("--{flag}: nivel desconocido {tok} (open, 3bet, 4bet, 5bet…)")),
            };
            if out.len() <= k {
                out.resize(k + 1, true);
            }
            out[k] = false;
        }
        out
    };
    if let Some(v) = args.get("no-call") { cfg.call_allowed = levels("no-call", v); }
    if let Some(v) = args.get("no-allin") {
        let mut a = levels("no-allin", v);
        for x in a.iter_mut().filter(|x| **x) {
            *x = cfg.allow_allin;
        }
        cfg.allin_allowed = a;
    }
    if let Some(v) = args.get("prune") {
        for entry in v.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let (path, label) = entry.split_once('@').unwrap_or_else(|| panic!("--prune: formato ruta@etiqueta (ruta vacía = raíz): {entry:?}"));
            cfg.pruned.push((path.trim().to_string(), label.trim().to_string()));
        }
    }
    if let Some(v) = args.get("limp-by") {
        let names = preflop_engine::tree::builder::seat_names(cfg.stacks.len());
        cfg.limp_seats = vec![false; cfg.stacks.len()];
        for pos in v.split([',', ';']).map(str::trim).filter(|p| !p.is_empty()) {
            let k = names.iter().position(|n| n.eq_ignore_ascii_case(pos)).unwrap_or_else(|| panic!("--limp-by: posición desconocida {pos} ({names:?})"));
            cfg.limp_seats[k] = true;
        }
    }
    if args.contains_key("no-limp") { cfg.allow_sb_limp = false; }
    if args.contains_key("limp") { cfg.allow_limp = true; }
    if let Some(v) = args.get("max-in-pot") { cfg.max_players_in_pot = num("max-in-pot", v); }
    // Push/fold con postflop: los botes limpeados (SB completa, BB pasa) se juegan con
    // apuestas geométricas (Geo 66 %), en vez de ir a showdown sin apuestas.
    if args.contains_key("postflop") && cfg.postflop.is_none() {
        cfg.postflop = Some(preflop_engine::abstraction::geometric::GeometricConfig::classic());
    }
    if args.contains_key("checkdown") { cfg.postflop = None; }
    if let Some(v) = args.get("leaf-infosets") { cfg.leaf_infosets = Some(v != "off"); }
    if let (Some(v), Some(geo)) = (args.get("pf-raise"), cfg.postflop.as_mut()) {
        if v == "geo" {
            geo.raise_geometric = true;
        } else {
            let m: f64 = num("pf-raise", v);
            geo.raise_mult = (m > 1.0).then_some(m);
            geo.raise_geometric = false;
        }
    }
    if let Some(geo) = cfg.postflop.as_mut() {
        if args.contains_key("pf-raise-multiway") {
            geo.raise_multiway = true;
        }
        if args.contains_key("no-pf-raise-multiway") {
            geo.raise_multiway = false;
        }
        if let Some(g) = args.get("geo-size-mw") {
            geo.geo_size_multiway = Some(num("geo-size-mw", g));
        }
        if let Some(g) = args.get("geo-size") {
            geo.geo_size = Some(num("geo-size", g));
        }
        if let Some(m) = args.get("pf-max-bets") {
            geo.max_bets = num("pf-max-bets", m);
        }
        if let Some(x) = args.get("allin-spr") {
            geo.allin_spr = num("allin-spr", x);
        }
        if args.contains_key("donk") {
            geo.donk = true;
        }
        if args.contains_key("donk-never") {
            geo.donk_limited = false;
        }
        // Apuestas postflop ≥ x·stack se convierten en all-in (1 = nunca:
        // solo añade el all-in como opción con SPR bajo).
        if let Some(v) = args.get("pf-jam") {
            geo.jam_threshold = num("pf-jam", v);
        }
        // Última calle con apuestas por número de jugadores (2, 3, 4…): "3,3,2,1,0".
        if let Some(v) = args.get("checkdown-after") {
            geo.checkdown_after = list(v).iter().map(|&x| x as u8).collect();
        }
        // Tamaños del flop en % del bote ("33,66"; apuestas / c-bets / donks /
        // Raises): sustituyen al geométrico en el flop.
        for (k, dst) in [("flop-bets", &mut geo.flop_bets), ("flop-cbets", &mut geo.flop_cbets), ("flop-donks", &mut geo.flop_donks), ("flop-raises", &mut geo.flop_raises)] {
            if let Some(v) = args.get(k) {
                *dst = list(v).iter().map(|x| x / 100.0).collect();
            }
        }
        if args.contains_key("no-reraise") {
            geo.reraise = false;
        }
    }

    if preview_mode {
        let tree = build_tree(cfg.clone());
        let names = preflop_engine::tree::builder::seat_names(cfg.stacks.len());
        let mut nodes = String::from("[");
        for k in 0..tree.actions.len() {
            if k > 0 {
                nodes.push(',');
            }
            let labels: Vec<String> = tree.labels[k].iter().map(|l| export::json_string(l)).collect();
            nodes.push_str(&format!(
                "{{\"path\":{},\"seat\":{},\"actions\":[{}]}}",
                export::json_string(&tree.paths[k]),
                export::json_string(names[tree.actions[k].player as usize]),
                labels.join(",")
            ));
        }
        nodes.push(']');
        let pf_nodes: usize = tree.leaves.iter().map(|l| l.nodes.len() + l.multi.as_ref().map_or(0, |m| m.nodes.len())).sum();
        let arena_mb = (tree.total_rows * 1328 * 2 * 2) as f64 / 1e6;
        let ram = system_memory_mb().map_or("null,\"ram_avail_mb\":null".to_string(), |(t, a)| format!("{t},\"ram_avail_mb\":{a}"));
        println!(
            "{{\"nodes\":{nodes},\"stats\":{{\"action_nodes\":{},\"terminals\":{},\"flop_leaves\":{},\"postflop_nodes\":{pf_nodes},\"arena_mb\":{:.2},\"ram_total_mb\":{ram}}}}}",
            tree.actions.len(),
            tree.terminals.len(),
            tree.leaves.len(),
            arena_mb,
        );
        return;
    }
    if icm_mode && args.get("fgs").is_none() {
        icm_table(&cfg, None, args.contains_key("json"));
        return;
    }
    // 0 = exacta (todos los boards por órbitas de palos).
    let eq_boards: u32 = match get("equity-boards", "exact").as_str() {
        "exact" => 0,
        v => num("equity-boards", v),
    };
    let cache = PathBuf::from(get("equity-cache", &if eq_boards == 0 { ".cache/equity_exact.bin".to_string() } else { format!(".cache/equity_{eq_boards}.bin") }));
    let t0 = Instant::now();
    let eq = Arc::new(EquityMatrix::load_or_build(&cache, eq_boards, 0x5EED, threads));
    eprintln!("matriz de equity: {} ({:.1}s)", if eq.boards == 0 { "exacta".to_string() } else { format!("{} boards", eq.boards) }, t0.elapsed().as_secs_f64());

    if let Some(d) = args.get("fgs") {
        if matches!(cfg.payoff, Payoff::ChipEv | Payoff::PkoChipEv { .. }) {
            eprintln!("aviso: --fgs solo tiene efecto con ICM (--payouts); en chip EV (con o sin bounties en fichas) se ignora");
        }
        let t0 = Instant::now();
        let spec = ChartSpec::new(cfg.sb, cfg.bb, cfg.ante, cfg.bb_ante);
        let charts = load_or_build_charts(std::path::Path::new(".cache"), eq.clone(), &spec);
        // Ranking de los rangos "top x %" del FGS: por defecto "Nash combinado" (push + call
        // del Nash push/fold HU en la malla de stacks);
        // `--fgs-ranking equity` para la equity contra una mano al azar.
        let ranges = match get("fgs-ranking", "nash").as_str() {
            "nash" => RangeEquity::from_matrix_with_order(&eq, 50, load_or_build_nash_ranking(std::path::Path::new(".cache"), eq.clone(), &spec)),
            "equity" => RangeEquity::from_matrix(&eq, 50),
            o => panic!("--fgs-ranking: valor desconocido «{o}» (nash | equity)"),
        };
        // `--fgs auto`: reglas de profundidad según los jugadores de la mesa.
        let depth: u8 = if d == "auto" { auto_depth(cfg.stacks.len()) } else { num("fgs", d) };
        let mode = match get("fgs-mode", "nash").as_str() {
            "charts" => FgsMode::Charts,
            "nash" => FgsMode::Nash,
            other => panic!("--fgs-mode: valor desconocido «{other}» (nash | charts)"),
        };
        cfg.fgs = Some(Arc::new(Fgs {
            depth,
            sb: cfg.sb,
            bb: cfg.bb,
            ante: cfg.ante,
            bb_ante: cfg.bb_ante,
            prune: 1e-4,
            charts: Arc::new(charts),
            ranges: Arc::new(ranges),
            field: None,
            mode,
            nash_iters: (200, 50),
            cache: Arc::default(),
            three: Arc::default(),
            next_level: args.get("fgs-next-level").map(|v| NextLevel::parse(v).unwrap_or_else(|e| panic!("--fgs-next-level: {e}"))),
        }));
        eprintln!(
            "FGS profundidad {depth} ({}): tablas push/fold y equity de rangos listas ({:.1}s)",
            if mode == FgsMode::Nash { "push/fold resuelto en cada estado" } else { "tablas chip EV" },
            t0.elapsed().as_secs_f64()
        );
    }
    if icm_mode {
        icm_table(&cfg, cfg.fgs.clone(), args.contains_key("json"));
        return;
    }
    let t0 = Instant::now();
    // Con ICM/PKO, tabla de fichas del mismo árbol para dar también el EV en fichas.
    // Con ICM/PKO hay dos unidades (ICM$ y fichas) y merece la pena mostrar las dos; PKO en
    // fichas ya está en fichas de por sí, como chip EV — nada que contrastar.
    let chip_utils = (!matches!(cfg.payoff, Payoff::ChipEv | Payoff::PkoChipEv { .. })).then(|| preflop_engine::tree::builder::chip_utils(&cfg));
    let tree = build_tree(cfg);
    let pf_nodes: usize = tree.leaves.iter().map(|l| l.nodes.len() + l.multi.as_ref().map_or(0, |m| m.nodes.len())).sum();
    eprintln!(
        "árbol: {} nodos de acción, {} terminales, {} hojas de flop ({} nodos postflop), {:.1} MB de arenas ({:.2}s)",
        tree.actions.len(),
        tree.terminals.len(),
        tree.leaves.len(),
        pf_nodes,
        (tree.total_rows * 1328 * 2 * 2) as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    let mut solver_cfg = SolverConfig::default();
    solver_cfg.threads = threads;
    if let Some(v) = args.get("regret-power") {
        solver_cfg.regret_power = Some(v.parse().unwrap_or_else(|_| panic!("--regret-power: número inválido {v}")));
    }
    if let Some(v) = args.get("avg-power") {
        solver_cfg.avg_power = v.parse().unwrap_or_else(|_| panic!("--avg-power: número inválido {v}"));
    }
    solver_cfg.simultaneous = args.get("updates").map(|u| u == "sim");
    solver_cfg.bunching = !args.contains_key("no-preflop-bunching");
    if let Some(v) = args.get("mw-adaptive") {
        solver_cfg.multiway_adaptive = v == "on";
    }
    if let Some(v) = args.get("mw-boards") {
        solver_cfg.multiway_boards_per_visit = num("mw-boards", v);
    }
    if let Some(v) = args.get("mw-pool") {
        solver_cfg.multiway_pool = num("mw-pool", v);
    }
    if let Some(v) = args.get("mw-eval-boards") {
        solver_cfg.multiway_eval_boards = num("mw-eval-boards", v);
    }
    let board_bunch = (tree.n >= 3 && solver_cfg.bunching && !args.contains_key("no-board-bunching")).then(|| {
        let (basis, name) = match get("board-basis", "poly").as_str() {
            "full" => (board_bunch::full_basis(), "full"),
            _ => (board_bunch::default_basis(), "poly"),
        };
        let t0 = Instant::now();
        let path = PathBuf::from(format!(".cache/boardbunch_{name}.bin"));
        let bb = board_bunch::BoardBunching::load_or_build(&path, basis, threads);
        eprintln!("board bunching: base {name} ({:.1}s)", t0.elapsed().as_secs_f64());
        Arc::new(bb)
    });
    if tree.cfg.postflop.is_some() && !tree.leaves.is_empty() {
        let b = list(&get("buckets", "256,256,256"));
        let pool_cfg = PoolConfig {
            boards: num("boards", &get("boards", "4096")),
            buckets: [b[0] as u16, b[1] as u16, b[2] as u16],
            threads,
            stratified: get("pool", "stratified") != "random",
            river_ochs: match get("river-buckets", "ochs").as_str() {
                "ochs" => true,
                "percentile" => false,
                o => panic!("--river-buckets: valor desconocido «{o}» (ochs | percentile)"),
            },
            ..PoolConfig::default()
        };
        let t0 = Instant::now();
        let pool = if args.contains_key("no-pool-cache") { BoardPool::build(&pool_cfg) } else { BoardPool::load_or_build(&pool_cfg, std::path::Path::new(".cache")) };
        eprintln!("pool postflop: {} boards, buckets {:?} ({:.1}s)", pool.boards.len(), pool.num_buckets, t0.elapsed().as_secs_f64());
        // Tabla de infosets a la medida: nodos de decisión × buckets, con holgura (lleno, se
        // perderían actualizaciones).
        let decisions: usize = tree
            .leaves
            .iter()
            .map(|l| {
                l.nodes.iter().filter(|n| matches!(n, preflop_engine::tree::postflop::PfNode::Decision { .. })).count()
                    + l.multi.as_ref().map_or(0, |m| m.nodes.iter().filter(|n| matches!(n, preflop_engine::tree::postflop::PfNode3::Decision { .. })).count())
            })
            .sum();
        let need = decisions * *pool.num_buckets.iter().max().unwrap_or(&32) as usize * 2;
        let log2 = (need.max(1 << 22) as f64).log2().ceil() as u32;
        let mut engine = PostflopEngine::new(pool, log2.min(28), num("boards-per-visit", &get("boards-per-visit", "64")));
        engine.bunching = !args.contains_key("no-bunching");
        engine.adaptive_boards = get("adaptive-boards", "on") == "on";
        engine.prune_mass = num("prune-mass", &get("prune-mass", "1e-3"));
        engine.threads = threads;
        // Reducción de varianza (VR-MCCFR) en el valor de las hojas heads-up al entrenar.
        engine.vr = get("vr", "on") == "on";
        engine.vr_boards = num("vr-boards", &get("vr-boards", "1024"));
        engine.avg_power = num("pf-avg-power", &get("pf-avg-power", "3"));
        engine.regret_plus = get("pf-regrets", "rm") == "plus";
        engine.regret_power = num("pf-regret-power", &get("pf-regret-power", "0"));
        engine.batch_updates = get("pf-batch", "off") == "on";
        // Boards al evaluar (EV, NashConv): todo el pool con pocas hojas; con muchas, un
        // presupuesto fijo repartido (6-max: 303 hojas → 256 boards, ~1 min por informe).
        // Presupuesto por nodos postflop (un recorrido de evaluación cuesta ~nodos × boards):
        // 4-max deep (3 114 nodos) → todo el pool; 6 jugadores con botes a 3 (14 116) → ~1 200.
        let pf_nodes: usize = tree.leaves.iter().map(|l| l.nodes.len() + l.multi.as_ref().map_or(0, |m| m.nodes.len())).sum();
        let auto_eval = ((1usize << 24) / pf_nodes.max(1)).clamp(256, usize::MAX);
        engine.eval_boards = args.get("eval-boards").map(|v| num("eval-boards", v)).unwrap_or(auto_eval);
        let mut s = Solver::new(tree, eq, engine, solver_cfg);
        s.board_bunch = board_bunch;
        run(s, &get, &shows, &locks, &freqs, 0, chip_utils);
    } else {
        let est = CheckDown { eq: eq.clone() };
        let mut s = Solver::new(tree, eq, est, solver_cfg);
        s.board_bunch = board_bunch;
        run(s, &get, &shows, &locks, &freqs, 1, chip_utils);
    }
}

/// Calculadora ICM / FGS (subcomando `icm`).
fn icm_table(cfg: &PreflopConfig, fgs: Option<Arc<Fgs>>, json: bool) {
    let payouts = match &cfg.payoff {
        Payoff::Icm { payouts } | Payoff::Pko { payouts, .. } => payouts.clone(),
        Payoff::ChipEv | Payoff::PkoChipEv { .. } => panic!("el subcomando icm necesita --payouts"),
    };
    if let Some(field) = preflop_engine::tree::builder::field_for(cfg) {
        let fgs = fgs.map(|f| Fgs { field: Some(field.clone()), ..(*f).clone() });
        return icm_table_field(cfg, &field, fgs.as_ref(), &payouts, json);
    }
    let all: Vec<f64> = cfg.stacks.iter().chain(&cfg.offtable_stacks).copied().collect();
    let mut icm = vec![0.0; all.len()];
    preflop_engine::math::icm_equity_precise(&all, &payouts, &mut icm);
    let mut fg = vec![0.0; all.len()];
    if let Some(f) = &fgs {
        f.equity(&(0..cfg.stacks.len()).collect::<Vec<_>>(), &all, &payouts, &mut fg);
    }
    // Bubble factor y risk premium (fila = quien decide, columna = rival).
    let n_table = cfg.stacks.len();
    let table: Vec<usize> = (0..n_table).collect();
    let model = |idx: &[usize], s: &[f64], p: &[f64], o: &mut [f64]| match &fgs {
        Some(f) => {
            let t: Vec<usize> = (0..idx.len()).filter(|&k| idx[k] < n_table).collect();
            f.equity(&t, s, p, o)
        }
        None => preflop_engine::math::icm_equity(s, p, o),
    };
    let bf = preflop_engine::math::bubble_factors_detail(&all, &payouts, &table, &model);
    let names = preflop_engine::tree::builder::seat_names(cfg.stacks.len());
    let total: f64 = all.iter().sum();
    let seat_name = |k: usize| if k < names.len() { names[k].to_string() } else { format!("otra{}", k - names.len() + 1) };
    if json {
        let seats: Vec<String> = (0..all.len())
            .map(|k| {
                let f = if fgs.is_some() { format!(",\"fgs\":{:.6}", fg[k]) } else { String::new() };
                format!("{{\"name\":{},\"stack\":{},\"chips\":{:.5},\"icm\":{:.6}{f}}}", export::json_string(&seat_name(k)), all[k], all[k] / total, icm[k])
            })
            .collect();
        println!("{{\"seats\":[{}],\"table\":{n_table},\"fgs\":{},\"bubble\":{},\"risk\":{},\"ev_base\":{},\"ev_win\":{},\"ev_lose\":{}}}", seats.join(","), fgs.as_ref().map_or(0, |f| f.depth), detail_mat(&bf, |d| d.bubble_factor), detail_mat(&bf, |d| d.risk_premium), detail_mat(&bf, |d| d.ev_base), detail_mat(&bf, |d| d.ev_win), detail_mat(&bf, |d| d.ev_lose));
        return;
    }
    println!("{:<8} {:>9} {:>8} {:>10}{}", "asiento", "stack", "% fichas", "ICM", if fgs.is_some() { "        FGS" } else { "" });
    for (k, &st) in all.iter().enumerate() {
        let name = seat_name(k);
        let f = if fgs.is_some() { format!(" {:>10.4}", fg[k]) } else { String::new() };
        println!("{:<8} {:>9.2} {:>7.2}% {:>10.4}{f}", name, st, st / total * 100.0, icm[k]);
    }
    print_bf_tables(&bf, &table, &names);
}

/// Serializa una matriz de [`preflop_engine::math::MatchupDetail`] a JSON extrayendo un campo.
fn detail_mat(bf: &[Vec<Option<preflop_engine::math::MatchupDetail>>], pick: impl Fn(&preflop_engine::math::MatchupDetail) -> f64) -> String {
    let rows: Vec<String> = bf.iter().map(|r| format!("[{}]", r.iter().map(|c| c.as_ref().map_or("null".into(), |d| format!("{:.4}", pick(d)))).collect::<Vec<_>>().join(","))).collect();
    format!("[{}]", rows.join(","))
}

/// Imprime las tablas de texto de bubble factor y risk premium (usadas por `icm_table` y
/// `icm_table_field`, que comparten el mismo formato).
fn print_bf_tables(bf: &[Vec<Option<preflop_engine::math::MatchupDetail>>], table: &[usize], names: &[&str]) {
    for (title, pick) in [("Bubble factor", 0usize), ("Risk premium (% de equity extra para pagar un all-in)", 1)] {
        println!("\n{title}");
        print!("{:<8}", "");
        for &j in table {
            print!(" {:>7}", names[j]);
        }
        println!();
        for (r, &i) in table.iter().enumerate() {
            print!("{:<8}", names[i]);
            for c in 0..table.len() {
                match &bf[r][c] {
                    Some(d) => {
                        if pick == 0 {
                            print!(" {:>7.2}", d.bubble_factor)
                        } else {
                            print!(" {:>6.1}%", d.risk_premium * 100.0)
                        }
                    }
                    None => print!(" {:>7}", "-"),
                }
            }
            println!();
        }
    }
}

/// Calculadora ICM para campos grandes (ICM de campo, `math::field_icm`): los asientos de
/// la mesa y una fila con la media de las otras mesas (su equity total es el resto de los
/// premios), más bubble factor y risk premium de la mesa.
fn icm_table_field(cfg: &PreflopConfig, field: &preflop_engine::math::field_icm::FieldIcm, fgs: Option<&Fgs>, payouts: &[f64], json: bool) {
    let n = cfg.stacks.len();
    let mut eq = vec![0.0; n];
    field.equity(&cfg.stacks, payouts.len(), &mut eq);
    // Con FGS (sobre la mesa, con el ICM de campo al final de cada rama).
    let mut fg = vec![0.0; n];
    if let Some(f) = fgs {
        f.equity(&(0..n).collect::<Vec<_>>(), &cfg.stacks, payouts, &mut fg);
    }
    let n_off = cfg.offtable_stacks.len();
    let off_total: f64 = cfg.offtable_stacks.iter().sum();
    let pool: f64 = payouts.iter().take(n + n_off).sum();
    let off_eq = (pool - eq.iter().sum::<f64>()) / n_off as f64;
    let total = field.total;
    let names = preflop_engine::tree::builder::seat_names(n);
    let model = |idx: &[usize], s: &[f64], p: &[f64], o: &mut [f64]| {
        let t: Vec<usize> = (0..idx.len()).filter(|&k| idx[k] < n).collect();
        let st: Vec<f64> = t.iter().map(|&k| s[k]).collect();
        let mut e = vec![0.0; t.len()];
        match fgs {
            Some(f) => f.equity(&(0..st.len()).collect::<Vec<_>>(), &st, p, &mut e),
            None => field.equity(&st, p.len(), &mut e),
        }
        o.fill(0.0);
        for (k, &i) in t.iter().enumerate() {
            o[i] = e[k];
        }
    };
    let all: Vec<f64> = cfg.stacks.iter().chain(&cfg.offtable_stacks).copied().collect();
    let table: Vec<usize> = (0..n).collect();
    let bf = preflop_engine::math::bubble_factors_detail(&all, payouts, &table, &model);
    if json {
        let mut seats: Vec<String> = (0..n)
            .map(|k| {
                let f = if fgs.is_some() { format!(",\"fgs\":{:.6}", fg[k]) } else { String::new() };
                format!("{{\"name\":{},\"stack\":{},\"chips\":{:.5},\"icm\":{:.6}{f}}}", export::json_string(names[k]), cfg.stacks[k], cfg.stacks[k] / total, eq[k])
            })
            .collect();
        seats.push(format!("{{\"name\":{},\"stack\":{:.2},\"chips\":{:.5},\"icm\":{:.6},\"group\":{n_off}}}", export::json_string(&format!("otras ({n_off}), media")), off_total / n_off as f64, off_total / total / n_off as f64, off_eq));
        println!("{{\"seats\":[{}],\"table\":{n},\"fgs\":{},\"field\":{},\"bubble\":{},\"risk\":{},\"ev_base\":{},\"ev_win\":{},\"ev_lose\":{}}}", seats.join(","), fgs.map_or(0, |f| f.depth), n + n_off, detail_mat(&bf, |d| d.bubble_factor), detail_mat(&bf, |d| d.risk_premium), detail_mat(&bf, |d| d.ev_base), detail_mat(&bf, |d| d.ev_win), detail_mat(&bf, |d| d.ev_lose));
        return;
    }
    println!("ICM de campo: {} jugadores ({} en la mesa), {} premios", n + n_off, n, payouts.len());
    println!("{:<14} {:>9} {:>8} {:>10}{}", "asiento", "stack", "% fichas", "ICM", if fgs.is_some() { "        FGS" } else { "" });
    for k in 0..n {
        let f = if fgs.is_some() { format!(" {:>10.4}", fg[k]) } else { String::new() };
        println!("{:<14} {:>9.2} {:>7.3}% {:>10.4}{f}", names[k], cfg.stacks[k], cfg.stacks[k] / total * 100.0, eq[k]);
    }
    println!("{:<14} {:>9.2} {:>7.3}% {:>10.4}", format!("otras ({n_off})"), off_total / n_off as f64, off_total / total / n_off as f64 * 100.0, off_eq);
    print_bf_tables(&bf, &table, &names);
}

/// Compara la solución con rangos de referencia (TSV `ruta \t asiento \t acción \t rango`,
/// el mismo formato que `--export`; el asiento se ignora). Por nodo y acción: frecuencia
/// media nuestra y de la referencia, y el % de combos en desacuerdo (|Δ| > 0.5).
fn compare<E: PostflopEstimator>(s: &Solver<E>, path: &str) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("no se pudo leer {path}: {e}"));
    println!("\n=== Comparación con {path} ===");
    println!("{:<34} {:<6} {:>8} {:>8} {:>9} {:>9}", "nodo", "acción", "nuestra", "ref", "Δmedia", "desac.%");
    let (mut worst, mut total, mut count) = (0.0f32, 0.0f32, 0usize);
    for line in text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 4 {
            eprintln!("línea ignorada (faltan columnas): {line}");
            continue;
        }
        let node = if cols[0] == "(raíz)" { "" } else { cols[0] };
        let Some(k) = s.tree.find(node) else {
            eprintln!("nodo no encontrado en nuestro árbol: {}", cols[0]);
            continue;
        };
        let Some(a) = s.tree.labels[k].iter().position(|l| l == cols[2]) else {
            eprintln!("acción {} no existe en {} ({:?})", cols[2], cols[0], s.tree.labels[k]);
            continue;
        };
        let reference = match parse_range(cols[3]) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("rango inválido en {}: {e}", cols[0]);
                continue;
            }
        };
        let ours = s.average_strategy(k);
        let row = &ours[a * 1328..(a + 1) * 1328];
        let (mut sum_d, mut disagree) = (0.0f32, 0usize);
        for h in 0..1326 {
            let d = (row[h] - reference[h]).abs();
            sum_d += d;
            if d > 0.5 {
                disagree += 1;
            }
        }
        let mean_d = sum_d / 1326.0;
        let dis = disagree as f32 / 13.26;
        worst = worst.max(dis);
        total += dis;
        count += 1;
        println!("{:<34} {:<6} {:>7.1}% {:>7.1}% {:>8.3} {:>8.1}%", cols[0], cols[2], overall_frequency(&ours, a) * 100.0, reference[..1326].iter().sum::<f32>() / 13.26, mean_d, dis);
    }
    if count > 0 {
        println!("{count} comparaciones · desacuerdo medio {:.1}% · máximo {:.1}%", total / count as f32, worst);
    }
}

fn run<E: PostflopEstimator>(mut s: Solver<E>, get: &dyn Fn(&str, &str) -> String, shows: &[String], locks: &[String], freqs: &[String], nc_default: u32, chip_utils: Option<Vec<f32>>) {
    // Depuración: `CONSERVE=1` (tras las iteraciones) imprime, por terminal, la suma sobre los
    // jugadores de su EV en fichas netas (debería ser 0).
    let conserve = |s: &mut Solver<E>| {
        if std::env::var("CONSERVE").is_err() {
            return;
        }
        let rows = match &chip_utils {
            Some(u) => s.with_utils(u.clone(), |s| s.terminal_breakdown()),
            None => s.terminal_breakdown(),
        };
        // Probabilidad del terminal según cada jugador (utilidad 1 en todos los outcomes).
        let ones = vec![1.0f32; s.tree.utils.len()];
        let probs = s.with_utils(ones, |s| s.terminal_breakdown());
        let stacks = s.tree.cfg.stacks.clone();
        let mut by_kind: std::collections::BTreeMap<String, (f64, f64, f64)> = Default::default();
        let mut rows: Vec<_> = rows
            .into_iter()
            .zip(probs)
            .map(|((p, idx, e), (_, _, pr))| {
                let net: Vec<f64> = (0..e.len()).map(|i| e[i] - pr[i] * stacks[i]).collect();
                let (lo, hi) = pr.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                (net.iter().sum::<f64>(), p, idx, net, pr, hi - lo)
            })
            .collect();
        for (sum, _, idx, _, pr, spread) in &rows {
            let t = &s.tree.terminals[*idx];
            let k = format!("{:?}/{}", t.kind, (t.alive_mask as u32).count_ones());
            let x = by_kind.entry(k).or_default();
            x.0 += sum;
            x.1 += pr.iter().sum::<f64>() / pr.len() as f64;
            x.2 += spread;
        }
        for (k, (sum, p, spread)) in &by_kind {
            println!("conservación {k:16} Σ fichas netas {sum:+.5} bb  P {p:.4}  Σ(Pmáx-Pmín) {spread:.5}");
        }
        rows.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
        for (sum, p, idx, net, pr, _) in rows.iter().take(25) {
            println!("  {sum:+.5}  {:?} {p}  netas {:?}  P {:?}", s.tree.terminals[*idx].kind, net.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>(), pr.iter().map(|x| (x * 1e5).round() / 1e5).collect::<Vec<_>>());
        }
    };
    // Depuración: `DUMP_TERM=ruta` imprime las utilidades de los terminales hijos de ese nodo
    // (`U[outcome][asiento]`) y termina.
    // Depuración: `CHECK_LEAVES=1` comprueba que en cada final de las hojas multiway la suma
    // de las utilidades (con chip EV: stacks finales) sea la misma en todos los outcomes.
    if std::env::var("CHECK_LEAVES").is_ok() {
        let n = s.tree.n;
        let (mut worst, mut cnt) = (0.0f32, 0usize);
        let mut sums = std::collections::BTreeMap::<i64, usize>::new();
        for leaf in &s.tree.leaves {
            let Some(m) = &leaf.multi else { continue };
            for node in &m.nodes {
                if let preflop_engine::tree::PfNode3::End { alive, util } = node {
                    let k = preflop_engine::tree::nodes::factorial(alive.count_ones() as usize);
                    for o in 0..k {
                        let row = &s.tree.utils[*util as usize + o * n..*util as usize + (o + 1) * n];
                        let t: f32 = row.iter().sum();
                        *sums.entry((t * 100.0).round() as i64).or_default() += 1;
                        cnt += 1;
                        let _ = &mut worst;
                    }
                }
            }
        }
        println!("finales multiway: {cnt} outcomes; sumas (centésimas de bb → veces): {sums:?}");
        return;
    }
    if let Ok(node) = std::env::var("DUMP_TERM") {
        let k = s.tree.find(&node).unwrap_or_else(|| panic!("nodo no encontrado: {node:?}"));
        for (a, label) in s.tree.labels[k].iter().enumerate() {
            let c = s.tree.actions[k].children[a];
            if c.kind() != preflop_engine::tree::NodeKind::Terminal {
                println!("{label}: nodo");
                continue;
            }
            let t = &s.tree.terminals[c.index()];
            println!("{label}: {:?} players {:?} outcomes {} seats {}", t.kind, t.players, t.num_outcomes, t.num_seats);
            for row in t.utilities(&s.tree.utils).chunks(t.num_seats as usize) {
                println!("  {}", row.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" "));
            }
        }
        return;
    }
    // Bloqueo de frecuencias / edges: "ruta@ACCIÓN=valor".
    for spec in freqs {
        let (kind, rest) = spec.split_at(1);
        let (node, rest) = rest.split_once('@').unwrap_or_else(|| panic!("falta '@' en {rest}"));
        let (label, value) = rest.split_once('=').unwrap_or_else(|| panic!("falta '=' en {rest}"));
        let k = s.tree.find(node).unwrap_or_else(|| panic!("nodo no encontrado: {node:?}"));
        let a = s.tree.labels[k].iter().position(|l| l == label).unwrap_or_else(|| panic!("acción {label} no existe en {node:?} ({:?})", s.tree.labels[k]));
        let x: f32 = value.parse().unwrap_or_else(|_| panic!("valor inválido: {value}"));
        if kind == "f" {
            s.lock_frequency(k, a, x);
            eprintln!("frecuencia bloqueada: {node:?} {label} = {x}");
        } else {
            s.set_edge(k, a, x);
            eprintln!("edge: {node:?} {label} = {x}");
        }
    }
    // Node locking: "ruta@ACCIÓN=rango" (ruta vacía = raíz), p.ej. "SB:A@C=QQ+,AK".
    for spec in locks {
        let (node, rest) = spec.split_once('@').unwrap_or_else(|| panic!("--lock: falta '@' en {spec}"));
        let (label, range) = rest.split_once('=').unwrap_or_else(|| panic!("--lock: falta '=' en {spec}"));
        let k = s.tree.find(node).unwrap_or_else(|| panic!("--lock: nodo no encontrado: {node:?}"));
        let a = s.tree.labels[k].iter().position(|l| l == label).unwrap_or_else(|| panic!("--lock: acción {label} no existe en {node:?} ({:?})", s.tree.labels[k]));
        let r = parse_range(range).unwrap_or_else(|e| panic!("--lock: {e}"));
        s.lock(k, a, &r);
        eprintln!("nodo bloqueado: {node:?} {label} = {range}");
    }
    if let Some(path) = Some(get("warm", "")).filter(|p| !p.is_empty()) {
        match s.warm_start_file(std::path::Path::new(&path)) {
            Ok(n) => eprintln!("arranque en caliente: {path}: {n} de {} nodos preflop (iteración {})", s.tree.paths.len(), s.iteration),
            Err(e) => panic!("--warm {path}: {e}"),
        }
    }
    if let Some(path) = Some(get("load", "")).filter(|p| !p.is_empty()) {
        match s.load(std::path::Path::new(&path)) {
            Ok(()) => eprintln!("solución cargada: {path} (iteración {})", s.iteration),
            Err(e) => panic!("--load {path}: {e}"),
        }
    }
    if let Some(p) = Some(get("subtree", "\u{1}")).filter(|p| p != "\u{1}") {
        s.set_subtree(Some(&p)).unwrap_or_else(|e| panic!("--subtree: {e}"));
        eprintln!("muestreo de subárbol: {p:?}");
    }
    // Con arranque en caliente, `--iters` cuenta las iteraciones nuevas.
    let warm_base = if get("warm", "").is_empty() { 0 } else { s.iteration };
    let iters: u32 = warm_base + num::<u32>("iters", &get("iters", "500"));
    let every: u32 = num::<u32>("report-every", &get("report-every", "100")).max(1);
    let nc_every: u32 = num("nashconv-every", &get("nashconv-every", &nc_default.to_string()));
    let mut reports = 0u32;
    let target: f64 = num("target", &get("target", "0"));
    let target_ci: f64 = num("target-ci", &get("target-ci", "0"));
    let reset_at: u32 = num("reset-strategies", &get("reset-strategies", "0"));
    let stop_file = get("stop-file", "");
    let export_path = get("export", "");
    let export_names = preflop_engine::tree::builder::seat_names(s.tree.n);
    // Se llama en cada informe (no solo al final) para que la interfaz pueda ir mostrando
    // la estrategia mientras converge, aunque todavía sea mala — como el árbol, que
    // también se actualiza en vivo.
    let write_export = |s: &Solver<_>| {
        if export_path.is_empty() {
            return;
        }
        let mut out = String::new();
        for k in 0..s.tree.actions.len() {
            let strat = s.average_strategy(k);
            let path_k = if s.tree.paths[k].is_empty() { "(raíz)" } else { &s.tree.paths[k] };
            for (a, label) in s.tree.labels[k].iter().enumerate() {
                let row = &strat[a * 1328..(a + 1) * 1328];
                out.push_str(&format!("{path_k}\t{}\t{label}\t{}\n", export_names[s.tree.actions[k].player as usize], format_range(row)));
            }
        }
        std::fs::write(&export_path, out).expect("escribir rangos");
    };
    let ev_path = get("export-ev", "");
    // Igual que `write_export`: el EV por mano también se ve en vivo en la interfaz (si no,
    // el panel de EV se queda vacío hasta terminar, aunque la estrategia ya se vea).
    let write_ev = |s: &mut Solver<_>| {
        if ev_path.is_empty() {
            return;
        }
        let evs = s.action_evs();
        let chips = match &chip_utils {
            Some(u) => s.with_utils(u.clone(), |s| s.action_evs()),
            None => evs.clone(),
        };
        std::fs::write(&ev_path, export::ev_tsv(s, &evs, &chips)).expect("escribir EV");
    };
    let mut snapshot = None;
    // CI con la fórmula (por defecto) o el indicador anterior (`--ci-legacy`).
    let ci_legacy = get("ci-legacy", "-") != "-";
    let mut ci_state: Option<preflop_engine::cfr::CiState> = None;
    if !ci_legacy {
        // Primera instantánea al empezar (su CI mide desde el inicio del cálculo).
        s.convergence_index(&mut ci_state);
    }
    let mut did_reset = false;
    let (mut t_solve, mut t_br, mut t_ev) = (0.0f64, 0.0f64, 0.0f64);
    let mut stopped = false;
    while s.iteration < iters {
        if !stop_file.is_empty() && std::path::Path::new(&stop_file).exists() {
            eprintln!("detenido por el usuario (checkpoint en la iteración {})", s.iteration);
            break;
        }
        let mut step = every.min(iters - s.iteration);
        if reset_at > 0 && !did_reset && s.iteration < reset_at {
            step = step.min(reset_at - s.iteration);
        }
        let t0 = Instant::now();
        // Comprueba --stop-file en cada iteración, no solo entre informes: con árboles
        // grandes esperar a todo un `--report-every` podía ser decenas de segundos sin
        // atender la parada, y el servidor acababa matando el proceso en seco a los 30s
        // (ver POST /api/stop en serve.rs). `run_until` construye el pool de hilos una
        // sola vez para el bloque entero, como `run`, así que esto no añade coste cuando
        // no se pide parar.
        s.run_until(step, || {
            let hit = !stop_file.is_empty() && std::path::Path::new(&stop_file).exists();
            stopped |= hit;
            hit
        });
        t_solve += t0.elapsed().as_secs_f64();
        if stopped {
            eprintln!("detenido por el usuario (checkpoint en la iteración {})", s.iteration);
            break;
        }
        if reset_at > 0 && !did_reset && s.iteration >= reset_at {
            s.reset_strategies();
            did_reset = true;
            snapshot = None;
            ci_state = None;
            if !ci_legacy {
                s.convergence_index(&mut ci_state);
            }
            eprintln!("estrategias reiniciadas (regrets conservados); la cuenta de iteraciones vuelve a 0");
            continue;
        }
        let ci = if ci_legacy { s.convergence_indicator(&mut snapshot) } else { s.convergence_index(&mut ci_state) };
        let ci_txt = ci.map(|c| format!("{c:.2}")).unwrap_or_else(|| "-".into());
        reports += 1;
        write_export(&s);
        // El EV por mano en vivo cuesta como una evaluación entera del árbol (en un 3-max
        // 20 bb, ~100 s frente a ~20 s de 25 iteraciones): con informes frecuentes dominaba
        // el tiempo total. Se recalcula solo si lo gastado en EV no pasa del 25 % del
        // tiempo de CFR (siempre en el primer informe y al terminar).
        if t_ev == 0.0 || t_ev <= 0.25 * t_solve {
            let t0 = Instant::now();
            write_ev(&mut s);
            t_ev += t0.elapsed().as_secs_f64();
        }
        // La NashConv (EV + mejor respuesta con el pool postflop entero) cuesta tanto como
        // decenas de iteraciones: con postflop se mide cada `nashconv-every` informes y
        // siempre en el último; el CI (barato) en todos.
        let last = s.iteration >= iters;
        if !last && (nc_every == 0 || !reports.is_multiple_of(nc_every)) {
            eprintln!("iter {:>6}  CFR {:>7.1}s  BR {:>6.1}s  CI = {ci_txt}", s.iteration, t_solve, t_br);
            if let (true, Some(c)) = (target_ci > 0.0, ci) {
                if c <= target_ci {
                    eprintln!("objetivo de CI alcanzado");
                    break;
                }
            }
            continue;
        }
        let t0 = Instant::now();
        let gains = s.best_response_gains();
        t_br += t0.elapsed().as_secs_f64();
        eprintln!(
            "iter {:>6}  CFR {:>7.1}s  BR {:>6.1}s  NashConv preflop = {:.5}  CI = {}  (por jugador {:?})",
            s.iteration,
            t_solve,
            t_br,
            gains.iter().sum::<f64>(),
            ci_txt,
            gains.iter().map(|g| (g * 1e4).round() / 1e4).collect::<Vec<_>>()
        );
        if target > 0.0 && gains.iter().sum::<f64>() <= target {
            eprintln!("objetivo de NashConv alcanzado");
            break;
        }
        if let (true, Some(c)) = (target_ci > 0.0, ci) {
            if c <= target_ci {
                eprintln!("objetivo de CI alcanzado");
                break;
            }
        }
    }
    // `expected_values` recorre todo el árbol a coste de un Best Response (igual que el
    // NashConv final) -- al parar a mano (--stop-file) no merece la pena pagarlo dos veces
    // (EV + EV en fichas) solo para un resumen: el usuario quiere parar ya, y el EV por
    // mano de la última iteración ya se ve en la interfaz (write_ev, por informe).
    conserve(&mut s);
    if !stopped {
        let ev = s.expected_values();
        println!("\nEV por asiento: {:?}", ev.iter().map(|x| (x * 1e3).round() / 1e3).collect::<Vec<_>>());
        // En fichas: bb netas por asiento (con ICM/PKO, la misma estrategia con la tabla de fichas).
        let chip_ev = match &chip_utils {
            Some(u) => s.with_utils(u.clone(), |s| s.expected_values()),
            None => ev.clone(),
        };
        println!("EV por asiento en fichas (bb netas): {:?}", chip_ev.iter().zip(&s.tree.cfg.stacks).map(|(x, st)| ((x - st) * 1e3).round() / 1e3).collect::<Vec<_>>());
    }
    if let Some(path) = Some(get("save", "")).filter(|p| !p.is_empty()) {
        s.save(std::path::Path::new(&path)).expect("guardar solución");
        eprintln!("solución guardada: {path}");
    }
    if let Some(path) = Some(get("html", "")).filter(|p| !p.is_empty()) {
        let depth: usize = num("html-depth", &get("html-depth", "2"));
        std::fs::write(&path, render_html(&s, &get("title", "Solución preflop"), depth)).expect("escribir HTML");
        eprintln!("informe HTML: {path}");
    }
    if let Some(path) = Some(get("compare", "")).filter(|p| !p.is_empty()) {
        compare(&s, &path);
    }
    if !export_path.is_empty() {
        write_export(&s);
        eprintln!("rangos exportados: {export_path}");
    }
    if !ev_path.is_empty() {
        write_ev(&mut s);
        eprintln!("EV exportados: {ev_path}");
    }
    let json_path = get("export-json", "");
    if !json_path.is_empty() {
        let evs = s.action_evs();
        // La misma estrategia medida en fichas (con ICM/PKO, con la tabla de fichas).
        let (chips, chip_seats) = match &chip_utils {
            Some(u) => s.with_utils(u.clone(), |s| (s.action_evs(), s.expected_values())),
            None => (evs.clone(), s.expected_values()),
        };
        std::fs::write(&json_path, export::solution_json(&s, &evs, &chips, &chip_seats)).expect("escribir JSON");
        eprintln!("solución en JSON: {json_path}");
    }
    if let Some(path) = Some(get("export-pio", "")).filter(|p| !p.is_empty()) {
        std::fs::write(&path, export::pio_tsv(&s)).expect("escribir flops");
        eprintln!("flops HU para PioSolver: {path}");
    }
    let names = preflop_engine::tree::builder::seat_names(s.tree.n);
    let nodes: Vec<usize> = if shows.is_empty() {
        (0..s.tree.actions.len()).filter(|&k| s.tree.paths[k].split(' ').filter(|x| !x.is_empty()).count() <= 1).collect()
    } else {
        shows.iter().filter_map(|p| s.tree.find(p).or_else(|| { eprintln!("nodo no encontrado: {p}"); None })).collect()
    };
    for k in nodes {
        let strat = s.average_strategy(k);
        let who = names[s.tree.actions[k].player as usize];
        let path = if s.tree.paths[k].is_empty() { "(raíz)" } else { &s.tree.paths[k] };
        println!("\n=== {path} → decide {who} ===");
        for (a, label) in s.tree.labels[k].iter().enumerate() {
            println!("-- {label}: {:.1}% de combos", overall_frequency(&strat, a) * 100.0);
            if label != "F" {
                print!("{}", format_grid(&strat, a));
            }
        }
    }
}

/// Número de una opción: admite coma decimal (`0,125`) y, si no se entiende, dice qué
/// opción falla en vez de un error de Rust.
fn num<T: std::str::FromStr>(name: &str, v: &str) -> T {
    let t = v.trim().replace(',', ".");
    t.parse().unwrap_or_else(|_| panic!("--{name}: número inválido «{v}»"))
}
