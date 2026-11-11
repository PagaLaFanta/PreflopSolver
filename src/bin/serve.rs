//! Servidor local de la interfaz gráfica (HTTP mínimo, sin dependencias).
//!
//! ```text
//! serve [--port 8765] [--ui ui]      → http://127.0.0.1:8765/
//! ```
//!
//! API (JSON; los POST requieren la cabecera `X-Solver: 1`, que un navegador no envía
//! desde otra web sin preflight CORS, así ninguna página ajena puede lanzar cálculos):
//!
//! | Ruta | Qué hace |
//! |---|---|
//! | `GET /` | `ui/app.html` (la interfaz) (`/guia`, la guía de uso) |
//! | `POST /api/icm` | cuerpo `clave=valor` por línea (`stacks`, `payouts`, `offtable`, `fgs`, `ante`, `bb-ante`): ICM / FGS, bubble factor y risk premium (con `solve icm --json`) |
//! | `POST /api/solve` | cuerpo: argumentos de `solve`, uno por línea; lanza el cálculo |
//! | `GET /api/progress?from=n` | líneas de registro desde la `n`, estado y código de salida |
//! | `POST /api/stop` | detiene el cálculo |
//! | `GET /api/nodes` | nodos de la solución: ruta, asiento, acciones y frecuencia global |
//!
//! Las rutas de solución (`nodes`, `node`, `flops`, `pio`, `export`, `export.json`) aceptan `job=g` para leer el cálculo `g` de la sesión (una mano abierta de
//! la interfaz); sin él, el último terminado. `--continue g` continúa el cálculo `g` y
//! `--warm-job g` arranca en caliente desde él (otro spot parecido).
//! | `GET /api/node?path=p` | estrategia y EV del nodo como rejilla 13×13 por acción |
//! | `GET /api/export`, `/api/export.json` | rangos (TSV) y solución completa (JSON) |
//! | `GET /api/flops`, `/api/pio?path=p` | flops heads-up y script de PioSolver de uno |
//! | `GET /api/spots`, `GET/POST /api/spot?name=` | ficheros `.spot` |
//!
//! El cálculo se ejecuta con el binario `solve` (junto a este) y la solución se lee de su
//! `--export`: la interfaz usa exactamente el mismo código que el CLI.

use preflop_engine::cfr::report::class_grid;
use preflop_engine::io::ranges::parse_range;
use preflop_engine::io::export::{json_string as q, pio_script};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Default)]
struct Job {
    lines: Vec<String>,
    child: Option<Child>,
    running: bool,
    exit: Option<i32>,
    export: Option<PathBuf>,
    /// Fichero de EV por acción de la última solución terminada.
    ev: Option<PathBuf>,
    /// Solución en JSON y flops HU (`--export-json`, `--export-pio`).
    json: Option<PathBuf>,
    pio: Option<PathBuf>,
    /// Solución binaria (`--save`) de la última ejecución terminada, para continuar.
    bin: Option<PathBuf>,
    generation: u64,
}

struct App {
    ui: PathBuf,
    solve: PathBuf,
    work: PathBuf,
    spots: PathBuf,
    job: Mutex<Job>,
    /// Último ping de la interfaz (modo aplicación: cierre al cerrar la ventana).
    last_ping: Mutex<Option<std::time::Instant>>,
}

fn main() {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
        }
        disable_power_throttling(unsafe { GetCurrentProcess() });
    }
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| argv.iter().position(|a| a == k).and_then(|p| argv.get(p + 1)).cloned();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "uso: serve [--app] [--port 8765] [--ui directorio] [--spots directorio] [--host 127.0.0.1]\n  \
             --app   modo aplicación (Windows): puerto libre, abre la interfaz en su propia ventana\n          \
             (Chrome o Edge en modo app, o el navegador) y se cierra al cerrar la ventana"
        );
        return;
    }
    let app_mode = argv.iter().any(|a| a == "--app");
    let exe = std::env::current_exe().expect("ruta del ejecutable");
    let solve = solve_exe(&exe);
    if !solve.exists() {
        eprintln!("aviso: no se encuentra {} (compila con `cargo build --release`)", solve.display());
    }
    // Carpeta de la aplicación (la que contiene `ui/`): junto al ejecutable (paquete),
    // dos niveles por encima (`target/release` del repositorio) o el directorio actual.
    // En modo aplicación se trabaja desde ahí: `solve` guarda su caché en `.cache/`.
    let root = app_root(&exe);
    if app_mode {
        if let Some(r) = &root {
            let _ = std::env::set_current_dir(r);
        }
    }
    let base = root.clone().unwrap_or_else(|| PathBuf::from("."));
    let ui = opt("--ui").map(PathBuf::from).unwrap_or_else(|| base.join("ui"));
    let spots = opt("--spots").map(PathBuf::from).unwrap_or_else(|| base.join("spots"));
    let _ = std::fs::create_dir_all(&spots);
    let work = std::env::temp_dir().join(format!("preflop_serve_{}", std::process::id()));
    std::fs::create_dir_all(&work).expect("directorio de trabajo");
    let app = Arc::new(App { ui, solve, work, spots, job: Mutex::new(Job::default()), last_ping: Mutex::new(None) });
    let port = opt("--port").unwrap_or_else(|| if app_mode { "0".into() } else { "8765".into() });
    let addr = format!("{}:{port}", opt("--host").unwrap_or_else(|| "127.0.0.1".into()));
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| panic!("no se pudo abrir {addr}: {e}"));
    let local = listener.local_addr().expect("dirección local");
    let url = format!("http://{local}/");
    eprintln!("interfaz en {url}");
    if app_mode {
        if !open_window(&url, &app.work) {
            eprintln!("no se pudo abrir una ventana: abre {url} en el navegador");
        }
        // Cierre automático: la interfaz hace ping cada 10 s; si deja de hacerlo (ventana
        // cerrada) y no hay cálculo en marcha, el servidor termina. Cualquier petición cuenta
        // como señal de vida y el fin de un cálculo reinicia la cuenta: con la ventana en
        // segundo plano, Chrome/Edge espacian los temporizadores hasta 1/min, y con 45 s el
        // servidor se cerraba justo al terminar un cálculo largo y borraba la solución
        // ("Failed to fetch" al navegar el árbol).
        let watch = app.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(5));
            let idle = watch.last_ping.lock().unwrap().is_some_and(|t| t.elapsed() > Duration::from_secs(180));
            if idle && !watch.job.lock().unwrap().running {
                eprintln!("ventana cerrada: fin");
                let _ = std::fs::remove_dir_all(&watch.work);
                std::process::exit(0);
            }
        });
    }
    for stream in listener.incoming().flatten() {
        let app = app.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(&app, stream) {
                eprintln!("petición fallida: {e}");
            }
        });
    }
}

/// Carpeta que contiene `ui/app.html`: junto al ejecutable, dos niveles por encima
/// (`target/release`) o el directorio actual.
fn app_root(exe: &std::path::Path) -> Option<PathBuf> {
    let dir = exe.parent()?.to_path_buf();
    let mut cands = vec![dir.clone()];
    if let Some(up) = dir.parent().and_then(|p| p.parent()) {
        cands.push(up.to_path_buf());
    }
    if let Ok(cwd) = std::env::current_dir() {
        cands.push(cwd);
    }
    cands.into_iter().find(|c| c.join("ui").join("app.html").exists())
}

/// Abre `url` en una ventana propia (Windows): Chrome o Edge en modo aplicación (sin
/// barras del navegador, con perfil propio en `work`) o, si no hay ninguno, el navegador
/// por defecto. En otros sistemas (solo desarrollo) no abre nada: basta con la URL.
fn open_window(url: &str, work: &std::path::Path) -> bool {
    if !cfg!(windows) {
        return false;
    }
    let mut cands: Vec<PathBuf> = Vec::new();
    for var in ["ProgramFiles", "ProgramFiles(x86)", "LocalAppData"] {
        if let Ok(base) = std::env::var(var) {
            let base = PathBuf::from(base);
            cands.push(base.join("Google").join("Chrome").join("Application").join("chrome.exe"));
            cands.push(base.join("Microsoft").join("Edge").join("Application").join("msedge.exe"));
        }
    }
    let profile = work.join("ventana");
    for c in cands.iter().filter(|c| c.is_file()) {
        let ok = Command::new(c)
            .arg(format!("--app={url}"))
            .arg(format!("--user-data-dir={}", profile.display()))
            .args(["--no-first-run", "--no-default-browser-check", "--window-size=1400,900"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .is_ok();
        if ok {
            return true;
        }
    }
    Command::new("cmd").args(["/C", "start", "", url]).spawn().is_ok()
}

/// En Windows, los procesos hijos (`solve`) sin ventana de consola propia.
fn no_console(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

/// Pide a Windows que no limite `handle` a los núcleos "E" (eficiencia) en una CPU
/// híbrida. Sin esto, Windows 11 trata un proceso sin ventana propia (`solve`, lanzado
/// con `CREATE_NO_WINDOW` arriba) como "de fondo" y lo deja solo en los núcleos lentos,
/// aunque el resto de la máquina esté libre -- medido en una CPU 6P+4E (12600KF): ~20 %
/// de CPU total (los 4 E a tope, los 12 P ociosos) en vez de ~85 % con todos los núcleos.
/// No hace nada (sin error) en CPUs sin núcleos heterogéneos ni en Windows anteriores a
/// la versión que soporta este campo. Sin dependencias externas: FFI directo a
/// `SetProcessInformation` de `kernel32.dll`, mismo patrón que `system_memory_mb` en
/// `solve.rs`.
#[cfg(windows)]
fn disable_power_throttling(handle: *mut std::ffi::c_void) {
    #[repr(C)]
    struct ProcessPowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn SetProcessInformation(h: *mut std::ffi::c_void, class: i32, info: *mut ProcessPowerThrottlingState, size: u32) -> i32;
    }
    const PROCESS_POWER_THROTTLING_EXECUTION_SPEED: u32 = 0x1;
    const PROCESS_POWER_THROTTLING_CURRENT_VERSION: u32 = 1;
    const PROCESS_POWER_THROTTLING: i32 = 4;
    let mut state = ProcessPowerThrottlingState {
        version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        control_mask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        state_mask: 0, // 0 = sin throttling
    };
    unsafe {
        SetProcessInformation(handle, PROCESS_POWER_THROTTLING, &mut state, std::mem::size_of::<ProcessPowerThrottlingState>() as u32);
    }
}

struct Request {
    method: String,
    path: String,
    query: String,
    trusted: bool,
    body: String,
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<Request> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut r = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let (mut len, mut trusted) = (0usize, false);
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => len = v.trim().parse().unwrap_or(0),
                "x-solver" => trusted = v.trim() == "1",
                _ => {}
            }
        }
    }
    let mut body = vec![0u8; len.min(1 << 20)];
    r.read_exact(&mut body)?;
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    Ok(Request { method, path, query, trusted, body: String::from_utf8_lossy(&body).into_owned() })
}

fn respond(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    write!(stream, "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n", body.len())?;
    stream.write_all(body)
}

fn json(stream: &mut TcpStream, body: &str) -> std::io::Result<()> {
    respond(stream, "200 OK", "application/json; charset=utf-8", body.as_bytes())
}

fn error(stream: &mut TcpStream, status: &str, msg: &str) -> std::io::Result<()> {
    respond(stream, status, "application/json; charset=utf-8", format!("{{\"error\":{}}}", q(msg)).as_bytes())
}

fn handle(app: &Arc<App>, mut stream: TcpStream) -> std::io::Result<()> {
    let req = read_request(&mut stream)?;
    *app.last_ping.lock().unwrap() = Some(std::time::Instant::now());
    if req.method == "POST" && !req.trusted {
        return error(&mut stream, "403 Forbidden", "falta la cabecera X-Solver");
    }
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => file(app, &mut stream, "app.html"),
        ("GET", "/guia") => file(app, &mut stream, "guia.html"),
        ("GET", "/api/ping") => {
            *app.last_ping.lock().unwrap() = Some(std::time::Instant::now());
            json(&mut stream, "{\"ok\":true}")
        }
        ("POST", "/api/icm") => match icm(app, &req.body) {
            Ok(s) => json(&mut stream, &s),
            Err(e) => error(&mut stream, "400 Bad Request", &e),
        },
        ("POST", "/api/hh") => match hand_history(&req.body) {
            Ok(s) => json(&mut stream, &s),
            Err(e) => error(&mut stream, "400 Bad Request", &e),
        },
        ("POST", "/api/preview") => match preview(app, &req.body) {
            Ok(s) => json(&mut stream, &s),
            Err(e) => error(&mut stream, "400 Bad Request", &e),
        },
        ("POST", "/api/solve") => match start(app, &req.body) {
            Ok(g) => json(&mut stream, &format!("{{\"job\":{g}}}")),
            Err(e) => error(&mut stream, "409 Conflict", &e),
        },
        ("POST", "/api/stop") => {
            // Parada ordenada: `solve` comprueba el fichero-señal entre informes y sale
            // por su propio código de guardado/exportación (así "seguir" tiene de dónde
            // partir). Si no termina en un tiempo razonable (atascado, o un informe muy
            // largo), se mata como respaldo.
            let g = {
                let mut job = app.job.lock().unwrap();
                job.lines.push("deteniendo (guardando lo hecho hasta ahora)...".into());
                job.generation
            };
            let stop = app.work.join(format!("stop_{g}.flag"));
            let _ = std::fs::write(&stop, b"");
            let watch = app.clone();
            std::thread::spawn(move || {
                for _ in 0..100 {
                    std::thread::sleep(Duration::from_millis(300));
                    let job = watch.job.lock().unwrap();
                    if job.generation != g || !job.running {
                        return;
                    }
                    drop(job);
                }
                let mut job = watch.job.lock().unwrap();
                if job.generation == g && job.running {
                    job.lines.push("no respondió a tiempo: forzando la parada (sin guardar)".into());
                    if let Some(c) = job.child.as_mut() {
                        let _ = c.kill();
                    }
                }
            });
            json(&mut stream, "{\"ok\":true}")
        }
        ("GET", "/api/progress") => {
            let from: usize = param(&req.query, "from").and_then(|v| v.parse().ok()).unwrap_or(0);
            let job = app.job.lock().unwrap();
            let lines: Vec<String> = job.lines.iter().skip(from).map(|l| q(l)).collect();
            json(
                &mut stream,
                &format!(
                    "{{\"job\":{},\"running\":{},\"exit\":{},\"total\":{},\"lines\":[{}]}}",
                    job.generation,
                    job.running,
                    job.exit.map_or("null".into(), |c| c.to_string()),
                    job.lines.len(),
                    lines.join(",")
                ),
            )
        }
        ("GET", "/api/nodes") => match solution(app, &req.query) {
            Ok(rows) => json(&mut stream, &nodes_json(&rows)),
            Err(e) => error(&mut stream, "404 Not Found", &e),
        },
        ("GET", "/api/node") => {
            let path = param(&req.query, "path").unwrap_or_default();
            match solution(app, &req.query).and_then(|rows| node_json(&rows, &solution_ev(app, &req.query), &path)) {
                Ok(s) => json(&mut stream, &s),
                Err(e) => error(&mut stream, "404 Not Found", &e),
            }
        }
        ("GET", "/api/flops") => {
            let rows = pio_rows(app, &req.query);
            json(&mut stream, &format!("[{}]", rows.iter().map(|r| q(&r[0])).collect::<Vec<_>>().join(",")))
        }
        ("GET", "/api/pio") => {
            let path = param(&req.query, "path").unwrap_or_default();
            match pio_rows(app, &req.query).into_iter().find(|r| r[0] == path) {
                Some(r) => {
                    let pot: f64 = r[3].parse().unwrap_or(0.0);
                    let eff: f64 = r[4].parse().unwrap_or(0.0);
                    respond(&mut stream, "200 OK", "text/plain; charset=utf-8", pio_script(&r[0], &r[1], &r[2], pot, eff, &r[5], &r[6]).as_bytes())
                }
                None => error(&mut stream, "404 Not Found", &format!("no hay flop heads-up en {path:?}")),
            }
        }
        ("GET", "/api/export.json") => {
            let path = job_file(app, &req.query, "json");
            match path.and_then(|p| std::fs::read(p).ok()) {
                Some(b) => respond(&mut stream, "200 OK", "application/json; charset=utf-8", &b),
                None => error(&mut stream, "404 Not Found", "todavía no hay solución"),
            }
        }
        ("GET", "/api/export") => {
            let path = job_file(app, &req.query, "export");
            match path.and_then(|p| std::fs::read(p).ok()) {
                Some(b) => respond(&mut stream, "200 OK", "text/tab-separated-values; charset=utf-8", &b),
                None => error(&mut stream, "404 Not Found", "todavía no hay solución"),
            }
        }
        ("GET", "/api/spots") => {
            let mut names: Vec<String> = std::fs::read_dir(&app.spots)
                .map(|d| d.flatten().filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".spot")).map(String::from)).collect())
                .unwrap_or_default();
            names.sort();
            json(&mut stream, &format!("[{}]", names.iter().map(|n| q(n)).collect::<Vec<_>>().join(",")))
        }
        ("GET", "/api/spot") => match spot_path(app, &param(&req.query, "name").unwrap_or_default()).and_then(|p| std::fs::read_to_string(p).map_err(|e| e.to_string())) {
            Ok(t) => json(&mut stream, &format!("{{\"text\":{}}}", q(&t))),
            Err(e) => error(&mut stream, "404 Not Found", &e),
        },
        ("POST", "/api/spot") => match spot_path(app, &param(&req.query, "name").unwrap_or_default()).and_then(|p| std::fs::write(p, &req.body).map_err(|e| e.to_string())) {
            Ok(()) => json(&mut stream, "{\"ok\":true}"),
            Err(e) => error(&mut stream, "400 Bad Request", &e),
        },
        _ => error(&mut stream, "404 Not Found", "ruta desconocida"),
    }
}

fn file(app: &App, stream: &mut TcpStream, name: &str) -> std::io::Result<()> {
    match std::fs::read(app.ui.join(name)) {
        Ok(b) => respond(stream, "200 OK", "text/html; charset=utf-8", &b),
        Err(e) => error(stream, "404 Not Found", &format!("{}: {e} (¿--ui?)", app.ui.join(name).display())),
    }
}

// ---------------------------------------------------------------------------------------
// Cálculo en segundo plano

/// Argumentos de `solve` del cuerpo de `/api/solve` y `/api/preview`: uno por línea. Las
/// líneas vacías del medio se conservan: son valores vacíos legítimos (`--show ""`, la
/// raíz); solo se descartan las del principio y el final.
fn body_args(body: &str) -> Vec<String> {
    let body = body.trim_matches(['\n', '\r']);
    if body.is_empty() {
        return Vec::new();
    }
    body.lines().map(|l| l.trim().to_string()).collect()
}

fn start(app: &Arc<App>, body: &str) -> Result<u64, String> {
    let mut job = app.job.lock().unwrap();
    if job.running {
        return Err("ya hay un cálculo en marcha".into());
    }
    let mut args: Vec<String> = body_args(body);
    if args.iter().any(|a| matches!(a.as_str(), "icm" | "--export" | "--export-ev" | "--export-json" | "--export-pio" | "--save" | "--load" | "--warm" | "--stop-file")) {
        return Err("argumento no permitido aquí: icm, --export*, --save, --load, --warm, --stop-file".into());
    }
    // `--warm-job g`: arranque en caliente desde la solución del cálculo `g` (otra mano).
    if let Some(k) = args.iter().position(|a| a == "--warm-job") {
        let g: u64 = args.get(k + 1).and_then(|v| v.parse().ok()).ok_or("--warm-job necesita un número de cálculo")?;
        let bin = Some(app.work.join(format!("sol_{g}.bin"))).filter(|p| p.exists()).ok_or("la mano de partida no tiene solución guardada")?;
        args.splice(k..k + 2, ["--warm".to_string(), bin.display().to_string()]);
    }
    // `--continue [g]`: carga la solución del cálculo `g` (o la última) y sigue iterando.
    if let Some(k) = args.iter().position(|a| a == "--continue") {
        let g: Option<u64> = args.get(k + 1).and_then(|v| v.parse().ok());
        let bin = match g {
            Some(g) => Some(app.work.join(format!("sol_{g}.bin"))).filter(|p| p.exists()),
            None => job.bin.clone(),
        }
        .ok_or("no hay solución anterior que continuar")?;
        let n = if g.is_some() { 2 } else { 1 };
        args.splice(k..k + n, ["--load".to_string(), bin.display().to_string()]);
    }
    job.generation += 1;
    let g = job.generation;
    let export = app.work.join(format!("sol_{g}.tsv"));
    let ev = app.work.join(format!("ev_{g}.tsv"));
    let bin = app.work.join(format!("sol_{g}.bin"));
    let js = app.work.join(format!("sol_{g}.json"));
    let pio = app.work.join(format!("pio_{g}.tsv"));
    let stop = app.work.join(format!("stop_{g}.flag"));
    let _ = std::fs::remove_file(&stop);
    for (k, v) in [("--export", &export), ("--export-ev", &ev), ("--export-json", &js), ("--export-pio", &pio), ("--save", &bin), ("--stop-file", &stop)] {
        args.push(k.into());
        args.push(v.display().to_string());
    }
    let mut child = no_console(&mut Command::new(&app.solve))
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("no se pudo lanzar {}: {e}", app.solve.display()))?;
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        disable_power_throttling(child.as_raw_handle());
    }
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    job.lines = vec![format!("$ solve {}", args.join(" "))];
    job.child = Some(child);
    job.running = true;
    job.exit = None;
    job.export = None;
    let generation = job.generation;
    drop(job);
    for pipe in [Box::new(out) as Box<dyn Read + Send>, Box::new(err)] {
        let shared = app.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                shared.job.lock().unwrap().lines.push(line);
            }
        });
    }
    let shared = app.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_millis(150));
        let mut job = shared.job.lock().unwrap();
        let status = match job.child.as_mut() {
            Some(c) => c.try_wait(),
            None => return,
        };
        match status {
            Ok(Some(st)) => {
                job.exit = Some(st.code().unwrap_or(-1));
                job.running = false;
                *shared.last_ping.lock().unwrap() = Some(std::time::Instant::now());
                job.child = None;
                if export.exists() {
                    job.export = Some(export.clone());
                    job.ev = ev.exists().then(|| ev.clone());
                    job.bin = bin.exists().then(|| bin.clone());
                    job.json = js.exists().then(|| js.clone());
                    job.pio = pio.exists().then(|| pio.clone());
                }
                return;
            }
            Ok(None) => {}
            Err(_) => {
                job.running = false;
                *shared.last_ping.lock().unwrap() = Some(std::time::Instant::now());
                job.child = None;
                return;
            }
        }
    });
    Ok(generation)
}

type Rows = Vec<(String, String, String, String)>;

/// Filas de un TSV `ruta \t asiento \t acción \t datos` (formato de `--export` y
/// `--export-ev`); la raíz se escribe `(raíz)`.
fn read_rows(text: &str) -> Rows {
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            (c.len() >= 4).then(|| (if c[0] == "(raíz)" { String::new() } else { c[0].to_string() }, c[1].to_string(), c[2].to_string(), c[3].to_string()))
        })
        .collect()
}

/// Fichero de una solución: el del cálculo `job=g` de la consulta (varias manos abiertas
/// en la interfaz, cada una con su cálculo) o, sin él, el del último cálculo terminado.
/// `kind`: export | ev | json | pio | bin.
fn job_file(app: &App, query: &str, kind: &str) -> Option<PathBuf> {
    if let Some(g) = param(query, "job").and_then(|v| v.parse::<u64>().ok()) {
        let name = match kind {
            "export" => format!("sol_{g}.tsv"),
            "ev" => format!("ev_{g}.tsv"),
            "json" => format!("sol_{g}.json"),
            "pio" => format!("pio_{g}.tsv"),
            _ => format!("sol_{g}.bin"),
        };
        let p = app.work.join(name);
        return p.exists().then_some(p);
    }
    let job = app.job.lock().unwrap();
    match kind {
        "export" => job.export.clone(),
        "ev" => job.ev.clone(),
        "json" => job.json.clone(),
        "pio" => job.pio.clone(),
        _ => job.bin.clone(),
    }
}

fn solution(app: &App, query: &str) -> Result<Rows, String> {
    let path = job_file(app, query, "export").ok_or("no hay solución para esta mano (resuélvela)")?;
    Ok(read_rows(&std::fs::read_to_string(&path).map_err(|e| e.to_string())?))
}

/// Filas de `--export-pio`: ruta, OOP, IP, bote, stack efectivo, rango OOP, rango IP.
fn pio_rows(app: &App, query: &str) -> Vec<Vec<String>> {
    let path = job_file(app, query, "pio");
    path.and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.lines().map(|l| l.split('\t').map(String::from).collect::<Vec<_>>()).filter(|c| c.len() >= 7).collect())
        .unwrap_or_default()
}

/// Filas de `--export-ev`: (ruta, acción, EV del modelo, EV en fichas netas); ficheros
/// antiguos sin la columna de fichas la dejan vacía.
fn solution_ev(app: &App, query: &str) -> Rows {
    let path = job_file(app, query, "ev");
    path.and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| {
            t.lines()
                .filter_map(|l| {
                    let c: Vec<&str> = l.split('\t').collect();
                    (c.len() >= 4).then(|| (if c[0] == "(raíz)" { String::new() } else { c[0].to_string() }, c[2].to_string(), c[3].to_string(), c.get(4).unwrap_or(&"").to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Lista JSON de 169 valores (vacío ⇒ `null`); `null` si no hay datos.
fn ev_list(cells: &str) -> String {
    if cells.is_empty() {
        return "null".into();
    }
    format!("[{}]", cells.split(',').map(|x| if x.is_empty() { "null" } else { x }).collect::<Vec<_>>().join(","))
}

/// Rejilla 13×13 (169 valores, fila a fila) de la frecuencia de un rango.
fn range_grid(r: &[f32]) -> Vec<f32> {
    class_grid(r, 0).iter().flatten().copied().collect()
}

fn range_freq(r: &[f32]) -> f32 {
    r[..1326].iter().sum::<f32>() / 1326.0
}

/// Probabilidad por combo de que el jugador que actúa en `path` llegue a ese nodo con cada
/// mano: producto de sus acciones anteriores en la ruta (las de los demás no dependen de su
/// mano). Los segmentos de la ruta son `asiento:acción`.
fn reach_of(index: &HashMap<(&str, &str), &str>, path: &str, seat: &str) -> Vec<f32> {
    let mut reach = vec![1.0f32; 1328];
    let segs: Vec<&str> = if path.is_empty() { Vec::new() } else { path.split(' ').collect() };
    for (k, seg) in segs.iter().enumerate() {
        let Some((who, label)) = seg.split_once(':') else { continue };
        if who != seat {
            continue;
        }
        let parent = segs[..k].join(" ");
        if let Some(r) = index.get(&(parent.as_str(), label)).and_then(|r| parse_range(r).ok()) {
            for (x, p) in reach.iter_mut().zip(&r) {
                *x *= p;
            }
        }
    }
    reach
}

fn range_index(rows: &Rows) -> HashMap<(&str, &str), &str> {
    rows.iter().map(|r| ((r.0.as_str(), r.2.as_str()), r.3.as_str())).collect()
}

/// Frecuencia de una acción dentro del rango que llega al nodo; sin rango que
/// llegue, la media sin ponderar.
fn reach_freq(r: &[f32], reach: &[f32]) -> f32 {
    let tot: f32 = reach[..1326].iter().sum();
    if tot <= 1e-9 {
        return range_freq(r);
    }
    r[..1326].iter().zip(&reach[..1326]).map(|(a, b)| a * b).sum::<f32>() / tot
}

fn nodes_json(rows: &Rows) -> String {
    let index = range_index(rows);
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        let path = &rows[i].0;
        let mut j = i;
        let (mut labels, mut freqs) = (Vec::new(), Vec::new());
        let reach = reach_of(&index, path, &rows[i].1);
        while j < rows.len() && &rows[j].0 == path {
            labels.push(q(&rows[j].2));
            freqs.push(format!("{:.4}", parse_range(&rows[j].3).map(|r| reach_freq(&r, &reach)).unwrap_or(0.0)));
            j += 1;
        }
        out.push(format!("{{\"path\":{},\"seat\":{},\"actions\":[{}],\"freqs\":[{}]}}", q(path), q(&rows[i].1), labels.join(","), freqs.join(",")));
        i = j;
    }
    format!("[{}]", out.join(","))
}

fn node_json(rows: &Rows, evs: &Rows, path: &str) -> Result<String, String> {
    let sel: Vec<_> = rows.iter().filter(|r| r.0 == path).collect();
    if sel.is_empty() {
        return Err(format!("nodo no encontrado: {path:?}"));
    }
    let reach = reach_of(&range_index(rows), path, &sel[0].1);
    let mut actions = Vec::new();
    for r in &sel {
        let v = parse_range(&r.3)?;
        let combos: f32 = v[..1326].iter().zip(&reach[..1326]).map(|(a, b)| a * b).sum();
        let cells: Vec<String> = range_grid(&v).iter().map(|x| format!("{x:.3}")).collect();
        // EV por clase (vacío ⇒ null); sin fichero de EV, `null`.
        let row = evs.iter().find(|e| e.0 == path && e.1 == r.2);
        let ev = row.map_or("null".to_string(), |e| ev_list(&e.2));
        let ev_chips = row.map_or("null".to_string(), |e| ev_list(&e.3));
        actions.push(format!("{{\"label\":{},\"freq\":{:.4},\"combos\":{combos:.1},\"range\":{},\"grid\":[{}],\"ev\":{ev},\"ev_chips\":{ev_chips}}}", q(&r.2), reach_freq(&v, &reach), q(&r.3), cells.join(",")));
    }
    let reach_cells: Vec<String> = range_grid(&reach).iter().map(|x| format!("{x:.3}")).collect();
    Ok(format!(
        "{{\"path\":{},\"seat\":{},\"reach\":[{}],\"reach_combos\":{:.1},\"actions\":[{}]}}",
        q(path),
        q(&sel[0].1),
        reach_cells.join(","),
        reach[..1326].iter().sum::<f32>(),
        actions.join(",")
    ))
}

/// `solve` junto al ejecutable. El paquete incluye también `solve-avx2` (compilado para
/// x86-64-v3: AVX2 y FMA, ~17 % más rápido); se usa si el procesador lo admite.
fn solve_exe(exe: &std::path::Path) -> PathBuf {
    let name = |base: &str| exe.with_file_name(if cfg!(windows) { format!("{base}.exe") } else { base.to_string() });
    #[cfg(target_arch = "x86_64")]
    {
        let fast = name("solve-avx2");
        if fast.exists() && std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("bmi2") {
            return fast;
        }
    }
    name("solve")
}

/// Fichero de spot a partir de un nombre sin rutas (letras, dígitos, `_`, `-`).
fn spot_path(app: &App, name: &str) -> Result<PathBuf, String> {
    if name.is_empty() || name.len() > 64 || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err(format!("nombre de spot inválido: {name:?} (letras, dígitos, _ y -)"));
    }
    Ok(app.spots.join(format!("{name}.spot")))
}

// ---------------------------------------------------------------------------------------
// Calculadora ICM

fn numbers(v: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for tok in v.split([',', ' ']).map(str::trim).filter(|t| !t.is_empty()) {
        match tok.split_once(['x', 'X']) {
            Some((n, s)) => {
                let n: usize = n.trim().parse().map_err(|_| format!("grupo inválido {tok}"))?;
                let s: f64 = s.trim().parse().map_err(|_| format!("grupo inválido {tok}"))?;
                out.extend(std::iter::repeat_n(s, n));
            }
            None => out.push(tok.parse().map_err(|_| format!("número inválido {tok}"))?),
        }
    }
    Ok(out)
}

/// Argumentos de `solve icm --json` a partir de `clave=valor` por línea (`stacks`,
/// `payouts`, `offtable`, `fgs`, `ante`, `bb-ante`), validados antes de lanzar nada.
fn icm_args(body: &str) -> Result<Vec<String>, String> {
    let get = |k: &str| body.lines().find_map(|l| l.split_once('=').filter(|(a, _)| a.trim() == k).map(|(_, v)| v.trim().to_string())).unwrap_or_default();
    let stacks = numbers(&get("stacks"))?;
    let payouts = numbers(&get("payouts"))?;
    let off = numbers(&get("offtable"))?;
    if stacks.len() < 2 || stacks.len() > 10 {
        return Err("entre 2 y 10 stacks en la mesa".into());
    }
    if payouts.is_empty() {
        return Err("faltan premios".into());
    }
    if stacks.iter().chain(&off).any(|&s| s <= 0.0) {
        return Err("los stacks deben ser positivos".into());
    }
    let join = |v: &[f64]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    let mut a: Vec<String> = vec!["icm".into(), "--json".into(), "--stacks".into(), join(&stacks), "--payouts".into(), join(&payouts)];
    if !off.is_empty() {
        a.extend(["--offtable".into(), join(&off)]);
    }
    match get("fgs").as_str() {
        "" | "0" => {}
        d @ ("1" | "2" | "3" | "auto") => a.extend(["--fgs".into(), d.to_string()]),
        d => return Err(format!("FGS inválido: {d}")),
    }
    // Un solo número: la coma es decimal (`0,125`).
    let ante_s = get("ante").trim().replace(',', ".");
    let ante: f64 = if ante_s.is_empty() { 0.0 } else { ante_s.parse().map_err(|_| format!("ante inválido: {}", get("ante")))? };
    if ante > 0.0 {
        let x = ante;
        a.extend(["--ante".into(), x.to_string()]);
    }
    if get("bb-ante") == "1" {
        a.push("--bb-ante".into());
    }
    Ok(a)
}

/// Calculadora ICM / FGS: la hace `solve icm --json` (misma lógica que el CLI).
fn icm(app: &App, body: &str) -> Result<String, String> {
    let out = no_console(&mut Command::new(&app.solve)).args(icm_args(body)?).stdin(Stdio::null()).output().map_err(|e| format!("no se pudo lanzar solve: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(err.lines().rev().find(|l| l.contains("panicked") || !l.trim().is_empty()).unwrap_or("error en solve icm").to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Una mano de historial (PokerStars, GGPoker o Winamax; la interfaz separa el fichero en
/// manos y manda solo la elegida) → ciegas, antes y asientos en orden de acción, en bb.
fn hand_history(body: &str) -> Result<String, String> {
    let h = preflop_engine::io::hh::parse_hand(body.trim_start())?;
    let hero = h.hero_index();
    let players: Vec<String> = h
        .players
        .iter()
        .enumerate()
        .map(|(k, (name, chips))| {
            let bounty = h.bounties.as_ref().map_or("null".to_string(), |b| b[k].to_string());
            format!("{{\"name\":{},\"chips\":{chips},\"bb\":{},\"bounty\":{bounty}}}", q(name), chips / h.bb)
        })
        .collect();
    Ok(format!(
        "{{\"site\":{},\"id\":{},\"sb\":{},\"bb\":{},\"ante\":{},\"bb_ante\":{},\"hero\":{},\"hero_cards\":{},\"players\":[{}]}}",
        q(&format!("{:?}", h.site)),
        q(&h.hand_id),
        h.sb / h.bb,
        h.bb,
        h.ante / h.bb,
        h.bb_ante,
        hero.map_or("null".to_string(), |k| k.to_string()),
        q(h.hero_cards.as_deref().unwrap_or("")),
        players.join(",")
    ))
}

/// Vista previa del árbol (sin resolver, para la poda de la interfaz antes
/// de lanzar el cálculo real): mismo formato de cuerpo que `/api/solve`, sin los argumentos
/// de exportación/cálculo (se ignoran si llegan, `preview` no los usa).
fn preview(app: &App, body: &str) -> Result<String, String> {
    let args: Vec<String> = body_args(body);
    let mut argv = vec!["preview".to_string()];
    argv.extend(args);
    let out = no_console(&mut Command::new(&app.solve)).args(&argv).stdin(Stdio::null()).output().map_err(|e| format!("no se pudo lanzar solve: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(err.lines().rev().find(|l| l.contains("panicked") || !l.trim().is_empty()).unwrap_or("error al previsualizar el árbol").to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

// ---------------------------------------------------------------------------------------
// Utilidades


fn param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        (k == key).then(|| url_decode(v))
    })
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => match std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                Some(v) => {
                    out.push(v);
                    i += 2;
                }
                None => out.push(b'%'),
            },
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reach_multiplies_own_earlier_actions() {
        let row = |p: &str, seat: &str, a: &str, r: &str| (p.to_string(), seat.to_string(), a.to_string(), r.to_string());
        // SB sube con AA siempre y con KK la mitad; BB vuelve a subir; SB decide.
        let rows: Rows = vec![
            row("", "SB", "F", "QQ-22,AKs-A2o"),
            row("", "SB", "R3", "AA,KK:0.5"),
            row("SB:R3", "BB", "F", "KK-22"),
            row("SB:R3", "BB", "A", "AA"),
            row("SB:R3 BB:A", "SB", "F", "KK"),
            row("SB:R3 BB:A", "SB", "C", "AA"),
        ];
        let index = range_index(&rows);
        let reach = reach_of(&index, "SB:R3 BB:A", "SB");
        let total: f32 = reach[..1326].iter().sum();
        assert!((total - 9.0).abs() < 1e-4, "6 combos de AA + 3 de KK: {total}");
        let c = parse_range("AA").unwrap();
        assert!((reach_freq(&c, &reach) - 6.0 / 9.0).abs() < 1e-4);
        // La BB no ha actuado antes: llega con todo.
        assert_eq!(reach_of(&index, "SB:R3", "BB")[..1326].iter().sum::<f32>(), 1326.0);
    }

    #[test]
    fn hand_history_endpoint_reports_seats_in_bb() {
        let hh = "Winamax Poker - Tournament \"X\" buyIn: 1€ level: 3 - HandId: #1-2-3 - Holdem no limit (25/100/200) - 2024/01/01 12:00:00 UTC
Table: 'X(1)#002' 3-max (real money) Seat #1 is the button
Seat 1: Ana (4000, 1€ bounty)
Seat 2: Hero (3000, 1€ bounty)
Seat 3: Bea (2000, 2€ bounty)
Hero posts small blind 100
Bea posts big blind 200
Dealt to Hero [As Ks]
*** PRE-FLOP ***
";
        let j = hand_history(hh).unwrap();
        assert!(j.contains("\"site\":\"Winamax\"") && j.contains("\"sb\":0.5") && j.contains("\"ante\":0.125"), "{j}");
        assert!(j.contains("\"hero\":1") && j.contains("{\"name\":\"Ana\",\"chips\":4000,\"bb\":20,\"bounty\":1}"), "{j}");
        assert!(hand_history("nada").is_err());
    }

    #[test]
    fn decodes_urls_and_escapes_json() {
        assert_eq!(url_decode("SB%3AR3+x"), "SB:R3 x");
        assert_eq!(url_decode("100%"), "100%");
        assert_eq!(q("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
    }

    #[test]
    fn icm_arguments_are_validated() {
        let a = icm_args("stacks=10,10,10\npayouts=50,30,20\nofftable=2x5\nfgs=1\nante=0.1\nbb-ante=1").unwrap();
        assert_eq!(a.join(" "), "icm --json --stacks 10,10,10 --payouts 50,30,20 --offtable 5,5 --fgs 1 --ante 0.1 --bb-ante");
        let a = icm_args("stacks=10,10,10\npayouts=50,30,20\nante=0,125").unwrap();
        assert!(a.join(" ").ends_with("--ante 0.125"), "coma decimal: {a:?}");
        assert!(icm_args("stacks=10\npayouts=1").is_err());
        assert!(icm_args("stacks=10,10\npayouts=1\nfgs=4").is_err());
        assert!(icm_args("stacks=10,10\npayouts=1\nfgs=auto").is_ok());
        assert!(icm_args("stacks=10,-1\npayouts=1").is_err());
    }

}
