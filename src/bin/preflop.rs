//! Lanzador de la aplicación de escritorio para Windows (`PreflopSolver.exe` en el
//! paquete): arranca `serve --app` sin ventana de consola y termina. `serve` abre la
//! interfaz en su propia ventana y se cierra solo al cerrarla.
#![cfg_attr(windows, windows_subsystem = "windows")]

use std::process::{Command, Stdio};

fn main() {
    let exe = std::env::current_exe().expect("ruta del ejecutable");
    let serve = exe.with_file_name(if cfg!(windows) { "serve.exe" } else { "serve" });
    let mut cmd = Command::new(&serve);
    cmd.arg("--app").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    if let Some(dir) = exe.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000 | 0x0000_0008); // CREATE_NO_WINDOW | DETACHED_PROCESS
    }
    if let Err(e) = cmd.spawn() {
        eprintln!("no se pudo arrancar {}: {e}", serve.display());
        std::process::exit(1);
    }
}
