//! Incrusta el icono de la app (`assets/icon.ico`) en los `.exe` de Windows usando el
//! `rc.exe` del Windows SDK que ya trae la toolchain MSVC — sin añadir ninguna
//! dependencia de compilación a Cargo.toml (el núcleo sigue siendo deliberadamente
//! libre de crates externos). Si no se encuentra el SDK o falla la compilación del
//! recurso, el build sigue adelante sin icono en vez de rompar la compilación.
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let icon = manifest.join("assets").join("icon.ico");
    if !icon.exists() {
        return;
    }
    let Some(rc_exe) = find_rc_exe() else {
        println!("cargo:warning=no se encontro rc.exe (Windows SDK); el .exe se compilara sin icono incrustado");
        return;
    };
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let rc_src = out_dir.join("app.rc");
    std::fs::write(&rc_src, format!("1 ICON \"{}\"\n", icon.display().to_string().replace('\\', "\\\\"))).expect("escribir app.rc");
    let res = out_dir.join("app.res");
    let status = Command::new(&rc_exe).args(["/nologo", "/fo"]).arg(&res).arg(&rc_src).status().expect("ejecutar rc.exe");
    if !status.success() {
        println!("cargo:warning=rc.exe fallo compilando el icono; el .exe se compilara sin el");
        return;
    }
    println!("cargo:rustc-link-arg-bins={}", res.display());
}

/// Busca `rc.exe` bajo el Windows Kits 10 instalado (no suele estar en PATH).
fn find_rc_exe() -> Option<PathBuf> {
    let kits = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&kits).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
    versions.sort();
    versions.reverse();
    for v in versions {
        for arch in ["x64", "x86"] {
            let p = v.join(arch).join("rc.exe");
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}
