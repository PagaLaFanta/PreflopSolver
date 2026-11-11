#!/bin/sh
# Empaqueta la aplicación para Windows en dist/PreflopSolver.zip compilando de forma
# cruzada desde Linux (requiere `rustup target add x86_64-pc-windows-gnu` y MinGW:
# `apt install gcc-mingw-w64-x86-64`). En Windows, usa scripts\package.ps1.
#   sh scripts/package.sh
set -e
cd "$(dirname "$0")/.."
target=x86_64-pc-windows-gnu
cargo build --release --bins --target "$target"
rel="target/$target/release"
# Segunda compilación de `solve` para procesadores con AVX2/FMA (serve la elige si puede).
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --bin solve --target "$target" --target-dir target/v3
name=PreflopSolver
out="dist/$name"
rm -rf "$out"
mkdir -p "$out"
cp "$rel/solve.exe" "$rel/serve.exe" "$out/"
cp "$rel/preflop.exe" "$out/$name.exe"
cp "target/v3/$target/release/solve.exe" "$out/solve-avx2.exe"
cp -r ui spots README.md README.es.md "$out/"
cp scripts/LEEME.txt docs/GUIA_DE_USO.md docs/USER_GUIDE.md "$out/"
# Sin símbolos de depuración (el perfil release los incluye para perfilar).
if command -v x86_64-w64-mingw32-strip >/dev/null 2>&1; then
    x86_64-w64-mingw32-strip "$out/solve.exe" "$out/solve-avx2.exe" "$out/serve.exe" "$out/$name.exe"
fi
cd dist
rm -f "$name.zip"
if command -v zip >/dev/null 2>&1; then zip -qr "$name.zip" "$name"; else python3 -m zipfile -c "$name.zip" "$name"; fi
echo "dist/$name.zip"
