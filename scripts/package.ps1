# Empaqueta la aplicación para Windows en dist\PreflopSolver.zip.
#   powershell -ExecutionPolicy Bypass -File scripts\package.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
# Sin rutas locales (usuario incluido) en los binarios: se reescriben a "~".
$remap = "--remap-path-prefix=$HOME=~"
$env:RUSTFLAGS = $remap
cargo build --release --bins
if ($LASTEXITCODE -ne 0) { throw "cargo build falló" }
# Segunda compilación de solve para procesadores con AVX2/FMA (serve la elige si puede).
$env:RUSTFLAGS = "-C target-cpu=x86-64-v3 $remap"
cargo build --release --bin solve --target-dir target\v3
$env:RUSTFLAGS = ""
if ($LASTEXITCODE -ne 0) { throw "cargo build (AVX2) falló" }
$name = "PreflopSolver"
$out = "dist\$name"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item target\release\solve.exe, target\release\serve.exe $out
Copy-Item target\release\preflop.exe "$out\$name.exe"
Copy-Item target\v3\release\solve.exe "$out\solve-avx2.exe"
Copy-Item -Recurse ui, spots $out
Copy-Item README.md, README.es.md, docs\GUIA_DE_USO.md, docs\USER_GUIDE.md, scripts\LEEME.txt $out
Compress-Archive -Force $out "dist\$name.zip"
Write-Output "dist\$name.zip"
