# Spots de ejemplo

Cada spot es un fichero `clave = valor` que acepta `solve --spot`. Las claves son las
opciones del CLI sin `--`; `show` y `lock` pueden repetirse.

Para comparar una solución con rangos de referencia propios: escribir los rangos de los
nodos de interés en un TSV con el formato de `--export` (`ruta 	 asiento 	 acción 	
rango`) y ejecutar:

```
./target/release/solve --spot spots/<nombre>.spot --compare referencia.tsv
```

Rutas: acciones separadas por espacio, `POS:F|C|X|A|R<total>` (p.ej. `LJ:F HJ:R2.5 CO:F`);
la raíz es `(raíz)`. Los rangos aceptan pesos (`AKo:0.5`).

| Fichero | Qué es |
|---|---|
| `ft6_icm_pushfold.spot` | mesa final 6 jugadores, ICM, push/fold |
| `bubble_sat.spot` | burbuja de satélite (3 premios iguales, 4 jugadores) |
| `pko_ft.spot` | mesa final PKO |
| `mtt_8max_multitable.spot` | 8 jugadores con otras mesas (ICM multimesa) |
| `hu_25bb_standard.spot` | heads-up 25 bb con opens, 3-bets y postflop |
| `6max_30bb_standard.spot` | 6-max 30 bb con opens/3-bets, flops en check-down |
| `3max_20bb_icm_pruned.spot` | 3-max 20 bb ICM con el árbol podado |

---

## Example spots (English)

Each spot is a `key = value` file accepted by `solve --spot`. Keys are the CLI options
without `--`; `show` and `lock` can be repeated. Paths are actions separated by spaces,
`POS:F|C|X|A|R<total>` (e.g. `LJ:F HJ:R2.5 CO:F`); the root is `(raíz)`. Ranges accept
weights (`AKo:0.5`). To compare a solution with your own reference ranges, write them in a
TSV with the `--export` format and run `solve --spot spots/<name>.spot --compare reference.tsv`.

| File | What it is |
|---|---|
| `ft6_icm_pushfold.spot` | 6-player final table, ICM, push/fold |
| `bubble_sat.spot` | satellite bubble (3 equal prizes, 4 players) |
| `pko_ft.spot` | PKO final table |
| `mtt_8max_multitable.spot` | 8 players with other tables (multi-table ICM) |
| `hu_25bb_standard.spot` | 25 bb heads-up with opens, 3-bets and postflop |
| `6max_30bb_standard.spot` | 30 bb 6-max with opens/3-bets, flops in check-down |
| `3max_20bb_icm_pruned.spot` | 3-max 20 bb ICM with a pruned tree |
