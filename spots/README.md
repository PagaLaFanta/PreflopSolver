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
