# PreflopSolver — qué hace por debajo

*[English version](FEATURES.md)*

Solver de preflop de torneos (CFR+ / MCCFR) escrito desde cero en Rust. El núcleo de
cálculo no tiene dependencias externas salvo dos: `rayon` para el paralelismo y
`mimalloc` como allocator. Todo lo demás (evaluador de manos, equity, ICM, abstracción
postflop, CFR+) está escrito a mano.

## Modelos de equidad

- **Chip EV** puro.
- **ICM** (Malmuth-Harville, DP exacta; Monte Carlo con números aleatorios comunes para
  campos grandes, sin ruido en el gradiente).
- **ICM multimesa**: el resto del torneo (otras mesas) entra en el cálculo, no solo tu mesa.
- **FGS** (Future Game Simulation): en vez de congelar el valor de las fichas en el ICM de
  esta mano, simula las manos siguientes resolviendo el push/fold de cada estado futuro,
  con overcalls a 3, botón móvil entre manos y subida de ciegas opcional. Así una SB corta
  que va a pasar a botón no se infravalora.
- **PKO / SKO**: bounties progresivos o estándar, reparto en empates, eliminaciones
  simultáneas, valor de las cabezas futuras repartido por fichas. Con o sin ICM: en chip EV
  los bounties van en fichas, para aislar su efecto sin presión de ICM.

## Árbol preflop

- Tamaños de open por asiento, con escalado automático por limpers y callers
  (`2.5bb+1bb`, `3x+1x`, `75%`, `ai`).
- Raise sobre limp de la BB con tamaño propio.
- Niveles de 3-bet, 4-bet y 5-bet independientes, con tamaños en y fuera de posición.
- **Flats por nivel**: cuántos pueden limpear u overlimpear y cuántos pueden pagar sin subir
  en cada nivel, más *closing flats* (quien cierra la acción siempre puede pagar) y la
  opción de prohibir cold calls frente a 3-bet o más.
- Umbral de all-in: un tamaño que comprometa más de cierto % del stack efectivo pasa a
  all-in solo; un tamaño por debajo del raise mínimo sube al mínimo.
- Máximo de jugadores por bote (2 a 6) independiente de los jugadores en la mesa: botes a
  3 con evaluación exacta y a 4+ con cálculo analítico por board, todo con botes laterales.
- **Poda antes de resolver**: quita acciones concretas de un nodo exacto sobre una vista
  previa del árbol, en vez de resolver primero y recortar a ciegas después.
- Perfiles Push/fold y Estándar, con tamaños por defecto que escalan con el stack efectivo.

## Postflop

- MCCFR con tamaños geométricos, all-in como opción con SPR bajo, raises y re-raises,
  check-down forzado según el número de jugadores, donk limitado y tamaños propios para el
  flop (c-bet, donk, raise).
- Botes a 3 y 4 jugadores postflop con el stack de cada uno y botes laterales.
- Abstracción de manos por calle: flop y turn por la distribución de fuerza final (k-means
  EMD), río por fuerza contra 8 grupos de rivales (OCHS). 256 o 1024 grupos por calle.
- **Reducción de varianza** (VR-MCCFR) en las hojas heads-up: a igual número de
  iteraciones, la NashConv baja ~2,3× con poco coste extra.
- Board bunching exacto: los rangos de los que foldearon cambian el board y las manos de
  los demás.
- Modo check-down para estudiar solo la estructura preflop muy rápido.

## Convergencia

- CFR+ con media ponderada, updates simultáneos con 3+ jugadores.
- **Indicador de convergencia (CI)** con una escala fija: < 10 estrategia estable, ≤ 3
  calidad de referencia. El usuario elige la calidad, no las iteraciones.
- NashConv (mejor respuesta exacta) para medir la distancia real al equilibrio.

## Rendimiento

- Paralelismo con `rayon` (work-stealing) en todo el recorrido del árbol.
- Vectores por mano en orden de fuerza, que el compilador vectoriza; binario AVX2 aparte,
  elegido solo si la CPU lo soporta.
- En CPU híbridas de Intel, el motor se saca del modo de eficiencia de Windows para no
  quedarse en los núcleos lentos.
- Parar un cálculo responde en menos de un segundo y guarda lo hecho para continuar.

## Interfaz

- Asistente en tres pasos, con **plantillas** de spots típicos, importación de
  **historiales de PokerStars, GGPoker y Winamax** (stacks, ante, ante de BB y bounties) y
  stacks en fichas o en bb.
- Vista previa del árbol completo antes de resolver, con poda.
- Calidad Rápido / Estudio / Referencia, con estimación del tiempo restante.
- Árbol con los folds aplanados, rejilla 13×13 con estrategia, EV y «EV frente a la mejor
  otra acción», panel por nodo con combos y EV.
- Bloqueo de rango o de frecuencia por nodo (node locking) y afinado de un subárbol.
- Calculadora ICM con bubble factors, también con campo y FGS.
- Exportación de rangos (TSV), solución completa (JSON), informe HTML y rangos de llegada a
  cada flop en formato PioSolver.

---

*Windows, portable (sin instalación), ~20 000 líneas de Rust.*
