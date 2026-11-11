# PreflopSolver — Guía de uso y práctica

*[English version](USER_GUIDE.md)*

Guía para usar el solver en el día a día: cómo montar un spot, qué ajustes elegir, cómo
leer los resultados y qué hacer cuando algo no cuadra. La descripción general y la
compilación están en el [README](../README.es.md); la lista de funciones, en
[FEATURES.es.md](../FEATURES.es.md).

## Índice

1. [Qué es y qué no es](#1-qué-es-y-qué-no-es)
2. [Instalación y primer arranque](#2-instalación-y-primer-arranque)
3. [Flujo de trabajo en la interfaz](#3-flujo-de-trabajo-en-la-interfaz)
4. [Elegir los ajustes](#4-elegir-los-ajustes)
5. [Leer los resultados](#5-leer-los-resultados)
6. [Recetas de estudio](#6-recetas-de-estudio)
7. [Línea de comandos](#7-línea-de-comandos)
8. [Rendimiento](#8-rendimiento)
9. [Problemas frecuentes](#9-problemas-frecuentes)
10. [Glosario](#10-glosario)

---

## 1. Qué es y qué no es

**Es** un solver preflop de torneos: calcula el equilibrio (Nash aproximado) de una mano
con stacks, ciegas, antes y estructura de premios concretos, en **chip EV**, **ICM**
(también multimesa) y **PKO**. Lo que pasa después del flop se juega con una abstracción
(MCCFR con tamaños geométricos y manos agrupadas en *buckets*), para que el
preflop tenga en cuenta cuánto vale de verdad ver un flop con cada mano.

**No es** un solver postflop. El postflop está ahí para valorar bien el preflop; sus
frecuencias no son material de estudio de postflop. Si quieres estudiar un flop concreto,
exporta los rangos de llegada (`--export-pio`) y llévalos a un solver postflop.

**Calidad.** Los modelos de premios (ICM, ICM multimesa, PKO) son exactos, y la
estrategia se valida con la NashConv (cuánto podría explotarla un rival perfecto) y con
comprobaciones exactas en spots pequeños. En spots push/fold e ICM típicos la solución es
prácticamente exacta; en spots profundos con postflop, las frecuencias de acciones
indiferentes pueden variar entre resoluciones sin que eso cueste EV (ver §5.4).

---

## 2. Instalación y primer arranque

### Requisitos

- Windows 10/11 de 64 bits. No hace falta instalar nada.
- RAM: 8 GB bastan para casi todo; los árboles grandes (6-max profundo con botes a 3-4 y
  1024 buckets) pueden pedir varios GB. La vista previa del árbol muestra la estimación
  antes de resolver.
- CPU: usa todos los núcleos. Si el procesador tiene AVX2, la interfaz elige sola
  `solve-avx2.exe`, que es más rápido.

### Contenido de la carpeta

| Fichero | Para qué |
|---|---|
| `PreflopSolver.exe` | Arranca la interfaz en una ventana propia (Chrome o Edge en modo aplicación). |
| `serve.exe` | Servidor local de la interfaz (`http://127.0.0.1:8765/`). Lo lanza el anterior. |
| `solve.exe`, `solve-avx2.exe` | El motor. Se pueden usar directamente desde la consola. |
| `ui/` | La interfaz. |
| `spots/` | Spots guardados desde la interfaz y spots de referencia. |
| `.cache/` | Se crea sola: matriz de equity, pool postflop, tablas FGS. No borrar. |

### Primer arranque

Doble clic en `PreflopSolver.exe`. **La primera vez** el motor calcula y guarda en `.cache/`:

- la matriz de equity exacta (~40 s),
- las matrices de *board bunching* (1-3 min, con 3 o más jugadores),
- el pool postflop de cada combinación de buckets (segundos),
- las tablas y el ranking del FGS, si lo usas (~1 min),
- con ICM multimesa de muchos jugadores, la tabla de campo (puede llegar a ~13 min en un
  MTT de 200+ jugadores; solo la primera vez para esa estructura).

A partir de ahí, todo sale de la caché. Si un primer cálculo parece "parado" en la fase de
preparación, casi siempre es esto: mira la pestaña **Registro**.

Al cerrar la ventana, el programa se cierra solo en menos de un minuto. Si hay un cálculo
en marcha, no se cierra hasta que termina.

---

## 3. Flujo de trabajo en la interfaz

El ciclo normal es: **＋ Nueva mano → asistente → revisar el árbol → ▶ Resolver → leer**.

### 3.1 Crear la mano: el asistente

**＋ Nueva mano** abre el asistente de tres pasos. Es el único sitio donde se configura una
mano (no hay otro formulario que pueda quedar desincronizado).

**Paso 1 · Juego y equidad**

- **Plantilla**: rellena el asistente con un spot típico (mesa final de MTT, SNG, final de
  SNG push/fold, mesa final PKO, heads-up, 6-max chip EV). Luego cambia stacks y premios a
  los de tu mano. Es la forma más rápida de empezar.
- **📋 Importar historial**: pega o abre un historial de PokerStars, GGPoker o Winamax,
  elige la mano y se rellenan jugadores, stacks en bb, SB, ante (también el ante de BB) y
  bounties. Tu asiento queda marcado con ★ y tus cartas.
- **Jugadores y stacks**, en bb y en orden de acción (UTG … BTN, SB, BB). Si prefieres
  teclear fichas tal como salen en el reloj del torneo, rellena **BB en fichas** y escribe
  los stacks en fichas: se convierten a bb al aceptar.
- **Modelo de equidad**: Chip EV, ICM o ICM multimesa. Con ICM, añade los **premios
  restantes** (+ Puesto). Con multimesa, los **stacks de las otras mesas** (sueltos o en la
  forma `NxS`, p. ej. `16x25,2x40`) o **🎲 Generar stacks** a partir de la media del torneo.
- **Bounty**: Off, PKO o SKO, con la cabeza de cada jugador (y la media de otras mesas).
  Con Chip EV, los bounties se valoran en fichas (útil para aislar el efecto del bounty
  sin presión de ICM).
- **FGS**: no, 1, 2, 3 o auto. Ver §4.4.

**Paso 2 · Betting preflop**

- **Perfil**: *Push/fold* (all-in o fold) o *Estándar* (opens, 3-bets, 4-bets…).
- **Tamaños de open por posición**, **ROL de la BB** (raise sobre limp), opens de la SB.
- **Por nivel**: tamaños de 3-bet/4-bet/5-bet en y fuera de posición.
  Gramática: `2.5bb`, `3x` (sobre el último raise), `75%` (del bote tras pagar), `ai`,
  `2.5bb+1bb` (+1 bb por cada limper/caller).
- **Umbral de all-in**: un raise que comprometa más de ese % del stack efectivo se convierte
  en all-in.
- **⚙ Avanzado**: *flats* por nivel (cuántos pueden pagar sin subir), *closing flats* (quien
  cierra la acción siempre puede pagar) y *cold calls*. Los valores por defecto son los
  buenos; no los toques salvo que quieras reproducir un árbol concreto.

**Paso 3 · Postflop y cálculo**

- **Postflop**: *Completo* (lo normal) o *Check-down* (nadie apuesta tras el
  flop: más rápido, menos preciso).
- **Precisión postflop**: *Normal* (256 grupos de manos por calle) o *Alta* (1024, más
  lenta). Ver §4.3.
- **⚙ Avanzado**: tamaño geométrico, apuestas máximas, raises postflop y
  raises en botes a 3. Solo para reproducir un árbol concreto.
- **Máx. jugadores activos**: cuántos pueden llegar a ver un flop o un all-in juntos (2-6).
- **Calidad**: *Rápido* (CI 20), *Estudio* (CI 10, la recomendada) o *Referencia* (CI 3).
  No hay que elegir iteraciones: el cálculo sigue hasta que el CI baja del objetivo (ver
  §4.5).
- **🌳 Previsualizar árbol**: construye el árbol sin resolver (instantáneo), con nodos,
  hojas de flop, nodos postflop, tamaño y RAM estimados.

**Aceptar** guarda la mano y muestra su árbol en la pestaña Estrategia, todavía sin
resolver.

### 3.2 Revisar y podar el árbol antes de resolver

Antes de pulsar Resolver, recorre el árbol. Cada ✕ junto a una acción la **poda** de ese
nodo concreto, viendo el árbol antes de gastar tiempo de cálculo. Úsalo para:

- quitar tamaños que no quieres en una posición (p. ej. la SB solo con un tamaño de open),
- recortar líneas absurdas que inflan el árbol.

No se puede podar la última acción de un nodo. Las podas se guardan con la mano.

> **Buena práctica**: un árbol más pequeño converge antes y mejor. Cada tamaño extra
> multiplica nodos y diluye las iteraciones. Empieza con pocos tamaños y añade solo los
> que cambien una decisión real.

### 3.3 Resolver, parar y continuar

- **▶ Resolver** lanza el cálculo. En la barra superior verás el **CI** en vivo, en color
  con los umbrales (rojo ≥ 10, ámbar < 10, verde ≤ 3), y una estimación del tiempo
  que falta para llegar a la calidad elegida.
- **■ Parar** detiene el cálculo en menos de un segundo y guarda lo calculado.
- **⏩ Continuar** carga la solución anterior y sigue iterando sobre el mismo árbol: si la
  paraste antes de tiempo, hasta la calidad elegida; si ya había llegado, hasta la mitad
  del CI que tenía.
- **Guardar spot** guarda la configuración en `spots/` para reutilizarla.
- **Duplicar** (panel de manos, ☰) copia la mano actual sin la solución: la forma rápida de
  estudiar variantes (otro stack, otro premio). Doble clic en una mano para renombrarla.
- **descargar** (barra superior, al terminar): rangos en TSV y solución en JSON.

### 3.4 Leer la solución (pestaña Estrategia)

- **Árbol**: los folds están aplanados. En la raíz ves directamente el open de
  cada posición. Clic en una fila muestra el rango de ese jugador en ese nodo; ▸ o doble
  clic, sus respuestas. Con el árbol enfocado: ← nodo padre, → entrar, ↑↓ acción hermana.
- **Rejilla 13×13**: color por acción y frecuencia por mano. La altura de cada celda es la
  parte de esa mano que llega al nodo (vacía: ya se foldeó antes). Modos:
  - **Estrategia**: frecuencias.
  - **EV**: EV de cada acción relativo a foldear, en $ con ICM o en
    bb (selector de unidades).
  - **EV vs mejor otra**: cuánto gana la acción elegida frente a la mejor alternativa. Es
    la vista que de verdad dice si una decisión importa.
- **Panel del nodo**: por acción, frecuencia, combos y EV medio de los combos que la juegan.
  Desde ahí también se bloquea el rango o la frecuencia de una acción.
- **✎ Pintar bloqueo**: pinta acciones sobre la rejilla y fija ese nodo con esa estrategia
  (*node locking*). Los bloqueos se aplican al resolver de nuevo; ver §6.4.
- **🎯 Afinar este spot** (panel del nodo, con la solución guardada): sigue iterando solo el
  subárbol de ese nodo; ver §6.5.
- **Imagen**: descarga la rejilla del nodo como PNG.

### 3.5 Otras pestañas y herramientas

- **ICM**: calculadora ICM (y FGS) de los stacks, sin resolver nada.
- **Registro**: la salida del motor. Es lo primero que hay que mirar si algo va lento o
  falla.

Las opciones de laboratorio (iteraciones, hilos, updates, intervalo de informes, NashConv
periódica, re-raise general, máximo de raises, α del bounty, río por percentil, buckets por
calle, boards del pool, bunching) no están en la interfaz: los valores por defecto son los
buenos. Siguen disponibles en la línea de comandos (`solve --help`).

---

## 4. Elegir los ajustes

Los valores por defecto son una buena elección en casi todos los casos. Esta sección explica cuándo cambiarlos.

### 4.1 Perfil: push/fold o estándar

| Stack efectivo | Perfil recomendado |
|---|---|
| ≤ 10-12 bb | **Push/fold**. Es lo que se juega y converge en segundos. |
| 12-20 bb | **Estándar** con all-in como opción (el umbral de all-in lo añade solo). Compara con push/fold si dudas. |
| > 20 bb | **Estándar**. |

En push/fold, los botes limpeados (complete de la SB) se pueden jugar postflop con la
opción *postflop*; sin ella se llevan a showdown.

### 4.2 Máx. jugadores activos

- **3** (por defecto) cubre casi todo lo relevante: cold calls y squeezes.
- **4** en mesas cortas con stacks cortos donde los all-in a 4 son frecuentes (push/fold
  de 4-6 jugadores). Cuesta bastante más en árboles estándar.
- **2** solo para estudios rápidos heads-up o si el árbol no cabe.

### 4.3 Precisión postflop

| Precisión | Uso |
|---|---|
| **Normal** (256 buckets) | El estándar. Recomendada. |
| Alta (1024 buckets) | Algo más precisa, más memoria y tiempo. Útil en PKO profundo, donde el valor de ver flop pesa más. |

### 4.4 FGS (Future Game Simulation)

El ICM puro "congela" el valor de las fichas en esta mano. El FGS simula las manos
siguientes en push/fold: así una SB corta que va a pasar a botón no se infravalora, y quien
va a pagar la BB en la mano siguiente no se sobrevalora.

- Úsalo en **mesas finales y SNG con stacks cortos** (por debajo de ~20 bb), donde la
  posición futura cambia decisiones.
- **auto** aplica las reglas: 2-4 jugadores → 3 manos; 5-7 → 2; resto → 1.
- **Subida de nivel en el FGS**: si las ciegas suben en breve, pon desde qué mano simulada
  y con qué ciegas (en bb del nivel actual), p. ej. `2@0.75/1.5/0.2`.
- Solo tiene efecto con ICM.

### 4.5 Calidad (CI)

El solver no pregunta cuántas iteraciones hacer: sigue hasta que el **CI** (indicador de
convergencia) baja del objetivo de la calidad elegida.

| Calidad | CI | Significado | Uso |
|---|---|---|---|
| Rápido | 20 | Todavía cambia algo | Una idea rápida del spot. |
| Estudio | 10 | Estable: opens, flats y 3-bets firmes | Estudio normal. Recomendada. |
| Referencia | 3 | Mezclas y manos marginales asentadas | Revisar a fondo o compartir. |

Tiempos orientativos hasta CI 10 con 16 hilos:

| Spot | Tiempo aproximado |
|---|---|
| 3 jugadores push/fold 10 bb | < 10 s |
| Heads-up 25 bb con postflop | ~30 s |
| 4-max 25 bb estándar con postflop | 1,5-2 min |
| 5-9 jugadores push/fold ICM | 2 min – 1 h (según jugadores y máx. activos) |
| 4-max profundo con all-in a 4 | horas |
| 6-max PKO MTT profundo | 45 min – 3 h |

Referencia (CI 3) suele tardar varias veces lo que Estudio.

### 4.6 Modelos de premios

- **Chip EV**: sin premios. Para cash-like, primeras fases o como base de comparación.
- **ICM**: premios restantes de la mesa. La última posición pagada es la última que
  escribas; los jugadores por debajo cobran 0.
- **ICM multimesa**: añade los stacks de las otras mesas. Imprescindible cerca de la
  burbuja de un MTT: con solo tu mesa, el ICM exagera o subestima la presión.
- **PKO**: por defecto progresivo al 50 % (la mitad de la cabeza se cobra y la otra mitad
  se suma a la tuya). Si la sala usa otro reparto, cámbialo (`--ko 100%` o el %
  de la sala); **SKO** cobra la cabeza entera.

---

## 5. Leer los resultados

### 5.1 CI y NashConv

- **CI** mide cuánto cambia todavía la estrategia. Es barato y sale en cada informe.
- **NashConv** (en el Registro, al final) mide cuánto podría ganar un rival que te
  explotara de forma óptima, sumado entre jugadores. Es la medida real de distancia al
  equilibrio. Valores orientativos: < 0,01 bb muy bueno en spots postflop, < 0,001 en
  push/fold.

Un CI bajo con una NashConv alta indica un árbol mal planteado (p. ej. un nodo que casi no
se alcanza y no converge), no un problema del solver.

### 5.2 Frecuencias mixtas

Una mano que sale al 50/50 entre dos acciones casi siempre está **indiferente**: las dos
acciones valen lo mismo. Mira **EV vs mejor otra**: si la diferencia es de céntimos o
centésimas de bb, cualquiera de las dos es correcta. No memorices mezclas exactas de manos
indiferentes; memoriza los umbrales donde la diferencia de EV es grande.

### 5.3 EV relativo a foldear

Los EV de la rejilla y de los exports son relativos a foldear en ese mismo nodo (foldear =
0). Un EV de +0,40 $ en un call significa que pagar vale 0,40 $ más que foldear, no que
ganes 0,40 $ en la mano. Si el nodo no tiene fold (p. ej. la BB tras limps), el EV es
absoluto.

### 5.4 Nodos improbables

Las líneas muy raras (open pequeño → shove → call → re-shove → el que pagó decide otra vez)
convergen mal **en cualquier solver**. Sus rangos pueden verse extraños y
no importa: casi no pesan en el EV total. Al comparar soluciones, pondera siempre por la
probabilidad de llegar al nodo.

### 5.5 Exportar

| Exportación | Contenido |
|---|---|
| `--export rangos.tsv` | Rangos de todos los nodos en notación estándar. |
| `--export-ev ev.tsv` | EV por acción y clase de mano, en $ y en bb. |
| `--export-json sol.json` | Solución completa: configuración, EV por asiento, frecuencias y EV por nodo. |
| `--export-pio flops.tsv` | Rangos de llegada a cada flop heads-up, en formato PioSolver. |
| `--html informe.html` | Informe con las rejillas coloreadas. |

---

## 6. Recetas de estudio

### 6.1 Revisar una mano jugada

1. ＋ Nueva mano → **📋 Importar historial** → elige la mano.
2. Añade los premios restantes (y otras mesas si es un MTT).
3. Perfil según el stack efectivo (§4.1). Previsualiza y poda tamaños que no se usen.
4. Resuelve hasta CI < 10.
5. Ve al nodo de tu decisión, busca tu mano y mira **EV vs mejor otra**: eso es lo que
   costó (o ganó) tu jugada.

### 6.2 Sensibilidad a un parámetro

Para saber si una decisión depende de un detalle (el stack de un rival, un premio):

1. Resuelve la mano base.
2. **Duplicar**, cambia solo ese parámetro y resuelve.
3. Compara las dos rejillas en el mismo nodo.

### 6.3 Burbuja y presión ICM

Resuelve el mismo spot en Chip EV y en ICM (o ICM multimesa). La diferencia entre los dos
rangos de call es la presión ICM; la de los de shove, el *fold equity* extra que te da.

### 6.4 Explotar a un rival (node locking)

1. Resuelve el equilibrio.
2. En el nodo del rival, **✎ Pintar bloqueo** con lo que crees que hace (p. ej. paga solo
   QQ+), o con `--freq` fija solo su frecuencia y deja que el solver elija los combos.
3. Resuelve de nuevo: tu estrategia en los nodos siguientes es la respuesta máxima contra
   ese rival bloqueado.

Bloquea con prudencia: una explotación contra una lectura errónea pierde más que jugar el
equilibrio.

### 6.5 Afinar un nodo concreto

Si un nodo importante no termina de converger, **🎯 Afinar este spot** en el panel del nodo
(en la línea de comandos, `--subtree "CO:R2 BTN:F"` con `--load`) itera solo ese subárbol
con los rangos de entrada de la solución, mucho más rápido que todo el árbol.

---

## 7. Línea de comandos

`solve.exe --help` lista todas las opciones. Recetas habituales:

```bash
# Push/fold heads-up chip EV
solve --stacks 10,10 --iters 500

# 6 jugadores ICM con otras mesas
solve --stacks 12,18,25,9,15,20 --payouts 50,30,20 --offtable 16x25,2x40 --iters 300

# Estándar 4-max 25 bb hasta CI 10, con exportación
solve --stacks 25,25,25,25 --profile standard --target-ci 10 --iters 3000 --export rangos.tsv --export-ev ev.tsv

# FGS automático en una mesa final
solve --stacks 8,25,12,30,6,19 --payouts 50,30,20 --fgs auto

# PKO con cabezas y otras mesas
solve --stacks 15,30,22 --payouts 50,30,20 --bounties 10,10,10,10 --offtable 25

# Desde un historial (cabezas incluidas si las tiene)
solve --hh mano.txt --hand 3 --payouts 50,30,20 --html informe.html

# Calculadora ICM sin resolver
solve icm --stacks 20,15,10 --payouts 50,30,20

# Guardar, continuar y afinar un subárbol
solve --spot mispot.spot --iters 500 --save sol.bin
solve --spot mispot.spot --load sol.bin --iters 1500 --save sol.bin
solve --spot mispot.spot --load sol.bin --subtree "CO:R2 BTN:F" --iters 500

# Ver el árbol sin resolver (JSON)
solve preview --stacks 25,25,25,25 --profile standard
```

Un **fichero de spot** (`--spot`) tiene una opción por línea, `clave = valor`, y `#` para
comentarios; es la forma de evitar líneas de comandos kilométricas:

```text
# 4-max 25 bb, ICM 50/30/20
stacks = 25,25,25,25
payouts = 50,30,20
profile = standard
max-in-pot = 3
target-ci = 10
iters = 3000
```

---

## 8. Rendimiento

- **Cierra otros cálculos**. Dos cálculos a la vez se reparten la CPU y los dos tardan el
  doble.
- **Árbol pequeño primero**. El tamaño del árbol es la palanca principal: tamaños de
  apuesta, máx. jugadores activos y buckets, por ese orden.
- **CPU híbridas (Intel 12.ª gen o posterior)**. El motor se saca solo del modo de
  eficiencia de Windows para no quedar en los núcleos E. Si aun así ves la CPU en torno al
  20-25 %, comprueba en el Administrador de tareas que `solve` no tiene activado el *Modo
  de eficiencia*.
- **Memoria**. El consumo crece hasta un techo y se mantiene; no hay fugas. Si la vista
  previa estima más RAM de la que tienes, reduce buckets o máx. jugadores activos.
- **Caché**. `.cache/` ahorra minutos en cada arranque. Si la mueves de carpeta, muévela
  con el programa.

---

## 9. Problemas frecuentes

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| El primer cálculo tarda minutos "sin avanzar" | Está construyendo la caché (§2) | Mira el Registro; solo pasa una vez. |
| CPU al 20-25 % | Windows ha puesto el proceso en los núcleos E | Ver §8. |
| Rangos muy raros en una línea larga | Nodo improbable sin convergir | Normal (§5.4). Si importa, `--subtree`. |
| El árbol no tiene una acción que esperabas | Umbral de all-in, flats o una poda | Revisa ⚙ Avanzado y las podas; la vista previa lo muestra. |
| Todos los opens salen como all-in | Umbral de all-in demasiado bajo para el stack | Sube el umbral o revisa los tamaños. |
| El perfil vuelve a Push/fold solo | Valor viejo guardado en el navegador | Cámbialo una vez a mano; se queda guardado. |
| "Failed to fetch" al navegar el árbol | El servidor se cerró (ventana cerrada mucho rato) | Vuelve a abrir `PreflopSolver.exe` y carga la mano. |
| El FGS no cambia nada | Estás en Chip EV | El FGS solo actúa con ICM. |

Si un problema persiste, guarda el **Registro** completo (pestaña Registro) antes de cerrar:
ahí se ve en qué fase está el cálculo (preparación, iteraciones o evaluación final).

---

## 10. Glosario

- **Bucket**: grupo de manos que el postflop trata igual en una calle.
- **Board bunching / card bunching**: efecto de las cartas que se llevaron los que
  foldearon sobre las manos de los demás y sobre el board.
- **CFR+ / MCCFR**: algoritmos de minimización de regret que convergen al equilibrio; MCCFR
  muestrea boards en vez de recorrerlos todos.
- **CI**: indicador de convergencia: cuánto cambia todavía la estrategia (§4.5).
- **Closing flats**: quien cierra la acción preflop siempre puede pagar.
- **FGS**: simulación de manos futuras para valorar las fichas mejor que el ICM puro.
- **Flats**: pagos sin subir permitidos por nivel.
- **ICM multimesa**: ICM que incluye los stacks de las otras mesas del torneo.
- **NashConv**: suma de lo que cada jugador ganaría explotando a los demás; 0 en el
  equilibrio exacto.
- **Node locking**: fijar la estrategia de un jugador en un nodo para calcular la mejor
  respuesta.
- **OCHS**: forma de agrupar las manos del río por su fuerza contra 8 grupos de rivales.
- **PKO / SKO**: bounty progresivo (parte de la cabeza crece) / estándar (se cobra entera).
- **Poda**: quitar una acción de un nodo concreto del árbol.
- **ROL**: raise sobre limp.
- **SPR**: relación stack/bote; con SPR bajo el postflop añade el all-in como opción.
