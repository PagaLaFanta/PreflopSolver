# PreflopSolver — User guide

*[Versión en español](GUIA_DE_USO.md)*

A practical guide to using the solver day to day: how to set up a spot, which settings to
choose, how to read the results and what to do when something looks wrong. The general
overview and build instructions are in the [README](../README.md); the full feature list is
in [FEATURES.md](../FEATURES.md).

> The application's interface is in Spanish. This guide names each button by its exact
> on-screen label, followed by an English translation in parentheses.

## Contents

1. [What it is and what it is not](#1-what-it-is-and-what-it-is-not)
2. [Installation and first run](#2-installation-and-first-run)
3. [Workflow in the interface](#3-workflow-in-the-interface)
4. [Choosing the settings](#4-choosing-the-settings)
5. [Reading the results](#5-reading-the-results)
6. [Study recipes](#6-study-recipes)
7. [Command line](#7-command-line)
8. [Performance](#8-performance)
9. [Troubleshooting](#9-troubleshooting)
10. [Glossary](#10-glossary)

---

## 1. What it is and what it is not

**It is** a preflop tournament solver: it computes the equilibrium (approximate Nash) of a
hand with specific stacks, blinds, antes and payout structure, in **chip EV**, **ICM**
(including multi-table) and **PKO**. Play after the flop uses an abstraction (MCCFR with
geometric bet sizes and hands grouped into *buckets*), so that the preflop strategy knows
how much seeing a flop is really worth with each hand.

**It is not** a postflop solver. The postflop game is there to value preflop decisions
correctly; its frequencies are not postflop study material. To study a specific flop,
export the ranges that reach it (`--export-pio`) and load them into a postflop solver.

**Quality.** The prize models (ICM, multi-table ICM, PKO) are exact, and the strategy is
validated with NashConv (how much a perfect opponent could exploit it) and with exact checks
on small spots. In typical push/fold and ICM spots the solution is practically exact; in
deep spots with postflop play, the frequencies of indifferent actions may vary between runs
without costing any EV (see §5.4).

---

## 2. Installation and first run

### Requirements

- 64-bit Windows 10/11. Nothing to install.
- RAM: 8 GB is enough for almost everything; large trees (deep 6-max with 3-4 way pots and
  1024 buckets) can need several GB. The tree preview shows the estimate before solving.
- CPU: all cores are used. If the processor supports AVX2, the interface automatically picks
  `solve-avx2.exe`, which is faster.

### Folder contents

| File | Purpose |
|---|---|
| `PreflopSolver.exe` | Opens the interface in its own window (Chrome or Edge in app mode). |
| `serve.exe` | Local server for the interface (`http://127.0.0.1:8765/`). Started by the one above. |
| `solve.exe`, `solve-avx2.exe` | The engine. Can also be used directly from a console. |
| `ui/` | The interface. |
| `spots/` | Spots saved from the interface and example spots. |
| `.cache/` | Created automatically: equity matrix, postflop pool, FGS tables. Do not delete. |

### First run

Double-click `PreflopSolver.exe`. **The first time**, the engine computes and stores in
`.cache/`:

- the exact equity matrix (~40 s),
- the *board bunching* matrices (1-3 min, with 3 or more players),
- the postflop pool for each bucket configuration (seconds),
- the FGS tables and ranking, if you use FGS (~1 min),
- with multi-table ICM and many players, the field table (up to ~13 min for an MTT with
  200+ players; only the first time for that structure).

After that everything comes from the cache. If a first calculation seems "stuck" in the
preparation phase, this is almost always why: check the **Registro** (Log) tab.

When you close the window, the program exits on its own within a minute. If a calculation
is running, it waits until it finishes.

---

## 3. Workflow in the interface

The normal cycle is: **＋ Nueva mano** (New hand) → wizard → review the tree →
**▶ Resolver** (Solve) → read the results.

### 3.1 Creating the hand: the wizard

**＋ Nueva mano** (New hand) opens the three-step wizard. It is the only place where a hand
is configured.

**Step 1 · Juego y equidad (Game and equity)**

- **Plantilla** (Template): fills the wizard with a typical spot (MTT final table, SNG,
  push/fold SNG endgame, PKO final table, heads-up, chip EV 6-max). Then change the stacks
  and prizes to those of your hand. It is the fastest way to start.
- **📋 Importar historial** (Import hand history): paste or open a PokerStars, GGPoker or
  Winamax hand history, pick the hand, and players, stacks in bb, SB, ante (including big
  blind ante) and bounties are filled in. Your seat is marked with ★ and your cards.
- **Jugadores** (Players) and **stacks**, in bb and in action order (UTG … BTN, SB, BB).
  If you prefer to type chips as shown by the tournament clock, fill in **BB en fichas**
  (BB in chips) and type the stacks in chips: they are converted to bb when you accept.
- **Modelo de equidad** (Equity model): Chip EV, ICM or multi-table ICM. With ICM, add the
  **remaining prizes** (+ Puesto adds a place). With multi-table ICM, the **stacks of the
  other tables** (individually or as `NxS`, e.g. `16x25,2x40`), or **🎲 Generar stacks**
  (Generate stacks) from the tournament average.
- **Bounty**: Off, PKO or SKO, with each player's bounty (and the average of other tables).
  With Chip EV, bounties are valued in chips (useful to isolate the bounty effect without
  ICM pressure).
- **FGS**: no, 1, 2, 3 or auto. See §4.4.

**Step 2 · Betting preflop**

- **Perfil** (Profile): *Push/fold* (all-in or fold) or *Estándar* (Standard: opens,
  3-bets, 4-bets…).
- **Open sizes by position**, **BB raise over limp**, SB opens.
- **Por nivel** (By level): 3-bet/4-bet/5-bet sizes in and out of position.
  Grammar: `2.5bb`, `3x` (of the last raise), `75%` (of the pot after calling), `ai`,
  `2.5bb+1bb` (+1 bb per limper/caller).
- **Umbral de all-in** (All-in threshold): a raise that commits more than this % of the
  effective stack becomes an all-in.
- **⚙ Avanzado** (Advanced): *flats* per level (how many players may call without raising),
  *closing flats* (whoever closes the action can always call) and *cold calls*. The
  defaults are good; only change them to reproduce a specific tree.

**Step 3 · Postflop y cálculo (Postflop and calculation)**

- **Postflop**: *Completo* (Full, the normal choice) or *Check-down* (nobody bets after the
  flop: faster, less accurate).
- **Precisión postflop** (Postflop precision): *Normal* (256 hand groups per street) or
  *Alta* (High: 1024, slower). See §4.3.
- **⚙ Avanzado** (Advanced): geometric size, maximum bets, postflop raises and raises in
  3-way pots. Only to reproduce a specific tree.
- **Máx. jugadores activos** (Max active players): how many players can reach a flop or an
  all-in together (2-6).
- **Calidad** (Quality): *Rápido* (Quick, CI 20), *Estudio* (Study, CI 10, recommended) or
  *Referencia* (Reference, CI 3). There is no iteration count to choose: the calculation
  runs until the CI drops below the target (see §4.5).
- **🌳 Previsualizar árbol** (Preview tree): builds the tree without solving (instant),
  with action nodes, flop leaves, postflop nodes, size and estimated RAM.

**Aceptar** (Accept) saves the hand and shows its tree in the **Estrategia** (Strategy) tab,
not solved yet.

### 3.2 Reviewing and pruning the tree before solving

Before pressing Solve, walk through the tree. Each ✕ next to an action **prunes** it from
that specific node, seeing the tree before spending any calculation time. Use it to:

- remove sizes you do not want for a position (e.g. a single open size for the SB),
- cut absurd lines that bloat the tree.

The last action of a node cannot be pruned. Prunes are saved with the hand.

> **Good practice**: a smaller tree converges faster and better. Every extra size
> multiplies the number of nodes and dilutes the iterations. Start with few sizes and only
> add those that change a real decision.

### 3.3 Solving, stopping and continuing

- **▶ Resolver** (Solve) starts the calculation. The top bar shows the **CI** live, colored
  by threshold (red ≥ 10, amber < 10, green ≤ 3), and an estimate of the time left to reach
  the chosen quality.
- **■ Parar** (Stop) stops the calculation within a second and keeps what was computed.
- **⏩ Continuar** (Continue) loads the previous solution and keeps iterating on the same
  tree: up to the chosen quality if you stopped early; if it had already got there, down to
  half of its CI.
- **Guardar spot** (Save spot) saves the configuration in `spots/` for reuse.
- **Duplicar** (Duplicate, hands panel ☰) copies the current hand without its solution:
  the quick way to study variants (another stack, another prize). Double-click a hand to
  rename it.
- **descargar** (download, top bar, when finished): ranges as TSV and the solution as JSON.

### 3.4 Reading the solution (Estrategia tab)

- **Tree**: folds are flattened, so the root directly shows every position's open. Click a
  row to see that player's range at that node; ▸ or double-click shows the responses. With
  the tree focused: ← parent node, → enter, ↑↓ sibling action.
- **13×13 grid**: color by action and frequency per hand. Each cell's height is the share of
  that hand reaching the node (empty: it was already folded). Modes:
  - **Estrategia** (Strategy): frequencies.
  - **EV**: EV of each action relative to folding, in $ with ICM or in bb (unit selector).
  - **EV vs mejor otra** (EV vs best alternative): how much the chosen action gains over the
    best alternative. This is the view that actually tells you whether a decision matters.
- **Node panel**: per action, frequency, combos and average EV of the combos that play it.
  You can also lock an action's range or frequency from here.
- **✎ Pintar bloqueo** (Paint lock): paint actions onto the grid and lock the node to that
  strategy (*node locking*). Locks apply when you solve again; see §6.4.
- **🎯 Afinar este spot** (Refine this spot, node panel, with a saved solution): keeps
  iterating only that node's subtree; see §6.5.
- **Imagen** (Image): downloads the node's grid as a PNG.

### 3.5 Other tabs

- **ICM**: ICM (and FGS) calculator for the stacks, without solving anything.
- **Registro** (Log): the engine's output. The first place to look if something is slow or
  fails.

Lab options (iterations, threads, update scheme, report interval, periodic NashConv,
general re-raise size, maximum raises, bounty α, percentile river buckets, per-street
buckets, pool boards, bunching) are not in the interface: the defaults are the right ones.
They are still available on the command line (`solve --help`).

---

## 4. Choosing the settings

The defaults are a good choice in almost every case. This section explains when to change
them.

### 4.1 Profile: push/fold or standard

| Effective stack | Recommended profile |
|---|---|
| ≤ 10-12 bb | **Push/fold**. It is what is actually played, and it converges in seconds. |
| 12-20 bb | **Standard** with all-in as an option (the all-in threshold adds it automatically). Compare with push/fold if in doubt. |
| > 20 bb | **Standard**. |

In push/fold, limped pots (SB complete) can be played postflop with the *postflop* option;
without it they go straight to showdown.

### 4.2 Max active players

- **3** (default) covers almost everything relevant: cold calls and squeezes.
- **4** at short-handed tables with short stacks where 4-way all-ins are common (push/fold
  with 4-6 players). Much more expensive in standard trees.
- **2** only for quick heads-up studies or when the tree does not fit.

### 4.3 Postflop precision

| Precision | Use |
|---|---|
| **Normal** (256 buckets) | The standard. Recommended. |
| High (1024 buckets) | Somewhat more accurate, more memory and time. Useful in deep PKO, where the value of seeing a flop matters more. |

### 4.4 FGS (Future Game Simulation)

Plain ICM "freezes" the value of the chips at this hand. FGS simulates the following hands
in push/fold: a short SB about to move to the button is not undervalued, and a player about
to pay the big blind next hand is not overvalued.

- Use it at **final tables and SNGs with short stacks** (below ~20 bb), where future
  position changes decisions.
- **auto** applies the rules: 2-4 players → 3 hands; 5-7 → 2; otherwise → 1.
- **Subida de nivel en el FGS** (Blind level increase in FGS): if blinds go up soon, enter
  from which simulated hand and with which blinds (in bb of the current level), e.g.
  `2@0.75/1.5/0.2`.
- Only has an effect with ICM.

### 4.5 Quality (CI)

The solver does not ask how many iterations to run: it continues until the **CI**
(convergence indicator) drops below the target of the chosen quality.

| Quality | CI | Meaning | Use |
|---|---|---|---|
| Rápido (Quick) | 20 | Still changing somewhat | A quick idea of the spot. |
| Estudio (Study) | 10 | Stable: opens, flats and 3-bets settled | Normal study. Recommended. |
| Referencia (Reference) | 3 | Mixes and marginal hands settled | In-depth review or sharing. |

Approximate times to CI 10 with 16 threads:

| Spot | Approximate time |
|---|---|
| 3-player push/fold, 10 bb | < 10 s |
| Heads-up 25 bb with postflop | ~30 s |
| 4-max 25 bb standard with postflop | 1.5-2 min |
| 5-9 player push/fold ICM | 2 min – 1 h (depending on players and max active) |
| Deep 4-max with 4-way all-ins | hours |
| Deep 6-max PKO MTT | 45 min – 3 h |

Reference (CI 3) usually takes several times as long as Study.

### 4.6 Prize models

- **Chip EV**: no prizes. For cash-like play, early stages or as a baseline.
- **ICM**: the table's remaining prizes. The last paid place is the last one you enter;
  players below it get 0.
- **Multi-table ICM**: adds the stacks of the other tables. Essential near an MTT bubble:
  with only your table, ICM over- or underestimates the pressure.
- **PKO**: progressive at 50 % by default (half of the bounty is paid out, the other half is
  added to yours). If the site uses a different split, change it (`--ko 100%` or the site's
  %); **SKO** pays the whole bounty.

---

## 5. Reading the results

### 5.1 CI and NashConv

- **CI** measures how much the strategy is still changing. It is cheap and reported at
  every progress report.
- **NashConv** (in the Log, at the end) measures how much an opponent exploiting you
  optimally could win, summed over players. It is the real measure of distance to
  equilibrium. Rough values: < 0.01 bb is very good in postflop spots, < 0.001 in
  push/fold.

A low CI with a high NashConv points to a badly designed tree (e.g. a node that is almost
never reached and does not converge), not to a solver problem.

### 5.2 Mixed frequencies

A hand that comes out 50/50 between two actions is almost always **indifferent**: both
actions are worth the same. Check **EV vs mejor otra** (EV vs best alternative): if the
difference is cents or hundredths of a bb, either action is correct. Do not memorize exact
mixes of indifferent hands; memorize the thresholds where the EV difference is large.

### 5.3 EV relative to folding

EVs in the grid and in exports are relative to folding at that same node (fold = 0). An EV
of +$0.40 for a call means calling is worth $0.40 more than folding, not that you win $0.40
in the hand. If the node has no fold (e.g. the BB after limps), the EV is absolute.

### 5.4 Improbable nodes

Very rare lines (small open → shove → call → re-shove → the original caller decides again)
converge poorly **in any solver**. Their ranges can look strange and it does not matter:
they carry almost no weight in the total EV. When comparing solutions, always weight by the
probability of reaching the node.

### 5.5 Exporting

| Export | Contents |
|---|---|
| `--export ranges.tsv` | Ranges of every node in standard notation. |
| `--export-ev ev.tsv` | EV per action and hand class, in $ and in bb. |
| `--export-json sol.json` | Full solution: configuration, EV per seat, frequencies and EV per node. |
| `--export-pio flops.tsv` | Ranges reaching each heads-up flop, in PioSolver format. |
| `--html report.html` | Report with colored grids. |

---

## 6. Study recipes

### 6.1 Reviewing a hand you played

1. ＋ Nueva mano → **📋 Importar historial** (Import hand history) → pick the hand.
2. Add the remaining prizes (and other tables for an MTT).
3. Profile according to the effective stack (§4.1). Preview the tree and prune unused sizes.
4. Solve to CI < 10.
5. Go to your decision node, find your hand and look at **EV vs mejor otra** (EV vs best
   alternative): that is what your play cost (or gained).

### 6.2 Sensitivity to a parameter

To find out whether a decision depends on a detail (an opponent's stack, a prize):

1. Solve the base hand.
2. **Duplicar** (Duplicate), change only that parameter and solve.
3. Compare both grids at the same node.

### 6.3 Bubble and ICM pressure

Solve the same spot in Chip EV and in ICM (or multi-table ICM). The difference between the
two calling ranges is the ICM pressure; between the shoving ranges, the extra *fold equity*
it gives you.

### 6.4 Exploiting an opponent (node locking)

1. Solve the equilibrium.
2. At the opponent's node, use **✎ Pintar bloqueo** (Paint lock) with what you think they
   do (e.g. call only QQ+), or use `--freq` to lock only their frequency and let the solver
   choose the combos.
3. Solve again: your strategy in the following nodes is the maximal response against that
   locked opponent.

Lock with care: exploiting a wrong read loses more than playing the equilibrium.

### 6.5 Refining a specific node

If an important node does not quite converge, **🎯 Afinar este spot** (Refine this spot) in
the node panel (on the command line, `--subtree "CO:R2 BTN:F"` with `--load`) iterates only
that subtree with the solution's incoming ranges, much faster than the whole tree.

---

## 7. Command line

`solve.exe --help` lists every option (help text in Spanish). Common recipes:

```bash
# Heads-up push/fold, chip EV
solve --stacks 10,10 --iters 500

# 6 players, ICM with other tables
solve --stacks 12,18,25,9,15,20 --payouts 50,30,20 --offtable 16x25,2x40 --iters 300

# Standard 4-max 25 bb to CI 10, with exports
solve --stacks 25,25,25,25 --profile standard --target-ci 10 --iters 3000 --export ranges.tsv --export-ev ev.tsv

# Automatic FGS at a final table
solve --stacks 8,25,12,30,6,19 --payouts 50,30,20 --fgs auto

# PKO with bounties and other tables
solve --stacks 15,30,22 --payouts 50,30,20 --bounties 10,10,10,10 --offtable 25

# From a hand history (bounties included if present)
solve --hh hand.txt --hand 3 --payouts 50,30,20 --html report.html

# ICM calculator without solving
solve icm --stacks 20,15,10 --payouts 50,30,20

# Save, continue and refine a subtree
solve --spot myspot.spot --iters 500 --save sol.bin
solve --spot myspot.spot --load sol.bin --iters 1500 --save sol.bin
solve --spot myspot.spot --load sol.bin --subtree "CO:R2 BTN:F" --iters 500

# Show the tree without solving (JSON)
solve preview --stacks 25,25,25,25 --profile standard
```

A **spot file** (`--spot`) has one option per line, `key = value`, and `#` for comments;
it is the way to avoid very long command lines:

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

## 8. Performance

- **Close other calculations**. Two calculations at once share the CPU and both take twice
  as long.
- **Small tree first**. Tree size is the main lever: bet sizes, max active players and
  buckets, in that order.
- **Hybrid CPUs (Intel 12th gen or later)**. The engine takes itself out of Windows
  efficiency mode so it does not get stuck on the E-cores. If CPU usage still sits around
  20-25 %, check in Task Manager that *Efficiency mode* is not enabled for `solve`.
- **Memory**. Usage grows up to a ceiling and stays there; there are no leaks. If the
  preview estimates more RAM than you have, reduce buckets or max active players.
- **Cache**. `.cache/` saves minutes on every start. If you move the program folder, move
  the cache with it.

---

## 9. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| The first calculation takes minutes "without progress" | Building the cache (§2) | Check the Log; it only happens once. |
| CPU at 20-25 % | Windows put the process on the E-cores | See §8. |
| Very strange ranges in a long line | Improbable node not converged | Normal (§5.4). If it matters, `--subtree`. |
| The tree lacks an action you expected | All-in threshold, flats or a prune | Check ⚙ Avanzado (Advanced) and the prunes; the preview shows them. |
| Every open becomes an all-in | All-in threshold too low for the stack | Raise the threshold or review the sizes. |
| The profile switches back to Push/fold | An old value stored in the browser | Change it once by hand; it stays saved. |
| "Failed to fetch" when browsing the tree | The server closed (window closed for a long time) | Reopen `PreflopSolver.exe` and load the hand. |
| FGS changes nothing | You are in Chip EV | FGS only acts with ICM. |

If a problem persists, save the full **Registro** (Log) before closing: it shows which
phase the calculation is in (preparation, iterations or final evaluation).

---

## 10. Glossary

- **Bucket**: a group of hands that the postflop game treats as identical on a street.
- **Board bunching / card bunching**: the effect of the cards held by players who folded on
  the other players' hands and on the board.
- **CFR+ / MCCFR**: regret-minimization algorithms that converge to equilibrium; MCCFR
  samples boards instead of enumerating all of them.
- **CI**: convergence indicator: how much the strategy is still changing (§4.5).
- **Closing flats**: whoever closes the preflop action can always call.
- **FGS**: simulation of future hands to value chips better than plain ICM.
- **Flats**: calls without raising allowed per level.
- **Multi-table ICM**: ICM that includes the stacks at the tournament's other tables.
- **NashConv**: the sum of what each player would gain by exploiting the others; 0 at the
  exact equilibrium.
- **Node locking**: fixing a player's strategy at a node to compute the best response.
- **OCHS**: a way to group river hands by their strength against 8 groups of opponents.
- **PKO / SKO**: progressive bounty (part of the bounty grows) / standard bounty (paid in
  full).
- **Pruning**: removing an action from a specific node of the tree.
- **Raise over limp**: a raise after one or more players limped in.
- **SPR**: stack-to-pot ratio; with a low SPR the postflop game adds all-in as an option.
