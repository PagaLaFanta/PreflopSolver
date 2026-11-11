# PreflopSolver — what it does under the hood

*[Versión en español](FEATURES.es.md)*

A preflop tournament solver (CFR+ / MCCFR) written from scratch in Rust. The computing core
has no external dependencies except two: `rayon` for parallelism and `mimalloc` as the
allocator. Everything else (hand evaluator, equity, ICM, postflop abstraction, CFR+) is
hand-written.

## Equity models

- Plain **chip EV**.
- **ICM** (Malmuth-Harville, exact DP; Monte Carlo with common random numbers for large
  fields, so the gradient has no noise).
- **Multi-table ICM**: the rest of the tournament (other tables) enters the calculation,
  not just your table.
- **FGS** (Future Game Simulation): instead of freezing the chips' value at this hand's
  ICM, it simulates the following hands by solving the push/fold game of each future state,
  with 3-way overcalls, a moving button between hands and an optional blind increase. A
  short SB about to move to the button is not undervalued.
- **PKO / SKO**: progressive or standard bounties, split pots, simultaneous eliminations,
  and the value of future bounties distributed by chips. With or without ICM: in chip EV
  the bounties are valued in chips, to isolate their effect without ICM pressure.

## Preflop tree

- Open sizes by seat, with automatic scaling by limpers and callers (`2.5bb+1bb`,
  `3x+1x`, `75%`, `ai`).
- Big blind raise over limp with its own size.
- Independent 3-bet, 4-bet and 5-bet levels, with in-position and out-of-position sizes.
- **Flats per level**: how many players may limp or overlimp and how many may call without
  raising at each level, plus *closing flats* (whoever closes the action can always call)
  and the option to forbid cold calls against 3-bets or more.
- All-in threshold: a size committing more than a given % of the effective stack becomes
  an all-in automatically; a size below the minimum raise is raised to the minimum.
- Maximum players per pot (2 to 6) independent of the number of players at the table:
  3-way pots evaluated exactly and 4+ way pots analytically per board, all with side pots.
- **Pruning before solving**: removes specific actions from an exact node on a preview of
  the tree, instead of solving first and trimming blindly afterwards.
- Push/fold and Standard profiles, with default sizes that scale with the effective stack.

## Postflop

- MCCFR with geometric sizes, all-in as an option at low SPR, raises and re-raises, forced
  check-down by number of players, limited donk bets and dedicated flop sizes (c-bet, donk,
  raise).
- 3- and 4-way postflop pots with each player's own stack and side pots.
- Hand abstraction per street: flop and turn by final strength distribution (EMD k-means),
  river by strength against 8 opponent groups (OCHS). 256 or 1024 groups per street.
- **Variance reduction** (VR-MCCFR) at heads-up leaves: at the same number of iterations,
  NashConv drops ~2.3× at little extra cost.
- Exact board bunching: the ranges of the players who folded change the board and the
  other players' hands.
- Check-down mode to study the preflop structure alone, very quickly.

## Convergence

- CFR+ with a weighted average, simultaneous updates with 3+ players.
- **Convergence indicator (CI)** on a fixed scale: < 10 stable strategy, ≤ 3 reference
  quality. The user picks the quality, not the number of iterations.
- NashConv (exact best response) to measure the real distance to equilibrium.

## Performance

- Parallelism with `rayon` (work stealing) across the whole tree traversal.
- Per-hand vectors in strength order that the compiler vectorizes; a separate AVX2 build,
  used only if the CPU supports it.
- On Intel hybrid CPUs, the engine takes itself out of Windows efficiency mode so it does
  not get stuck on the slow cores.
- Stopping a calculation responds within a second and keeps the work done so it can be
  continued.

## Interface

- Three-step wizard with **templates** for typical spots, import of **PokerStars, GGPoker
  and Winamax hand histories** (stacks, ante, big blind ante and bounties) and stacks in
  chips or in bb.
- Preview of the full tree before solving, with pruning.
- Quick / Study / Reference quality, with an estimate of the remaining time.
- Tree with flattened folds, 13×13 grid with strategy, EV and "EV vs best alternative",
  node panel with combos and EV.
- Range or frequency locking per node (node locking) and subtree refinement.
- ICM calculator with bubble factors, also with a field and FGS.
- Export of ranges (TSV), the full solution (JSON), an HTML report and the ranges reaching
  each flop in PioSolver format.

---

*Windows, portable (no installation), ~20,000 lines of Rust.*
