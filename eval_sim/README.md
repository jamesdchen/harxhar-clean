# `eval_sim` — Monte Carlo first-passage simulator for prop-trading evaluations

A vectorized Monte Carlo engine that estimates the probability of **passing a proprietary-trading
evaluation** ("prop eval") whose account carries a **trailing drawdown** boundary. It answers questions a
naive spec can't: *given a candidate trading strategy's PnL distribution, what fraction of eval accounts
pass? how long does it take? how deep do drawdowns get? how does that change with position size?*

---

## What it models

An eval is a first-passage problem. Track account equity `E_t` step by step (a step is a trade or an
intraday bar). Let `M_t` be the running high-water mark (HWM) of equity, `U` the profit target barrier,
and `L_t` the (possibly moving) drawdown floor:

```
U   = initial + profit_target                 (upper barrier — the payout target)
M_t = max_{s <= t} E_s                         (high-water mark)
L_t = M_t - max_drawdown                        (trailing floor — the classic prop rule)
```

Each path terminates in exactly one of three outcomes:

| Outcome   | Condition                                                              |
|-----------|-----------------------------------------------------------------------|
| `PASS`    | `E_t >= U` **and** at least `min_days` trading days have completed     |
| `FAIL`    | `E_t <= L_t` (drawdown breach) **or** a daily-loss-limit breach        |
| `TIMEOUT` | neither happens within the horizon (`max_days` / `SimConfig.max_steps`)|

`TIMEOUT` is the **censored** case a first-passage-with-horizon *must* track — collapsing it into
"not passed" silently biases the pass-rate. The engine reports it explicitly.

---

## The eval-rule generalizations (why they matter)

Real firms differ on the fine print, and the fine print moves the pass probability materially. Every rule
is a first-class, frozen config field (`eval_sim/config.py`), so a sweep over firms is just a sweep over
`EvalConfig`s.

### Drawdown type — the floor's basis (`DrawdownType`)

The floor `L_t` is defined per `dd_type`. With an optional **lock** (see below) it is always
`L = min(trailing_floor, lock_level)`:

| `dd_type`           | `trailing_floor`                                  | Meaning                                         |
|---------------------|---------------------------------------------------|-------------------------------------------------|
| `STATIC`            | `initial - max_drawdown`  (constant)              | fixed floor; never moves                        |
| `TRAILING_INTRADAY` | `M_t - max_drawdown`, `M_t = max_{s<=t} E_s`      | floor trails the **intraday** equity HWM        |
| `TRAILING_EOD`      | `M^{eod}_t - max_drawdown`                         | floor trails the **end-of-day balance** HWM     |

where `M^{eod}_t = max` over *completed days* of that day's closing balance. Trailing-intraday is the
strictest (an unrealized spike lifts the floor); trailing-EOD only ratchets on closed-day balances;
static is the loosest. **Ordering:** for the same equity path, `L^{static} <= L^{eod} <= L^{intraday}`, so
`pass_rate` is (weakly) monotone: `static >= eod >= intraday` — a property the test suite asserts.

### Trailing lock (Apex-style breakeven lock) — `lock_level`

Some firms let the trailing floor rise **only up to a cap**, then freeze it:

```
L = min(trailing_floor, lock_level)
```

Once the trailing floor reaches `lock_level` it stops rising. Setting `lock_level = initial` gives the
common **breakeven lock**: the floor trails until it hits the starting balance, then locks there — after
that the account can never be stopped out above breakeven. `lock_level = None` means a pure, never-locking
trail. (The lock only applies to trailing floors, not `STATIC`.)

### Daily loss limit — `daily_loss_limit`

An intraday circuit breaker: `FAIL` if the **within-day** cumulative PnL drops to `<= -daily_loss_limit`,
independent of the trailing floor. The engine accumulates `day_pnl` and resets it each end-of-day
(`steps_per_day` controls how many PnL periods make a day).

### Trading-day windows — `min_days` / `max_days`

- `min_days`: a PASS **only counts** after this many completed trading days (you can't hit the target on
  day 1 and cash out — firms impose a minimum activity window).
- `max_days`: a hard eval time limit in days; `horizon_steps = min(SimConfig.max_steps, max_days *
  steps_per_day)`. `None` runs to the MC horizon.

---

## Design choices beyond a naive spec

A naive simulator would draw i.i.d. PnL, loop, and return one bit (passed / didn't). This one makes
deliberately richer choices:

- **Three outcomes incl. censored `TIMEOUT`.** Pass-rate is computed over *decided* paths and the timeout
  mass is reported, not silently reclassified. A horizon too short to decide most paths is a visible bug,
  not a quiet bias.
- **Rich per-path output**, not a bit: first-passage time (`stop_step`), `peak_equity`, `max_dd`,
  `terminal_equity`, `days`. This lets us report *how long* passes take and *how deep* drawdowns get — the
  numbers a trader actually needs to size and schedule an attempt.
- **Block bootstrap** (`pnl.BlockBootstrap`). Trade PnL is **autocorrelated** — volatility clustering and
  the intraday/close-auction regime (see research link below) create loss *runs*, and loss runs are what
  breach a trailing drawdown. I.i.d. resampling shuffles those runs apart and **mis-states drawdown
  risk** (too optimistic). The moving-block bootstrap preserves dependence up to ~`block` steps. The test
  suite verifies it retains lag-1 autocorrelation where the i.i.d. sampler destroys it.
- **Common random numbers (CRN) across the sizing grid.** The position-sizing multiplier `size_mult`
  scales the *same* drawn PnL at a fixed seed (`run` draws once, then multiplies). Sweeping `size_mult`
  over a grid therefore compares sizings on identical market draws — the variance of the *difference*
  collapses, so the optimal size is resolved with far fewer paths than independent runs would need.
- **Wilson score CI on the pass-rate** (`metrics.wilson_ci`). A pass-rate is a binomial proportion from
  `n_paths` draws; report the correct small-sample interval, not a bare point estimate.
  `metrics.paths_for_precision` inverts the half-width to plan `n_paths` up front.
- **Gambler's-ruin verification.** The engine is anchored to a closed form: a symmetric `+/-1` walk with
  static symmetric barriers passes with probability `a / (a + b)`. The test suite checks the empirical
  rate against this analytic value to within 3 standard errors — a real correctness proof, not a smoke
  test.
- **Vectorized-across-paths engine** (`sim_core.simulate`). The time loop is sequential (the HWM is
  path-dependent) but **all paths step together**, so each iteration is `O(n_paths)` numpy work →
  `O(T * P)` total, pure and allocation-light, with an early exit once all paths terminate.

---

## Module map

| Module                    | Role                                                                                     |
|---------------------------|------------------------------------------------------------------------------------------|
| `config.py`               | Frozen `EvalConfig` (eval rules) + `SimConfig` (MC controls) + `DrawdownType`. No deps.   |
| `pnl.py`                  | PnL samplers: `EmpiricalIID`, `BlockBootstrap`, `from_log` (`.sample(n_paths, n_steps, rng)`). |
| `sim_core.py`             | Pure vectorized engine: `simulate(pnl, cfg)` + `run(sampler, eval_cfg, sim_cfg)` → `PathResults`. |
| `metrics.py`              | `summarize(res)` (point estimates + Wilson CI), `wilson_ci`, `standard_error`, `paths_for_precision`. |
| `grid.py`                 | HPC layer: enumerate the sizing / config / seed grid into per-task specs.                 |
| `hpc_worker.py`           | HPC layer: run one grid task (from the array index), write per-path parquet + summary json. |
| `aggregate.py`            | HPC layer: collect all task outputs into one table + rollup.                              |

The pure core (`config`, `pnl`, `sim_core`, `metrics`) is numpy-only and trivially movable; the HPC layer
(`grid`, `hpc_worker`, `aggregate`) is scheduler-agnostic and the only place pandas/pyarrow appear.

---

## Usage

### Local (single batch)

```bash
python -c "
from eval_sim.pnl import from_log
from eval_sim.grid import LOG_PATH
from eval_sim.config import EvalConfig, SimConfig
from eval_sim.sim_core import run
from eval_sim.metrics import summarize
import json
res = run(from_log(LOG_PATH), EvalConfig(), SimConfig(n_paths=200_000, seed=0))
print(json.dumps(summarize(res), indent=2))
"
```

`summarize` returns the pass/fail/timeout rates, the Wilson 95% CI on the pass-rate, first-passage time
stats, and drawdown/equity summaries.

### HPC array (one task per grid cell)

**Slurm:**

```bash
sbatch --array=0-35 --wrap="python -m eval_sim.hpc_worker --out-dir results/eval --n-paths 200000"
# each task reads $SLURM_ARRAY_TASK_ID to pick its grid cell
```

**SGE (Hoffman2-style) equivalent** — same worker, different array variable:

```bash
#!/bin/bash
#$ -t 1-36
#$ -cwd
python -m eval_sim.hpc_worker --out-dir results/eval --n-paths 200000
# the worker reads $SGE_TASK_ID (1-based) and maps it to a grid cell
```

Then combine:

```bash
python -m eval_sim.aggregate --results-dir results/eval
```

---

## Output schema

**Per-path parquet** (one row per simulated path, one file per grid task):

| column            | dtype   | meaning                                             |
|-------------------|---------|-----------------------------------------------------|
| `outcome`         | int8    | `1`=PASS, `2`=FAIL, `3`=TIMEOUT                      |
| `stop_step`       | int32   | 0-based period index at termination                 |
| `terminal_equity` | float64 | equity at stop                                      |
| `peak_equity`     | float64 | max equity reached (`M`)                            |
| `max_dd`          | float64 | max `(peak - equity)` reached                       |
| `days`            | int32   | completed trading days at stop                      |
| `size_mult`,`seed`| —       | grid coordinates identifying the task               |

**Summary json** (`metrics.summarize`, one per task, keys):
`n`, `pass_rate`, `fail_rate`, `timeout_rate`, `wilson_lo`, `wilson_hi`,
`mean_steps_pass`, `median_steps_pass`, `mean_steps_fail`, `median_steps_fail`,
`mean_max_dd`, `p95_max_dd`, `mean_terminal_equity` — plus the grid coordinates (`size_mult`, `seed`).

---

## Strategy / research link

The trade log is generated by a **simple strategy that trades the `cumrv x close` edge** from the vol
research (`build_cumrv_pnl.py` → `eval_sim/data/cumrv_close_pnl.npy`). The finding (FWL attribution):
`cumrv_x_close` — within-day cumulative abnormal RV gated to the 16–19 close/after-hours window — carries
a **negative** coefficient; after a high-intraday-vol day the close/AH realized vol mean-reverts *below*
the HAR baseline. The strategy (causal, long-short):

```
signal_t = cumrv_x_close_t - E_causal[cumrv]     # how far above its expanding mean
r_t      = RV_realized_t - HAR_forecast_t         # close-vol surprise (HAR proxy = har_ma_5)
pnl_t    = -signal_t * r_t                         # short vol when cumrv high
```

aggregated to one trade per day. On the historical cache this realizes the edge: corr(cumrv, r) = −0.30,
positive daily mean, positive skew (occasional large wins from vol collapses). The daily PnL is scaled to
a target $ std for a prop account and **block-bootstrapped** (block ≈ a trading week) as the eval's PnL
source — a real research edge, not a synthetic shape.

**Caveat:** the log is a *frictionless statistical-edge* PnL (no costs/slippage, a HAR proxy, in-sample
scaling), so its Sharpe is optimistic — a realistic PnL *shape* for stress-testing the eval, not a live
track record. Swap in `BlockBootstrap` on a real fills log (same `.sample(...)` protocol) when available.
