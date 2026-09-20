"""37 - the 15:30 exit rule, decided by the combined next-bar forecast.

The standalone (``writeup/make_dh_causal_standalone_tex.py``) sells the
nearest-OTM SPX 0DTE straddle at 11:00 ET, delta-hedges the package on the
vendor spot every 30 minutes, and at 15:30 lets each forecast model decide on
its own whether to buy the straddle back ("exit if s>0", with
s = rv_hat_{15:30} - slice_{15:30}) or to hold through cash settlement.  The
signal it spends is the block-diagonal ridge's, read at 15:30 -- and 15:30 is
one of the four clocks at which proposal 35's own QLIKE table says the ridge
LOSES to the implied slice and a causal combination of the two beats both.

This proposal asks the one question that follows: does the combined forecast
make a better exit rule at 15:30, and does it change the hold-versus-exit
verdict?  Nothing is re-derived -- proposal 35's Part A combination and
proposal 32/33's book, quotes and hedge are imported by path and gated before
a single new number is computed.

The decision is taken at 15:30 with 15:30 information only.  A day is either
held to cash settlement (hedge running to the 16:00 settlement) or bought back
at 15:30 (hedge stopping at 15:30).  Two books carry it:

  primary    the buy-back pays the QUOTED 15:30 ask of the 11:00 strikes and
             the entry was taken at the 11:00 quoted bid -- fully crossed;
  secondary  the buy-back pays the Black-76 15:30 model mark, the entry still
             the quoted bid: the standalone's own crossed column, carried so
             the rules tie to its published table.

Returns are one unit a day, in units of the 11:00 midpoint entry premium.
Writing r_hold and r_exit for the two reference days, every rule here is a
per-day HOLD FRACTION phi in [0, 1] and earns

    r = (1 - phi) r_exit + phi r_hold = r_exit + phi L,   L = r_hold - r_exit,

so L is exactly the last-half-hour P&L of the settlement leg: what holding
past 15:30 earns over buying the straddle back there.  phi = 1 is always hold,
phi = 0 is always exit, and "exit if s>0" is phi = 0 when the signal is
positive.  A stamp with no signal holds, which is the standalone's own
default.

The rules:

  R0  always hold                 phi = 1                     (reference)
  R1  always exit                 phi = 0                     (reference)
  R2  exit if s>0                 s from the ridge (the standalone's own
                                  rule, reproduced), the baseline OLS, XGBoost,
                                  LightGBM, and the COMBINED forecast
                                  s_comb = f_comb_{15:30} - slice_{15:30};
  R3  margin, combined            exit when s_comb > theta slice_{15:30},
                                  theta in THETA_GRID (a negative theta exits
                                  more often); theta = 0 IS R2's combined cell;
  R4  sizing, combined            phi = clip(1 - s_comb / slice, 0, 1): buy
                                  back a fraction of the straddle at 15:30 in
                                  proportion to how hot the forecast calls the
                                  last bar.  Reported, never adopted: a
                                  fractional straddle only exists at two or
                                  more contracts.

The implied slice carries no exit rule of its own: s is a forecast MINUS the
slice, so the slice has no signal against itself, and it is skipped.

Selection is proposal 30's purged, embargoed walk-forward (63-session minimum,
5-day embargo) over the family (forecast choice x theta).  Significance is
proposal 30's run-length-preserving placebo on WHICH DAYS are exited: each
draw exits exactly as many days, in runs of exactly the same lengths, and only
the days change.  A rule is adopted only if, out of sample on the primary
book, it beats BOTH references -- always hold and always exit -- and its
placebo p-value is below PLACEBO_ALPHA.

Run:  python writeup/intraday_proposals/37_exit_rule_combined.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(path: Path, name: str) -> Any:
    """The repo's read-only import: a script is imported by path, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
p35 = _load_module(HERE / "35_combined_forecast_toggle.py", "p37_p35")
p30 = p35.p30
p32 = p35.p32
asl = p35.asl
standalone = _load_module(
    ROOT / "writeup" / "make_dh_causal_standalone_tex.py", "p37_sa"
)

HOLD = p32.HOLD
OUT = HOLD / "proposals" / "37"

ENTRY = p32.ENTRY
CLOSE = p32.CLOSE
SESSION: tuple[str, ...] = p32.SESSION
ERA0 = p32.ERA0
ANN = p30.ANN

#: The four forecast panels that carry a 15:30 signal, ridge first: the
#: standalone's own rule is the ridge's and is reproduced exactly.
RIDGE_TAG = "blk2"
MODEL_TAGS: tuple[str, ...] = (RIDGE_TAG, "a0", "xgb", "lgbm")
#: Proposal 35's Part A combination this proposal spends (asserted to be the
#: one its own pooled QLIKE ranking picks, never hard-chosen here).
BEST_COMBINATION = "C2_log_fit_weight"
#: The margin grid on the combined signal: exit when s_comb > theta x slice.
#: A negative theta exits more often, theta = 0 is plain "exit if s>0".
THETA_GRID: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
#: The hold fraction is a share of one straddle and is clipped to [0, 1].
PHI_LO, PHI_HI = 0.0, 1.0

#: Proposal 30's constants, reused verbatim so this proposal is scored on the
#: same day sets against the same standard.
WARMUP_SESSIONS = p30.WARMUP_SESSIONS
EMBARGO_DAYS = p30.EMBARGO_DAYS
N_PLACEBO = p30.N_PLACEBO
PLACEBO_SEED = p30.PLACEBO_SEED
PLACEBO_ALPHA = p30.PLACEBO_ALPHA

#: Published reference numbers, reproduced before anything new is computed.
GATE_N_DAYS = p32.GATE_N_DAYS
#: Proposal 35's Part A pooled QLIKE on the common cells (whole, era).
GATE_QLIKE: dict[str, tuple[float, float]] = {
    BEST_COMBINATION: (0.150514, 0.138981),
    "ridge": (0.182647, 0.184102),
    "implied slice": (0.180916, 0.160617),
}
#: Proposal 35's own reference row: the standalone's ridge exit rule scored on
#: the PRIMARY book (buy back at the quoted 15:30 ask), whole and era.
GATE_P35_ASK: tuple[float, float] = (3.275677, 3.349368)
#: The tolerance for a number published to six decimals.
GATE_TOL = 1e-6
#: The standalone's published table, Sharpe_ann at the model mark, midpoint
#: entry and crossed entry, and the exit count -- published to two decimals.
GATE_STANDALONE: dict[str, tuple[float, float, int]] = {
    "hold": (4.20, 3.58, 0),
    RIDGE_TAG: (4.53, 3.87, 344),
    "a0": (4.30, 3.66, 285),
    "xgb": (4.64, 3.99, 274),
    "lgbm": (4.68, 4.03, 286),
}
#: The bar a number published to two decimals can be gated to.
GATE_TOL_2DP = 5e-3
#: Two independent delta-hedge implementations (the live engine's replay and
#: the standalone's own tape) agree on the four legs to this.
EXACT_TOL = 1e-9
#: The identity s_comb = w s_ridge is a float path and is asserted against the
#: scale of the signal it is an identity about.
IDENT_RTOL = 1e-12

#: The columns every rule table prints.
RULE_COLS = [
    "book",
    "rule",
    "n_days",
    "frac_exited",
    "mean",
    "sd",
    "Sharpe_ann",
    "t",
    "MaxDD",
    "worst_day",
    "worst_date",
    "mean_last30_exited_without_rule",
    "mean_last30_exited_with_rule",
    "mean_last30_held",
    "mean_saved_vs_hold",
    "hit_rate_exit",
]


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 260, "display.max_columns", 80):
        print(use.to_string(index=False))


def write(df: pd.DataFrame, name: str, title: str) -> None:
    """Persist a table under the proposal's own directory."""
    df.to_csv(OUT / name, index=False)
    print(f"wrote {name}: {len(df)} rows, {len(df.columns)} columns  [{title}]")


# ------------------------------------------------------------- the books ----
def book_legs(g: dict[str, Any]) -> dict[str, np.ndarray]:
    """The hold day, and the two exit days, in units of the entry premium.

    Every leg enters at the 11:00 QUOTED BID and divides by the 11:00 midpoint
    premium, which is proposal 32's crossed convention.  ``hold`` cash-settles
    against the official close with the hedge running to settlement;
    ``exit_quoted`` buys the straddle back at the QUOTED 15:30 ask and
    ``exit_mark`` at the Black-76 15:30 model mark, both with the hedge
    stopping at 15:30.  The two midpoint-entry legs are carried only for the
    standalone's published table.
    """
    tape = g["tape"]
    em = tape["entry_mid"]
    eb = tape["entry_bid"]
    return {
        "hold": (-(tape["settle"] - eb) + tape["hedge_hold"]) / em,
        "exit_quoted": (-(g["ask"][:, -1] - eb) + tape["hedge_ref"]) / em,
        "exit_mark": (-(tape["mark"] - eb) + tape["hedge_ref"]) / em,
        "hold_mid": (-(tape["settle"] - em) + tape["hedge_hold"]) / em,
        "exit_mark_mid": (-(tape["mark"] - em) + tape["hedge_ref"]) / em,
    }


def signal_1530(g: dict[str, Any], dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Each model's 15:30 signal s = rv_hat_{15:30} - slice_{15:30}.

    The forecast panels are bar-end labelled, so the row that carries the
    forecast ISSUED at 15:30 is stamped 16:00; ``p35.clock_grid`` does that
    shift and keeps the fit mask's rows only.  The slice is the book's own,
    ``IV_hourly^2 x h_rem x w_slice`` at 15:30, where h_rem is the half hour to
    settlement and the remaining-share weight is one by construction at the
    last clock.
    """
    paths = asl.yhat_paths(ROOT)
    cols = list(SESSION)
    sl = g["slice"][:, -1]
    out: dict[str, np.ndarray] = {}
    for tag in MODEL_TAGS:
        panel = asl.load_yhat_panel(paths[tag]).set_index("t")
        rv = (
            p35.clock_grid(panel, "rv_hat")
            .reindex(index=dates, columns=cols)
            .to_numpy(float)
        )
        out[tag] = rv[:, -1] - sl
    return out


# ------------------------------------------------------------- the rules ----
def phi_hold(n: int) -> np.ndarray:
    """Always hold: the whole straddle carries into the settlement leg."""
    return np.full(n, PHI_HI)


def phi_sign(s: np.ndarray, sl: np.ndarray, theta: float) -> np.ndarray:
    """Exit if s > theta x slice; a stamp without a signal holds."""
    live = np.isfinite(s) & np.isfinite(sl) & (sl > 0.0)
    return np.where(live & (s > theta * sl), PHI_LO, PHI_HI)


def phi_size(s: np.ndarray, sl: np.ndarray) -> np.ndarray:
    """Hold clip(1 - s / slice, 0, 1) of the straddle; no signal holds it all."""
    live = np.isfinite(s) & np.isfinite(sl) & (sl > 0.0)
    ratio = np.where(live, s / np.where(live, sl, 1.0), 0.0)
    return np.where(live, np.clip(1.0 - ratio, PHI_LO, PHI_HI), PHI_HI)


def build_rules(
    sig: dict[str, np.ndarray], s_comb: np.ndarray, sl: np.ndarray, n: int
) -> list[dict[str, Any]]:
    """Every candidate hold fraction, references first.

    The combined margin cell at theta = 0 IS the combined "exit if s>0" rule,
    so it is built once and named once; the rule list therefore carries
    len(MODEL_TAGS) model sign rules, len(THETA_GRID) combined margin cells and
    one sizing cell.
    """
    rules: list[dict[str, Any]] = [
        {
            "family": "reference",
            "param": float("nan"),
            "name": "always hold (cash settlement)",
            "phi": phi_hold(n),
            "adoptable": False,
        },
        {
            "family": "reference",
            "param": float("nan"),
            "name": "always exit at 15:30 (the exit book)",
            "phi": np.full(n, PHI_LO),
            "adoptable": False,
        },
    ]
    for tag in MODEL_TAGS:
        rules.append(
            {
                "family": "sign_model",
                "param": 0.0,
                "name": f"exit if s>0 ({asl.YHAT_LABEL[tag]})",
                "phi": phi_sign(sig[tag], sl, 0.0),
                "adoptable": True,
            }
        )
    for theta in THETA_GRID:
        zero = theta == 0.0
        tail = " (= exit if s>0, combined)" if zero else ""
        rules.append(
            {
                "family": "margin_combined",
                "param": float(theta),
                "name": (
                    f"exit if s_comb > {theta:+.2f} x slice (combined "
                    f"{BEST_COMBINATION}){tail}"
                ),
                "phi": phi_sign(s_comb, sl, float(theta)),
                "adoptable": True,
            }
        )
    rules.append(
        {
            "family": "sizing_combined",
            "param": float("nan"),
            "name": (
                "hold clip(1 - s_comb / slice, 0, 1) of the straddle into the "
                "last half hour (fractional straddle)"
            ),
            "phi": phi_size(s_comb, sl),
            "adoptable": False,
        }
    )
    return rules


# ------------------------------------------------------------- the stats ----
def rule_stats(
    phi: np.ndarray,
    r_hold: np.ndarray,
    r_exit: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Calendar-day statistics of one hold-fraction rule on one book.

    Every rule trades every day -- it only decides the last half hour -- so all
    rules are scored on the same calendar days and their means compare
    directly.  ``L = r_hold - r_exit`` is the settlement leg: what holding past
    15:30 earns over buying the straddle back there, so the rule keeps
    ``phi L`` of it and gives up ``(1 - phi) L``.
    """
    lst = r_hold - r_exit
    r = r_exit + phi * lst
    n = int(r.size)
    mean = float(r.mean()) if n else float("nan")
    sd = float(r.std(ddof=1)) if n >= 2 else float("nan")
    live = bool(np.isfinite(sd)) and sd > 0.0
    exited = phi < PHI_HI
    held = ~exited
    n_ex = int(exited.sum())
    return {
        "n_days": n,
        "n_exited": n_ex,
        "frac_exited": float(exited.mean()) if n else float("nan"),
        "mean_hold_fraction": float(phi.mean()) if n else float("nan"),
        "mean": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN if live else float("nan"),
        "t": float(np.sqrt(n)) * mean / sd if live else float("nan"),
        "MaxDD": p30.maxdd(r),
        "worst_day": float(r.min()) if n else float("nan"),
        "worst_date": str(pd.Timestamp(dates[int(np.argmin(r))]).date()) if n else "",
        "mean_last30_exited_without_rule": float(lst[exited].mean())
        if n_ex
        else float("nan"),
        "mean_last30_exited_with_rule": float((phi * lst)[exited].mean())
        if n_ex
        else float("nan"),
        "mean_last30_held": float(lst[held].mean())
        if int(held.sum())
        else float("nan"),
        "mean_saved_vs_hold": float(((phi - PHI_HI) * lst).mean())
        if n
        else float("nan"),
        "hit_rate_exit": float((lst[exited] < 0.0).mean()) if n_ex else float("nan"),
    }


# ------------------------------------------------------ the walk-forward ----
def walk_forward(
    series: np.ndarray, phis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expanding, embargoed rule choice; only the out-of-sample day is scored.

    Proposal 30's walk-forward, unchanged in shape: for day i the training
    window is days [0, i - EMBARGO_DAYS), the candidate with the best in-window
    calendar-day Sharpe is applied to day i, candidate 0 is always hold and
    ``np.argmax`` keeps the first maximum so a tie goes to holding, and days
    whose training window is shorter than WARMUP_SESSIONS are not scored.
    """
    n = series.shape[1]
    curves = np.vstack([p30.expanding_sharpe(row) for row in series])
    curves = np.where(np.isfinite(curves), curves, -np.inf)
    scored = np.zeros(n, dtype=bool)
    pick = np.full(n, -1, dtype=int)
    phi = np.full(n, PHI_HI)
    for i in range(n):
        k = i - EMBARGO_DAYS
        if k < WARMUP_SESSIONS:
            continue
        c = int(np.argmax(curves[:, k]))
        pick[i] = c
        phi[i] = phis[c, i]
        scored[i] = True
    return scored, phi, pick, curves


# ---------------------------------------------------------- the placebo -----
def placebo_row(
    phi: np.ndarray,
    r_hold: np.ndarray,
    r_exit: np.ndarray,
    sel: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Where the rule's Sharpe falls among day-shuffled draws of itself.

    Proposal 30's block shuffle is applied to the mask of days the rule does
    not hold in full, so every draw exits on exactly as many days, in runs of
    exactly the same lengths; the rule's own hold fractions are then laid down,
    in their order, on those days.  Only WHICH days are exited changes.
    """
    ph = phi[sel]
    lst = (r_hold - r_exit)[sel]
    base = r_exit[sel]
    s_rule = p30.sharpe(base + ph * lst)
    active = ph < PHI_HI
    if not active.any() or bool(active.all()):
        return {
            "n_placebo": 0,
            "sharpe_rule": s_rule,
            "placebo_mean": float("nan"),
            "placebo_sd": float("nan"),
            "placebo_q05": float("nan"),
            "placebo_q50": float("nan"),
            "placebo_q95": float("nan"),
            "pct_rank": float("nan"),
            "p_value": float("nan"),
        }
    masks = p30.block_shuffled_masks(active, N_PLACEBO, rng)
    vals = ph[active]
    assert int(masks.sum(axis=1).min()) == vals.size, "a draw changed the exit count"
    draws = np.full((N_PLACEBO, ph.size), PHI_HI)
    draws[masks] = np.tile(vals, N_PLACEBO)
    rr = base[None, :] + draws * lst[None, :]
    mean = rr.mean(axis=1)
    sd = rr.std(axis=1, ddof=1)
    s_plac = np.where(sd > 0.0, mean / np.where(sd > 0.0, sd, 1.0) * ANN, np.nan)
    good = s_plac[np.isfinite(s_plac)]
    ge = int((good >= s_rule).sum())
    return {
        "n_placebo": int(good.size),
        "sharpe_rule": s_rule,
        "placebo_mean": float(good.mean()),
        "placebo_sd": float(good.std(ddof=1)),
        "placebo_q05": float(np.quantile(good, 0.05)),
        "placebo_q50": float(np.quantile(good, 0.50)),
        "placebo_q95": float(np.quantile(good, 0.95)),
        "pct_rank": float((good < s_rule).mean()),
        "p_value": float((1 + ge) / (1 + good.size)),
    }


# --------------------------------------------------------------- gates ------
def gate_part_a(pooled: pd.DataFrame, best: str) -> None:
    """Proposal 35's Part A: the pooled next-bar QLIKE on the common cells."""
    print("\nGATE - proposal 35 Part A, pooled QLIKE on the common cells")
    for name, (want_w, want_e) in GATE_QLIKE.items():
        for smp, want in (("whole", want_w), ("daily_era", want_e)):
            row = pooled[
                (pooled["sample"] == smp)
                & (pooled["cell_set"] == "common")
                & (pooled["forecast"] == name)
            ]
            assert len(row) == 1, (name, smp, len(row))
            have = float(row["QLIKE_pooled"].iloc[0])
            assert abs(have - want) < GATE_TOL, (name, smp, have, want)
            print(f"  {name:<20s} {smp:<10s} {have:.9f}  reference {want:.6f}  OK")
    assert best == BEST_COMBINATION, (best, BEST_COMBINATION)
    print(
        f"  proposal 35's own pooled ranking picks {best}, which is the "
        f"combination this proposal spends  OK"
    )
    print("GATE PASSED")


def gate_standalone(
    g: dict[str, Any],
    legs: dict[str, np.ndarray],
    sig: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> None:
    """The standalone's published table, and its own tape, reproduced.

    ``make_dh_causal_standalone_tex.hold_mark_1100`` builds the same four legs
    from the standalone's own delta-hedge machinery (censored vendor implied
    re-inverted from the straddle midpoint, delta rebalanced every 30 minutes);
    the book here builds them from the live engine's day replays.  The two are
    independent implementations and are asserted against each other, then the
    five published rows of ``writeup/generated_dh_causal/table_dh_causal.tex``
    are reproduced, exit counts included.
    """
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    r_h, r_f, r_hx, r_fx = standalone.hold_mark_1100(pkg)
    print("\nGATE - the standalone's own tape and its published table")
    pairs = (
        ("hold, midpoint entry", legs["hold_mid"], r_h),
        ("hold, crossed entry", legs["hold"], r_hx),
        ("exit at the model mark, midpoint entry", legs["exit_mark_mid"], r_f),
        ("exit at the model mark, crossed entry", legs["exit_mark"], r_fx),
    )
    for label, have, want in pairs:
        ref = pd.Series(want).reindex(dates).to_numpy(float)
        assert np.isfinite(ref).all(), label
        d = float(np.max(np.abs(have - ref)))
        assert d < EXACT_TOL, (label, d)
        print(f"  {label:<40s} max |difference| {d:.3e}  OK")
    assert len(dates) == GATE_N_DAYS, (len(dates), GATE_N_DAYS)
    print(f"  n = {len(dates)} expirations  reference {GATE_N_DAYS}  OK")
    for key, (want_mid, want_x, want_n) in GATE_STANDALONE.items():
        if key == "hold":
            phi = phi_hold(len(dates))
            label = "hold through cash settlement"
        else:
            phi = phi_sign(sig[key], g["slice"][:, -1], 0.0)
            label = f"exit if s>0 ({asl.YHAT_LABEL[key]})"
        n_ex = int((phi < PHI_HI).sum())
        got_mid = p30.sharpe(
            legs["exit_mark_mid"] + phi * (legs["hold_mid"] - legs["exit_mark_mid"])
        )
        got_x = p30.sharpe(legs["exit_mark"] + phi * (legs["hold"] - legs["exit_mark"]))
        assert n_ex == want_n, (key, n_ex, want_n)
        assert abs(got_mid - want_mid) < GATE_TOL_2DP, (key, got_mid, want_mid)
        assert abs(got_x - want_x) < GATE_TOL_2DP, (key, got_x, want_x)
        print(
            f"  {label:<48s} n_exit {n_ex:>3d} ({want_n})  Sharpe_ann "
            f"{got_mid:.4f} ({want_mid:.2f})  crossed {got_x:.4f} "
            f"({want_x:.2f})  OK"
        )
    # proposal 35's own reference row: the ridge rule on the PRIMARY book
    phi_r = phi_sign(sig[RIDGE_TAG], g["slice"][:, -1], 0.0)
    r = legs["exit_quoted"] + phi_r * (legs["hold"] - legs["exit_quoted"])
    era = np.asarray(dates >= ERA0, dtype=bool)
    for smp, m, want in (
        ("whole", np.ones(len(dates), dtype=bool), GATE_P35_ASK[0]),
        ("daily_era", era, GATE_P35_ASK[1]),
    ):
        have = p30.sharpe(r[m])
        assert abs(have - want) < GATE_TOL, (smp, have, want)
        print(
            f"  the ridge rule at the QUOTED 15:30 ask, {smp:<10s} {have:.9f}  "
            f"reference (proposal 35) {want:.6f}  OK"
        )
    print("GATE PASSED")


def gate_combination_identity(
    w15: np.ndarray,
    sl_part_a: np.ndarray,
    sl_book: np.ndarray,
    s_comb: np.ndarray,
    s_ridge: np.ndarray,
) -> None:
    """The levels combination cannot move the SIGN of s, and that is asserted.

    Proposal 35's best combination is convex in LEVELS,
    ``f_comb = w rv_hat + (1 - w) slice``, so at any clock

        s_comb = f_comb - slice = w (rv_hat - slice) = w s_ridge,

    and a weight in (0, 1] is a positive rescaling: "exit if s>0" on the
    combined forecast reads exactly the same days as the ridge's own rule.
    Only a rule that reads the MAGNITUDE of s -- a margin theta other than
    zero, or the sizing rule -- can part company with it.  The identity, the
    weight's range at 15:30 and the count of days the two rules disagree on
    are all asserted here rather than asserted in prose.
    """
    both = np.isfinite(sl_part_a) & np.isfinite(sl_book)
    d_slice = float(np.max(np.abs(sl_part_a[both] - sl_book[both])))
    assert d_slice < EXACT_TOL, d_slice
    live = np.isfinite(s_comb) & np.isfinite(s_ridge) & np.isfinite(w15)
    scale = float(np.max(np.abs(s_ridge[live])))
    d_id = float(np.max(np.abs(s_comb[live] - w15[live] * s_ridge[live])))
    assert d_id <= IDENT_RTOL * scale, (d_id, scale)
    n_dis = int(((s_comb[live] > 0.0) != (s_ridge[live] > 0.0)).sum())
    assert n_dis == 0, n_dis
    w_live = w15[live]
    print("\nGATE - the combination's 15:30 signal against the ridge's")
    print(
        f"  Part A's implied slice and the book's agree on all {int(both.sum())} "
        f"15:30 cells to {d_slice:.3e}  OK"
    )
    print(
        f"  s_comb = w_(15:30) s_ridge on all {int(live.sum())} days that carry "
        f"both, to {d_id:.3e} against a signal scale of {scale:.3e}  OK"
    )
    print(
        f"  the 15:30 weight on the ridge lies in "
        f"[{float(w_live.min()):.6f}, {float(w_live.max()):.6f}], median "
        f"{float(np.median(w_live)):.6f}; it is strictly positive on "
        f"{int((w_live > 0.0).sum())} of {int(live.sum())} days"
    )
    print(
        f"  so 'exit if s>0' on the combined forecast and 'exit if s>0' on the "
        f"ridge disagree on {n_dis} of {int(live.sum())} days  OK"
    )
    print("GATE PASSED")


# ---------------------------------------------------------------- main ------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------- Part A, imported ----
    work, clocks, _prof = p35.forecast_frame()
    p35.gate_qlike(work)
    fc, weights, y = p35.combination_grids(work, clocks)
    _by_clock, pooled, _wt, _boot, best = p35.part_a(fc, weights, y)
    gate_part_a(pooled, best)

    # --------------------------------------------------- the book, gated ---
    g = p35.book_grids()
    dates = pd.DatetimeIndex(g["dates"])
    cols = list(SESSION)
    p32.gate(g["tape"], g["mid"], g["ask"], dates)
    legs = book_legs(g)
    sig = signal_1530(g, dates)
    gate_standalone(g, legs, sig, dates)

    sl = g["slice"][:, -1]
    comb = fc[best].reindex(index=dates, columns=cols).to_numpy(float)[:, -1]
    s_comb = comb - sl
    gate_combination_identity(
        weights[best].reindex(index=dates, columns=cols).to_numpy(float)[:, -1],
        fc["implied slice"].reindex(index=dates, columns=cols).to_numpy(float)[:, -1],
        sl,
        s_comb,
        sig[RIDGE_TAG],
    )
    print(
        f"\nthe 15:30 decision: the combined forecast covers "
        f"{int(np.isfinite(comb).sum())} of {len(dates)} sessions (the other "
        f"{int((~np.isfinite(comb)).sum())} are the expanding weight's "
        f"{WARMUP_SESSIONS}-session warm-up and HOLD, the standalone's own "
        f"default); the implied slice covers {int(np.isfinite(sl).sum())}; the "
        f"implied slice carries no exit rule of its own, because s is a "
        f"forecast minus the slice and the slice has no signal against itself"
    )

    rules = build_rules(sig, s_comb, sl, len(dates))
    ridge_exit = phi_sign(sig[RIDGE_TAG], sl, 0.0) < PHI_HI
    comb_exit = phi_sign(s_comb, sl, 0.0) < PHI_HI
    warm = ~np.isfinite(comb)
    assert not (comb_exit & warm).any(), "a warm-up day exited on the combination"
    assert np.array_equal(ridge_exit[~warm], comb_exit[~warm]), (
        "the two sign rules part company on a day both cover"
    )
    print(
        f"  the combined sign rule and the ridge's therefore differ on the "
        f"{int(warm.sum())} warm-up days only, where the combined rule holds "
        f"and the ridge exits on {int((ridge_exit & warm).sum())} of them: "
        f"{int(comb_exit.sum())} exits against {int(ridge_exit.sum())} over the "
        f"whole sample, the SAME {int(comb_exit[~warm].sum())} exits over the "
        f"{int((~warm).sum())} covered days"
    )

    n_cells = len([r for r in rules if r["family"] != "reference"])
    n_adoptable = len([r for r in rules if r["adoptable"]])
    era = np.asarray(dates >= ERA0, dtype=bool)
    samples: tuple[tuple[str, np.ndarray], ...] = (
        ("whole", np.ones(len(dates), dtype=bool)),
        ("daily_era", era),
    )
    books: tuple[tuple[str, str], ...] = (
        ("primary_quoted_ask", "exit_quoted"),
        ("secondary_model_mark", "exit_mark"),
    )
    print(
        f"\n{n_cells} exit cells tried ({len(MODEL_TAGS)} model sign rules, "
        f"{len(THETA_GRID)} combined margin cells of which theta = 0 IS the "
        f"combined sign rule, and {n_cells - len(MODEL_TAGS) - len(THETA_GRID)} "
        f"sizing cell), {n_adoptable} of them adoptable (the sizing cell needs a "
        f"fractional straddle and is reported only), each scored on "
        f"{len(books)} books x {len(samples)} samples"
    )
    for rule in rules:
        if rule["family"] == "reference":
            continue
        n_ex = int((rule["phi"] < PHI_HI).sum())
        print(
            f"  {rule['family']:<16s} {rule['name'][:74]:<74s} exits "
            f"{n_ex:>3d} of {len(dates)}"
        )

    # ------------------------------------------------------- rules.csv -----
    rows: list[dict[str, Any]] = []
    for smp, m in samples:
        for bname, bkey in books:
            for rule in rules:
                rows.append(
                    {
                        "sample": smp,
                        "book": bname,
                        "family": rule["family"],
                        "param": rule["param"],
                        "rule": rule["name"],
                        "adoptable": rule["adoptable"],
                        **rule_stats(
                            rule["phi"][m], legs["hold"][m], legs[bkey][m], dates[m]
                        ),
                    }
                )
    rules_df = pd.DataFrame(rows)
    write(rules_df, "rules.csv", "every 15:30 exit rule, both books")
    for smp, _ in samples:
        for bname, _ in books:
            show(
                rules_df[(rules_df["sample"] == smp) & (rules_df["book"] == bname)],
                f"rules.csv  |  sample {smp}  |  book {bname}  |  entry at the "
                f"11:00 quoted bid, one unit a day in entry premia",
                RULE_COLS,
            )

    # --------------------------------------------------------- oos.csv -----
    cands_all = [rules[0]] + [r for r in rules if r["adoptable"]]
    families: dict[str, list[dict[str, Any]]] = {
        "sign_model": [rules[0]] + [r for r in rules if r["family"] == "sign_model"],
        "margin_combined": [rules[0]]
        + [r for r in rules if r["family"] == "margin_combined"],
        "all_adoptable": cands_all,
    }
    primary = legs["exit_quoted"]
    selected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    orows: list[dict[str, Any]] = []
    for fam, cands in families.items():
        series = np.vstack(
            [primary + c["phi"] * (legs["hold"] - primary) for c in cands]
        )
        phis = np.vstack([c["phi"] for c in cands])
        scored, wf_phi, pick, _curves = walk_forward(series, phis)
        selected[fam] = (scored, wf_phi)
        names = np.array([c["name"] for c in cands])
        for smp, m in samples:
            selm = scored & m
            if not selm.any():
                continue
            st = rule_stats(
                wf_phi[selm], legs["hold"][selm], primary[selm], dates[selm]
            )
            hold_st = rule_stats(
                phi_hold(int(selm.sum())),
                legs["hold"][selm],
                primary[selm],
                dates[selm],
            )
            exit_st = rule_stats(
                np.full(int(selm.sum()), PHI_LO),
                legs["hold"][selm],
                primary[selm],
                dates[selm],
            )
            counts = pd.Series(names[pick[selm]]).value_counts()
            orows.append(
                {
                    "block": "walk_forward_selected",
                    "sample": smp,
                    "family": fam,
                    "param": float("nan"),
                    "rule": (
                        f"{fam}: cell chosen on an expanding window, "
                        f"{EMBARGO_DAYS}-day embargo"
                    ),
                    "adoptable": True,
                    **st,
                    "Sharpe_hold": hold_st["Sharpe_ann"],
                    "Sharpe_exit": exit_st["Sharpe_ann"],
                    "beats_hold": bool(st["Sharpe_ann"] > hold_st["Sharpe_ann"]),
                    "beats_exit": bool(st["Sharpe_ann"] > exit_st["Sharpe_ann"]),
                    "beats_both": bool(
                        st["Sharpe_ann"] > hold_st["Sharpe_ann"]
                        and st["Sharpe_ann"] > exit_st["Sharpe_ann"]
                    ),
                    "modal_pick": str(counts.index[0]),
                    "modal_pick_share": float(counts.iloc[0] / counts.sum()),
                    "hold_share": float(counts.get(rules[0]["name"], 0) / counts.sum()),
                    "n_candidates": len(cands),
                }
            )
    scored_any = selected["all_adoptable"][0]
    for smp, m in samples:
        selm = scored_any & m
        hold_st = rule_stats(
            phi_hold(int(selm.sum())), legs["hold"][selm], primary[selm], dates[selm]
        )
        exit_st = rule_stats(
            np.full(int(selm.sum()), PHI_LO),
            legs["hold"][selm],
            primary[selm],
            dates[selm],
        )
        for rule in rules:
            st = rule_stats(
                rule["phi"][selm], legs["hold"][selm], primary[selm], dates[selm]
            )
            orows.append(
                {
                    "block": "fixed_rule_oos",
                    "sample": smp,
                    "family": rule["family"],
                    "param": rule["param"],
                    "rule": rule["name"],
                    "adoptable": rule["adoptable"],
                    **st,
                    "Sharpe_hold": hold_st["Sharpe_ann"],
                    "Sharpe_exit": exit_st["Sharpe_ann"],
                    "beats_hold": bool(st["Sharpe_ann"] > hold_st["Sharpe_ann"]),
                    "beats_exit": bool(st["Sharpe_ann"] > exit_st["Sharpe_ann"]),
                    "beats_both": bool(
                        st["Sharpe_ann"] > hold_st["Sharpe_ann"]
                        and st["Sharpe_ann"] > exit_st["Sharpe_ann"]
                    ),
                    "modal_pick": rule["name"],
                    "modal_pick_share": 1.0,
                    "hold_share": float("nan"),
                    "n_candidates": 1,
                }
            )
    oos_df = pd.DataFrame(orows)
    write(oos_df, "oos.csv", "the purged, embargoed out-of-sample table")
    ocols = [
        "block",
        "family",
        "param",
        "rule",
        "adoptable",
        "n_days",
        "frac_exited",
        "mean",
        "sd",
        "Sharpe_ann",
        "Sharpe_hold",
        "Sharpe_exit",
        "beats_both",
        "t",
        "MaxDD",
        "worst_day",
        "mean_last30_exited_without_rule",
        "mean_last30_held",
        "mean_saved_vs_hold",
        "hit_rate_exit",
        "modal_pick",
        "modal_pick_share",
        "hold_share",
    ]
    print(
        f"\nout-of-sample day set (primary book): expanding training window, "
        f"{EMBARGO_DAYS}-day embargo, minimum {WARMUP_SESSIONS} training days, so "
        f"the first {WARMUP_SESSIONS + EMBARGO_DAYS} days are never scored "
        f"({int(scored_any.sum())} of {len(dates)} days scored)"
    )
    for smp, _ in samples:
        show(oos_df[oos_df["sample"] == smp], f"oos.csv  |  sample {smp}", ocols)

    # ----------------------------------------------------- placebo.csv -----
    rng = np.random.default_rng(PLACEBO_SEED)
    prows: list[dict[str, Any]] = []
    for smp, m in samples:
        for day_set, selm in (("full", m), ("oos", m & scored_any)):
            for rule in rules:
                if rule["family"] == "reference":
                    continue
                prows.append(
                    {
                        "sample": smp,
                        "day_set": day_set,
                        "block": "fixed_rule",
                        "family": rule["family"],
                        "param": rule["param"],
                        "rule": rule["name"],
                        "adoptable": rule["adoptable"],
                        **placebo_row(rule["phi"], legs["hold"], primary, selm, rng),
                    }
                )
        for fam in families:
            wf_scored, wf_phi = selected[fam]
            prows.append(
                {
                    "sample": smp,
                    "day_set": "oos",
                    "block": "walk_forward_selected",
                    "family": fam,
                    "param": float("nan"),
                    "rule": (
                        f"{fam}: cell chosen on an expanding window, "
                        f"{EMBARGO_DAYS}-day embargo"
                    ),
                    "adoptable": True,
                    **placebo_row(wf_phi, legs["hold"], primary, m & wf_scored, rng),
                }
            )
    plac_df = pd.DataFrame(prows)
    write(plac_df, "placebo.csv", "the day-shuffled placebo, primary book")
    pcols = [
        "day_set",
        "block",
        "family",
        "param",
        "rule",
        "n_placebo",
        "sharpe_rule",
        "placebo_mean",
        "placebo_sd",
        "placebo_q05",
        "placebo_q50",
        "placebo_q95",
        "pct_rank",
        "p_value",
    ]
    print(
        f"\nplacebo: {N_PLACEBO} draws per rule, seed {PLACEBO_SEED}, primary book; "
        f"each draw redraws WHICH days are exited, cutting the rule's own exit "
        f"mask into its maximal runs and permuting them, so the exit-day count "
        f"and the run lengths are preserved exactly and only the days change; "
        f"the rule's own hold fractions are laid down in order on the redrawn "
        f"days; p = (1 + #(placebo Sharpe >= rule Sharpe)) / (1 + n_placebo)"
    )
    for smp, _ in samples:
        show(plac_df[plac_df["sample"] == smp], f"placebo.csv  |  sample {smp}", pcols)

    # -------------------------------------------------------- gate.csv -----
    key = ["sample", "block_key", "family", "rule"]
    left = oos_df.copy()
    left["block_key"] = left["block"].replace({"fixed_rule_oos": "fixed_rule"})
    right = plac_df.loc[plac_df["day_set"] == "oos"].copy()
    right["block_key"] = right["block"]
    gate_df = left[left["family"] != "reference"].merge(
        right[[*key, "p_value", "sharpe_rule"]], on=key, how="left"
    )
    assert not gate_df["p_value"].isna().all(), "the gate lost its placebo leg"
    same = gate_df["sharpe_rule"].notna() & gate_df["Sharpe_ann"].notna()
    assert np.allclose(
        gate_df.loc[same, "sharpe_rule"],
        gate_df.loc[same, "Sharpe_ann"],
        rtol=0.0,
        atol=EXACT_TOL,
    ), "the placebo and the out-of-sample table disagree on the rule's own Sharpe"
    gate_df["adopted"] = (
        gate_df["beats_both"]
        & (gate_df["p_value"] < PLACEBO_ALPHA)
        & gate_df["adoptable"]
    )
    write(
        gate_df,
        "gate.csv",
        "adoption: out of sample beats BOTH references AND p < alpha",
    )
    show(
        gate_df,
        f"gate.csv  |  adopted = adoptable AND out of sample beats BOTH "
        f"references AND placebo p < {PLACEBO_ALPHA}  |  primary book",
        [
            "sample",
            "block",
            "family",
            "param",
            "rule",
            "adoptable",
            "Sharpe_ann",
            "Sharpe_hold",
            "Sharpe_exit",
            "beats_hold",
            "beats_exit",
            "beats_both",
            "p_value",
            "adopted",
        ],
    )
    fixed = gate_df[gate_df["block_key"] == "fixed_rule"]
    print(f"\nrules adopted on any sample: {int(gate_df['adopted'].sum())}")
    print(
        f"fixed-rule cells read by the gate: {len(fixed)} "
        f"({int(fixed['adoptable'].sum())} adoptable); adopted "
        f"{int(fixed['adopted'].sum())}"
    )
    for fam in families:
        sub = gate_df[
            (gate_df["block"] == "walk_forward_selected") & (gate_df["family"] == fam)
        ]
        for _, row in sub.iterrows():
            verdict = (
                "BEATS BOTH REFERENCES"
                if row["beats_both"]
                else "DOES NOT BEAT BOTH REFERENCES"
            )
            print(
                f"  family verdict, {row['sample']:<9s} {fam:<16s} walk-forward "
                f"{row['Sharpe_ann']:7.3f} against always hold "
                f"{row['Sharpe_hold']:7.3f} and always exit "
                f"{row['Sharpe_exit']:7.3f}  {verdict}  (placebo p "
                f"{row['p_value']:.4f})"
            )
    n_read = len(fixed)
    print(
        f"\nmultiple testing: {n_cells} exit cells were tried, scored on "
        f"{len(books)} books x {len(samples)} samples, so the gate above reads "
        f"{n_read} fixed-rule cells on the primary book and a "
        f"{PLACEBO_ALPHA:.0%} level expects {n_read * PLACEBO_ALPHA:.1f} false "
        f"adoptions there by chance."
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
