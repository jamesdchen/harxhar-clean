"""The fast path's gate: the card it produces IS the path of record's card.

    python -m live.close_signal.fastpath_check --check \
        --sessions 2026-09-23,2026-09-24,2026-08-31

For each session it rebuilds the state as it stood on that day (the committed
panel cut at the stamp: rows up to 15:00 for the precompute phase, up to 15:30
for the card; later rows and placeholders dropped) and runs

* the PATH OF RECORD -- ``forecast.forecast_session``: 13 spec processes; and
* the FAST PATH exactly as the daily run does it -- a ``FastContext`` (warm
  ``arms_shared --serve`` server), the ``canary`` on the 15:00 panel, then
  ``forecast_session_fast`` on the 15:30 panel through the same server;

and requires, with nothing tolerated:

* the 13 results CSVs equal frame for frame (``DataFrame.equals``: bit-identical);
* rv_hat, yhat, baseline and the MZ pieces equal (relative deviation 0);
* the card equal: P*, the strikes, the counts and the decision of
  ``signal.build_instruction`` (every field of the record), at the session's
  15:29 ^GSPC close when Yahoo still serves it (else a fixed nominal spot --
  both paths price at the same spot either way) and a non-zero capital.

It also reports what the canary shows about dependence: the largest relative
change the 15:30 row makes to the PRIOR sessions' forecasts (why no arm output
computed before 15:30 can be reused).  Writes a JSON report; exits 1 on any
mismatch.  About 5 minutes per session on a 12-core Windows machine (the path
of record dominates).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from live.close_signal import feeds, forecast, schedule
from live.close_signal.common import stamp
from live.close_signal.signal import build_instruction
from live.close_signal.state import StateStore

REPO = Path(__file__).resolve().parents[2]
STATE_DIR = REPO / "live" / "close_signal" / "state"
FOMC_CSV = STATE_DIR / "fomc_statement_dates.csv"
NOMINAL_SPOT = 6500.0
CAPITAL = 100_000.0


def as_of(src: Path, dst: Path, cutoff: pd.Timestamp) -> StateStore:
    """A state directory holding the committed panel's rows up to ``cutoff``."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    p = pd.read_parquet(Path(src) / "panel_free.parquet")
    keep = (pd.to_datetime(p["endbartime"]) <= cutoff) & ~p["placeholder"].astype(bool)
    p[keep].reset_index(drop=True).to_parquet(dst / "panel_free.parquet", index=False)
    return StateStore(dst)


def session_spot(session: pd.Timestamp) -> tuple[float, str]:
    try:
        b = feeds.spx_minute_bars(session)
        s = b.frame["close"]
        s = s[
            (s.index.normalize() == session.normalize())
            & (s.index <= stamp(session, "15:29"))
        ]
        if len(s):
            return float(s.iloc[-1]), f"^GSPC close of the {s.index[-1]:%H:%M} bar"
    except Exception as e:  # noqa: BLE001 -- the spot only has to be the same on both sides
        return NOMINAL_SPOT, f"nominal ({type(e).__name__})"
    return NOMINAL_SPOT, "nominal (no bar)"


def rel(a: float, b: float) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(b), 1e-300)


def csv_equal(a: dict[str, Path], b: dict[str, Path]) -> dict[str, bool]:
    return {k: pd.read_csv(a[k]).equals(pd.read_csv(b[k])) for k in forecast.BARS}


def record_equal(x: dict[str, Any], y: dict[str, Any]) -> list[str]:
    bad = []
    for k in x:
        u, v = x[k], y[k]
        same = u == v or (
            isinstance(u, float)
            and isinstance(v, float)
            and math.isnan(u)
            and math.isnan(v)
        )
        if not same:
            bad.append(f"{k}: {u!r} != {v!r}")
    return bad


def prior_dependence(
    canary_csvs: dict[str, Path], card_csvs: dict[str, Path], session: pd.Timestamp
) -> dict[str, float]:
    """Max relative change of pred_adj on rows before the session: canary vs card."""
    worst, rows, differ = 0.0, 0, 0
    for b in forecast.BARS:
        x = pd.read_csv(canary_csvs[b], parse_dates=["date"]).set_index("date")[
            "pred_adj"
        ]
        y = pd.read_csv(card_csvs[b], parse_dates=["date"]).set_index("date")[
            "pred_adj"
        ]
        j = pd.concat([x, y], axis=1, join="inner", keys=["a", "b"])
        j = j[j.index < session.normalize()]
        d = (j["a"] - j["b"]).abs() / j["b"].abs().clip(lower=1e-300)
        rows += len(j)
        differ += int((j["a"] != j["b"]).sum())
        worst = max(worst, float(d.max()) if len(d) else 0.0)
    return {
        "prior_rows": rows,
        "prior_rows_changed": differ,
        "prior_max_rel_pred_adj": worst,
    }


def check_session(
    session: pd.Timestamp,
    scratch: Path,
    state_dir: Path,
    workers: int,
    fast_workers: int,
) -> dict[str, Any]:
    day = scratch / f"{session:%Y%m%d}"
    day.mkdir(parents=True, exist_ok=True)
    # --- path of record
    st_full = as_of(state_dir, day / "state_full", stamp(session, "15:30"))
    t = time.perf_counter()
    fc_full = forecast.forecast_session(
        REPO, st_full, session, day / "full", FOMC_CSV, workers=workers
    )
    s_full = time.perf_counter() - t
    csv_full = {
        b: day
        / "full"
        / "arms"
        / b
        / "causal_tune_linear"
        / forecast.ESTIMATOR
        / forecast.BUCKET
        / f"results_{b}.csv"
        for b in forecast.BARS
    }
    # --- fast path, as the daily run does it
    t = time.perf_counter()
    ctx = forecast.FastContext(REPO, day / "fast", workers=fast_workers)
    try:
        st_pre = as_of(state_dir, day / "state_pre", stamp(session, "15:00"))
        can = forecast.canary(ctx, st_pre, session, FOMC_CSV)
        s_pre = time.perf_counter() - t
        st_card = as_of(state_dir, day / "state_card", stamp(session, "15:30"))
        t = time.perf_counter()
        fc_fast = forecast.forecast_session_fast(
            REPO, st_card, session, day / "fast", FOMC_CSV, ctx=ctx
        )
        s_fast = time.perf_counter() - t
    finally:
        ctx.close()
    base = day / "fast" / "fast_arms"
    csv_fast = {
        b: base
        / "card"
        / "causal_tune_linear"
        / forecast.ESTIMATOR
        / forecast.BUCKET
        / f"results_{b}.csv"
        for b in forecast.BARS
    }
    csv_canary = {
        b: base
        / "canary"
        / "causal_tune_linear"
        / forecast.ESTIMATOR
        / forecast.BUCKET
        / f"results_{b}.csv"
        for b in forecast.BARS
    }
    eq = csv_equal(csv_fast, csv_full)
    devs = {k: rel(fc_fast[k], fc_full[k]) for k in fc_full}
    spot, spot_src = session_spot(session)
    flags = schedule.calendar_flags(session.date())
    kw = dict(
        session=session.date(),
        spot=spot,
        flags=flags,
        capital=CAPITAL,
        input_mode="replay",
    )
    i_full = build_instruction(rv_hat=fc_full["rv_hat"], **kw).as_record()  # type: ignore[arg-type]
    i_fast = build_instruction(rv_hat=fc_fast["rv_hat"], **kw).as_record()  # type: ignore[arg-type]
    card_bad = record_equal(i_fast, i_full)
    ok = all(eq.values()) and all(v == 0.0 for v in devs.values()) and not card_bad
    return {
        "session": str(session.date()),
        "ok": ok,
        "csv_identical": f"{sum(eq.values())}/{len(eq)}",
        "fc_full": fc_full,
        "fc_fast": fc_fast,
        "max_rel_dev": devs,
        "spot": spot,
        "spot_source": spot_src,
        "month_end": flags["month_end"],
        "card_full": {
            k: i_full[k]
            for k in (
                "decision",
                "kc_spx",
                "kp_spx",
                "p_star_spx",
                "kc_xsp",
                "kp_xsp",
                "p_star_xsp",
                "n_spx_at_pstar",
                "n_xsp_at_pstar",
            )
        },
        "card_fields_differing": card_bad,
        "seconds_path_of_record": round(s_full, 1),
        "seconds_precompute_canary": round(s_pre, 1),
        "seconds_fast_forecast": round(s_fast, 1),
        "canary": can,
        "dependence": prior_dependence(csv_canary, csv_fast, session),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="run the gate")
    ap.add_argument("--sessions", default="2026-09-23,2026-09-24,2026-08-31")
    ap.add_argument(
        "--state-dir", default=str(STATE_DIR), help="read only; copies are cut from it"
    )
    ap.add_argument(
        "--scratch",
        default=str(REPO / "live" / "close_signal" / ".scratch" / "fastpath_check"),
    )
    ap.add_argument(
        "--workers", type=int, default=forecast.WORKERS, help="path of record"
    )
    ap.add_argument("--fast-workers", type=int, default=forecast.FAST_WORKERS)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    if not a.check:
        ap.print_help()
        return 0
    scratch = Path(a.scratch).resolve()  # the spec processes run in their own cwd
    out = []
    for s in a.sessions.split(","):
        r = check_session(
            pd.Timestamp(s), scratch, Path(a.state_dir), a.workers, a.fast_workers
        )
        out.append(r)
        print(json.dumps(r, indent=1, default=str), flush=True)
    path = Path(a.out_json) if a.out_json else scratch / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str))
    ok = all(r["ok"] for r in out)
    print(
        ("FAST PATH == PATH OF RECORD on " if ok else "MISMATCH on ")
        + ", ".join(f"{r['session']} ({'ok' if r['ok'] else 'FAIL'})" for r in out),
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
