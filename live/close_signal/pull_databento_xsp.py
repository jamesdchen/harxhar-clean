"""XSP 0DTE quotes at the close from Databento OPRA (streaming API), estimate first.

    python live/close_signal/pull_databento_xsp.py --estimate            # sample days, extrapolated cost
    python live/close_signal/pull_databento_xsp.py --pull [--max-usd 100] # every session day

What it fetches, per session day D from 2023-03-28 (OPRA.PILLAR's first day):

1. ``definition`` for the parent ``XSP.OPT`` on D -> the contracts expiring on
   D (the 0DTE set), their strikes and OCC symbols; saved as
   ``defs_<D>.parquet`` (the strike grid the card assumes -- 1 point -- is
   read off these, not assumed).
2. ``cbbo-1m`` (consolidated NBBO sampled each minute, with sizes) for the
   0DTE contracts whose strike lies within a band around the spot, from
   15:20 to 16:16 ET -- the decision minute, the chase window and the
   settlement print; saved as ``cbbo1m_<D>.parquet``.

The spot band: the SPXW chain's 15:30 spot / 10 (data/spxw_spot.parquet)
with +-2 % where the chain has the day; otherwise the ES front-month close
at 15:24 ET from the purchased minutes (data/archive/es_v0_ohlcv1m_databento.csv)
with +-3 % -- ES sits within 1.5 % of SPX, so the ATM straddle is always inside.

Costs are summed as the pull goes and the run stops at --max-usd.  The key is
read from DATABENTO_API_KEY or asked for with a hidden prompt: run this in
your own terminal.  Output lands under data/archive/ (git-ignored).  Days
already on disk are skipped, so a stopped run resumes.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "archive" / "xsp_opra"
SPOT_PATH = REPO / "data" / "spxw_spot.parquet"
ES_PATH = REPO / "data" / "archive" / "es_v0_ohlcv1m_databento.csv"
DATASET = "OPRA.PILLAR"
PARENT = "XSP.OPT"
OPRA_START = pd.Timestamp("2023-03-28")
ET = "America/New_York"
WINDOW = ("15:20", "16:16")
BAND_CHAIN = 0.02
BAND_ES = 0.03
XSP_SCALE = 0.1


def session_days(today: pd.Timestamp) -> pd.DataFrame:
    """Session days with a spot proxy: ``day``, ``spot_xsp`` (SPX / 10 scale), ``band``, ``source``."""
    rows: list[dict[str, object]] = []
    if SPOT_PATH.exists():
        s = pd.read_parquet(SPOT_PATH)
        t = pd.to_datetime(s["timestamp"])
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_localize(
                None
            )  # the chain's stamps are naive ET carrying a fake UTC
        s = s.assign(t=t)
        s = s[s["t"].dt.strftime("%H:%M") == "15:30"]
        for d, spot in zip(s["t"].dt.normalize(), s["spot"].astype(float)):
            if d >= OPRA_START and np.isfinite(spot):
                rows.append(
                    {
                        "day": d,
                        "spot_xsp": spot * XSP_SCALE,
                        "band": BAND_CHAIN,
                        "source": "chain",
                    }
                )
    have = {r["day"] for r in rows}
    if ES_PATH.exists():
        es = pd.read_csv(ES_PATH, usecols=["ts_event", "close"])
        t = (
            pd.to_datetime(es["ts_event"], utc=True)
            .dt.tz_convert(ET)
            .dt.tz_localize(None)
        )
        es = es.assign(t=t)
        at = es[es["t"].dt.strftime("%H:%M") == "15:24"]
        for d, px in zip(at["t"].dt.normalize(), at["close"].astype(float)):
            if d >= OPRA_START and d not in have and np.isfinite(px):
                rows.append(
                    {
                        "day": d,
                        "spot_xsp": px * XSP_SCALE,
                        "band": BAND_ES,
                        "source": "es",
                    }
                )
    out = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    return out[out["day"] <= today.normalize()]


def et_window(day: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo = pd.Timestamp(f"{day.date()} {WINDOW[0]}", tz=ET).tz_convert("UTC")
    hi = pd.Timestamp(f"{day.date()} {WINDOW[1]}", tz=ET).tz_convert("UTC")
    return lo, hi


def occ_expiry(raw_symbol: pd.Series) -> pd.Series:
    """The expiry DATE encoded in an OCC symbol: ``XSP   230331P00400000`` -> 2023-03-31."""
    return pd.to_datetime(
        raw_symbol.astype(str).str.slice(6, 12), format="%y%m%d", errors="coerce"
    )


def zero_dte(defs: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The contracts expiring on ``day``: options only, with a strike.

    The expiry day is read off the OCC symbol.  Databento's ``expiration``
    field is the expiry DATE stamped at 00:00 UTC; converted to ET it lands on
    the previous evening, and the first pull (2026-09-23, 41 days) took every
    day's next-day contracts and nothing on Fridays.
    """
    d = defs.copy()
    exp = occ_expiry(d["raw_symbol"])
    d = d[(exp == day.normalize()) & d["instrument_class"].isin(["C", "P"])]
    d = d[np.isfinite(d["strike_price"].astype(float))]
    return d.drop_duplicates("raw_symbol")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--estimate", action="store_true", help="price a sample of days, extrapolate"
    )
    g.add_argument(
        "--pull", action="store_true", help="fetch every session day not yet on disk"
    )
    ap.add_argument("--sample", type=int, default=6, help="days priced in --estimate")
    ap.add_argument("--max-usd", type=float, default=100.0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--start", default=str(OPRA_START.date()))
    # OPRA.PILLAR after 13:30 UTC of the current day needs a live licence
    # (403 license_not_found_unauthorized, 2026-09-23): yesterday is the default end
    ap.add_argument(
        "--end", default=str((pd.Timestamp.today() - pd.Timedelta(days=1)).date())
    )
    a = ap.parse_args(argv)
    try:
        import databento as db
    except ImportError:
        print("pip install databento first", file=sys.stderr)
        return 2
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = session_days(pd.Timestamp(a.end))
    days = days[days["day"] >= pd.Timestamp(a.start)].reset_index(drop=True)
    if days.empty:
        print("no session days with a spot proxy in the range", file=sys.stderr)
        return 3
    print(
        f"{len(days)} session days {days['day'].min().date()} .. {days['day'].max().date()} "
        f"(chain spot on {int((days['source'] == 'chain').sum())}, ES proxy on "
        f"{int((days['source'] == 'es').sum())})"
    )
    key = os.environ.get("DATABENTO_API_KEY") or getpass.getpass(
        "Databento API key (hidden): "
    )
    client = db.Historical(key)

    if a.estimate:
        pick = days.iloc[np.linspace(0, len(days) - 1, a.sample).round().astype(int)]
    else:
        pick = days
    spent = 0.0
    priced = 0
    failed: list[str] = []
    for _, r in pick.iterrows():
        day = pd.Timestamp(r["day"])
        tag = day.strftime("%Y-%m-%d")
        f_defs = out_dir / f"defs_{tag}.parquet"
        f_cbbo = out_dir / f"cbbo1m_{tag}.parquet"
        # a saved definition set is good only if every symbol expires on the day
        # (the first pull saved the next day's contracts -- those are refetched)
        defs_ok = False
        if f_defs.exists():
            saved = pd.read_parquet(f_defs)
            defs_ok = bool(len(saved)) and bool(
                (occ_expiry(saved["raw_symbol"]) == day.normalize()).all()
            )
            if not defs_ok:
                f_defs.unlink()
                if f_cbbo.exists():
                    f_cbbo.unlink()
        if a.pull and f_cbbo.exists() and defs_ok:
            continue
        d0 = pd.Timestamp(day.date(), tz="UTC")
        kw_def = dict(
            dataset=DATASET,
            schema="definition",
            symbols=[PARENT],
            stype_in="parent",
            start=d0,
            end=d0 + pd.Timedelta(days=1),
        )
        c_def = float(client.metadata.get_cost(**kw_def))
        if f_defs.exists():
            defs = pd.read_parquet(f_defs)
        else:
            try:
                defs = (
                    client.timeseries.get_range(**kw_def)
                    .to_df(price_type="float", pretty_ts=True, map_symbols=True)
                    .reset_index()
                )
            except Exception as e:  # noqa: BLE001 -- one day's refusal must not end the run
                print(
                    f"{tag}: definitions refused ({type(e).__name__}: {str(e)[:120]}); skipped"
                )
                failed.append(tag)
                continue
            spent += c_def
            z = zero_dte(defs, day)
            z.to_parquet(f_defs, index=False)
            defs = z
        z = zero_dte(defs, day) if "expiration" in defs.columns else defs
        if z.empty:
            print(
                f"{tag}: no XSP contract expiring that day (holiday / no daily listing); skipped"
            )
            continue
        lo_k, hi_k = r["spot_xsp"] * (1 - r["band"]), r["spot_xsp"] * (1 + r["band"])
        near = z[
            (z["strike_price"].astype(float) >= lo_k)
            & (z["strike_price"].astype(float) <= hi_k)
        ]
        syms = sorted(near["raw_symbol"].astype(str).unique().tolist())
        lo, hi = et_window(day)
        kw_q = dict(
            dataset=DATASET,
            schema="cbbo-1m",
            symbols=syms,
            stype_in="raw_symbol",
            start=lo,
            end=hi,
        )
        c_q = float(client.metadata.get_cost(**kw_q)) if syms else 0.0
        step = (
            float(np.min(np.diff(np.unique(near["strike_price"].astype(float)))))
            if len(near) > 2
            else float("nan")
        )
        print(
            f"{tag}: 0DTE contracts {len(z)}, in band {len(syms)} (strike step {step:g}), "
            f"defs USD {c_def:.4f}, cbbo-1m USD {c_q:.4f}"
        )
        priced += 1
        if a.estimate:
            spent += c_q
            continue
        if spent + c_q > a.max_usd:
            print(
                f"stopping: spent {spent:.2f} + next {c_q:.2f} would pass --max-usd {a.max_usd}"
            )
            break
        if syms:
            try:
                q = (
                    client.timeseries.get_range(**kw_q)
                    .to_df(price_type="float", pretty_ts=True, map_symbols=True)
                    .reset_index()
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"{tag}: quotes refused ({type(e).__name__}: {str(e)[:120]}); skipped"
                )
                failed.append(tag)
                continue
            q.to_parquet(f_cbbo, index=False)
            spent += c_q
    if a.estimate:
        per_day = spent / max(priced, 1)
        print(
            f"\nESTIMATE: {priced} days priced, mean USD {per_day:.4f} per day "
            f"-> {len(days)} days ~ USD {per_day * len(days):.2f} (definitions pulled for the sample: paid)"
        )
    else:
        print(f"\nPULL: spent USD {spent:.2f}; files in {out_dir}")
    if failed:
        print(f"skipped days ({len(failed)}): {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
