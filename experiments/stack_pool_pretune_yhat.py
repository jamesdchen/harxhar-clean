"""Stack the pre-tuned POOLED linear forecasts (arm f) into 16:00 yhat tables for the master table.

User decision (2026-10-06): the pooled linear rows also use the pre-tuned version -- one protocol
for every linear model, decided before the pooled results were seen.  Source: the Hoffman2 run of
record of results/pool_linear_pretune_2026-10-06/ (experiments/pool_linear_pretune.py; its
SUMMARY.md and gates.csv): forecasts.parquet, series

  pool_linear_pretune/<est>_<bucket>_f_retune_margin_bb    (all_features, lean_vol_vix)
  pool_linear_pretune/<est>_baseline_f_retune_margin        (baseline: every column backbone)

for est in ridge / lasso / enet: date = the naive-ET 16:00 stamp, pred_adj = the adjusted-scale
forecast of the pooled model at the 16:00 bar, 1087 rows per series (2020-01-03 .. 2024-04-30).

This script writes, WITHOUT touching the stored one-penalty pooled tables,

  results/spxw_pnl/yhat_poolpt_<est>_<bucket>.parquet

with the stored pooled table's schema (t UTC, yhat, baseline, rv_raw) on every 16:00 stamp of the
stored results/spxw_pnl/yhat_pool_<est>_<bucket> table (its t, baseline (B) and rv_raw): yhat is
the f-arm pred_adj on the export's stamps and NaN before them (the pre-tuned arm starts at the
trade period; the master table scores finite forecasts and gates the stamps, finite or not).

GATES (asserted; results/pool_linear_pretune_2026-10-06/stack_gates.csv):
  * every export stamp is a 16:00 stamp of the stored pooled table; the new table has the stored
    table's 16:00 stamps, finite exactly on the export's;
  * the stacked yhat equals the f-arm export bit for bit;
  * t and B equal the stored pooled table's on those stamps;
  * arm (a) of the study (one penalty, the spec's causal re-choice) reproduces the stored pooled
    yhat on the same stamps to within A_TOL relative (the study: bit for bit except all_features
    ridge at 3.2e-7 -- the Sherman-Morrison running solve on another machine).

Run:  python experiments/stack_pool_pretune_yhat.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results" / "pool_linear_pretune_2026-10-06"
SPXW = ROOT / "results" / "spxw_pnl"
EXPORT = STUDY / "forecasts.parquet"
GATES_OUT = STUDY / "stack_gates.csv"
ESTS = ("ridge", "lasso", "enet")
BUCKETS = ("baseline", "all_features", "lean_vol_vix")  # the paper's three input sets
A_TOL = 1e-6  # arm (a) vs the stored pooled table, relative (the study: <= 3.2e-7)


def f_series(est: str, bucket: str) -> str:
    """The export series of arm (f): backbone 1/100 except on baseline (all backbone)."""
    arm = "f_retune_margin" if bucket == "baseline" else "f_retune_margin_bb"
    return f"pool_linear_pretune/{est}_{bucket}_{arm}"


def naive_et(t: pd.Series) -> pd.DatetimeIndex:
    et = pd.to_datetime(t).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return pd.DatetimeIndex(et.to_numpy()).as_unit("ns")


def main() -> None:
    ex = pd.read_parquet(EXPORT)
    gates: list[dict] = []

    def gate(name: str, item: str, value: object, ok: bool) -> None:
        gates.append({"gate": name, "item": item, "value": value, "ok": bool(ok)})
        assert ok, (name, item, value)

    for est in ESTS:
        for bucket in BUCKETS:
            sid = f_series(est, bucket)
            f = ex[ex["series_id"] == sid].sort_values("date")
            assert len(f) > 0 and f["date"].is_unique and f["pred_adj"].notna().all(), (
                sid
            )
            fx = pd.DatetimeIndex(pd.to_datetime(f["date"]).to_numpy()).as_unit("ns")
            src = SPXW / f"yhat_pool_{est}_{bucket}.parquet"
            st = pd.read_parquet(src)
            et = naive_et(st["t"])
            m16 = (et.hour == 16) & (et.minute == 0)
            st16 = st[m16].copy()
            st16.index = et[m16]
            assert not st16.index.duplicated().any(), src
            have = fx.isin(st16.index)
            gate(
                "export stamps are 16:00 stamps of the stored pooled table",
                f"{est}_{bucket}",
                f"{int(have.sum())} of {len(fx)}",
                have.all(),
            )
            st16 = st16.sort_index()
            fy = pd.Series(f["pred_adj"].to_numpy(), index=fx)
            out = pd.DataFrame(
                {
                    "t": st16["t"].to_numpy(),
                    "yhat": fy.reindex(st16.index).to_numpy(),  # NaN before the export
                    "baseline": st16["baseline"].to_numpy(),
                    "rv_raw": st16["rv_raw"].to_numpy(),
                }
            )
            dst = SPXW / f"yhat_poolpt_{est}_{bucket}.parquet"
            assert dst.name != src.name
            out.to_parquet(dst, index=False)
            back = pd.read_parquet(dst)
            bk = pd.Series(back["yhat"].to_numpy(), index=naive_et(back["t"]))
            fin = bk[np.isfinite(bk)]
            exact = bool(
                fin.index.equals(fx)
                and (fin.to_numpy() == f["pred_adj"].to_numpy()).all()
            )
            gate(
                "stacked yhat = the f-arm export, bit for bit (finite exactly on its stamps)",
                f"{est}_{bucket}",
                f"{len(fin)} finite of {len(bk)}",
                exact,
            )
            same = bool(
                (back["baseline"].to_numpy() == st16["baseline"].to_numpy()).all()
                and (
                    pd.to_datetime(back["t"]).to_numpy()
                    == pd.to_datetime(st16["t"]).to_numpy()
                ).all()
            )
            gate(
                "t and B = the stored pooled table's (every 16:00 stamp)",
                f"{est}_{bucket}",
                "identical" if same else "differ",
                same,
            )
            a = ex[ex["series_id"] == f"pool_linear_pretune/{est}_{bucket}_a_causal"]
            a = pd.Series(
                a["pred_adj"].to_numpy(),
                index=pd.DatetimeIndex(pd.to_datetime(a["date"]).to_numpy()).as_unit(
                    "ns"
                ),
            )
            j = a.reindex(fx)
            rel = float(
                np.nanmax(np.abs(j.to_numpy() / st16.loc[fx, "yhat"].to_numpy() - 1.0))
            )
            gate(
                f"arm (a) = the stored pooled table (max rel <= {A_TOL:g})",
                f"{est}_{bucket}",
                rel,
                j.notna().all() and rel <= A_TOL,
            )
            print(f"{dst.name}: {len(out)} rows; arm (a) vs stored max rel {rel:.1e}")
    pd.DataFrame(gates).to_csv(GATES_OUT, index=False, lineterminator="\n")
    print(f"{len(gates)} gates ok -> {GATES_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
