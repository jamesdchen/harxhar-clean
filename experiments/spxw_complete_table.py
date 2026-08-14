"""One-pass mid-IV a0/blk2/tree resign + exog increment for every sweep part."""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "results", "spxw_pnl", "parts")
BARS_PER_YEAR = 252.0 * 48.0


def _yh(path: str, tag: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["t"] = pd.to_datetime(df["t"], utc=True).astype("datetime64[ns, UTC]")
    return df.sort_values("t")[["t", "yhat", "baseline"]].rename(
        columns={"yhat": f"y_{tag}", "baseline": f"b_{tag}", "t": f"t_{tag}"}
    )


def _join(tr: pd.DataFrame, yh: pd.DataFrame, tag: str) -> pd.DataFrame:
    return pd.merge_asof(
        tr, yh, left_on="t0", right_on=f"t_{tag}", direction="backward"
    )


def _iv(m: pd.DataFrame) -> np.ndarray:
    exp = pd.to_datetime(m["expiration"])
    close = (
        exp.dt.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    ).dt.tz_convert("UTC")
    tau = (close - m["t0"]).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600)
    return m["entry"].to_numpy(float) / (
        m["spot"].to_numpy(float)
        * np.sqrt(np.maximum(tau, 1e-8))
        * np.sqrt(2.0 / np.pi)
    )


def _sh(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0:
        return float("nan")
    return float(x.mean() / x.std())


def main() -> None:
    files = sorted(
        glob.glob(os.path.join(PARTS, "h*_sweep.parquet")),
        key=lambda p: int(os.path.basename(p).split("_")[0][1:]),
    )
    models = {}
    for tag, fn in (
        ("a0", "yhat_a0.parquet"),
        ("b2", "yhat_blk2.parquet"),
        ("t0", "yhat_tree00.parquet"),
        ("t16", "yhat_tree16.parquet"),
    ):
        p = os.path.join(ROOT, "results", "spxw_pnl", fn)
        if os.path.exists(p):
            models[tag] = _yh(p, tag)
    print("h n settle long_sh a0_sh b2_sh t00_sh t16_sh incr_sh sign+a0", flush=True)
    rows = []
    for path in files:
        tr = pd.read_parquet(path)
        tr["t0"] = pd.to_datetime(tr["t0"], utc=True).astype("datetime64[ns, UTC]")
        d = tr["d_long"].to_numpy(float)
        h = int(tr["h"].iloc[0])
        nset = int((tr["how"] == "settle").sum()) if "how" in tr.columns else 0
        m = tr.sort_values("t0")
        for tag, yh in models.items():
            m = _join(m, yh, tag)
        iv = _iv(m)
        hh = m["h"].to_numpy(float)
        out = {"h": h, "n": len(m), "settle": nset, "long_sh": _sh(d)}
        for tag in models:
            pred = (
                hh
                * (m[f"y_{tag}"].to_numpy(float) ** 2)
                * m[f"b_{tag}"].to_numpy(float)
            )
            impl = (iv**2) * hh / BARS_PER_YEAR
            sgn = np.sign(pred - impl)
            sgn[~np.isfinite(pred) | ~np.isfinite(impl)] = 0.0
            out[f"{tag}_sh"] = _sh(sgn * d)
            if tag == "a0":
                out["sign+a0"] = float((sgn > 0).mean())
        if "a0" in models and "b2" in models:
            gap = hh * (
                (m["y_b2"].to_numpy(float) ** 2) * m["b_b2"].to_numpy(float)
                - (m["y_a0"].to_numpy(float) ** 2) * m["b_a0"].to_numpy(float)
            )
            z = gap / (np.nanstd(gap) + 1e-18)
            out["incr_sh"] = _sh(z * d)
        rows.append(out)
        print(
            f"{h:2d} {out['n']:5d} {out['settle']:5d} {out['long_sh']:+.3f} "
            f"{out.get('a0_sh', float('nan')):+.3f} {out.get('b2_sh', float('nan')):+.3f} "
            f"{out.get('t0_sh', float('nan')):+.3f} {out.get('t16_sh', float('nan')):+.3f} "
            f"{out.get('incr_sh', float('nan')):+.3f} {out.get('sign+a0', float('nan')):.3f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(ROOT, "results", "spxw_pnl", "complete_table.csv"), index=False
    )
    print("wrote results/spxw_pnl/complete_table.csv", flush=True)


if __name__ == "__main__":
    main()
