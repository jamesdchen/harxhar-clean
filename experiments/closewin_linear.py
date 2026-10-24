"""Close-only window sweep, pooled linear models (results/closeonly_wsweep_2014start/RULE.txt section 2):
experiments/linear_train3way.py's CLOSE arm with its window WD set to W, nothing else changed.  The close design is
derived (derive_close) from the pooled sweep's W design (the /home1 copy made by experiments/closewin.py stage), the
penalties re-tuned on fold 3 with the native rule (pretune, select), the walk refits at every session's 16:00 origin
(walk1), export writes forecasts_rth.parquet.

usage: python experiments/closewin_linear.py <linear_train3way stage> --W <W> --arm CLOSE --out <dir> [its options]
       python experiments/closewin_linear.py gate --out <W = 3,000 out dir> --ref <the three-way CLOSE out dir>
  gate (RULE.txt gate L): the W = 3,000 run against the three-way CLOSE arm -- derived caches md5-identical, frozen
  penalties and pretune_mse.csv identical, forecasts_rth.parquet values identical -> <out>/gate.json, exit 3 otherwise

2026-10-24 addendum B (owner: "go on linear w=1k+1"): the W = 1,000 point with the penalized models at a 1,001-close
window (STANDARDIZE_PENALTY's walk needs a window > 1,000 rows) and OLS at 1,000:
       python experiments/closewin_linear.py derive_shift --from <W = 1,000 out dir> --out <dir> --rows 1001
         the W = 1,000 close cache (X, y, dates_all, rth13_rows unchanged) with its first OOS row moved one close later
         (b_oos and dates_oos without their first entry, config window 1,001), so that design row 1,001 is the first
         row with a stored B and every window holds 1,001 closes; the scaler / late-start rule stay the W = 1,000
         design's
       python experiments/closewin_linear.py <stage> --W 1000 --rows 1001 --arm CLOSE --out <dir> [...]
         the linear_train3way stages with the runner's --window-rows 1001 (pretune / select / walk1 on that cache)
       python experiments/closewin_linear.py export_1k --pen <the 1,001 out dir> --ols <the W = 1,000 out dir>
         <pen>/forecasts_rth.parquet: ridge / lasso / elastic net of both sets from the 1,001-close walks, OLS from the
         1,000-close walk (linear_train3way.frame, unchanged)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closewin import WS, md5  # noqa: E402

CACHE = ("X.npy", "y.npy", "b_oos.npy", "dates_oos.csv", "dates_all.csv", "rth13_rows.npy")
FC_COLS = ("date", "alpha", "pred_adj", "true_adj", "B", "plain", "pred_raw_duan", "true_raw", "rv_actual")


def gate(out: Path, ref: Path) -> int:
    g: dict = {"out": str(out), "ref": str(ref)}
    ok = True
    for s in ("baseline", "all_features"):
        a, b = out / "_work" / f"design_{s}", ref / "_work" / f"design_{s}"
        r = {fn: [md5(a / fn), md5(b / fn)] for fn in CACHE}
        ja, jb = (json.loads((d / "design.json").read_text()) for d in (a, b))
        # the keys the three-way's gate (i) compared (host, numpy, rerun, part record where the source was built)
        same_meta = {
            k: ja.get(k) == jb.get(k) for k in ("names", "shape", "hp", "config", "raw_cols", "bucket", "f_order")
        }
        dfa = {k: v for k, v in ja["derived_from"].items() if k != "design"}
        dfb = {k: v for k, v in jb["derived_from"].items() if k != "design"}
        r_ok = all(v[0] == v[1] for v in r.values()) and all(same_meta.values()) and dfa == dfb
        g[f"cache_{s}"] = dict(md5=r, meta_equal=all(same_meta.values()), derived_from_equal=dfa == dfb, pass_=r_ok)
        ok &= r_ok
    fa, fb = (json.loads((d / "frozen_penalties.json").read_text()) for d in (out, ref))
    pen = all(fa[m] == fb[m] for m in ("ridge", "lasso", "enet", "ols"))
    ext = fa["info"].get("grid_extensions") == fb["info"].get("grid_extensions") and fa["info"].get(
        "off_spec_grid"
    ) == fb["info"].get("off_spec_grid")
    pa, pb = pd.read_csv(out / "pretune_mse.csv"), pd.read_csv(ref / "pretune_mse.csv")
    mse = pa.shape == pb.shape and bool((pa.astype(str).to_numpy() == pb.astype(str).to_numpy()).all())
    g["penalties"] = dict(
        equal=pen, grid_rule_equal=ext, pretune_mse_equal=mse, mine={m: fa[m] for m in fa if m != "info"}
    )
    ok &= pen and ext and mse
    xa, xb = pd.read_parquet(out / "forecasts_rth.parquet"), pd.read_parquet(ref / "forecasts_rth.parquet")
    g["forecasts_md5"] = [md5(out / "forecasts_rth.parquet"), md5(ref / "forecasts_rth.parquet")]
    keys = ["model", "predictor_set"]
    same_rows = xa.shape == xb.shape and xa[keys].equals(xb[keys])
    cols = {}
    for c in FC_COLS:
        u, v = xa[c].to_numpy(), xb[c].to_numpy()
        cols[c] = bool(same_rows and (np.array_equal(u, v) if c == "date" else np.array_equal(u, v, equal_nan=True)))
    g["forecasts"] = dict(rows=len(xa), same_rows=same_rows, identical=cols)
    ok &= same_rows and all(cols.values())
    g["pass"] = bool(ok)
    (out / "gate.json").write_text(json.dumps(g, indent=1))
    print(f"GATE L {'PASS' if ok else 'FAIL'}: {json.dumps(g)}", flush=True)
    return 0 if ok else 3


def derive_shift(src: Path, out: Path, n: int) -> int:
    """Addendum B: the close cache of src with its first OOS row moved to design row n (see the module docstring)."""
    import shutil

    for s in ("baseline", "all_features"):
        a, o = src / "_work" / f"design_{s}", out / "_work" / f"design_{s}"
        o.mkdir(parents=True, exist_ok=True)
        meta = json.loads((a / "design.json").read_text())
        wc = int(meta["config"]["window"])
        d = n - wc
        assert d > 0, (wc, n)
        for fn in ("X.npy", "y.npy", "dates_all.csv", "rth13_rows.npy"):
            shutil.copyfile(a / fn, o / fn)
            assert md5(a / fn) == md5(o / fn), fn
        b = np.load(a / "b_oos.npy")
        np.save(o / "b_oos.npy", np.ascontiguousarray(b[d:]))
        oos = pd.read_csv(a / "dates_oos.csv", dtype=str)
        oos.iloc[d:].to_csv(o / "dates_oos.csv", index=False)
        da = pd.read_csv(o / "dates_all.csv", dtype=str)["date"]
        assert da.iloc[n:].reset_index(drop=True).equals(oos["date"].iloc[d:].reset_index(drop=True)), s
        assert len(da) == n + len(b) - d
        out_meta = dict(
            meta,
            config=dict(meta["config"], window=n),
            derived_from=dict(
                meta["derived_from"],
                shift=dict(
                    source=str(a),
                    rule=f"the source cache with its first {d} OOS row(s) dropped from b_oos / dates_oos: design row "
                    f"{n} is the first row with a stored B, every window {n} closes (addendum B)",
                    source_window=wc,
                ),
            ),
        )
        (o / "design.json").write_text(json.dumps(out_meta))
        print(f"derive_shift {s}: window {wc} -> {n}; first OOS {oos['date'].iloc[d]} (was {oos['date'].iloc[0]})")
    return 0


def export_1k(pen: Path, ols: Path) -> int:
    """Addendum B: <pen>/forecasts_rth.parquet = the penalized models' 1,001-close walks + OLS's 1,000-close walk."""
    import linear_train3way as L3

    orch = L3.O
    core = pd.read_parquet(L3.REPO / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
    rv_bar = pd.Series(core["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(core["endbartime"])))
    rv_bar = rv_bar[~rv_bar.index.duplicated()]
    L3.WD = 1000
    parts = []
    runs: tuple = (
        (pen, 1001, [(s, m) for s in orch.SETS for m in orch.EST]),
        (ols, None, [("baseline", "ols")]),
    )
    for out, rows, models in runs:
        if rows:
            L3.ARMS["CLOSE"]["rows"] = rows
        else:
            L3.ARMS["CLOSE"].pop("rows", None)
        L3.configure("CLOSE", out)
        assert orch.W == (rows or 1000), orch.W
        fr = json.loads((out / "frozen_penalties.json").read_text())
        B = pd.Series(np.load(orch.design_dir("baseline") / "b_oos.npy"), index=pd.DatetimeIndex(orch.oos_dates()))
        for s, m in models:
            f = L3.frame(s, m, fr[m][s], B, rv_bar)
            f["window_rows"] = orch.W
            parts.append(f)
    fc = pd.concat(parts, ignore_index=True)
    fc.to_parquet(pen / "forecasts_rth.parquet", index=False)
    print(f"export_1k: {len(fc)} rows, {fc.groupby(['model', 'predictor_set', 'window_rows']).size().to_dict()}")
    return 0


def arg(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    if sys.argv[1] == "gate":
        return gate(Path(arg("--out")).resolve(), Path(arg("--ref")).resolve())
    if sys.argv[1] == "derive_shift":
        return derive_shift(Path(arg("--from")).resolve(), Path(arg("--out")).resolve(), int(arg("--rows")))
    if sys.argv[1] == "export_1k":
        return export_1k(Path(arg("--pen")).resolve(), Path(arg("--ols")).resolve())
    i = sys.argv.index("--W")
    W = int(sys.argv[i + 1])
    del sys.argv[i : i + 2]
    assert W in WS, W
    a = sys.argv
    assert a[a.index("--arm") + 1] == "CLOSE", "the close arm only"
    import linear_train3way as L3

    L3.WD = W  # the arm's window (closes) and the 13-bar source design's window (13 W rows) in derive_close
    if "--rows" in a:  # addendum B: a row-count window on a derive_shift cache (the runner's --window-rows)
        j = a.index("--rows")
        L3.ARMS["CLOSE"]["rows"] = int(a[j + 1])
        del a[j : j + 2]
        assert (W, L3.ARMS["CLOSE"]["rows"]) == (1000, 1001), "addendum B: W = 1,000 at 1,001 closes only"
    L3.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
