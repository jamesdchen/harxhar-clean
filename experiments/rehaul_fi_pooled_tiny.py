"""Tiny end-to-end case of experiments/rehaul_fi_pooled_compute.py (linear + reduce) and writeup/make_rehaul_fi_pooled.py
on a synthetic design: 14 predictors of the pooled design's name forms, 48 intervals a day, a 60-day window, both
samples, five months.  Seconds; run before every push.

    python experiments/rehaul_fi_pooled_tiny.py [DIR]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NAMES = [
    "har_ma_1", "har_ma_2", "har_ma_4", "DOW_0", "DOW_1", "hour", "is_overnight",
    "adj_vix_ma_1", "adj_vix_ma_4", "vix_avail_ma_4",
    "adj_sumret_ma_1", "sumret_active_ma_4", "adj_vvix_ma_1", "adj_vvix_ma_2",
]


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="rhfp_tiny_"))
    dz = root / "design"
    dz.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range("2019-06-03", "2020-05-29")
    labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    dates = [f"{d.date()} {t}:00" for d in days for t in labels]
    n, p = len(dates), len(NAMES)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    X[:, NAMES.index("adj_vvix_ma_2")] = X[:, NAMES.index("adj_vvix_ma_1")]  # a byte-duplicate column (masked)
    y = X[:, 0] * 0.5 + X[:, 7] * 0.2 + X[:, 10] * 0.05 + rng.standard_normal(n)
    np.save(dz / "X.npy", X)
    np.save(dz / "y.npy", y)
    pd.DataFrame({"date": dates}).to_csv(dz / "dates_all.csv", index=False)
    (dz / "design.json").write_text(json.dumps(dict(names=NAMES, shape=[n, p], env={})))
    pen = root / "frozen_penalties.json"
    pen.write_text(json.dumps(dict(ridge=dict(alpha=1e3), lasso=dict(alpha=1e-2), r=0.01, columns=NAMES[:-1])))
    py = [sys.executable, str(REPO / "experiments" / "rehaul_fi_pooled_compute.py")]
    for s in ("48", "13"):
        o = root / "res" / s
        common = ["--out", str(o)]
        subprocess.run(py + ["linear", *common, "--design", str(dz), "--sample", s, "--penalties", str(pen),
                             "--start", "2020-01-03", "--end", "2020-05-29", "--tw-days", "60"], check=True)
        subprocess.run(py + ["reduce", *common], check=True)
        S = json.loads((o / "summary.json").read_text())
        assert S["n_refits"] == 5 and S["p"] == p - 1, S
        assert S["lasso"]["probe_n"] == 2, S["lasso"]
        assert "adj_vvix_ma_1" in pd.read_csv(o / "lasso_columns.csv")["column"].tolist()
        assert S["lasso"]["series_ever"] >= 2
    subprocess.run([sys.executable, str(REPO / "writeup" / "make_rehaul_fi_pooled.py"), "--res", str(root / "res"),
                    "--out", str(root / "numbers.tex")], check=True)
    tex = (root / "numbers.tex").read_text()
    assert "\\rhfpAllLassoW" in tex and "\\rhfpRthRidgeW" in tex
    print(f"tiny end-to-end OK: {root}")


if __name__ == "__main__":
    main()
