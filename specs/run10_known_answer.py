# Run-#10 known-answer draft — audited against the run-#10 template.
#
# WHAT THIS IS: the real-but-small known-answer analysis the template header
# asks for — reproduce the deployed baseline's QLIKE LIVE (the deployed ridge
# config, same walk-forward engine) and score ONE existing exog family
# (market_ew) against it. The three pinned sections are copied VERBATIM from
# the template; the two variable sections below carry the analysis.
#
# VERDICT BOUNDARY: sections show evidence; no section states a conclusion.
# RECEIPTS: no emitter in this env — evidence is printed/shown, never
# assert-gated.

# %%
# Path bootstrap (PREAMBLE — outside every audit section by design; interactive
# kernels default cwd to this file's directory, and `src.*` imports need the
# repo root). Walks up from the file (or cwd when __file__ is absent, e.g. an
# interactive cell) to the first directory containing src/.
import sys
from pathlib import Path

import os

_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _START
while not (_ROOT / "src").is_dir() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if (_ROOT / "src").is_dir():  # only normalize when the repo root was found
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.chdir(_ROOT)  # relative data literals ("data") resolve from the root,
    # and stay bare literals for the executes-live lint's surface.

# %%
# hpc-audit-section: data-selection
# [PINNED — copy verbatim; auto-clears. The pinned loader, path literal
# under input_roots, shape shown as evidence.]
from src.data.loading import load_raw_data

DATA_PATH = "data"
df = load_raw_data(DATA_PATH, allow_missing=True)
print(f"loaded: rows={len(df)} cols={len(df.columns)}")

# %%
# hpc-audit-section: target-construction
# [PINNED — copy verbatim; auto-clears. Mirrors the production call
# (src/backtest/executor.py prepare step): the invariant transform CALLED,
# never re-derived — diurnal -> sqrt -> winsorize, is_target=True; the
# multiplicative baseline kept for raw-space reconstruction.]
from src.features.transforms.target import robust_transform

adj_rv, rv_baseline = robust_transform(df, "RV", is_target=True)
df["adj_RV"] = adj_rv
df["baseline"] = rv_baseline
print(f"target adj_RV: non-null={int(adj_rv.notna().sum())} (diurnal->sqrt->winsorize, is_target=True)")

# %%
# hpc-audit-section: feature-construction
# Candidate through the SAME production path as the baseline (nudge-revised:
# no inline rebuild): src.models.ridge.run, config-varied ONLY. The candidate
# arm is the deployed config (configs/ridge_market_ew_prod.yaml) with the ONE
# knob under test flipped — exog_cols cleared to "" (HAR-only) — so the
# candidate-vs-baseline delta isolates the market_ew moment family (the six
# *_ewstock columns) with no path differences to poison the known-answer
# check. The exog transforms (robust_transform per column, diurnal_mode as
# configured), the horizon-1 shift (apply_horizon_shift), and the pinned
# walk-forward engine (MultiStageBacktest via fit_predict_ridge) are all the
# executor's own — invoked through the production entry point, never
# re-derived. Output redirected under results/run10_audit/ so the live run
# cannot clobber the deployed artifact.
import pandas as pd
import yaml

from src.models.ridge import run as ridge_run

CONFIG_PATH = "configs/ridge_market_ew_prod.yaml"
with open(CONFIG_PATH) as fh:
    cand_cfg = yaml.safe_load(fh)
cand_cfg.pop("model")
cand_cfg["exog_cols"] = ""  # the ONE varied knob: drop the market_ew family
cand_cfg["output_file"] = "results/run10_audit/candidate/run.json"
print({k: cand_cfg[k] for k in ("horizon", "train_window", "exog_cols", "alpha", "overnight_fill")})

cand_metrics = ridge_run(**cand_cfg)
print(f"candidate (HAR-only) LIVE qlike (raw-scale): {cand_metrics['qlike']:.6f}")

cand_df = pd.read_csv(Path(cand_cfg["output_file"]).with_name("results.csv"))
print(f"candidate walk-forward rows: {len(cand_df)}")

# %%
# hpc-audit-section: baseline
# The deployed baseline reproduced LIVE: configs/ridge_market_ew_prod.yaml
# (last touched at commit 9b8131b), executed UNMODIFIED through the SAME
# production path as the candidate (src.models.ridge.run — identical path,
# config-varied only) — never a number quoted from results/. The run's
# output_file is redirected under results/run10_audit/ so the live rerun
# cannot clobber the deployed artifact; its qlike (raw-scale, from
# src/evaluation/metrics.py::calculate_metrics) is the known-answer number.
# END STATE: pred_raw / true_raw / baseline_pred_raw — three aligned 1-D
# raw-variance arrays per the header contract, date-inner-joined between the
# candidate rows and the live baseline rows.
import numpy as np

with open(CONFIG_PATH) as fh:
    base_cfg = yaml.safe_load(fh)
base_cfg.pop("model")
base_cfg["output_file"] = "results/run10_audit/baseline/run.json"
print({k: base_cfg[k] for k in ("horizon", "train_window", "exog_cols", "alpha", "overnight_fill")})

deployed_metrics = ridge_run(**base_cfg)
print(f"deployed-baseline LIVE qlike (raw-scale): {deployed_metrics['qlike']:.6f}")

base_df = pd.read_csv(Path(base_cfg["output_file"]).with_name("results.csv"))
cand = cand_df[["date", "pred_raw", "true_raw"]].copy()
cand["date"] = pd.to_datetime(cand["date"])
base = base_df[["date", "pred_raw"]].rename(columns={"pred_raw": "baseline_pred_raw"})
base["date"] = pd.to_datetime(base["date"])
merged = cand.merge(base, on="date", how="inner")
pred_raw = merged["pred_raw"].to_numpy(dtype=np.float64)
true_raw = merged["true_raw"].to_numpy(dtype=np.float64)
baseline_pred_raw = merged["baseline_pred_raw"].to_numpy(dtype=np.float64)
print(
    f"aligned rows: candidate={len(cand)} baseline={len(base_df)} merged={len(merged)}"
)

# %%
# hpc-audit-section: metrics
# [PINNED — copy verbatim; auto-clears GIVEN the interface contract.
# Claimed units computed by src/evaluation/metrics.py, never inline:
# QLIKE raw-space (primary), hmse/oos_r2 (level-robust), MZ calibration.
# A NameError below means the variable sections above were not filled —
# DELIBERATE: an unfilled draft must die here, never emit a vacuous report.]
from src.evaluation.metrics import forecast_metrics

report = forecast_metrics(pred_raw, true_raw, benchmark=baseline_pred_raw)
print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in report.items()})
