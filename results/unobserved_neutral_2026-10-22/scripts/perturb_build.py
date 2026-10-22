"""(c) Future-perturbation test of UNOBSERVED_NEUTRAL (latestart/perturb_build.py with the unobs worktree's src/,
UNOBSERVED_NEUTRAL and LATE_START_WARMUP at their defaults (on) unless overridden; read-only on audit_B/data_small).

Builds the 13-bar (rth13, global lags, base-5 ladder, 1,000-day window) prescaled design on audit_B's reduced bucket
with the runner's run_executor arguments, once on the data as it is and once with raw data after T* changed.  A
causal pipeline leaves every design row dated <= T* (+30 min for X) bit for bit unchanged.

  python perturb_build.py <scenario> <out.npz> [FLAG=0|1 ...]
    scenario  base   | all    every perturbable raw value after T_ALL (2016-06-15 15:30) perturbed
              | avail  as 'all', plus 30% of the prints of every late series after T_ALL deleted
              | stop   the feeds of volatility demand (both open-only), VIX, StockTwits sentiment and sumvolume stop
                       at T_ALL (every print after it deleted): their averages after T* go unobserved
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path(r"C:\Users\james\AppData\Local\Temp\un_wt")
sys.path.insert(0, str(WT))
AB = HERE.parent / "audit_B"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import src.backtest.executor as EX  # noqa: E402
import src.data.loading as LD  # noqa: E402

assert Path(EX.__file__).resolve().is_relative_to(WT), EX.__file__
scen, out = sys.argv[1], Path(sys.argv[2])
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    assert isinstance(getattr(EX, k), bool), k
    setattr(EX, k, v == "1")
DATA = str(AB / "data_small")
T_ALL = pd.Timestamp("2016-06-15 15:30")
BUCKET = [
    "sumret",
    "sumret3",
    "numobs",
    "sumvolume",
    "vix",
    "vvix",
    "vix3m",
    "voldemand_spx_open_only",
    "voldemand_all_open_only",
    "stocktwits_sentiment",
    "stocktwits_sentcount",
    "stocktwits_attention",
    "fomc_release",
    "fomc_day",
    "fomc_until_inv",
    "fomc_since_inv",
]
PERTURB = [c for c in BUCKET if not c.startswith("fomc")] + [
    "RV"
]  # FOMC = the published calendar
LATE = [
    "vix",
    "vvix",
    "vix3m",
    "voldemand_spx_open_only",
    "voldemand_all_open_only",
    "stocktwits_sentiment",
    "stocktwits_sentcount",
    "stocktwits_attention",
]
STOP = [
    "voldemand_spx_open_only",
    "voldemand_all_open_only",
    "vix",
    "stocktwits_sentiment",
    "sumvolume",
]
for k in (
    "DROP_ACTIVITY_MA",
    "CALENDAR_THROUGH_OVERNIGHT",
    "DROP_OVERNIGHT_INDICATOR",
    "LEAN_GLOBAL_LAGS",
):
    setattr(EX, k, True)
print(
    "UNOBSERVED_NEUTRAL",
    EX.UNOBSERVED_NEUTRAL,
    "LATE_START_WARMUP",
    EX.LATE_START_WARMUP,
    flush=True,
)
orig_load = LD.load_raw_data


def load_perturbed(data_path, allow_missing=False, **kw):
    df = orig_load(data_path, allow_missing=allow_missing, **kw)
    if scen == "base":
        return df
    m = (df["t"] > T_ALL).to_numpy()
    if scen == "stop":
        for c in STOP:
            x = df[c].to_numpy(dtype=np.float64, copy=True)
            print(
                f"stop: {c} {int((m & np.isfinite(x)).sum())} prints deleted",
                flush=True,
            )
            x[m] = np.nan
            df[c] = x
        return df
    rng = np.random.default_rng(7)
    for c in PERTURB:
        x = df[c].to_numpy(dtype=np.float64, copy=True)
        k = m & np.isfinite(x)
        if c in (
            "sumret",
            "sumret3",
            "stocktwits_sentiment",
        ):  # signed: shift and rescale
            x[k] = x[k] * rng.uniform(0.2, 5.0, k.sum()) + rng.normal(
                0, np.nanstd(x), k.sum()
            )
        elif c == "numobs":
            x[k] = np.clip(np.round(x[k] * rng.uniform(0.3, 1.0, k.sum())), 1, None)
        else:  # positive series: multiplicative
            x[k] = x[k] * rng.uniform(0.2, 5.0, k.sum())
        if scen == "avail" and c in LATE:
            x[k & (rng.uniform(size=len(x)) < 0.3)] = np.nan
        df[c] = x
    return df


EX.load_raw_data = load_perturbed  # type: ignore[assignment]
cap: dict = {}
orig_bs, orig_smear = EX._backtest_and_save, EX.apply_duan_smearing


def bs_spy(df, feature_names, *a, **kw):
    burn = kw.get("burn_in")
    max_lag = 3125 if burn is None else int(burn)
    cap["dates"] = pd.to_datetime(df["t"].iloc[max_lag:]).to_numpy()
    cap["baseline_all"] = df["baseline"].to_numpy(np.float64)[max_lag:]
    return orig_bs(df, feature_names, *a, **kw)


def smear_spy(preds, y_oos, b_oos):
    cap["b_oos"] = np.array(b_oos, dtype=np.float64)
    return orig_smear(preds, y_oos, b_oos)


def fit_predict(X_chunk, y_chunk, train_win_periods, hp):
    cap["X"] = np.array(X_chunk, dtype=np.float64)
    cap["y"] = np.array(y_chunk, dtype=np.float64)
    cap["names"] = list(hp["_feature_names"])
    cap["W"] = int(train_win_periods)
    return np.zeros(len(X_chunk) - train_win_periods)


EX._backtest_and_save, EX.apply_duan_smearing = bs_spy, smear_spy
t = time.time()
tmp = HERE / "tmp_out" / out.stem / "results.csv"
EX.run_executor(
    method_name="audit",
    fit_predict=fit_predict,
    hyperparams={"_refit_frequency": 1},
    data_path=DATA,
    output_file=str(tmp),
    horizon=1,
    train_window=1000,
    start=0,
    end=-1,
    halo=0,
    exog_cols=BUCKET,
    segment="rth13",
    lag_scope="global",
    har_lags=[1, 5, 25, 125, 625, 3125],
    add_calendar=True,
    target_use_diurnal=True,
    target_winsor_window=240,
    dropna_with_exog=False,
    overnight_fill=True,
    impute_indicate=True,
    diurnal_mode="divide",
    prescale=True,
    seed=42,
)
assert len(cap["dates"]) == len(cap["X"]), (len(cap["dates"]), cap["X"].shape)
np.savez(
    out,
    X=cap["X"],
    y=cap["y"],
    names=np.array(cap["names"]),
    W=cap["W"],
    dates=cap["dates"],
    b_oos=cap["b_oos"],
    baseline_all=cap["baseline_all"],
)
print(
    f"{scen}: X {cap['X'].shape}, W {cap['W']}, {time.time() - t:.0f} s -> {out.name}",
    flush=True,
)
