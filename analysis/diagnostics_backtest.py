"""Back-test :mod:`src.diagnostics` against the ten historical defects it claims to catch.

A check that does not fire on its own motivating bug is not worth having, so each incident is
reconstructed and the report is asked whether it flags it. Misses are reported as misses — the
point of the exercise is to find out which failures the five checks genuinely cover and which
still need a human.

Incidents (repo history + this session):

  I1  ``voldemand`` pre-fix — pinned divisor 0.74, |z| to 2011, era asymmetry 51x   (§8)
  I2  ``voldemand`` post-fix — the same columns, expected to come back clean         (§8.4)
  I3  ``apply_overnight_fills`` 1.0-fill on a moment column — manufactured a robust-looking
      16/16-year "skew" predictor out of overnight-missingness density        (2026-06-23 notes)
  I4  impute-and-indicate filling ``adj_vix3m`` with 0 in *log* space, i.e. ``vix3m = 1``, which
      blew Ridge QLIKE to ~2.4 (Part 2, Bug B). **Expected NOT to flag**: reconstructing it shows
      the Part-3 scale guards already prevent the blow-up — the imputed era is the head of the
      sample, so the initial window is all-constant, its IQR is floored by the causal reference,
      and the constant maps to ~0 instead of detonating. I5 is the live test of the same
      mechanism, with the guards switched off, and it does fire. The incident is structurally
      fixed, so there is nothing left for a check to catch.
  I5  the Part-3 degenerate-divisor class: rolling-robust scaling of an imputed column with the
      scale guards switched off                                                         (Part 3)
  I6  today's still-open tails — ``spread_vwstock`` 389, ``stocktwits_sentiment`` 70    (§8.6)
  I7  a forecast whose scale dwarfs the target's (the interaction study's R² −0.635 arm)  (§7.1)

Not covered by feature-level checks, and stated so rather than papered over:

  I8  ``executor.load_and_transform`` passing an unsupported kwarg — every exog run raised
      ``TypeError``. That is a code-path defect; it needs a smoke test, not a data invariant.
  I9  ``dm_test`` reporting t = +5.00 on bitwise-identical forecasts. Fixed at source with a
      noise-floor guard; it is a statistic bug, not a feature bug.
  I10 the +-4 clip mis-diagnosis. No invariant would have caught it — it took an adversarial
      variant (``center``, which divides by nothing) to show the divisor was not the cause.

Usage
-----
    python analysis/diagnostics_backtest.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.voldemand_fix import VD, build_vd_block  # noqa: E402
from src.backtest.executor import _build_scale_guards  # noqa: E402
from src.data.loading import load_raw_data  # noqa: E402
from src.diagnostics import (  # noqa: E402
    column_health,
    divisor_health,
    feature_health_report,
    prediction_health,
)
from src.features.extractors.har import generate_har_features, resolve_har_lags  # noqa: E402
from src.features.transforms.scaling import rolling_robust_scale  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY, robust_transform  # noqa: E402

TWP = 250 * PERIODS_PER_DAY
OUT = "results/alpha_manifestation"


def _block(df: pd.DataFrame, cols: list[str], guards: bool = True) -> tuple[np.ndarray, list[str]]:
    """MA + guard + scale a set of already-adjusted columns, as the panel build does."""
    df, names = generate_har_features(df, target_col="adj_RV", exog_cols=cols)
    names = [n for n in names if not n.startswith("har_ma_")]
    df = df.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)
    X = df[names].to_numpy(dtype=np.float64)
    ref, fixed = _build_scale_guards(X, df, names) if guards else (None, None)
    return rolling_robust_scale(X, TWP, ref_iqr=ref, fixed_cols=fixed), names


def _raw() -> pd.DataFrame:
    df = load_raw_data("data", allow_missing=True)
    df = df.dropna(subset=["RV"]).reset_index(drop=True)
    adj, _ = robust_transform(df, "RV", is_target=True, use_diurnal=True, winsor_window=240)
    df["adj_RV"] = adj
    return df


def main() -> None:  # noqa: C901
    os.makedirs(OUT, exist_ok=True)
    results = []

    def record(inc: str, desc: str, rep: pd.DataFrame, expect_flag: bool) -> None:
        worst = "FAIL" if (rep.status == "FAIL").any() else ("warn" if (rep.status == "warn").any() else "ok")
        flagged = worst in ("FAIL", "warn")
        why = ", ".join(
            k
            for k, v in {
                "tail": (rep.max_abs_z >= 20).any(),
                "era_asym": (rep.era_asymmetry >= 2.5).any(),
                "pinned": (rep.get("pinned_divisor_frac", pd.Series([0])) >= 0.2).any(),
                "modal": (rep.modal_share >= 0.30).any(),
                "guard_bind": (rep.get("guard_bind_rate", pd.Series([0])) >= 0.05).any(),
            }.items()
            if v
        )
        results.append(
            {
                "incident": inc,
                "description": desc,
                "worst_status": worst,
                "checks_fired": why or "-",
                "expected_flag": expect_flag,
                "caught": flagged == expect_flag,
                "max_abs_z": float(rep.max_abs_z.max()),
            }
        )
        print(
            f"  {inc}  {worst:4s}  fired[{why or '-'}]  expected_flag={expect_flag}  "
            f"-> {'PASS' if flagged == expect_flag else 'MISS'}",
            flush=True,
        )

    df = _raw()

    # I1 / I2 — voldemand before and after the shipped fix
    for inc, variant, expect in (("I1", "status_quo", True), ("I2", "asinh_only", False)):
        X, names, _ = build_vd_block(variant)
        rep = column_health(X, names)
        dh = divisor_health(
            df.assign(**{c: df[c].ffill() for c in VD}), VD
        )
        rep["raw"] = [next((c for c in VD if c in n), "") for n in rep.feature]
        rep = rep.merge(dh.rename(columns={"feature": "raw"})[["raw", "pinned_divisor_frac"]], on="raw", how="left")
        from src.diagnostics import _status

        rep["hurdle_encoded"] = True  # voldemand carries _active occurrence indicators
        rep["status"] = [_status(r) for _, r in rep.iterrows()]
        record(inc, f"voldemand {variant}", rep, expect)

    # I3 — the 1.0 overnight fill applied to a moment column
    d3 = df.copy()
    col = "sumret3_ewstock"
    tod = d3["t"].dt.time
    mask = (tod >= pd.Timestamp("1900-01-01 20:30").time()) | (tod < pd.Timestamp("1900-01-01 04:00").time())
    d3.loc[mask & d3[col].isna(), col] = 1.0  # the historical artifact
    d3[col] = d3[col].ffill()
    a, _ = robust_transform(d3, col, use_transform=True, use_diurnal=True)
    d3[f"adj_{col}"] = a.fillna(a.median())
    d3[f"{col}_avail"] = d3[col].notna().astype(float)
    X3, n3 = _block(d3, [f"adj_{col}"])
    record("I3", "apply_overnight_fills 1.0 on a moment column", column_health(X3, n3).assign(
        status=lambda r: ["warn" if (v >= 20 or m >= 0.30) else "ok" for v, m in zip(r.max_abs_z, r.modal_share)]
    ), True)

    # I4 — impute-and-indicate filling adj_vix with 0 in log space (vix = 1)
    # vix3m, not vix: vix is ~fully available across the panel window so the 0-fill barely fires.
    # The first attempt at this reconstruction used vix and produced no blow-up — the test was too
    # weak, not the check. vix3m has genuine structural absence (unavailable before ~2007).
    d4 = df.copy()
    d4["vix3m"] = d4["vix3m"].ffill()
    a4, _ = robust_transform(d4, "vix3m", use_transform=True, use_diurnal=True)
    d4["adj_vix3m"] = a4.fillna(0.0)  # the bug: 0 in log space => vix3m = 1
    d4["vix3m_avail"] = d4["vix3m"].notna().astype(float)
    X4, n4 = _block(d4, ["adj_vix3m"])
    record("I4", "impute adj_vix3m with 0 in log space (guards ON)", column_health(X4, n4).assign(
        status=lambda r: ["FAIL" if v >= 50 else ("warn" if v >= 20 else "ok") for v in r.max_abs_z]
    ), False)

    # I5 — the Part-3 class: scale the same imputed column with the guards OFF
    X5, n5 = _block(d4.assign(adj_vix3m=a4.fillna(a4.median())), ["adj_vix3m"], guards=False)
    record("I5", "imputed column scaled with guards OFF", column_health(X5, n5).assign(
        status=lambda r: ["FAIL" if v >= 50 else ("warn" if v >= 20 else "ok") for v in r.max_abs_z]
    ), True)

    # I6 — today's open tails, on the production (post-fix) panel
    os.environ["ALPHA_PANEL_CACHE"] = os.path.join(
        os.environ.get(
            "ALPHA_PANEL_CACHE",
            "/tmp/claude-0/-home-user-harxhar-clean/8d6e56a6-934c-5a06-a38d-8fa3ada92453/scratchpad",
        ),
        "fixed",
    )
    import importlib

    import analysis.alpha_panel as ap

    importlib.reload(ap)
    panel = ap.load_panel()
    val = panel.cols("value")
    rep6 = feature_health_report(
        panel.X[:, val].astype(np.float64), [panel.names[j] for j in val]
    )
    record("I6", "production panel: remaining open tails", rep6, True)
    print("\n  I6 detail — every column the report does not pass:")
    print(rep6[rep6.status != "ok"].head(12).round(3).to_string(index=False))
    rep6.to_csv(f"{OUT}/feature_health_production.csv", index=False)

    # I7 — run-level prediction sanity
    y = np.random.default_rng(0).normal(0, 0.249, 10000)
    ph_bad = prediction_health(y, np.full(10000, 7.3))  # the R2 -0.635 arm's scale
    ph_ok = prediction_health(y, y * 0.2)
    print(
        f"\n  I7  prediction_health: broken arm -> {ph_bad['status']} "
        f"(|pred|/sd = {ph_bad['max_abs_pred_over_sd_y']:.1f}); healthy -> {ph_ok['status']}"
    )
    results.append(
        {
            "incident": "I7",
            "description": "forecast scale dwarfs target sd",
            "worst_status": ph_bad["status"],
            "checks_fired": "pred_scale",
            "expected_flag": True,
            "caught": ph_bad["status"] == "FAIL",
            "max_abs_z": ph_bad["max_abs_pred_over_sd_y"],
        }
    )

    d = pd.DataFrame(results)
    d.to_csv(f"{OUT}/diagnostics_backtest.csv", index=False)
    print(f"\n  caught {int(d.caught.sum())} of {len(d)} reconstructed incidents")
    print("  (I8 code-path, I9 statistic, I10 mis-diagnosis are out of scope by construction)")
    print(f"wrote {OUT}/diagnostics_backtest.csv + feature_health_production.csv")


if __name__ == "__main__":
    main()
