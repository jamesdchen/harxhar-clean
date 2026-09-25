"""Model diagnostics for the 15:30 sign(s) trade: the per-bar ridge's weights.

Two stages (``python experiments/model_diagnostics_1530.py capture|analyze``):

capture  Re-runs the deck's per-bar arm (bar ending 16:00, ridge, live_feasible
         bucket, 2000-session rolling window, HAR lags on the whole panel) through
         the SAME spec class and the SAME run_executor call the research campaign
         used (specs/causal_tune_linear.py, executed read-only from its source) on
         a scratch data dir holding only the four vendor files (the vendor_root
         rule of live/close_signal/forecast.py).  A subclass of the spec's
         RollingTunedLinear records, at every prediction, the coefficient vector,
         the feature row, and the trailing-window mean and sd of every feature.
         Nothing in src/ or specs/ is edited.

analyze  Gate (captured predictions == the stored research table at the 16:00
         rows; coef . x + intercept == prediction), then the join to the 866 deck
         days and the diagnostics: standardized weights over time, weights and
         contributions vs the day's P&L (mid and crossed), money by contribution
         quintile, and the split of s = rv_hat - iv_var into its two parts.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "model_diagnostics_1530"
SCRATCH = Path(
    os.environ.get(
        "MD_SCRATCH", str(REPO / "results" / "model_diagnostics_1530" / "_scratch")
    )
)
VENDOR_FILES = (
    "core_stats.parquet",
    "vix_and_voldemand.parquet",
    "releases.parquet",
    "time_categories.parquet",
)
SEGMENT = "bar1600"
BUCKET = "live_feasible"
ESTIMATOR = "ridge"
TRAIN_WIN = 2000
ET = "America/New_York"


# --------------------------------------------------------------------------- capture
def _spec_namespace(estimator: str = ESTIMATOR) -> dict:
    """Execute the spec's estimator section (imports, constants, the class) only.

    The section runs from the feature-construction cell's first import to just
    before the arm loop; the arm loop and the incumbent are not run.
    """
    env = {
        "HPC_KW_SEGMENT": SEGMENT,
        "HPC_KW_LAG_SCOPE": "global",
        "HPC_KW_ESTIMATOR": estimator,
        "HPC_KW_EXOG_BUCKET": BUCKET,
        "HPC_KW_TRAIN_WIN": str(TRAIN_WIN),
        "HPC_KW_START": "0",
        "HPC_KW_END": "-1",
        "HPC_KW_HALO": "0",
    }
    os.environ.update(env)
    src = (REPO / "specs" / "causal_tune_linear.py").read_text(encoding="utf-8")
    a = src.index(
        "import numpy as np\nimport pandas as pd\n\nfrom src.models.reclasso_har"
    )
    b = src.index("arm_results: dict = {}")
    ns: dict = {"__name__": "causal_tune_linear_section", "os": os}
    exec(compile(src[a:b], str(REPO / "specs" / "causal_tune_linear.py"), "exec"), ns)
    return ns


def capture_path(estimator: str = ESTIMATOR, alpha: float | None = None) -> Path:
    """capture_bar1600.npz is the arm of record (ridge, tuned penalty); other arms get a tag."""
    if estimator == ESTIMATOR and alpha is None:
        return OUT / "capture_bar1600.npz"
    tag = estimator if alpha is None else f"{estimator}_fixed{alpha:g}"
    return OUT / f"capture_bar1600_{tag}.npz"


def capture(estimator: str = ESTIMATOR, alpha: float | None = None) -> None:
    """Record every prediction's coefficients.  estimator: the spec's arm (ridge / reclasso);
    alpha: hold the penalty at this one grid value (the counterfactual that separates
    penalty switches from changes in the data) instead of re-choosing it every 250 refits."""
    OUT.mkdir(parents=True, exist_ok=True)
    data = SCRATCH / "data"
    data.mkdir(parents=True, exist_ok=True)
    for f in VENDOR_FILES:
        if not (data / f).exists():
            shutil.copy2(REPO / "data" / f, data / f)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    ns = _spec_namespace(estimator)
    Base = ns["RollingTunedLinear"]
    MultiStageBacktest = ns["MultiStageBacktest"]
    IdentityResidualizer = ns["IdentityResidualizer"]
    run_executor = ns["run_executor"]
    grid = ns["ESTIMATOR_GRIDS"][estimator]
    if alpha is not None:
        grid = [g for g in grid if g[1] == alpha]
        assert len(grid) == 1, (estimator, alpha, ns["ESTIMATOR_GRIDS"][estimator])
    Base.grid = grid
    assert ns["TRAIN_WIN"] == TRAIN_WIN and ns["SEGMENT"] == SEGMENT

    rec: dict[str, list] = {
        "th": [],
        "x": [],
        "mu": [],
        "sd": [],
        "pred": [],
        "alpha": [],
        "ybar": [],
        "n_solve": [],
        "n_reseed": [],
        "n_masked": [],
    }

    class Capture(Base):  # type: ignore[misc, valid-type]
        def predict_one(self, x):  # noqa: D401 - same contract as the base
            p = super().predict_one(x)
            Xw = self._X[:, :-1]  # the ring = the current training window, raw rows
            rec["th"].append(self._th.copy())
            rec["x"].append(np.asarray(x, dtype=np.float64).copy())
            rec["mu"].append(Xw.mean(axis=0))
            rec["sd"].append(Xw.std(axis=0, ddof=1))
            rec["ybar"].append(float(self._y.mean()))
            rec["pred"].append(p)
            rec["alpha"].append(float(self.alpha_))
            rec["n_solve"].append(
                int(self._n_solve)
            )  # tune (cold reseed) when n_solve % TUNE_PER == 1
            rec["n_reseed"].append(
                len(Base.reseed_trace)
            )  # between-tune mask additions so far
            rec["n_masked"].append(int(self._maskout.sum()))
            return p

    names_box: list = []

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        names_box.append(list(hyperparams["_feature_names"]))
        Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
        bt = MultiStageBacktest(
            residualizer=IdentityResidualizer(),
            regressor_factory=Capture,
            refit_frequency=int(hyperparams.get("_refit_frequency", 1)),
        )
        return bt.run(X_chunk, y_chunk, train_win_periods, desc="capture")

    arm_dir = (
        "arm"
        if (estimator == ESTIMATOR and alpha is None)
        else f"arm_{capture_path(estimator, alpha).stem}"
    )
    out_csv = SCRATCH / arm_dir / "results.csv"
    run_executor(
        method_name=f"lin_tuned_{estimator}",
        fit_predict=fit_predict,
        hyperparams={"_refit_frequency": ns["REFIT_FREQUENCY"]},
        data_path=str(data),
        output_file=str(out_csv),
        horizon=ns["HORIZON"],
        train_window=TRAIN_WIN,
        start=0,
        end=-1,
        halo=0,
        exog_cols=ns["get_bucket"](BUCKET),
        segment=SEGMENT,
        lag_scope="global",
        har_lags=ns["HAR_LAGS"],
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=ns["SEED"],
    )
    stem = out_csv.with_name(f"results_{SEGMENT}.csv")
    res = pd.read_csv(stem if stem.exists() else out_csv)
    names = names_box[-1]
    n = len(rec["pred"])
    assert n == len(res), (n, len(res))
    np.savez_compressed(
        capture_path(estimator, alpha),
        th=np.array(rec["th"]),
        x=np.array(rec["x"]),
        mu=np.array(rec["mu"]),
        sd=np.array(rec["sd"]),
        ybar=np.array(rec["ybar"]),
        pred=np.array(rec["pred"]),
        alpha=np.array(rec["alpha"]),
        n_solve=np.array(rec["n_solve"]),
        n_reseed=np.array(rec["n_reseed"]),
        n_masked=np.array(rec["n_masked"]),
        names=np.array(names),
        date=res["date"].astype(str).to_numpy().astype("U32"),
        pred_adj=res["pred_adj"].to_numpy(float),
        true_adj=res["true_adj"].to_numpy(float),
    )
    print(f"captured {n} predictions, {len(names)} features", flush=True)
    print("features:", names, flush=True)


# --------------------------------------------------------------------------- analyze
BASE_WORDS = {
    "har_ma": "realized variance (target)",
    "sumret": "signed return",
    "sumabsret": "absolute return",
    "sumret3": "cubed returns (skew)",
    "sumret4": "4th-power returns (tails)",
    "sumpret2": "upside squared returns",
    "sumbipow": "bipower variation",
    "sumautocov": "return autocovariance",
    "sumvolume": "ES volume",
    "numobs": "ES prints per bar",
    "vix": "VIX",
    "vvix": "VVIX",
    "vix3m": "VIX3M",
    "fomc_release": "FOMC statement flag",
    "fomc_day": "FOMC day flag",
    "fomc_until_inv": "1 / days to next FOMC",
    "fomc_since_inv": "1 / days since last FOMC",
    "calendar": "calendar flags",
}
LAGS = (1, 5, 25, 125, 625, 3125)


def parse_feature(name: str) -> tuple[str, str, int | None]:
    """(group, kind, lag): group = the source series; kind = level/avail/active/x_gate/calendar."""
    import re

    m = re.fullmatch(r"har_ma_(\d+)", name)
    if m:
        return "har_ma", "level", int(m.group(1))
    m = re.fullmatch(r"har_ma_(\d+)_x_(open|close)", name)
    if m:
        return "har_ma", f"x_{m.group(2)}", int(m.group(1))
    m = re.fullmatch(r"adj_(.+)_ma_(\d+)", name)
    if m:
        return m.group(1), "level", int(m.group(2))
    m = re.fullmatch(r"(.+)_(avail|active)_ma_(\d+)", name)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return "calendar", "calendar", None


def plain(name: str) -> str:
    g, k, lag = parse_feature(name)
    if g == "calendar":
        return f"calendar: {name}"
    base = BASE_WORDS.get(g, g)
    lw = "last bar" if lag == 1 else f"mean of last {lag} bars"
    if k == "level":
        return f"{base}, {lw}"
    if k == "avail":
        return f"{base} is-present flag, {lw}"
    if k == "active":
        return f"{base} is-nonzero flag, {lw}"
    return f"{base}, {lw} x {k[2:]} gate"


def var_stem(g: str, names: list[str]) -> str:
    """The design's own column name for source series g, lag suffix as * (e.g. adj_sumabsret_ma_*)."""
    import re

    if g == "calendar":
        return "calendar dummies (DOW_*, is_*, hour, days_to_opex)"
    stems = {
        re.sub(r"_ma_\d+$", "_ma_*", x)
        for x in names
        if parse_feature(x)[:2] == (g, "level")
    }
    assert len(stems) == 1, (g, stems)
    return stems.pop()


def _corr_cols(F: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of F with every column of T; (..., k, m)."""
    Fc = F - F.mean(axis=-2, keepdims=True)
    Tc = T - T.mean(axis=-2, keepdims=True)
    num = np.einsum("...nk,...nm->...km", Fc, Tc)
    den = (
        np.sqrt((Fc**2).sum(axis=-2))[..., :, None]
        * np.sqrt((Tc**2).sum(axis=-2))[..., None, :]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def boot_corr(F, T, rng, B, asl):
    """Point correlation and circular-block-bootstrap 95% interval; block = ceil(n^(1/3))."""
    n = F.shape[0]
    blen = int(np.ceil(n ** (1.0 / 3.0)))
    idx = asl.circular_block_bootstrap_idx(rng, n, blen, B)
    pt = _corr_cols(F, T)
    draws = np.empty((B,) + pt.shape)
    step = max(1, int(2e7 // max(1, n * (F.shape[1] + T.shape[1]))))
    for a in range(0, B, step):
        ii = idx[a : a + step]
        draws[a : a + step] = _corr_cols(F[ii], T[ii])
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return pt, lo, hi, blen


def _diff_boot(c, win, idxs, rng, B, asl):
    """Mean contribution on winning minus losing days in the subset idxs, block-bootstrap 95% CI."""
    cw_, ww_ = c[idxs], win[idxs]
    diff = cw_[ww_].mean() - cw_[~ww_].mean()
    bl = int(np.ceil(len(idxs) ** (1 / 3)))
    bi = asl.circular_block_bootstrap_idx(rng, len(idxs), bl, B)
    cw, ww = cw_[bi], ww_[bi]
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(ww, cw, 0).sum(1) / ww.sum(1) - np.where(~ww, cw, 0).sum(1) / (
            ~ww
        ).sum(1)
    lo, hi = np.nanpercentile(dd, [2.5, 97.5])
    return diff, lo, hi


def analyze() -> None:  # noqa: C901 - one linear report
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "notebooks"))
    import atm_straddle_lib as asl  # type: ignore

    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    z = np.load(OUT / "capture_bar1600.npz", allow_pickle=True)
    names = [str(v) for v in z["names"]]
    th, X, mu, sd = z["th"], z["x"], z["mu"], z["sd"]
    pred = z["pred"]
    stamp_et = pd.DatetimeIndex(pd.to_datetime(np.asarray(z["date"]).astype(str)))
    assert stamp_et.tz is None, "panel stamps are naive ET"
    assert ((stamp_et.hour == 16) & (stamp_et.minute == 0)).all()

    # ---- GATE 1: captured predictions vs the stored research table (16:00 rows)
    tab = pd.read_parquet(
        REPO / "results" / "spxw_pnl" / "yhat_sub_ridge_live_feasible.parquet"
    )
    t_et = pd.DatetimeIndex(tab["t"]).tz_convert(ET).tz_localize(None).as_unit("ns")
    is16 = (t_et.hour == 16) & (t_et.minute == 0)
    stored = pd.Series(tab["yhat"].to_numpy(float)[is16], index=t_et[is16])
    cap = pd.Series(pred, index=stamp_et.as_unit("ns"))
    common = stored.index.intersection(cap.index)
    rel = np.abs(cap.loc[common].to_numpy() - stored.loc[common].to_numpy()) / np.abs(
        stored.loc[common].to_numpy()
    )
    gate1 = float(rel.max())
    say(
        f"GATE 1: stored 16:00 rows {len(stored)}, captured {len(cap)}, common {len(common)}; "
        f"max rel diff {gate1:.1e}; vs the arm's own pred_adj {np.abs(pred - z['pred_adj']).max():.1e}"
    )
    if len(common) != len(stored) or gate1 > 1e-6:
        raise SystemExit("GATE 1 FAILED -- weights not interpreted")

    # ---- GATE 2: coef . x + intercept reproduces the prediction
    recon = np.einsum("nk,nk->n", th[:, :-1], X) + th[:, -1]
    gate2 = float(np.max(np.abs(recon - pred) / np.abs(pred)))
    level = th[:, -1] + np.einsum(
        "nk,nk->n", th[:, :-1], mu
    )  # forecast at the window-mean row
    C = th[:, :-1] * (X - mu)  # contributions: pred = level + C.sum(1)
    gate3 = float(np.max(np.abs(level + C.sum(1) - pred) / np.abs(pred)))
    lvl_vs_ybar = float(np.max(np.abs(level - z["ybar"]) / np.abs(z["ybar"])))
    say(
        f"GATE 2: max rel |coef.x + intercept - prediction| {gate2:.1e}; "
        f"level + sum of contributions {gate3:.1e}; level vs window-mean target {lvl_vs_ybar:.1e}"
    )
    if gate2 > 1e-9:
        raise SystemExit("GATE 2 FAILED")
    Zs = th[:, :-1] * sd  # standardized weights (coef x trailing-window sd)

    # ---- the deck days
    d = pd.read_parquet(
        REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
    )
    d.index = pd.DatetimeIndex(d.index).as_unit("ns")
    me = pd.read_csv(
        REPO / "results" / "atm_straddle_0dte_1530" / "monthend_1530_longstraddle.csv",
        index_col=0,
        parse_dates=True,
    )["month_end"]
    me.index = pd.DatetimeIndex(me.index).as_unit("ns")
    d["month_end"] = me.reindex(d.index)
    assert d["month_end"].notna().all()
    d["month_end"] = d["month_end"].astype(bool)
    d["pnl_mid"] = d["pos"] * d["R"]
    bid = d["bid_c"].astype(float) + d["bid_p"].astype(float)
    ask = d["ask_c"].astype(float) + d["ask_p"].astype(float)
    d["pnl_x"] = asl.crossed_premium_return(d["pos"], d["exit"], bid, ask)
    n_untr = asl.crossed_untradeable_count(d["pos"], bid, ask)
    day = pd.Series(np.arange(len(stamp_et)), index=stamp_et.normalize().as_unit("ns"))
    assert not day.index.duplicated().any()
    miss = d.index.difference(day.index)
    say(
        f"JOIN: deck days {len(d)}, found in the capture {len(d) - len(miss)}; "
        f"month-ends {int(d.month_end.sum())}; crossed untradeable {n_untr}"
    )
    assert len(miss) == 0
    ii = day.reindex(d.index).to_numpy(int)
    rho = pd.Series(pred[ii]).rank().corr(pd.Series(d["rv_hat"].to_numpy()).rank())
    say(f"JOIN check: rank corr(arm yhat, deck rv_hat) on deck days {rho:.3f}")

    me_m = d["month_end"].to_numpy()
    say(
        "P&L per premium, sum over deck days: mid "
        f"{d['pnl_mid'].sum():+.2f} (month-ends {d.loc[me_m, 'pnl_mid'].sum():+.2f}, "
        f"other {d.loc[~me_m, 'pnl_mid'].sum():+.2f}); crossed {d['pnl_x'].sum():+.2f} "
        f"(month-ends {d.loc[me_m, 'pnl_x'].sum():+.2f}, other {d.loc[~me_m, 'pnl_x'].sum():+.2f}); "
        f"buy days {int((d.pos > 0).sum())} (month-ends {int((d.pos[me_m] > 0).sum())} of {int(me_m.sum())})"
    )

    Zd, Cd = Zs[ii], C[ii]
    active = np.flatnonzero(
        (sd[ii] > 0).any(axis=0) & (np.abs(th[ii, :-1]) > 0).any(axis=0)
    )
    say(
        f"features: {len(names)} in the design, {len(active)} carry a nonzero weight on the deck days "
        "(the rest are constant or duplicated inside the window and are masked to weight 0)"
    )
    alphas = pd.Series(z["alpha"][ii]).value_counts().sort_index()
    say(
        "ridge penalty in force on deck days: "
        + ", ".join(f"{a:g} ({c} days)" for a, c in alphas.items())
    )

    # ---- (a) weights table
    rows = []
    for j in active:
        g, kind, lag = parse_feature(names[j])
        zz = Zd[:, j]
        rows.append(
            dict(
                feature=names[j],
                plain=plain(names[j]),
                group=g,
                kind=kind,
                lag=lag,
                std_weight_mean=zz.mean(),
                std_weight_sd=zz.std(ddof=1),
                frac_positive=(zz > 0).mean(),
                frac_zero=(zz == 0).mean(),
                contrib_sd=Cd[:, j].std(ddof=1),
                contrib_mean_abs=np.abs(Cd[:, j]).mean(),
            )
        )
    W = pd.DataFrame(rows).sort_values("contrib_sd", ascending=False)
    W.to_csv(OUT / "weights_table.csv", index=False)

    # group contributions: the sum over lags and flags of one source series
    gset = sorted({parse_feature(names[j])[0] for j in active})
    gcols = {g: [j for j in active if parse_feature(names[j])[0] == g] for g in gset}
    groups = sorted(gset, key=lambda g: -Cd[:, gcols[g]].sum(1).std(ddof=1))
    G = np.column_stack([Cd[:, gcols[g]].sum(1) for g in groups])
    gname = [BASE_WORDS.get(g, g) for g in groups]
    vname = [
        var_stem(g, list(names)) for g in groups
    ]  # figure labels: the design's column names
    lvl_d = level[ii]
    say(
        f"sd on deck days (forecast units): forecast {pred[ii].std(ddof=1):.3f}, level {lvl_d.std(ddof=1):.3f}; "
        + "; ".join(f"{gname[k]} {G[:, k].std(ddof=1):.3f}" for k in range(len(groups)))
    )
    Gdf = pd.DataFrame(G, columns=gname, index=d.index)
    Gdf["level"] = lvl_d
    Gdf["forecast"] = pred[ii]
    Gdf.to_csv(OUT / "group_contributions_deck_days.csv")

    # ---- small multiples: one panel per source series, one line per lag
    panel_groups = [g for g in groups]
    ncol = 4
    nrow = int(np.ceil(len(panel_groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 1.9 * nrow + 0.6), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    lag_cols: dict[int | None, str] = dict(
        zip(LAGS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"])
    )
    dates_all = stamp_et.normalize()
    deck_lo, deck_hi = d.index.min(), d.index.max()
    for a_i, g in enumerate(panel_groups):
        ax = axes[a_i]
        for j in gcols[g]:
            _, kind, lag = parse_feature(names[j])
            if g == "calendar":
                ax.plot(dates_all, Zs[:, j], lw=0.7)
            elif kind == "level":
                ax.plot(dates_all, Zs[:, j], lw=0.9, color=lag_cols.get(lag, "k"))
            else:
                ax.plot(
                    dates_all, Zs[:, j], lw=0.6, color=lag_cols.get(lag, "k"), ls=":"
                )
        ax.axvspan(deck_lo, deck_hi, color="0.9", zorder=0)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(var_stem(g, list(names)), fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[len(panel_groups) :]:
        ax.axis("off")
    h = [plt.Line2D([], [], color=c, lw=1.2) for c in lag_cols.values()]
    fig.legend(
        h,
        [f"*_ma_{k}" for k in lag_cols],
        loc="lower center",
        ncol=6,
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "weights_small_multiples.png", dpi=150)
    plt.close(fig)

    # ---- (b) weights and contributions vs money, three splits, block bootstrap
    rng = np.random.default_rng(20260924)
    B = 2000
    T_all = np.column_stack(
        [
            d["pnl_mid"].to_numpy(float),
            d["pnl_x"].to_numpy(float),
            d["pos"].to_numpy(float),
        ]
    )
    tnames = ["pnl_mid", "pnl_crossed", "buy_decision"]
    splits = {"all": np.ones(len(d), bool), "month_end": me_m, "other": ~me_m}
    okx = np.isfinite(T_all).all(1)
    Fg = np.column_stack([G, lvl_d, pred[ii]])
    fg_names = gname + ["level (window-mean target)", "forecast (sum)"]
    grows, frows = [], []
    for sname, m in splits.items():
        mm = m & okx
        pt, lo, hi, blen = boot_corr(Fg[mm], T_all[mm], rng, B, asl)
        for k, gn in enumerate(fg_names):
            for t, tn in enumerate(tnames):
                grows.append(
                    dict(
                        split=sname,
                        n=int(mm.sum()),
                        block=blen,
                        series=gn,
                        target=tn,
                        corr=pt[k, t],
                        lo=lo[k, t],
                        hi=hi[k, t],
                    )
                )
        F2 = np.column_stack([Zd[mm][:, active], Cd[mm][:, active]])
        pt, lo, hi, blen = boot_corr(F2, T_all[mm], rng, B, asl)
        na = len(active)
        for q, j in enumerate(active):
            for t, tn in enumerate(tnames):
                frows.append(
                    dict(
                        split=sname,
                        n=int(mm.sum()),
                        feature=names[j],
                        plain=plain(names[j]),
                        target=tn,
                        corr_std_weight=pt[q, t],
                        lo_w=lo[q, t],
                        hi_w=hi[q, t],
                        corr_contribution=pt[na + q, t],
                        lo_c=lo[na + q, t],
                        hi_c=hi[na + q, t],
                    )
                )
    GC = pd.DataFrame(grows)
    GC.to_csv(OUT / "corr_group_contribution_vs_pnl.csv", index=False)
    FC = pd.DataFrame(frows)
    FC.to_csv(OUT / "corr_feature_vs_pnl.csv", index=False)
    say(
        f"bootstrap: circular moving blocks, block length ceil(n^(1/3)), B={B}, seed 20260924"
    )

    # feature-level: how many weight/contribution correlations exclude zero (vs 5% by chance)
    for sname in splits:
        for tn in ("pnl_mid", "pnl_crossed"):
            s_ = FC[(FC.split == sname) & (FC.target == tn)]
            nw = int(((s_.lo_w > 0) | (s_.hi_w < 0)).sum())
            nc = int(((s_.lo_c > 0) | (s_.hi_c < 0)).sum())
            say(
                f"(b) {sname:9s} {tn:11s}: intervals excluding 0 -- std weight {nw}/{len(s_)}, "
                f"contribution {nc}/{len(s_)} (chance ~{0.05 * len(s_):.0f})"
            )

    # ---- (c) money by contribution quintile, top 5 series
    top = list(range(min(5, len(groups))))
    qrows = []
    pos_ = d["pos"].to_numpy()
    pm, px = d["pnl_mid"].to_numpy(), d["pnl_x"].to_numpy()
    for k in top:
        c = G[:, k]
        qs = pd.qcut(c, 5, labels=False)
        for q in range(5):
            sel = qs == q
            qrows.append(
                dict(
                    series=gname[k],
                    quintile=q + 1,
                    n=int(sel.sum()),
                    n_month_end=int((sel & me_m).sum()),
                    contrib_mean=c[sel].mean(),
                    share_buy=(pos_[sel] > 0).mean(),
                    pnl_mid=pm[sel].mean(),
                    pnl_crossed=np.nanmean(px[sel]),
                    pnl_mid_other=pm[sel & ~me_m].mean(),
                    pnl_crossed_other=np.nanmean(px[sel & ~me_m]),
                )
            )
    Q = pd.DataFrame(qrows)
    Q.to_csv(OUT / "pnl_by_contribution_quintile.csv", index=False)

    # winners vs losers among buy days and among sell days
    win = pm > 0
    wrows = []
    for sname, m in splits.items():
        for side, sm in (("buy", pos_ > 0), ("sell", pos_ < 0)):
            sel = m & sm
            idxs = np.flatnonzero(sel)
            nw, nl = int((sel & win).sum()), int((sel & ~win).sum())
            for k in range(len(fg_names)):
                c = Fg[:, k]
                if nw >= 2 and nl >= 2:
                    diff, lo, hi = _diff_boot(c, win, idxs, rng, B, asl)
                else:
                    diff = lo = hi = np.nan
                wrows.append(
                    dict(
                        split=sname,
                        side=side,
                        series=fg_names[k],
                        n_win=nw,
                        n_lose=nl,
                        mean_win=c[sel & win].mean() if nw else np.nan,
                        mean_lose=c[sel & ~win].mean() if nl else np.nan,
                        diff=diff,
                        lo=lo,
                        hi=hi,
                    )
                )
    WL = pd.DataFrame(wrows)
    WL.to_csv(OUT / "winners_vs_losers_contribution.csv", index=False)

    # ---- (d) s split into the forecast part and the implied part
    lr = np.log(d["rv_hat"].to_numpy(float))
    li = np.log(d["iv_var"].to_numpy(float))
    L = lr - li
    assert (np.sign(L) == pos_).all()
    vr, vi, cv = np.var(lr, ddof=1), np.var(li, ddof=1), np.cov(lr, li)[0, 1]
    say(
        f"(d) var log(rv_hat/iv_var) {np.var(L, ddof=1):.3f} = var log rv_hat {vr:.3f} + var log iv_var {vi:.3f} "
        f"- 2 cov {2 * cv:.3f}; corr(log rv_hat, log iv_var) {cv / np.sqrt(vr * vi):.3f}"
    )
    Dd = np.column_stack([lr, li, L])
    dnames = ["log forecast rv_hat", "log implied iv_var", "log(rv_hat / iv_var)"]
    drows = []
    for sname, m in splits.items():
        mm = m & okx
        pt, lo, hi, blen = boot_corr(Dd[mm], T_all[mm], rng, B, asl)
        for k, nm in enumerate(dnames):
            for t, tn in enumerate(tnames):
                drows.append(
                    dict(
                        split=sname,
                        n=int(mm.sum()),
                        series=nm,
                        target=tn,
                        corr=pt[k, t],
                        lo=lo[k, t],
                        hi=hi[k, t],
                    )
                )
    # the forecast in the weights' units (yhat, a diurnally-normalized root variance) vs its
    # raw-variance scale: log rv_hat = log baseline + the recalibrated yhat part
    bl = pd.Series(tab["baseline"].to_numpy(float)[is16], index=t_et[is16]).reindex(
        stamp_et.as_unit("ns")
    )
    lb = np.log(bl.to_numpy()[ii])
    ly = (
        lr - lb
    )  # everything in log rv_hat that is not the baseline (the yhat part after the map)
    say(
        f"(d) log rv_hat = log baseline + rest: var {vr:.3f} = {np.var(lb, ddof=1):.3f} + {np.var(ly, ddof=1):.3f} "
        f"+ 2 cov {2 * np.cov(lb, ly)[0, 1]:.3f}; rank corr(rest, arm yhat) "
        f"{pd.Series(ly).rank().corr(pd.Series(pred[ii]).rank()):.3f}"
    )
    for sname, m in splits.items():
        mm = m & okx
        pt, lo, hi, blen = boot_corr(
            np.column_stack([lb, ly, lb - li, ly])[mm], T_all[mm], rng, B, asl
        )
        for k, nm in enumerate(
            [
                "log baseline (diurnal scale)",
                "log rv_hat - log baseline (the yhat part)",
                "log baseline - log iv_var",
            ]
        ):
            for t, tn in enumerate(tnames):
                drows.append(
                    dict(
                        split=sname,
                        n=int(mm.sum()),
                        series=nm,
                        target=tn,
                        corr=pt[k, t],
                        lo=lo[k, t],
                        hi=hi[k, t],
                    )
                )
    DD = pd.DataFrame(drows)
    DD.to_csv(OUT / "s_decomposition_corr.csv", index=False)
    dr, di = np.diff(lr), -np.diff(li)
    flip = np.sign(L[1:]) != np.sign(L[:-1])
    say(
        f"(d) the decision changes on {int(flip.sum())} of {len(flip)} consecutive deck-day pairs; on those, the forecast "
        f"moved more than the implied on {int((np.abs(dr[flip]) > np.abs(di[flip])).sum())}; var of the daily change: "
        f"forecast {np.var(dr, ddof=1):.3f}, implied {np.var(di, ddof=1):.3f}"
    )

    # ---- figures for (b) and (c)
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.3 * len(fg_names) + 1.3), sharey=True)
    y = np.arange(len(fg_names))
    for a_i, sname in enumerate(splits):
        ax = axes[a_i]
        for tn, off, col, mk in (
            ("pnl_mid", -0.15, "#1f77b4", "o"),
            ("pnl_crossed", 0.15, "#d62728", "s"),
        ):
            sub = (
                GC[(GC.split == sname) & (GC.target == tn)]
                .set_index("series")
                .loc[fg_names]
            )
            ax.errorbar(
                sub["corr"],
                y + off,
                xerr=[sub["corr"] - sub["lo"], sub["hi"] - sub["corr"]],
                fmt=mk,
                ms=3,
                lw=0.8,
                color=col,
                label="mid" if tn == "pnl_mid" else "crossed",
            )
        ax.axvline(0, color="0.5", lw=0.6)
        nn = int(GC[(GC.split == sname)]["n"].iloc[0])
        ax.set_title(f"{sname.replace('_', '-')} days (n={nn})", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(
            vname + ["ybar (window-mean target)", "yhat (forecast)"], fontsize=7
        )
        ax.tick_params(axis="x", labelsize=7)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "corr_contribution_vs_pnl.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(top), figsize=(11, 2.5), sharey=True)
    for a_i, k in enumerate(top):
        ax = axes[a_i]
        sub = Q[Q.series == gname[k]]
        ax.bar(
            sub["quintile"] - 0.2,
            sub["pnl_mid"],
            width=0.4,
            color="#1f77b4",
            label="all days",
        )
        ax.bar(
            sub["quintile"] + 0.2,
            sub["pnl_mid_other"],
            width=0.4,
            color="#9ecae1",
            label="excl. month-ends",
        )
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_title(vname[k], fontsize=8)
        ax.set_xlabel("contribution quintile (1 = pushes s down most)", fontsize=6)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("mean P&L per premium (mid)", fontsize=7)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "pnl_by_contribution_quintile.png", dpi=150)
    plt.close(fig)

    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(
        W.head(20)[
            ["plain", "std_weight_mean", "std_weight_sd", "frac_positive", "contrib_sd"]
        ].to_string()
    )
    print(GC[GC.target != "buy_decision"].round(3).to_string())
    print(GC[GC.target == "buy_decision"].round(3).to_string())
    print(Q.round(3).to_string())
    print(WL[WL.split != "month_end"].round(4).to_string())
    print(WL[WL.split == "month_end"].round(4).to_string())
    print(DD.round(3).to_string())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if stage == "capture":
        # capture [ridge|reclasso] [fixed penalty]; no arguments = the arm of record
        est = sys.argv[2] if len(sys.argv) > 2 else ESTIMATOR
        capture(est, float(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        analyze()
