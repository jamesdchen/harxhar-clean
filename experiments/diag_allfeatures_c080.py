"""Scratch diagnostic: a_bucket_all_features chunk 080 forecast incident.

Read-only over results/unification_carc + data/*.parquet. Phases via argv:
    facts    (default) chunk-level bar facts + meta disclosures for c080/c063
    scan     max|yhat| across all chunks of the arm (locate every elevated bar)
    chain    full causal-chain rescore of arm + a0, per-chunk deltas (3 smears)
    amp      amplification decomposition for c080 (+ c081 spillover), c063
    parquet  raw-parquet extremeness check at the bad timestamps
Run: C:/Users/james/miniconda3/envs/285J/python.exe experiments/diag_allfeatures_c080.py [phases...]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evaluation.diebold_mariano import qlike_per_bar  # noqa: E402

ROOT = os.path.join(_ROOT, "results", "unification_carc")
ARM = "a_bucket_all_features"
A0 = "a0_ols_har"


def load(arm: str, idx: int) -> dict:
    path = os.path.join(ROOT, arm, f"chunk_{idx:03d}.npz")
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.asarray(z[k]) for k in ("row_id", "t", "y_fit", "yhat", "rv_raw", "baseline", "valid_mask")}
        d["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    d["valid"] = d.pop("valid_mask").astype(bool)
    return d


def ts(t) -> str:
    return str(np.datetime64(t, "s"))


# ── facts ─────────────────────────────────────────────────────────────────────

def phase_facts() -> None:
    for idx in (80, 63):
        c = load(ARM, idx)
        a = load(A0, idx)
        assert np.array_equal(c["row_id"], a["row_id"])
        ay = np.abs(c["yhat"])
        print(f"\n=== {ARM} chunk_{idx:03d}: rows [{c['row_id'][0]}, {c['row_id'][-1]+1}) "
              f"n={len(ay)}  t=[{ts(c['t'][0])} .. {ts(c['t'][-1])}]")
        print(f"  |yhat|: max={ay.max():.3f} p99.9={np.percentile(ay,99.9):.3f} "
              f"p99={np.percentile(ay,99):.3f} median={np.median(ay):.3f}; "
              f"max|y_fit|={np.abs(c['y_fit']).max():.3f}")
        top = np.argsort(-ay)[:12]
        top = top[np.argsort(c["row_id"][top])]
        print("  top-|yhat| bars (row, t, yhat, a0_yhat, y_fit, rv_raw, baseline):")
        for i in top:
            print(f"    row={c['row_id'][i]}  {ts(c['t'][i])}  yhat={c['yhat'][i]:+10.3f}  "
                  f"a0={a['yhat'][i]:+7.3f}  y_fit={c['y_fit'][i]:+7.3f}  "
                  f"rv={c['rv_raw'][i]:.3e}  B={c['baseline'][i]:.3e}")
        # structure: where are the elevated bars? consecutive runs?
        thr = float(np.abs(c["y_fit"]).max())  # forecast beyond ANY in-chunk target
        bad = np.flatnonzero(ay > thr)
        print(f"  bars with |yhat| > max|y_fit| ({thr:.3f}): {len(bad)} at chunk-pos {bad.tolist()}")
        if len(bad):
            runs = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1)
            print(f"  run structure: {[(int(r[0]), int(r[-1]), len(r)) for r in runs]} (pos first,last,len)")
            # local context around the worst bar
            w = int(np.argmax(ay))
            sl = slice(max(0, w - 6), min(len(ay), w + 7))
            print(f"  context around worst bar (pos {w}):")
            for i in range(sl.start, sl.stop):
                print(f"    pos={i} row={c['row_id'][i]} {ts(c['t'][i])} yhat={c['yhat'][i]:+9.3f} "
                      f"y_fit={c['y_fit'][i]:+6.3f} a0={a['yhat'][i]:+6.3f}")
        # meta disclosures
        m = c["meta"]
        print(f"  meta: n_design_cols={m.get('n_design_cols')} dropped={len(m.get('ols_dropped_cols', []))} "
              f"collinear={len(m.get('ols_collinear_cols', []))} masked={len(m.get('ols_masked_cols', {}))}")
        mc = m.get("ols_masked_cols", {})
        sticky = {k: v for k, v in mc.items() if "sticky_collinear" in v.get("reasons", [])}
        print(f"  sticky_collinear entries: {len(sticky)}")
        for k, v in sticky.items():
            print(f"    {k}: {v}")
        # non-sticky mask entries whose first/last falls near the bad rows
        if len(bad):
            lo_r, hi_r = int(c["row_id"][bad[0]]) - 60, int(c["row_id"][bad[-1]]) + 60
            near = {k: v for k, v in mc.items()
                    if (lo_r <= v["first"] <= hi_r) or (lo_r <= v["last"] <= hi_r)}
            print(f"  mask entries with first/last within ±60 rows of bad bars: {len(near)}")
            for k, v in sorted(near.items(), key=lambda kv: kv[1]["last"]):
                print(f"    {k}: {v}")
        reasons: dict[str, int] = {}
        for v in mc.values():
            for r in v.get("reasons", []):
                reasons[r] = reasons.get(r, 0) + 1
        print(f"  mask reason counts: {reasons}")


# ── scan ──────────────────────────────────────────────────────────────────────

def phase_scan() -> None:
    print("\n=== per-chunk max|yhat| across the arm ===")
    rows = []
    for idx in range(100):
        p = os.path.join(ROOT, ARM, f"chunk_{idx:03d}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=False) as z:
            yh = np.asarray(z["yhat"])
            yf = np.asarray(z["y_fit"])
        rows.append((idx, float(np.abs(yh).max()), int((np.abs(yh) > 5).sum()),
                     int((np.abs(yh) > 2).sum()), float(np.abs(yf).max())))
    rows.sort(key=lambda r: -r[1])
    print("  idx  max|yhat|  n>|5|  n>|2|  max|y_fit|")
    for r in rows[:12]:
        print(f"  {r[0]:3d}  {r[1]:9.3f}  {r[2]:5d}  {r[3]:5d}  {r[4]:9.3f}")


# ── chain rescore (contract + duan + none), per-chunk deltas ─────────────────

def _score_chain(arm: str) -> dict[int, dict]:
    """Replicates experiments/score_unification.py exactly, keeping per-chunk
    pooled means for all three smear conventions + per-bar arrays."""
    idxs = sorted(int(f[6:9]) for f in os.listdir(os.path.join(ROOT, arm))
                  if f.startswith("chunk_") and f.endswith(".npz"))
    out: dict[int, dict] = {}
    prev = None
    prev_idx = None
    for idx in idxs:
        c = load(arm, idx)
        n = len(c["yhat"])
        y_raw = np.full(n, np.nan)
        ok = np.isfinite(c["rv_raw"]) & np.isfinite(c["baseline"]) & (c["baseline"] > 0) & (c["rv_raw"] >= 0)
        y_raw[ok] = np.sqrt(c["rv_raw"][ok] / c["baseline"][ok])
        p = prev if prev_idx == idx - 1 else None
        a, b = 0.0, 1.0
        if p is not None:
            s = p["valid"] & np.isfinite(p["y_raw"]) & np.isfinite(p["yhat"])
            X = np.column_stack([np.ones(s.sum()), p["yhat"][s]])
            coef, _, rank, _ = np.linalg.lstsq(X, p["y_raw"][s], rcond=None)
            if rank == 2 and np.all(np.isfinite(coef)):
                a, b = float(coef[0]), float(coef[1])
        m = a + b * c["yhat"]
        if p is not None:
            s = p["valid"] & np.isfinite(p["y_raw"]) & np.isfinite(p["m"])
            sigma2 = float(np.mean((p["y_raw"][s] - p["m"][s]) ** 2))
        else:
            s0 = c["valid"] & np.isfinite(y_raw)
            sigma2 = float(np.mean((y_raw[s0] - m[s0]) ** 2))
        f = np.full(n, np.nan)
        okf = c["valid"] & np.isfinite(m) & np.isfinite(c["baseline"])
        f[okf] = (m[okf] ** 2 + sigma2) * c["baseline"][okf]
        loss = qlike_per_bar(f, c["rv_raw"]); loss[~c["valid"]] = np.nan
        # duan (in-chunk fit-scale scalar smear) + none
        s2_fit = float(np.mean((c["y_fit"][c["valid"]] - c["yhat"][c["valid"]]) ** 2))
        f_d = (c["yhat"] ** 2 + s2_fit) * c["baseline"]
        f_n = c["yhat"] ** 2 * c["baseline"]
        loss_d = qlike_per_bar(f_d, c["rv_raw"]); loss_d[~c["valid"]] = np.nan
        loss_n = qlike_per_bar(f_n, c["rv_raw"]); loss_n[~c["valid"]] = np.nan
        out[idx] = {"loss": loss, "loss_duan": loss_d, "loss_none": loss_n,
                    "row_id": c["row_id"], "yhat": c["yhat"], "y_fit": c["y_fit"],
                    "rv_raw": c["rv_raw"], "baseline": c["baseline"], "valid": c["valid"],
                    "ab": (a, b), "sigma2": sigma2, "s2_fit": s2_fit, "y_raw": y_raw, "m": m}
        prev = {"yhat": c["yhat"], "y_raw": y_raw, "m": m, "valid": c["valid"]}
        prev_idx = idx
    return out


def phase_chain() -> None:
    arm = _score_chain(ARM)
    a0 = _score_chain(A0)
    print("\n=== per-chunk pooled-QLIKE delta (arm - a0), three conventions ===")
    deltas = []
    for idx in sorted(set(arm) & set(a0)):
        d_c = np.nanmean(arm[idx]["loss"]) - np.nanmean(a0[idx]["loss"])
        d_d = np.nanmean(arm[idx]["loss_duan"]) - np.nanmean(a0[idx]["loss_duan"])
        d_n = np.nanmean(arm[idx]["loss_none"]) - np.nanmean(a0[idx]["loss_none"])
        deltas.append((idx, d_c, d_d, d_n))
    worst = sorted(deltas, key=lambda r: -r[1])[:8]
    print("  worst by CONTRACT delta:  idx  d_contract  d_duan   d_none")
    for r in worst:
        print(f"    {r[0]:3d}  {r[1]:+9.4f}  {r[2]:+9.4f}  {r[3]:+9.4f}")
    worst_d = sorted(deltas, key=lambda r: -r[2])[:8]
    print("  worst by DUAN delta:")
    for r in worst_d:
        print(f"    {r[0]:3d}  {r[1]:+9.4f}  {r[2]:+9.4f}  {r[3]:+9.4f}")
    np.save(os.path.join(_ROOT, "experiments", "_diag_c080_deltas.npy"), np.array(deltas))


# ── amplification decomposition ───────────────────────────────────────────────

def _pooled(loss: np.ndarray) -> float:
    return float(np.nanmean(loss))


def phase_amp() -> None:
    arm = _score_chain(ARM)
    a0 = _score_chain(A0)
    for idx in (80, 63):
        A, Z = arm[idx], a0[idx]
        n = len(A["loss"])
        thr = float(np.abs(A["y_fit"]).max())
        ext = np.abs(A["yhat"]) > thr
        for name in ("loss", "loss_duan", "loss_none"):
            dl = A[name] - Z[name]
            tot = np.nansum(dl) / n
            direct = np.nansum(dl[ext]) / n
            print(f"  c{idx:03d} {name:9s}: total_delta={tot:+.4f}  "
                  f"direct(ext bars, n={ext.sum()})={direct:+.4f}  "
                  f"ordinary={tot - direct:+.4f}")
        # duan counterfactual: smear computed EXCLUDING extreme bars
        v = A["valid"]
        s2_cf = float(np.mean((A["y_fit"][v & ~ext] - A["yhat"][v & ~ext]) ** 2))
        f_cf = (A["yhat"] ** 2 + s2_cf) * A["baseline"]
        loss_cf = qlike_per_bar(f_cf, A["rv_raw"]); loss_cf[~v] = np.nan
        d_cf = np.nansum((loss_cf - Z["loss_duan"])[~ext]) / n
        print(f"  c{idx:03d} duan smear: s2_fit={A['s2_fit']:.4f} -> {s2_cf:.4f} excl-ext; "
              f"ordinary-bar delta {np.nansum((A['loss_duan']-Z['loss_duan'])[~ext])/n:+.4f} -> {d_cf:+.4f} counterfactual")
        # contract-side story: calibration/sigma2 for THIS chunk come from prev chunk
        print(f"  c{idx:03d} contract maps: (a,b)={A['ab']}  sigma2={A['sigma2']:.4f}  "
              f"(a0: (a,b)={Z['ab']} sigma2={Z['sigma2']:.4f})")
        # spillover into idx+1 via the causal maps
        nxt = idx + 1
        if nxt in arm:
            N, Zn = arm[nxt], a0[nxt]
            print(f"  c{nxt:03d} (spillover): contract delta={_pooled(N['loss'])-_pooled(Zn['loss']):+.4f}  "
                  f"(a,b)={N['ab']}  sigma2={N['sigma2']:.4f} vs a0 sigma2={Zn['sigma2']:.4f}")
            # counterfactual: refit next-chunk maps excluding this chunk's extreme bars
            s = A["valid"] & np.isfinite(A["y_raw"]) & ~ext
            X = np.column_stack([np.ones(s.sum()), A["yhat"][s]])
            coef, _, _, _ = np.linalg.lstsq(X, A["y_raw"][s], rcond=None)
            a_cf, b_cf = float(coef[0]), float(coef[1])
            m_cf_prev = a_cf + b_cf * A["yhat"]  # not exact (m applied there came from c79 fit) but indicative
            sig_cf = float(np.mean((A["y_raw"][s] - A["m"][s]) ** 2))
            m_n = N["ab"][0] + N["ab"][1] * N["yhat"]  # actual
            m_cf = a_cf + b_cf * N["yhat"]
            f_cf = (m_cf ** 2 + sig_cf) * N["baseline"]
            loss_cf = qlike_per_bar(f_cf, N["rv_raw"]); loss_cf[~N["valid"]] = np.nan
            print(f"    counterfactual next-chunk maps: (a,b)=({a_cf:+.4f},{b_cf:+.4f}) sigma2={sig_cf:.4f} "
                  f"-> delta {_pooled(loss_cf)-_pooled(Zn['loss']):+.4f} (actual {_pooled(N['loss'])-_pooled(Zn['loss']):+.4f})")


# ── raw parquet extremeness at the bad timestamps ────────────────────────────

def phase_parquet() -> None:
    import pandas as pd

    c = load(ARM, 80)
    ay = np.abs(c["yhat"])
    thr = float(np.abs(c["y_fit"]).max())
    bad_t = pd.to_datetime(c["t"][ay > thr])
    c63 = load(ARM, 63)
    bad_t63 = pd.to_datetime(c63["t"][np.abs(c63["yhat"]) > float(np.abs(c63["y_fit"]).max())])
    print(f"\nbad timestamps c080: {list(bad_t)}")
    print(f"bad timestamps c063: {list(bad_t63)}")
    files = ["core_stats.parquet", "ewstock_stats.parquet", "vwstock_stats.parquet",
             "spy_and_sentiment.parquet", "vix_and_voldemand.parquet",
             "time_categories.parquet", "releases.parquet", "optionm_spx_spot.parquet"]
    for fn in files:
        path = os.path.join(_ROOT, "data", fn)
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            tc = next((c_ for c_ in df.columns if "time" in c_.lower() or "date" in c_.lower()), None)
            if tc is None:
                print(f"[{fn}] no datetime index/col; cols={list(df.columns)[:8]}...")
                continue
            df = df.set_index(pd.to_datetime(df[tc])).drop(columns=[tc])
        num = df.select_dtypes(include=[np.number])
        print(f"\n[{fn}] rows={len(df)} cols={num.shape[1]} span={df.index.min()}..{df.index.max()}")
        for t0 in list(bad_t) + list(bad_t63):
            if t0 < df.index.min() or t0 > df.index.max():
                continue
            # window: trailing 24000 bars ending at t0 (or all history if shorter)
            hist = num.loc[:t0]
            tail = hist.tail(24000)
            if t0 not in df.index:
                # nearest at-or-before
                pos = df.index.get_indexer([t0], method="ffill")[0]
                t_use = df.index[pos]
            else:
                t_use = t0
            row = num.loc[t_use]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            mx, mn = tail.max(), tail.min()
            hi = row[(row >= mx) & (tail.notna().sum() > 100) & row.notna()]
            lo_ = row[(row <= mn) & (tail.notna().sum() > 100) & row.notna()]
            iqr = (tail.quantile(0.75) - tail.quantile(0.25)).replace(0, np.nan)
            z = (row - tail.median()) / iqr
            zbig = z[np.abs(z) > 10]
            if len(hi) or len(lo_) or len(zbig):
                print(f"  @ {t0} (row used {t_use}):")
                for cname in hi.index:
                    print(f"    NEW-24k-MAX {cname}: {row[cname]:.6g} (prev max {tail[cname].iloc[:-1].max():.6g})")
                for cname in lo_.index:
                    print(f"    NEW-24k-MIN {cname}: {row[cname]:.6g} (prev min {tail[cname].iloc[:-1].min():.6g})")
                for cname in zbig.index.difference(hi.index).difference(lo_.index):
                    print(f"    |robust z|>10 {cname}: {row[cname]:.6g} (z={z[cname]:+.1f})")


def phase_arms() -> None:
    """yhat at the bad bar across every arm that has chunk_080 — localizes the
    offending column's bucket (single-bucket arms partition the exog columns)."""
    bad_row = 200318
    print(f"\n=== yhat at row {bad_row} (2020-12-24T18:00) across arms ===")
    for arm in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, arm, "chunk_080.npz")
        if not os.path.isdir(os.path.join(ROOT, arm)) or not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=False) as z:
            rid = np.asarray(z["row_id"])
            yh = np.asarray(z["yhat"])
        i = np.flatnonzero(rid == bad_row)
        if len(i):
            i = int(i[0])
            print(f"  {arm:28s} yhat={yh[i]:+10.3f}   max|yhat| in chunk={np.abs(yh).max():8.3f}")


def phase_probe() -> None:
    import pandas as pd

    # 1. collinear/dropped-name set diffs: c080 vs neighbors (and c063 vs c064)
    metas = {i: load(ARM, i)["meta"] for i in (62, 63, 64, 79, 80, 81)}
    base = set(metas[79]["ols_collinear_cols"])
    for i in (80, 81, 63):
        cur = set(metas[i]["ols_collinear_cols"])
        ref = base if i in (80, 81) else set(metas[62]["ols_collinear_cols"])
        print(f"c{i:03d} collinear={len(cur)}  added_vs_ref={sorted(cur - ref)}")
        print(f"      removed_vs_ref={sorted(ref - cur)}")
    # in particular: cols collinear in NEIGHBORS but NOT in c080 (kept there!)
    kept_only_c080 = (set(metas[79]["ols_collinear_cols"]) & set(metas[81]["ols_collinear_cols"])) - set(metas[80]["ols_collinear_cols"])
    print(f"\ncollinear in BOTH c079+c081 but KEPT in c080: {sorted(kept_only_c080)}")

    def _dt(fn: str):
        df = pd.read_parquet(os.path.join(_ROOT, "data", fn))
        if not isinstance(df.index, pd.DatetimeIndex):
            tc = next(c_ for c_ in df.columns if "time" in c_.lower() or "date" in c_.lower())
            df = df.set_index(pd.to_datetime(df[tc])).drop(columns=[tc])
        return df

    # 2. raw bar grid on the early-close days vs a normal day
    core = _dt("core_stats.parquet")
    for day in ("2020-12-24", "2020-11-27", "2020-12-23"):
        sl = core.loc[day]
        print(f"\ncore_stats {day}: {len(sl)} bars, index {sl.index.min()}..{sl.index.max()}")
        seg = core.loc[f"{day} 16:30":f"{day} 19:30"]
        print(seg.to_string(max_cols=10))

    # 3. panel bars + yhat on 2020-11-27 (the other early close inside c080)
    c = load(ARM, 80)
    t = c["t"].astype("datetime64[s]")
    for day in ("2020-11-26", "2020-11-27", "2020-12-24", "2020-12-25"):
        m = (t >= np.datetime64(day)) & (t < np.datetime64(day) + np.timedelta64(1, "D"))
        if m.sum() == 0:
            print(f"\npanel {day}: NO bars")
            continue
        print(f"\npanel {day}: {m.sum()} bars  max|yhat|={np.abs(c['yhat'][m]).max():.3f}")
        idx = np.flatnonzero(m)
        for i in idx:
            hh = str(t[i])[11:16]
            if "17:0" in str(t[i]) or "18:" in str(t[i]) or "19:0" in str(t[i]):
                print(f"    {t[i]}  yhat={c['yhat'][i]:+8.3f} y_fit={c['y_fit'][i]:+6.3f} rv={c['rv_raw'][i]:.2e} B={c['baseline'][i]:.2e}")

    # 4. vix + time_categories + sentiment at the bad bar
    for fn in ("vix_and_voldemand.parquet", "time_categories.parquet", "spy_and_sentiment.parquet"):
        df = _dt(fn)
        seg = df.loc["2020-12-24 16:30":"2020-12-24 19:30"]
        print(f"\n{fn} around the bad bar (cols: {list(df.columns)}):")
        print(seg.to_string(max_cols=12))

    # 5. stocktwits series: where does it live, when did it die/go-live?
    for fn in ("spy_and_sentiment.parquet",):
        df = _dt(fn)
        st = [c_ for c_ in df.columns if "stocktwit" in c_.lower() or "attention" in c_.lower()]
        print(f"\n{fn} stocktwits cols: {st}")
        for cn in st:
            s = df[cn]
            notna = s.notna()
            print(f"  {cn}: first valid {s.first_valid_index()}, last valid {s.last_valid_index()}, n_valid={notna.sum()}")
            seg = s.loc["2020-11-01":"2020-12-31"]
            print(f"    2020-11..12: n_valid={seg.notna().sum()}, last valid in seg={seg.dropna().index.max() if seg.notna().any() else None}")
            seg2 = s.loc["2020-12-24"]
            print(f"    2020-12-24 values: {seg2.dropna().to_dict()}")

    # 6. sentiment-bucket arm meta for c080 (same panel, bucket-only design)
    for arm in ("a_bucket_sentiment", "a_bucket_vol_demand", "a_bucket_implied_vol"):
        p = os.path.join(ROOT, arm, "chunk_080.npz")
        if not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
        cc = m.get("ols_collinear_cols", [])
        st_cc = [c_ for c_ in cc if "stocktwit" in c_ or "attention" in c_]
        print(f"\n{arm} c080: n_design={m.get('n_design_cols')} collinear={len(cc)} "
              f"stocktwits-collinear={st_cc} masked={len(m.get('ols_masked_cols', {}))}")


def phase_design() -> None:
    """Rebuild the local b2 panel (executor's own prep path), reproduce the c080
    entry QR + the window solve at the bad bar, and decompose yhat by column.
    CAVEAT: local panel may diverge from the campaign panel (known 9-vs-16
    calendar-col divergence) — parity is checked against the npz disclosures."""
    os.environ.setdefault(
        "UNIFY_CACHE_DIR",
        r"C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-clean\340c14de-7d7a-4383-9e9d-54a267acfe00\scratchpad",
    )
    os.environ.setdefault("PREP_ROWS", "205000")
    import src.unification as U

    p = U._load_panel()
    meta = load(ARM, 80)["meta"]
    c = load(ARM, 80)
    BAD = 200318
    print(f"panel n={len(p.y)} cols={len(p.names)}")
    print(f"t parity at BAD: panel={p.t[BAD]} npz={c['t'][BAD - 199147]}")
    print(f"y parity at BAD: panel y={p.y[BAD]:.6f} npz y_fit={c['y_fit'][BAD - 199147]:.6f}")
    cal = [nm for nm in p.names if U._classify(nm)[0] == "calendar"]
    print(f"local calendar cols ({len(cal)}): {cal}")

    F, kept, dropped = U._ols_design(p, U.ARMS[ARM])
    print(f"design: kept={len(kept)} dropped={len(dropped)} (campaign: 703+? dropped={len(meta['ols_dropped_cols'])})")
    F2, kept2, coll = U._eliminate_exact_collinear(F, kept, 24000, 199147)
    camp = meta["ols_collinear_cols"]
    print(f"entry-QR collinear: local={len(coll)} campaign={len(camp)}")
    print(f"  local-only: {sorted(set(coll) - set(camp))}")
    print(f"  campaign-only: {sorted(set(camp) - set(coll))}")
    del F

    W = 24000
    for t_bar in (BAD - 1, BAD, BAD + 1):
        Xw = F2[t_bar - W : t_bar]
        mu = Xw.mean(0)
        Xc = Xw - mu
        yw = p.y[t_bar - W : t_bar]
        coef, _, rank, sv = np.linalg.lstsq(Xc, yw - yw.mean(), rcond=None)
        b0 = yw.mean()
        x = F2[t_bar] - mu
        yh = float(x @ coef + b0)
        print(f"\nbar {t_bar} ({p.t[t_bar]}): local OLS yhat={yh:+.3f} "
              f"(npz {c['yhat'][t_bar - 199147]:+.3f})  rank={rank}/{Xc.shape[1]}  "
              f"cond~{sv[0] / sv[-1]:.2e}")
        contrib = coef * x
        topi = np.argsort(-np.abs(contrib))[:14]
        print("  top |contribution| columns (name, contrib, coef, x_t-mu, win_sd, x_prev-mu):")
        xp = F2[t_bar - 1] - mu
        sd = Xw.std(0)
        for j in topi:
            print(f"    {kept2[j]:42s} c={contrib[j]:+10.3f} coef={coef[j]:+12.3f} "
                  f"dx={x[j]:+10.5f} sd={sd[j]:.5f} dx_prev={xp[j]:+10.5f}")


def phase_avail() -> None:
    """Availability-desync check straight from the raw parquets on the loader
    grid (weekend-trimmed 30-min grid; no panel reconstruction needed).
    avail[stem] at grid row r = notna(raw[stem]) at r-1 (the loader's shift(1))."""
    import pandas as pd

    # loader grid t for rows ~172000..201400 from the persisted chunk t arrays
    ts_all, rid_all = [], []
    for idx in range(66, 82):
        c = load(ARM, idx)
        ts_all.append(c["t"]); rid_all.append(c["row_id"])
    t = pd.to_datetime(np.concatenate(ts_all))
    rid = np.concatenate(rid_all)

    def _dt(fn):
        df = pd.read_parquet(os.path.join(_ROOT, "data", fn))
        if not isinstance(df.index, pd.DatetimeIndex):
            tc = next(c_ for c_ in df.columns if "time" in c_.lower() or "date" in c_.lower())
            df = df.set_index(pd.to_datetime(df[tc])).drop(columns=[tc])
        return df

    spy = _dt("spy_and_sentiment.parquet")
    core = _dt("core_stats.parquet")
    ew = _dt("ewstock_stats.parquet")
    vix = _dt("vix_and_voldemand.parquet")
    # notna at the PREVIOUS grid bar (loader shift(1) on the grid)
    def avail(df, col):
        s = df[col].reindex(t)  # value AT each grid bar
        return s.notna().to_numpy()[:-1], t[1:]  # availability applying to next bar

    a_st, tt = avail(spy, "stocktwits_attention")
    a_core, _ = avail(core, "sumret2")
    a_ew, _ = avail(ew, "sumret2_ewstock")
    a_vix, _ = avail(vix, "vix")
    r = rid[1:]
    W0 = 175147  # c080 entry-window start
    BAD = 200318
    for nm, a_o in (("core.sumret2", a_core), ("ew.sumret2", a_ew), ("vix", a_vix)):
        d = a_st != a_o
        dr = r[d]
        in_win = dr[(dr >= W0 - 2048) & (dr < 199147)]
        after = dr[(dr >= 199147) & (dr <= BAD + 30)]
        print(f"stocktwits vs {nm}: desync bars in [W0-2048, W0)={len(dr[(dr>=W0-2048)&(dr<W0)])}, "
              f"in entry window [W0, 199147)={len(in_win)}, in chunk up to BAD+30: rows {after.tolist()}")
        if len(after):
            for rr in after[:6]:
                i = np.flatnonzero(r == rr)[0]
                print(f"    row {rr} {tt[i]}: stocktwits={a_st[i]} {nm}={a_o[i]}")
    # gap structure that kept ma_512/1024/2048 alive in the entry window
    d_any = a_st != a_core
    dr = r[d_any & (r >= W0 - 2048) & (r < W0)]
    print(f"stocktwits-vs-core desync rows in [W0-2048, W0): {dr.tolist()[:40]}")
    # and stocktwits own gaps (avail=0) in entry window
    z = r[(~a_st) & (r >= W0 - 2048) & (r < 199147)]
    print(f"stocktwits avail==0 rows in [W0-2048, 199147): n={len(z)}, first={z[:5].tolist()}, last={z[-5:].tolist()}")
    zc = r[(~a_core) & (r >= W0 - 2048) & (r < 199147)]
    print(f"core avail==0 rows in same range: n={len(zc)} (holiday/half-day gap bars)")


if __name__ == "__main__":
    phases = sys.argv[1:] or ["facts", "scan", "chain", "amp"]
    for ph in phases:
        {"facts": phase_facts, "scan": phase_scan, "chain": phase_chain,
         "amp": phase_amp, "parquet": phase_parquet, "arms": phase_arms,
         "probe": phase_probe, "design": phase_design, "avail": phase_avail}[ph]()
