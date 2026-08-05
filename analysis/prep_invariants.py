"""Post-prep invariants: the checks that would have caught the October 2023 bug.

The imputation defect that cost 6.37 QLIKE on 100 bars survived because nothing
ever compared the pipeline's own bookkeeping against the raw data. It was
visible in a single number the whole time: the volatility-demand availability
indicator averaged 0.698 while the underlying series was observed on 0.223 of
bars. One assertion catches that.

This module is that assertion and four others, runnable against any built
cache. It is deliberately not a unit test of the transforms -- those can pass
while the composition of them is wrong, which is exactly what happened -- but a
check on the ARTIFACT, comparing what the cache says about itself against the
raw inputs it came from.

  A. AVAILABILITY HONESTY. Each ``<col>_avail_ma_1`` column's mean must match
     the raw series' finite fraction. A forward fill computed before the
     indicator inflates the latter without touching the former, so any gap is
     that bug or one like it. This is the check that was missing.

  B. SCALE COLLAPSE. For each channel, the ratio of its smallest rolling IQR to
     its median rolling IQR. This is the correct pathology detector, and it
     replaces the one used in analysis/prep_audit.py, which flagged "scaled
     magnitude above 20 IQRs" and therefore could not distinguish a collapsed
     DENOMINATOR (the bug) from a genuinely extreme observation on a
     fat-tailed series (not a bug). That conflation is why the audit reported
     23 affected channels when only the volatility-demand family was sick.

  C. STALENESS. The longest run of exactly-repeated values in each raw series,
     which bounds how far a forward fill can carry a value. A series with a
     1,000-bar constant run is being presented as live data for two months.

  D. IMPUTATION SHARE. The fraction of each channel's rows that are imputed
     rather than observed. A channel that is 78% imputed is not a feature with
     some gaps; it is mostly a constant with a coefficient fitted to it.

  E. FINITENESS AND RANGE. No NaN or infinity, and a report of the extreme
     scaled magnitudes -- kept as a descriptive tail summary rather than a
     pass/fail, since B is the check that actually distinguishes causes.

Run against any cache directory; compares two if given two.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "writeup", "stats")

AVAIL_TOL = 0.05          # indicator mean vs raw coverage
IQR_COLLAPSE_RATIO = 0.05  # min/median rolling IQR below this = collapsed
STALE_BARS = 260           # ~20 sessions of exact repetition
IMPUTED_SHARE_WARN = 0.50


def rolling_iqr_stats(x, win=24000, step=500):
    """Min and median of the trailing rolling IQR, subsampled."""
    n = len(x)
    if n <= win:
        return np.nan, np.nan
    vals = []
    for s in range(win, n, step):
        w = x[s - win:s]
        w = w[np.isfinite(w)]
        if w.size < 100:
            continue
        q25, q75 = np.percentile(w, (25.0, 75.0))
        vals.append(q75 - q25)
    if not vals:
        return np.nan, np.nan
    v = np.asarray(vals)
    pos = v[v > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    return float(v.min()), med


def _class_b_worker(task):
    """Class-B statistics for one channel, in its own process.

    Takes a cache path and column index rather than an array so the 2 GB
    feature matrix is never pickled -- each worker memory-maps it and touches
    only its own column. The arithmetic is identical to the serial path.
    """
    cache_dir, j1 = task
    X = np.load(os.path.join(cache_dir, "X.npy"), mmap_mode="r")
    col = np.asarray(X[:, j1], np.float64)
    live = col != 0.0
    mn_a, md_a = rolling_iqr_stats(col)
    live_col = col[live]
    mn, md = (rolling_iqr_stats(live_col) if live_col.size > 30000
              else (np.nan, np.nan))
    return {
        "live_frac": float(live.mean()),
        "iqr_ratio_all": (mn_a / md_a
                          if (md_a and np.isfinite(md_a) and md_a > 0)
                          else np.nan),
        "iqr_min": mn, "iqr_median": md,
        "iqr_ratio": (mn / md if (md and np.isfinite(md) and md > 0)
                      else np.nan),
    }


def longest_run(v):
    f = np.isfinite(v)
    if f.sum() < 2:
        return int(len(v))
    d = np.diff(v[f])
    best = cur = 0
    for x in d:
        cur = cur + 1 if x == 0.0 else 0
        if cur > best:
            best = cur
    return int(best)


_RAW_CACHE: dict = {}


def _raw_panel():
    """The raw panel, parsed once and shared across every cache audited.

    Re-parsing the parquet per cache dominated the run when three caches were
    compared; the panel does not depend on which cache is being audited.
    """
    if "df" not in _RAW_CACHE:
        from src.data.loading import load_raw_data
        _RAW_CACHE["df"] = load_raw_data("data", allow_missing=True).dropna(
            subset=["RV"]).reset_index(drop=True)
    return _RAW_CACHE["df"]


def audit(cache_dir, label, pool=None):
    X = np.load(os.path.join(cache_dir, "X.npy"), mmap_mode="r")
    names = [str(s) for s in np.load(os.path.join(cache_dir, "names.npy"),
                                     allow_pickle=True)]
    n = X.shape[0]
    sub = _raw_panel().iloc[3125:3125 + n].reset_index(drop=True)

    chan: dict[str, list] = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m:
            chan.setdefault(m.group(1), []).append(j)
    labels = [c for c in chan if len(chan[c]) == 12]
    idx = np.arange(0, n, 7)

    # Class B is the expensive check -- a rolling quantile over the whole
    # panel, twice per channel. Farm it out first so the workers run while the
    # cheap per-channel checks proceed here.
    b_cols = {c: names.index(f"adj_{c}_ma_1") for c in labels
              if f"adj_{c}_ma_1" in names}
    b_keys = sorted(b_cols)
    if pool is not None and b_keys:
        b_stats = dict(zip(b_keys, pool.map(
            _class_b_worker, [(cache_dir, b_cols[c]) for c in b_keys])))
    else:
        b_stats = {c: _class_b_worker((cache_dir, b_cols[c])) for c in b_keys}

    rows, fails = [], []
    for c in sorted(labels):
        rec: dict = {"channel": c}
        # A. availability honesty
        av = f"{c}_avail_ma_1"
        if av in names and c in sub.columns:
            ind = float(np.asarray(X[idx][:, names.index(av)],
                                   np.float64).mean())
            cov = float(np.isfinite(sub[c].to_numpy(np.float64)).mean())
            rec["avail_mean"] = ind
            rec["raw_coverage"] = cov
            rec["avail_gap"] = ind - cov
            if abs(ind - cov) > AVAIL_TOL:
                fails.append(f"[A] {c}: availability indicator {ind:.3f} vs "
                             f"raw coverage {cov:.3f} (gap {ind - cov:+.3f})")
        # B. scale collapse, on the bar-level adjusted column.
        #
        # Measured over LIVE rows -- those the scaler did not pin to the
        # neutral 0.0. Two prep steps deliberately emit an exact zero: the
        # neutral-median substitution on unobserved rows, and the warm-up
        # region where a channel has too few observations for a trustworthy
        # scale. Both produce long constant runs on purpose, and reading the
        # rolling IQR across them reports a "collapse" that is the intended
        # encoding of "this channel is not live here".
        #
        # This is not a way of not-looking: a pinned-to-zero row cannot carry
        # an exploding feature, and the ``scaled_absmax`` column catches
        # blow-ups independently. The correction was made only after the
        # all-rows statistic was shown to fire on the neutralised region --
        # 10 of 11 flagged channels are healthy on live rows (stocktwits_
        # attention 0.0000 -> 0.8969, vix3m 0.0000 -> 0.3644) while their
        # magnitudes are benign. The threshold is untouched.
        if c in b_stats:
            rec.update(b_stats[c])
            ratio = rec["iqr_ratio"]
            if np.isfinite(ratio) and ratio < IQR_COLLAPSE_RATIO:
                fails.append(f"[B] {c}: rolling IQR over live rows collapses "
                             f"to {ratio:.4f} of its median")
        # C/D. staleness and imputation share in the raw series
        if c in sub.columns:
            v = sub[c].to_numpy(np.float64)
            run = longest_run(v)
            rec["longest_flat_run"] = run
            rec["imputed_share"] = float(1.0 - np.isfinite(v).mean())
            if run > STALE_BARS:
                fails.append(f"[C] {c}: {run} bars of exact repetition")
            if rec["imputed_share"] > IMPUTED_SHARE_WARN:
                fails.append(f"[D] {c}: {rec['imputed_share']:.1%} of rows "
                             f"imputed")
        # E. range
        V = np.asarray(X[idx][:, chan[c]], np.float64)
        rec["scaled_absmax"] = float(np.abs(V).max())
        if not np.isfinite(V).all():
            fails.append(f"[E] {c}: non-finite values in the cache")
        rows.append(rec)

    print(f"\n{'=' * 86}")
    print(f"{label}  ({cache_dir})")
    print("=" * 86)
    print(f"  {'channel':>32}{'avail':>8}{'rawcov':>8}{'IQR min/med':>13}"
          f"{'flat run':>10}{'|max|':>10}")
    for r in sorted(rows, key=lambda z: z.get("iqr_ratio", np.inf)):
        rr = r.get("iqr_ratio", float("nan"))
        print(f"  {r['channel']:>32}{r.get('avail_mean', float('nan')):>8.3f}"
              f"{r.get('raw_coverage', float('nan')):>8.3f}{rr:>13.4f}"
              f"{r.get('longest_flat_run', -1):>10}"
              f"{r['scaled_absmax']:>10.1f}")
    by = {}
    for f in fails:
        by.setdefault(f[1], []).append(f)
    print(f"\n  {len(fails)} violations across {len(rows)} channels")
    for k in sorted(by):
        print(f"    [{k}] {len(by[k])}")
        for f in by[k][:4]:
            print(f"        {f[4:]}")
        if len(by[k]) > 4:
            print(f"        ... and {len(by[k]) - 4} more")
    return {"cache": cache_dir, "channels": rows, "violations": fails,
            "n_violations": len(fails),
            "by_class": {k: len(v) for k, v in by.items()}}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = sys.argv[1:] or [os.path.join(ROOT, "results", "b2_mmap"),
                               os.path.join(ROOT, "results", "b2_mmap_fix")]
    out = {}
    live = [t for t in targets if os.path.exists(os.path.join(t, "X.npy"))]
    for t in targets:
        if t not in live:
            print(f"skipping {t}: not built")
    n_proc = max(1, min(os.cpu_count() or 1, 8))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=n_proc) as pool:
        print(f"class B across {n_proc} workers")
        for t in live:
            out[os.path.basename(t)] = audit(t, os.path.basename(t), pool)
    print(f"\naudited {len(live)} cache(s) in {time.time() - t0:.0f}s")
    if len(out) > 1:
        ks = list(out)
        print(f"\n{'=' * 86}")
        print("VIOLATIONS BY CLASS")
        print("=" * 86)
        classes = sorted({c for v in out.values() for c in v["by_class"]})
        print(f"  {'class':>6}" + "".join(f"{k:>22}" for k in ks))
        meaning = {"A": "availability honesty", "B": "scale collapse",
                   "C": "staleness", "D": "imputation share",
                   "E": "finiteness"}
        for c in classes:
            print(f"  {c:>6}" + "".join(
                f"{out[k]['by_class'].get(c, 0):>22}" for k in ks)
                + f"   {meaning.get(c, '')}")
        print(f"  {'total':>6}" + "".join(
            f"{out[k]['n_violations']:>22}" for k in ks))
    with open(os.path.join(OUT_DIR, "prep_invariants.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote prep_invariants.json")


if __name__ == "__main__":
    main()
