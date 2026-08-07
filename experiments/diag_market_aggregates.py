"""Diagnose why a_bucket_market_ew / a_bucket_market_vw HURT under literal OLS.

Read-only on results/. Scores per-chunk via experiments/score_unification._score_chunk
(the paper's smearing contract, reused not re-derived), localizes the damage vs
a0_ols_har per-chunk and per-bar, characterizes yhat pathology, then inspects the
ew/vw bucket columns in the prep cache (memory-mapped zip-store access — the local
box is memory-tight) and cross-references extreme design values with the worst bars.

Usage:  C:/Users/james/miniconda3/envs/285J/python.exe experiments/diag_market_aggregates.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from score_unification import _score_chunk  # noqa: E402

ROOT = os.path.join(_ROOT, "results", "unification_carc")
CACHE = os.path.join(_ROOT, "results", "prep_cache_all_features_b2.npz")
ARMS = ["a0_ols_har", "a_bucket_market_ew", "a_bucket_market_vw"]
N_CHUNK = 100


# ── mmap access to one member of an UNCOMPRESSED npz (zip-store) ──────────────
def npz_memmap(path: str, member: str):
    """Memory-map one array from an uncompressed .npz without loading the file."""
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(member + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError(f"{member} is compressed; cannot memmap")
    # local file header: 30 fixed bytes + name + extra
    fh = open(path, "rb")
    fh.seek(info.header_offset)
    hdr = fh.read(30)
    nlen = int.from_bytes(hdr[26:28], "little")
    elen = int.from_bytes(hdr[28:30], "little")
    data_start = info.header_offset + 30 + nlen + elen
    fh.seek(data_start)
    version = np.lib.format.read_magic(fh)
    shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
    offset = fh.tell()
    fh.close()
    zf.close()
    order = "F" if fortran else "C"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order=order)


# ── 1+2. harvest, score, localize ─────────────────────────────────────────────
def harvest(arm: str) -> dict:
    per_chunk = []  # (idx, qlike, sigma2, t0, t1, n)
    rows, losses, yhats, fs, rvs, ts, yfits, bases = [], [], [], [], [], [], [], []
    for i in range(N_CHUNK):
        path = os.path.join(ROOT, arm, f"chunk_{i:03d}.npz")
        c = _score_chunk(path)
        with np.load(path, allow_pickle=False) as z:
            t = np.asarray(z["t"])
            y_fit = np.asarray(z["y_fit"], dtype=np.float64)
            base = np.asarray(z["baseline"], dtype=np.float64)
        q = float(np.nanmean(c["loss"]))
        per_chunk.append((i, q, c["sigma2"], t[0], t[-1], len(t)))
        rows.append(c["row_id"]); losses.append(c["loss"]); yhats.append(c["yhat"])
        fs.append(c["f"]); rvs.append(c["rv_raw"]); ts.append(t)
        yfits.append(y_fit); bases.append(base)
    cat = lambda a: np.concatenate(a)  # noqa: E731
    return {
        "per_chunk": per_chunk,
        "row_id": cat(rows), "loss": cat(losses), "yhat": cat(yhats),
        "f": cat(fs), "rv": cat(rvs), "t": cat(ts), "y_fit": cat(yfits),
        "baseline": cat(bases),
    }


def part7() -> None:
    """Reproduce the c057 explosion with the executor's own rolling gram walk,
    and scan ALL design columns (indicators included) for extreme magnitudes."""
    names = [str(x) for x in np.load(CACHE, allow_pickle=True)["names"]]
    X = npz_memmap(CACHE, "X")
    y = np.load(CACHE, allow_pickle=True)["y"].astype(np.float64)
    W, LO, HI = 24000, 148792, 150981

    val_re = re.compile(r"^adj_(.+)_ma_(\d+)$")
    ind_re = re.compile(r"^(.+)_(avail|active)_ma_(\d+)$")
    har_re = re.compile(r"^har_ma_(\d+)$")
    reg_re = re.compile(r"^har_ma_(\d+)_x_(open|close)$")

    def classify(nm):
        if reg_re.match(nm):
            return ("regime", "")
        if har_re.match(nm):
            return ("har", "")
        m = ind_re.match(nm)
        if m:
            return ("indicator", m.group(1))
        m = val_re.match(nm)
        if m:
            return ("value", m.group(1))
        return ("calendar", "")

    for arm, fam in (("a_bucket_market_ew", "ewstock"),
                     ("a_bucket_market_vw", "vwstock")):
        stems = {f"{s}_{fam}" for s in ("sumret2", "sumret3", "sumret4",
                                        "sumabsret", "sumbipow", "sumpret2")}
        backbone = [j for j, nm in enumerate(names)
                    if classify(nm)[0] in ("har", "regime", "calendar")]
        bucket = [j for j, nm in enumerate(names)
                  if classify(nm)[0] in ("value", "indicator")
                  and classify(nm)[1] in stems]
        with np.load(os.path.join(ROOT, arm, "chunk_057.npz"),
                     allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            yhat_dep = np.asarray(z["yhat"], dtype=np.float64)
        dropped = set(meta["ols_dropped_cols"])
        cols = [j for j in backbone + bucket if names[j] not in dropped]
        cn = [names[j] for j in cols]
        print(f"\n[{arm}] reconstructed design: {len(cols)} cols "
              f"(meta n_design_cols={meta['n_design_cols']})")

        # extreme-magnitude scan over the chunk's rolled span (window + chunk)
        span = np.asarray(X[LO - W:HI, cols], dtype=np.float64)
        mx = np.abs(span).max(0)
        order = np.argsort(-mx)[:8]
        print("  max|x| over rows [124792,150981), top:")
        for j in order:
            am = int(np.abs(span[:, j]).argmax())
            print(f"    {cn[j]:36s} {mx[j]:12.4g}  at global row {LO - W + am}")

        # executor-faithful rolling walk (no support masks — result was
        # mask-invariant); compare to the deployed yhat
        from src.unification import _walk_ols
        F = np.ascontiguousarray(
            np.asarray(X[LO - W:HI, cols], dtype=np.float64))
        yv = y[LO - W:HI]
        # _walk_ols slices F[lo-window:hi] with GLOBAL indices; pass lo=W
        yh, masked = _walk_ols(F, np.concatenate([yv]), W, W, W + (HI - LO), cn)
        dmax = np.abs(yh - yhat_dep).max()
        print(f"  reroll vs deployed yhat: max|diff|={dmax:.3e}  "
              f"reroll min/max {yh.min():+.1f}/{yh.max():+.1f}  "
              f"deployed {yhat_dep.min():+.1f}/{yhat_dep.max():+.1f}")
        k = int(np.abs(yhat_dep).argmax())
        print(f"  at deployed-worst bar (chunk idx {k}): "
              f"deployed {yhat_dep[k]:+.2f} reroll {yh[k]:+.2f}")
        if masked:
            mk = {k2: v["reasons"] for k2, v in list(masked.items())[:6]}
            print(f"  reroll masked cols: {len(masked)} e.g. {mk}")
        del span, F


def main() -> None:
    if "--part7" in sys.argv:
        part7()
        return
    data = {arm: harvest(arm) for arm in ARMS}
    a0 = data["a0_ols_har"]
    # row sets are identical by construction; assert then use directly
    for arm in ARMS[1:]:
        assert np.array_equal(data[arm]["row_id"], a0["row_id"]), arm

    for arm in ("a_bucket_market_ew", "a_bucket_market_vw"):
        d = data[arm]
        print("=" * 100)
        print(f"ARM {arm}: pooled QLIKE={np.nanmean(d['loss']):.5f}  "
              f"a0={np.nanmean(a0['loss']):.5f}")
        diff = d["loss"] - a0["loss"]  # per-bar differential (NaN-safe pairwise)
        fin = np.isfinite(diff)
        total_damage = np.nansum(diff[fin])
        print(f"  per-bar diff: n={fin.sum()}  sum={total_damage:.1f}  "
              f"mean={np.nanmean(diff[fin]):.5f}  median={np.nanmedian(diff[fin]):.6f}")
        q = np.nanpercentile(diff[fin], [1, 25, 50, 75, 99, 99.9])
        print(f"  diff quantiles p1/p25/p50/p75/p99/p99.9: "
              + " ".join(f"{v:.4f}" for v in q))
        # concentration: what share of total damage from top-k bars?
        sd = np.sort(diff[fin])[::-1]
        for k in (20, 100, 1000, int(0.01 * fin.sum())):
            print(f"    top {k:>6d} bars carry {sd[:k].sum():10.1f} "
                  f"({100 * sd[:k].sum() / total_damage:5.1f}% of net damage)")
        pos_frac = float(np.mean(diff[fin] > 0))
        print(f"  fraction of bars where arm loses to a0: {pos_frac:.3f}")

        # per-chunk table: worst 12 chunks by qlike differential
        pc_arm = d["per_chunk"]; pc_a0 = a0["per_chunk"]
        deltas = []
        for (i, qa, s2a, t0, t1, n), (_, q0, s20, _, _, _) in zip(pc_arm, pc_a0):
            deltas.append((qa - q0, i, qa, q0, s2a, s20, t0, t1))
        deltas.sort(reverse=True)
        print("  worst chunks (dQLIKE, chunk, q_arm, q_a0, sigma2_arm, sigma2_a0, dates):")
        for dq, i, qa, q0, s2a, s20, t0, t1 in deltas[:12]:
            print(f"    {dq:+9.4f}  c{i:03d}  {qa:8.4f} {q0:7.4f}  "
                  f"s2 {s2a:9.4f} vs {s20:7.4f}  "
                  f"{str(t0)[:10]} .. {str(t1)[:10]}")
        n_hurt = sum(1 for x in deltas if x[0] > 0.001)
        print(f"  chunks with dQLIKE > 0.001: {n_hurt}/100 ; "
              f"sum of top-5 chunk contributions = "
              f"{sum(x[0] * pc_arm[x[1]][5] for x in deltas[:5]):.1f} bars-loss "
              f"of total {total_damage:.1f}")

        # worst 20 bars
        order = np.argsort(np.where(fin, diff, -np.inf))[::-1][:20]
        print("  worst 20 bars (t, dloss, yhat_arm, yhat_a0, y_fit, f_arm, f_a0, rv_raw):")
        for j in order:
            print(f"    {str(d['t'][j])[:19]}  dl={diff[j]:9.2f}  "
                  f"yh={d['yhat'][j]:+9.4f} vs {a0['yhat'][j]:+7.4f}  "
                  f"yfit={d['y_fit'][j]:7.4f}  f={d['f'][j]:11.4e} vs "
                  f"{a0['f'][j]:9.3e}  rv={d['rv'][j]:9.3e}  row={d['row_id'][j]}")

        # yhat characterization, overall + on the worst chunk
        yh, yh0 = d["yhat"], a0["yhat"]
        print(f"  sqrt-space yhat: arm min/max {yh.min():+.3f}/{yh.max():+.3f} "
              f"sd {yh.std():.4f} | a0 {yh0.min():+.3f}/{yh0.max():+.3f} sd {yh0.std():.4f}")
        print(f"    n(yhat<0): arm {int((yh < 0).sum())} a0 {int((yh0 < 0).sum())} ; "
              f"n(|yhat|>3): arm {int((np.abs(yh) > 3).sum())} "
              f"a0 {int((np.abs(yh0) > 3).sum())} ; "
              f"n(|yhat|>10): arm {int((np.abs(yh) > 10).sum())}")
        wc = deltas[0][1]
        m = np.zeros(len(yh), bool)
        # rebuild chunk membership from per-chunk sizes
        sizes = [x[5] for x in pc_arm]; starts = np.cumsum([0] + sizes)
        m[starts[wc]:starts[wc + 1]] = True
        print(f"    worst chunk c{wc:03d}: yhat_arm quantiles "
              + str(np.round(np.percentile(yh[m], [0, 1, 50, 99, 100]), 3))
              + "  a0 " + str(np.round(np.percentile(yh0[m], [0, 1, 50, 99, 100]), 3)))

    # ── 3. panel columns ──────────────────────────────────────────────────────
    print("=" * 100)
    print("PANEL COLUMN INSPECTION (memmapped prep cache)")
    with zipfile.ZipFile(CACHE) as zf:
        comp = {i.filename: i.compress_type for i in zf.infolist()}
    print(f"  cache members: {comp}")
    names_arr = np.load(CACHE, allow_pickle=True)["names"]
    names = [str(x) for x in names_arr]
    X = npz_memmap(CACHE, "X")
    n_rows = X.shape[0]
    row_id = a0["row_id"]
    t_by_row = {int(r): a0["t"][k] for k, r in enumerate(row_id)}

    def date_of(r):
        return str(t_by_row.get(int(r), f"row{r}(pre-OOS)"))[:19]

    val_re = re.compile(r"^adj_(.+)_ma_(\d+)$")
    for fam, tag in (("ewstock", "market_ew"), ("vwstock", "market_vw")):
        cols = [
            (j, nm) for j, nm in enumerate(names)
            if (m := val_re.match(nm)) and fam in m.group(1)
            and not any(x in m.group(1) for x in ("turnover", "spread", "ofi"))
        ]
        print(f"\n  {tag}: {len(cols)} value columns")
        stats = []
        for j, nm in cols:
            col = np.asarray(X[:, j], dtype=np.float64)  # one column at a time
            amax = int(np.argmax(np.abs(col)))
            top = np.argsort(np.abs(col))[::-1][:5]
            stats.append((float(np.abs(col).max()), nm, amax,
                          float(col.std()), int((np.abs(col) > 10).sum()),
                          int((np.abs(col) > 50).sum()),
                          [(int(r), float(col[r])) for r in top]))
            del col
        stats.sort(reverse=True)
        print("    max|z|   sd      n>|10|  n>|50|  column / argmax date / top rows")
        for mx, nm, amax, sd, n10, n50, top in stats[:10]:
            tops = ", ".join(f"{v:+.0f}@{date_of(r)}" for r, v in top[:3])
            print(f"    {mx:7.1f} {sd:7.3f} {n10:6d} {n50:6d}  {nm:32s} {tops}")

    # ── 4. cross-reference worst bars with design values ──────────────────────
    print("\nCROSS-REFERENCE: design values at the worst bars (|x|>5 shown)")
    for arm, fam in (("a_bucket_market_ew", "ewstock"), ("a_bucket_market_vw", "vwstock")):
        d = data[arm]
        diff = d["loss"] - a0["loss"]
        fin = np.isfinite(diff)
        order = np.argsort(np.where(fin, diff, -np.inf))[::-1][:10]
        cols = [
            (j, nm) for j, nm in enumerate(names)
            if (m := val_re.match(nm)) and fam in m.group(1)
            and not any(x in m.group(1) for x in ("turnover", "spread", "ofi"))
        ]
        print(f"  {arm}:")
        for k in order:
            r = int(d["row_id"][k])
            xr = np.asarray(X[r, :], dtype=np.float64)
            hits = [(nm, xr[j]) for j, nm in cols if abs(xr[j]) > 5]
            hits.sort(key=lambda z: -abs(z[1]))
            hs = ", ".join(f"{nm}={v:+.1f}" for nm, v in hits[:6]) or "(none >5)"
            print(f"    {date_of(r)} dl={diff[k]:8.2f} yh={d['yhat'][k]:+8.3f}: {hs}")

    # ── 5. REFIT DIAGNOSIS of the c057 explosion (coefficient pathology) ──────
    # Rebuild the exact OLS design (backbone + bucket, dedup per the persisted
    # dropped-cols disclosure), fit the 24000-bar window ending at an explosion
    # bar two ways — the executor's gram path and stable lstsq — and decompose
    # yhat into per-column contributions. Contrast with a benign window.
    W = 24000
    ind_re = re.compile(r"^(.+)_(avail|active)_ma_(\d+)$")
    har_re = re.compile(r"^har_ma_(\d+)$")
    reg_re = re.compile(r"^har_ma_(\d+)_x_(open|close)$")

    def classify(nm):
        if reg_re.match(nm):
            return "regime"
        if har_re.match(nm):
            return "har"
        m = ind_re.match(nm)
        if m:
            return "indicator", m.group(1)
        m = val_re.match(nm)
        if m:
            return "value", m.group(1)
        return "calendar"

    STEMS = {
        "a_bucket_market_ew": {f"{s}_ewstock" for s in
                               ("sumret2", "sumret3", "sumret4", "sumabsret",
                                "sumbipow", "sumpret2")},
        "a_bucket_market_vw": {f"{s}_vwstock" for s in
                               ("sumret2", "sumret3", "sumret4", "sumabsret",
                                "sumbipow", "sumpret2")},
    }

    def design_cols(arm):
        stems = STEMS[arm]
        backbone = [j for j, nm in enumerate(names)
                    if classify(nm) in ("har", "regime", "calendar")]
        bucket = [j for j, nm in enumerate(names)
                  if isinstance(c := classify(nm), tuple) and c[1] in stems]
        with np.load(os.path.join(ROOT, arm, "chunk_057.npz"),
                     allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        dropped = set(meta["ols_dropped_cols"])
        cols = [j for j in backbone + bucket if names[j] not in dropped]
        return cols, [names[j] for j in cols]

    def refit(arm, r):
        cols, cn = design_cols(arm)
        Xw = np.asarray(X[r - W:r, cols], dtype=np.float64)
        xt = np.asarray(X[r, cols], dtype=np.float64)
        yw = np.load(CACHE, allow_pickle=True)["y"][r - W:r].astype(np.float64)
        live = Xw.max(0) != Xw.min(0)  # window identifiability mask
        Xl, xl = Xw[:, live], xt[live]
        mu = Xl.mean(0)
        Xc = Xl - mu
        yc = yw - yw.mean()
        # (a) executor path: normal equations
        gram = Xc.T @ Xc
        sv = np.linalg.svd(Xc, compute_uv=False)
        try:
            beta_g = np.linalg.solve(gram, Xc.T @ yc)
        except np.linalg.LinAlgError:
            beta_g = None
        # (b) stable path
        beta_l = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        ln = [nm for nm, lv in zip(cn, live) if lv]
        out = {}
        for tag, b in (("gram", beta_g), ("lstsq", beta_l)):
            if b is None:
                out[tag] = None
                continue
            yh = float((xl - mu) @ b + yw.mean())
            contrib = (xl - mu) * b
            top = np.argsort(-np.abs(contrib))[:8]
            out[tag] = (yh, b, contrib, top)
        print(f"\n  {arm} @ row {r} ({date_of(r)}):  design {Xl.shape[1]} live cols, "
              f"cond(X)={sv[0] / sv[-1]:.2e}  cond(gram)={(sv[0] / sv[-1]) ** 2:.2e}")
        for tag in ("gram", "lstsq"):
            if out[tag] is None:
                print(f"    [{tag}] singular")
                continue
            yh, b, contrib, top = out[tag]
            print(f"    [{tag}] yhat={yh:+.3f}  max|coef|={np.abs(b).max():.3e}  "
                  f"sum|coef|={np.abs(b).sum():.3e}")
            for j in top:
                print(f"       {ln[j]:34s} coef={b[j]:+12.4e} x-mu={xl[j] - mu[j]:+9.3f} "
                      f"contrib={contrib[j]:+10.3f}")
        # near-null-space: smallest right singular vector of the centered design
        _, svals, Vt = np.linalg.svd(Xc, full_matrices=False)
        v = Vt[-1]
        top = np.argsort(-np.abs(v))[:8]
        print(f"    smallest singular value {svals[-1]:.3e}; null-vector weights:")
        for j in top:
            print(f"       {ln[j]:34s} {v[j]:+.4f}")
        del Xw, Xc, Xl
        return sv[0] / sv[-1]

    print("\nREFIT DIAGNOSIS (explosion window vs benign window)")
    refit("a_bucket_market_ew", 150144)   # 2017-01-16 explosion bar
    refit("a_bucket_market_vw", 150112)   # 2017-01-15 explosion bar
    refit("a_bucket_market_ew", 160000)   # benign contrast (mid-2018)
    refit("a_bucket_market_vw", 160000)

    # conditioning trajectory: cond of the ew design window at each chunk start
    print("\n  cond(X_window) at each chunk start (ew design), chunks 50..65:")
    cols, cn = design_cols("a_bucket_market_ew")
    for c in range(50, 66):
        r0 = a0["per_chunk"][c][0]
        # first row_id of chunk c
        sizes = [x[5] for x in a0["per_chunk"]]
        r = int(a0["row_id"][int(np.cumsum([0] + sizes)[c])])
        Xw = np.asarray(X[r - W:r, cols], dtype=np.float64)
        live = Xw.max(0) != Xw.min(0)
        Xc = Xw[:, live] - Xw[:, live].mean(0)
        sv = np.linalg.svd(Xc, compute_uv=False)
        print(f"    c{c:03d} row {r} {date_of(r)}: cond={sv[0] / sv[-1]:.3e} "
              f"smin={sv[-1]:.3e} live={int(live.sum())}")
        del Xw, Xc

    # ── 6. feed-regime scan: raw print pattern of the stock aggregates ────────
    print("\nFEED REGIME (raw parquet print pattern by month, sumret2 stem):")
    import pandas as pd
    for pq, col in (("data/ewstock_stats.parquet", "sumret2_ewstock"),
                    ("data/vwstock_stats.parquet", "sumret2_vwstock")):
        df = pd.read_parquet(os.path.join(_ROOT, pq))
        tcol = "endbartime" if "endbartime" in df.columns else df.columns[0]
        t = pd.to_datetime(df[tcol])
        ok = df[col].notna()
        sub = (t >= "2014-01-01") & (t < "2018-01-01")
        bym = ok[sub].groupby(t[sub].dt.to_period("M")).agg(["sum", "count"])
        print(f"  {pq} [{col}] monthly printed/total:")
        line = "   "
        for p, row in bym.iterrows():
            line += f" {p}:{int(row['sum'])}/{int(row['count'])}"
            if len(line) > 150:
                print(line); line = "   "
        print(line)
        # time-of-day print pattern before/after the c057 window midpoint
        for lab, m in (("2016H1", (t >= "2016-01-01") & (t < "2016-07-01")),
                       ("2016H2", (t >= "2016-07-01") & (t < "2017-01-01")),
                       ("2017H1", (t >= "2017-01-01") & (t < "2017-07-01"))):
            frac = ok[m].mean() if m.any() else float("nan")
            n_hours = t[m & ok].dt.hour.nunique() if (m & ok).any() else 0
            print(f"    {lab}: print-frac {frac:.3f}  distinct print hours {n_hours}")
        del df


if __name__ == "__main__":
    main()
