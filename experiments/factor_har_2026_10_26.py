"""Factor-augmented HAR on Table 1's design: results/channels_2026-10-26/RULE.txt section 4 (2026-10-26).

QUESTION  On ordinary (non-FOMC-statement) days the full predictor set beats the HAR benchmark by a few percent of QLIKE;
          the ablations suggest that many correlated proxies of one shared volatility signal carry the gain.  The
          diffusion-index test (Stock and Watson 2002): OLS on the HAR set plus the first k principal components of the
          non-FOMC exogenous value averages (k = 1, 2, 3, 5, 10, 20), and a supervised variant with partial least squares
          components (k = 1, 2, 3).  R_k = (L_HAR - L_k) / (L_HAR - L_full): the share of the full set's gain that k
          components recover.

DESIGN    Table 1's (D13; results/window_sweep_2014start_fomcfix/w3000/run_info.json): the 13 regular-hours bars pooled,
          base-5 HAR ladder on the 48-interval grid, rolling window W = 39,000 rows (3,000 days); forecasts from the
          250-solve boundary at or before the first bar of 2 January 2014 (the walk start of Table 1's linear walks) to
          30 April 2024.
  HAR set   har_ma_1 .. har_ma_3125, DOW_0 .. DOW_4, hour (Table 1's 12-column baseline design).
  Block     the value averages adj_<v>_ma_<k>, k = 1 .. 3125, of the 43 exogenous series outside the FOMC group (258
            columns); no availability averages, no HAR or calendar column.
  Map       once a trading day, from the window ending at the bar before the day's first bar (= that bar's estimation
            window): the block's window means m and standard deviations s (ddof 0); the eigenvectors V of the active
            columns' correlation matrix (descending eigenvalues) -> PCA scores f = V_k' diag(1/s) (z - m); the SIMPLS
            weights R of the standardized block against the target on the same window (its cross-products) -> PLS
            scores R_k' diag(1/s) (z - m).  The map (standardization and loadings together) is held for the day's bars.
  Active    a block column enters a day's map when its window s.d. exceeds STD_MIN = 1e-12 and its series' k = 1 column
            has at least 250 rows off its window mode: Table 1's entry rule for window-standardized columns
            (experiments/pool_linear_pretune.py STD_MIN, STD_MIN_EVIDENCE, evid_ref; RULE_std.txt addenda 1-2).  The
            RULE fixes no rule for a column that cannot be standardized (constant in the window: the late-start
            warm-up zeros); this one was chosen before any scoring.  --entry sd keeps the s.d. condition only (a
            sensitivity variant, labelled in every output).
  Fit       min-norm OLS on [HAR set, first k scores] with a free intercept, re-estimated at every bar on the bar's
            39,000-row window exactly as Table 1's OLS-HAR (pool_linear_pretune.make_ols_class): eigh pseudo-inverse of
            the centred Gram (eigenvalues below 1e-15 x the largest dropped) times the centred cross-product.  The
            window sums of [HAR, block] and y roll rank-one every bar and are rebuilt exactly every 250 solves from the
            walk start (Table 1's schedule); the centred Gram of [HAR, scores] is Q' G Q of the rolled centred Gram G,
            Q = blockdiag(I_12, diag(1/s) [V | R]), so no 39,000 x 258 score matrix is formed at any bar.  k = 0 is
            Table 1's OLS-HAR.

GATES (before any scoring; gates.json)
  G1  k = 0 reproduces Table 1's OLS-HAR pred_adj (window_sweep_2014start_fomcfix/w3000 forecasts.parquet, ols /
      baseline) to 1e-10 relative on every bar, and the target equals its true_adj.
  G2  on six spread-out days (their first and last bar), the PCA k = 3 and PLS k = 3 forecasts of the walk equal a
      direct np.linalg.lstsq on materialized standardized scores (np.linalg.eigh of the materialized window's
      correlation matrix; NIPALS PLS1 on the materialized block; the evidence rule through pool_linear_pretune.col_mode)
      within 1e-8 relative.
  G3  the scored set is Table 1's 33,527 bars (2,579 sessions with all 13 bars; experiments/ablate_score.py's
      evaluation_days and scored); EVENT = the 81 scheduled FOMC statement days of the span (data/releases.parquet
      'fomc release' + loading.FOMC_MISSING_STATEMENTS - loading.UNSCHEDULED_FOMC_DAYS), every one a scored session.

SCORING   QLIKE (plain = pred_adj^2 B against the bar's rv_actual) and MSE on ORDINARY days (primary), every bar and EVENT
          days; Diebold-Mariano (src/evaluation/diebold_mariano.dm_test: HLN, daily mean losses, HAC lag 8, two-sided)
          against Table 1's OLS-HAR and against the full set (the lower-QLIKE of Table 1's patched full-set ridge and
          elastic net on the span: L_full); R_k on each span; the RULE's reading on ORDINARY days.

usage (repository root, the 285J environment):
  python experiments/factor_har_2026_10_26.py design --out D     Table 1's design D13 rebuilt: 12 column parts D/p01 ..
                                                                 D/p12 through pool_linear_pretune.py design --part K/12
                                                                 (one at a time: laptop memory); the merged matrix is not
                                                                 written (disk), its md5 is streamed in merge's column
                                                                 order and compared with the CARC build
                                                                 (results/fomcfix_2026-10-26/gates/G3.json) -> D/design_check.json
  python experiments/factor_har_2026_10_26.py walk --design D [--entry table1|sd]
  python experiments/factor_har_2026_10_26.py gates --design D [--entry table1|sd]
  python experiments/factor_har_2026_10_26.py score
Outputs: results/channels_2026-10-26/factor/ (forecasts.parquet, factor_days.parquet, walk.json, gates.json, scores.csv,
reading.json, SUMMARY.md; the --entry sd variant: forecasts_entry_sd.parquet, factor_days_entry_sd.parquet,
walk_entry_sd.json).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_arms as AA  # noqa: E402

OUT = ROOT / "results" / "channels_2026-10-26" / "factor"
WS = ROOT / "results" / "window_sweep_2014start_fomcfix" / "w3000"
WS_FCST = WS / "forecasts.parquet"
G3_JSON = ROOT / "results" / "fomcfix_2026-10-26" / "gates" / "G3.json"
PLP_PY = HERE / "pool_linear_pretune.py"
W = 39_000
TUNE_PER = 250  # Table 1's exact rebuild of the window sums (reinit_every)
FIRST_DAY = pd.Timestamp("2014-01-02")
PINV_RCOND = 1e-15  # pool_linear_pretune.PINV_RCOND (numpy.linalg.pinv's default)
STD_MIN = 1e-12  # pool_linear_pretune.STD_MIN
STD_MIN_EVIDENCE = 250  # pool_linear_pretune.STD_MIN_EVIDENCE
K_PCA = (1, 2, 3, 5, 10, 20)
K_PLS = (1, 2, 3)
NPC, NPLS = max(K_PCA), max(K_PLS)
MODELS = ["k0"] + [f"pca{k}" for k in K_PCA] + [f"pls{k}" for k in K_PLS]
N_PARTS = 12
G1_TOL, G2_TOL = 1e-10, 1e-8
N_SCORED, N_DAYS, N_EVENT = 33_527, 2_579, 81
HAC_LAG = 8
SPANS = ("ordinary", "all", "event")
GROUPS7 = [g for g in AA.GROUPS if g != "fomc"]


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(p: Path, lf: bool = False) -> str:
    b = p.read_bytes()
    return hashlib.md5(b.replace(b"\r\n", b"\n") if lf else b).hexdigest()


def sfx(entry: str) -> str:
    return "" if entry == "table1" else f"_entry_{entry}"


# ============================================================================ design
def run_cfg() -> list[str]:
    return list(json.loads((WS / "run_info.json").read_text())["runner_cfg"])


_EXOG = re.compile(r"_ma_\d+$")  # pool_linear_pretune._EXOG


class Parts:
    """The column parts of pool_linear_pretune.py design --part K/N, read as the merged design (pool_linear_pretune
    merge's column order) without writing the merged matrix (2026-10-26: the laptop's disk was full)."""

    def __init__(self, d: Path):
        self.parts = sorted(p for p in d.glob("p[0-9][0-9]") if (p / "design.json").is_file())
        self.metas = [json.loads((p / "design.json").read_text()) for p in self.parts]
        assert [m["part"] for m in self.metas] == [[k + 1, N_PARTS] for k in range(N_PARTS)], "parts 1 .. N"
        assert all(m["config"] == self.metas[0]["config"] and m["bucket"] == "all_features" for m in self.metas)
        self.y = np.load(self.parts[0] / "y.npy")
        self.dates_bytes = (self.parts[0] / "dates_oos.csv").read_bytes()
        for p in self.parts[1:]:  # rows and target are the parts' common columns (as merge asserts)
            assert np.array_equal(np.load(p / "y.npy"), self.y)
            assert np.array_equal(np.load(p / "b_oos.npy"), np.load(self.parts[0] / "b_oos.npy"))
            assert (p / "dates_oos.csv").read_bytes() == self.dates_bytes
        self.src: dict[str, tuple[int, int]] = {}
        for i, m in enumerate(self.metas):
            for j, nm in enumerate(m["names"]):
                self.src.setdefault(nm, (i, j))
        self.X = [np.load(p / "X.npy", mmap_mode="r") for p in self.parts]
        self.n = len(self.y)

    def order(self) -> list[str]:
        """pool_linear_pretune.merge's column order: target HAR, every value ladder, every availability ladder (every
        activity ladder), calendar."""
        raw = [c for m in self.metas for c in m["raw_cols"]]
        common = [nm for nm in self.metas[0]["names"] if not _EXOG.search(nm) or nm.startswith("har_ma_")]
        har = [nm for nm in common if nm.startswith("har_ma_")]
        cal = [nm for nm in common if not nm.startswith("har_ma_")]
        lad = [int(nm.rsplit("_", 1)[1]) for nm in har]
        exog = {nm for m in self.metas for nm in m["names"] if nm not in common}
        out = list(har)
        for f in ("adj_{}", "{}_avail", "{}_active"):
            out += [f"{f.format(c)}_ma_{k}" for c in raw for k in lad if f"{f.format(c)}_ma_{k}" in exog]
        out += cal
        assert set(out) == exog | set(common) and len(out) == len(set(out)), "column order"
        return out

    def common_equal(self) -> bool:
        """The HAR and calendar columns are identical in every part (as merge asserts)."""
        common = [nm for nm in self.metas[0]["names"] if not _EXOG.search(nm) or nm.startswith("har_ma_")]
        return all(
            np.array_equal(
                self.X[i][:, self.metas[i]["names"].index(nm)], self.X[0][:, self.metas[0]["names"].index(nm)]
            )
            for i in range(1, len(self.parts))
            for nm in common
        )

    def cols(self, names: list[str]) -> np.ndarray:
        out = np.empty((self.n, len(names)), dtype=np.float64)
        for c, nm in enumerate(names):
            i, j = self.src[nm]
            out[:, c] = self.X[i][:, j]
        return out

    def md5_merged(self) -> str:
        """md5 of the X.npy that merge would write (F order: np.save's header, then the columns in turn), streamed."""
        order = self.order()
        hdr = {"descr": "<f8", "fortran_order": True, "shape": (self.n, len(order))}
        return _stream_md5(hdr, (self.X[self.src[nm][0]][:, self.src[nm][1]] for nm in order))


def _stream_md5(hdr: dict, columns) -> str:
    """md5 of np.save's file for an F-ordered float64 matrix: its own header writer, then each column's bytes."""
    import io

    from numpy.lib import format as npf

    buf = io.BytesIO()
    npf._write_array_header(buf, hdr, None)  # type: ignore[attr-defined]
    h = hashlib.md5(buf.getvalue())
    for col in columns:
        h.update(np.ascontiguousarray(col, dtype="<f8").tobytes())
    return h.hexdigest()


def cmd_design(a) -> int:
    """Table 1's design D13 rebuilt with the repository's own builder, one column part at a time (memory); the merged
    matrix is not written (the parts are read in its order: the laptop's disk was full), its md5 is streamed and
    compared with the CARC build of the patched design."""
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = run_cfg()
    log = []
    for k in range(1, N_PARTS + 1):
        d = out / f"p{k:02d}"
        cmd = [sys.executable, str(PLP_PY), "design", "--bucket", "all_features", "--part", f"{k}/{N_PARTS}"]
        cmd += ["--out", str(d), *cfg]
        if not (d / "design.json").is_file():
            t0 = time.time()
            say(f"design part {k}/{N_PARTS}")
            subprocess.run(cmd, check=True, cwd=ROOT)
            log.append(dict(part=k, sec=round(time.time() - t0, 1)))
    P = Parts(out)
    X0 = P.X[0]
    ok_stream = md5(P.parts[0] / "X.npy") == _stream_md5(
        {"descr": "<f8", "fortran_order": True, "shape": X0.shape}, (X0[:, j] for j in range(X0.shape[1]))
    )
    ref = json.loads(G3_JSON.read_text())["all_features"]["md5_new"]
    got = {"X.npy": P.md5_merged(), "y.npy": md5(P.parts[0] / "y.npy"), "b_oos.npy": md5(P.parts[0] / "b_oos.npy")}
    got["dates_oos.csv"] = md5(P.parts[0] / "dates_oos.csv", lf=True)  # pandas writes CRLF on Windows
    carc_names = json.loads((WS / "frozen_penalties.json").read_text())["designs"]["all_features"]["columns"]
    # the HAR set alone = Table 1's 12-column baseline design (its own cache, the same column order)
    har_md5 = _stream_md5(
        {"descr": "<f8", "fortran_order": True, "shape": (P.n, 12)},
        (P.X[P.src[nm][0]][:, P.src[nm][1]] for nm in AA.HAR12),
    )
    har_ref = json.loads(G3_JSON.read_text())["baseline"]["md5_new"]["X.npy"]
    rec = dict(
        design=str(out),
        command=f"pool_linear_pretune.py design --bucket all_features --part K/{N_PARTS} <run_info runner_cfg>",
        runner_cfg=cfg,
        merged_written=False,
        merged_order_eq_table1_columns=P.order() == carc_names,
        common_columns_equal_across_parts=P.common_equal(),
        stream_md5_checked_on_part1=ok_stream,
        md5=got,
        md5_carc_patched=ref,
        md5_equal={f: got[f] == ref[f] for f in ref},
        har12_md5=har_md5,
        har12_md5_eq_carc_baseline_X=har_md5 == har_ref,
        parts_built_now=log,
        host=platform.node(),
        numpy=np.__version__,
    )
    (out / "design_check.json").write_text(json.dumps(rec, indent=1))
    say(f"design: {out}; md5 equal to the CARC patched build: {rec['md5_equal']}")
    return 0


# ============================================================================ the walk
def block_names(names: list[str], gs: dict[str, list[str]]) -> list[str]:
    fomc = set(gs["fomc"])
    out = []
    for n in names:
        m = re.match(r"^adj_(.+)_ma_(\d+)$", n)
        if m and not n.startswith("har_ma_") and m.group(1) not in fomc:
            out.append(n)
    return out


def series_group(gs: dict[str, list[str]]) -> dict[str, str]:
    return {s: g for g, ss in gs.items() for s in ss}


def load_design(design: Path) -> tuple[dict, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """(meta, X, y, dates): a merged design cache (X memory-mapped), or a directory of the design --part K/N parts
    (cmd_design), of which only the HAR set and the block are read, into memory, with meta["names"] = those columns."""
    if (design / "X.npy").is_file():
        meta = json.loads((design / "design.json").read_text())
        X = np.load(design / "X.npy", mmap_mode="r")
        y = np.load(design / "y.npy")
        dfile = design / "dates_oos.csv"
    else:
        P = Parts(design)
        names = list(AA.HAR12) + block_names(P.order(), AA.group_series())
        meta = dict(P.metas[0], names=names, part=None, parts=[str(p) for p in P.parts])
        X, y, dfile = P.cols(names), P.y, P.parts[0] / "dates_oos.csv"
    assert int(meta["config"]["window"]) == W and meta["config"]["periods_per_day"] == 13, meta["config"]
    d = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(dfile, dtype=str)["date"]))
    assert X.shape[0] == W + len(d) == len(y), (X.shape, len(d), len(y))
    return meta, X, y, d


def walk_start(dates: pd.DatetimeIndex) -> int:
    """Table 1's walk start (experiments/pooled_linear_rerun_2026_10_19_13.walk_start): the 250-solve boundary at or
    before the first bar of FIRST_DAY, a design row."""
    first_row = W + int(np.flatnonzero(dates >= FIRST_DAY)[0])
    return (first_row - W) // TUNE_PER * TUNE_PER + W


def simpls(C: np.ndarray, s: np.ndarray, K: int) -> np.ndarray:
    """SIMPLS weights (de Jong 1993) of one response from the cross-products: C = the standardized block's correlation
    matrix (Z'Z / n), s = Z'y / n; returns R (p x K), the scores T = Z R (unit variance, mutually uncorrelated)."""
    p = len(s)
    R = np.zeros((p, K))
    V = np.zeros((p, K))
    s = s.copy()
    for a in range(K):
        r = s.copy()
        r /= np.sqrt(float(r @ C @ r))
        pl = C @ r
        v = pl - V[:, :a] @ (V[:, :a].T @ pl)
        v /= np.linalg.norm(v)
        s = s - v * float(v @ s)
        R[:, a], V[:, a] = r, v
    return R


class Evidence:
    """Off-mode row counts of the k = 1 value columns over a window, exactly (= pool_linear_pretune.col_mode's
    n - count of the modal value) where they can fall below STD_MIN_EVIDENCE: a value can be a window's mode with fewer
    than STD_MIN_EVIDENCE rows off it only if it fills more than half the window, so only values whose count over the
    whole chunk exceeds W - STD_MIN_EVIDENCE are tracked (cumulative counts); every other window has >= W / 2 off-mode
    rows."""

    def __init__(self, A1: np.ndarray):
        self.cands: list[list[np.ndarray]] = []
        for j in range(A1.shape[1]):
            vals, cnt = np.unique(A1[:, j], return_counts=True)
            big = vals[cnt > W - STD_MIN_EVIDENCE]
            self.cands.append([np.concatenate([[0], np.cumsum(A1[:, j] == v)]) for v in big])

    def off_mode(self, t: int) -> np.ndarray:
        """Off-mode counts over the chunk rows [t - W, t)."""
        out = np.full(len(self.cands), W // 2, dtype=np.int64)
        for j, cs in enumerate(self.cands):
            for c in cs:
                n_v = int(c[t] - c[t - W])
                if n_v > W // 2:
                    out[j] = min(out[j], W - n_v)
        return out


def day_map(S, c, sx, sy, ev_ok: np.ndarray | None, p: int) -> tuple[np.ndarray, dict]:
    """The day's map from the window sums (rows [t - W, t)): P (p x (NPC + NPLS)) with scores = P' (z - m) up to a
    constant, and its diagnostics."""
    n = float(W)
    mZ = sx[12:] / n
    G = S[12:, 12:] - n * np.outer(mZ, mZ)
    sd = np.sqrt(np.maximum(np.diag(G) / n, 0.0))
    act = sd > STD_MIN
    if ev_ok is not None:
        act &= ev_ok
    a = np.flatnonzero(act)
    sa = sd[a]
    R = G[np.ix_(a, a)] / (n * np.outer(sa, sa))
    R = 0.5 * (R + R.T)
    lam, V = np.linalg.eigh(R)
    lam, V = lam[::-1], V[:, ::-1]
    V = V[:, :NPC] * np.where(V[:, :NPC].sum(axis=0) < 0, -1.0, 1.0)  # sign: positive sum (reporting only)
    gy = c[12:][a] - n * mZ[a] * (sy / n)
    Rp = simpls(R, gy / (n * sa), NPLS)
    P = np.zeros((p, NPC + NPLS))
    P[a, :NPC] = V / sa[:, None]
    P[a, NPC:] = Rp / sa[:, None]
    r1 = Rp[:, 0] / np.linalg.norm(Rp[:, 0])
    r1 = r1 * (1.0 if r1.sum() >= 0 else -1.0)
    diag = dict(n_active=len(a), trace=float(lam.sum()), lam=lam[:NPC].copy(), act=act, v1=V[:, 0].copy(), r1=r1)
    return P, diag


def model_index() -> list[np.ndarray]:
    ix = [np.arange(12)]
    ix += [np.r_[np.arange(12), 12 + np.arange(k)] for k in K_PCA]
    ix += [np.r_[np.arange(12), 12 + NPC + np.arange(k)] for k in K_PLS]
    return ix


def minnorm_pred(G: np.ndarray, c: np.ndarray, x: np.ndarray, ybar: float) -> tuple[float, int]:
    """Table 1's min-norm OLS on centred sums (make_ols_class.solve), predicted at the centred row x."""
    lam, V = np.linalg.eigh(0.5 * (G + G.T))
    keep = lam > PINV_RCOND * max(float(lam.max()), 0.0)
    Vk = V[:, keep]
    b = Vk @ ((Vk.T @ c) / lam[keep])
    return ybar + float(x @ b), int(keep.sum())


def cmd_walk(a) -> int:
    t_wall, t_cpu = time.time(), time.process_time()
    design = Path(a.design).resolve()
    meta, X, y, dates = load_design(design)
    names = meta["names"]
    gs = AA.group_series()
    sg = series_group(gs)
    Z = block_names(names, gs)
    assert len(Z) == 258 and len({re.match(r"^adj_(.+)_ma_\d+$", z).group(1) for z in Z}) == 43, len(Z)  # type: ignore[union-attr]
    zser = [re.match(r"^adj_(.+)_ma_\d+$", z).group(1) for z in Z]  # type: ignore[union-attr]
    zgrp = np.array([sg[s] for s in zser])
    k1 = {s: Z.index(f"adj_{s}_ma_1") for s in dict.fromkeys(zser)}
    ser_list = list(k1)
    zk1 = np.array([ser_list.index(s) for s in zser])  # each block column -> its series' k = 1 column
    hi = [names.index(cn) for cn in AA.HAR12]
    zi = [names.index(cn) for cn in Z]
    n_rows = X.shape[0]
    S0 = walk_start(dates)
    lo = S0 - W
    p = len(Z)
    say(f"design {design} {X.shape}; walk start row {S0} ({dates[S0 - W]}); block {p} columns; entry {a.entry}")
    A = np.empty((n_rows - lo, 12 + p), dtype=np.float64)
    A[:, :12] = X[lo:, hi]
    A[:, 12:] = X[lo:, zi]
    yc = np.ascontiguousarray(y[lo:], dtype=np.float64)
    nt = n_rows - S0
    fdates = dates[S0 - W :]
    assert len(fdates) == nt
    fday = fdates.normalize()
    ev = Evidence(A[:, 12 + np.array([k1[s] for s in ser_list])]) if a.entry == "table1" else None
    ixs = model_index()
    preds = np.empty((nt, len(MODELS)))
    ranks = np.empty((nt, len(MODELS)), dtype=np.int16)
    days: list[dict] = []
    P = np.zeros((p, NPC + NPLS))
    S = c = sx = None
    sy = 0.0
    G35 = np.empty((12 + NPC + NPLS, 12 + NPC + NPLS))
    c35 = np.empty(12 + NPC + NPLS)
    x35 = np.empty(12 + NPC + NPLS)
    n = float(W)
    t_map = 0.0
    for i in range(nt):
        t = W + i
        if i % TUNE_PER == 0:
            Aw, yw = A[t - W : t], yc[t - W : t]
            S, c, sx, sy = Aw.T @ Aw, Aw.T @ yw, Aw.sum(axis=0), float(yw.sum())
        else:
            u, v = A[t - 1], A[t - 1 - W]
            S += np.outer(u, u) - np.outer(v, v)
            c += u * float(yc[t - 1]) - v * float(yc[t - 1 - W])
            sx += u - v
            sy += float(yc[t - 1]) - float(yc[t - 1 - W])
        if i == 0 or fday[i] != fday[i - 1]:
            t0 = time.perf_counter()
            ev_ok = None
            if ev is not None:
                ev_ok = ev.off_mode(t)[zk1] >= STD_MIN_EVIDENCE
            P, dg = day_map(S, c, sx, sy, ev_ok, p)
            rec = dict(day=fday[i], first_row=S0 + i, n_active=dg["n_active"], trace=dg["trace"])
            rec.update({f"lam{j + 1}": float(dg["lam"][j]) for j in range(NPC)})
            act = dg["act"]
            for g in GROUPS7:
                gm = zgrp[act] == g
                rec[f"n_{g}"] = int(gm.sum())
                rec[f"pc1_abs_{g}"] = float(np.abs(dg["v1"][gm]).sum())
                rec[f"pc1_sq_{g}"] = float((dg["v1"][gm] ** 2).sum())
                rec[f"pls1_abs_{g}"] = float(np.abs(dg["r1"][gm]).sum())
                rec[f"pls1_sq_{g}"] = float((dg["r1"][gm] ** 2).sum())
            days.append(rec)
            t_map += time.perf_counter() - t0
        m = sx / n
        ybar = sy / n
        Gc = S - n * np.outer(m, m)
        cc = c - n * m * ybar
        d = A[t] - m
        GzP = Gc[12:, 12:] @ P
        G35[:12, :12] = Gc[:12, :12]
        G35[:12, 12:] = Gc[:12, 12:] @ P
        G35[12:, :12] = G35[:12, 12:].T
        G35[12:, 12:] = P.T @ GzP
        c35[:12], c35[12:] = cc[:12], P.T @ cc[12:]
        x35[:12], x35[12:] = d[:12], P.T @ d[12:]
        for j, ix in enumerate(ixs):
            preds[i, j], ranks[i, j] = minnorm_pred(G35[np.ix_(ix, ix)], c35[ix], x35[ix], ybar)
        if i % 5000 == 0:
            say(f"  bar {i:,} / {nt:,} ({fdates[i]}); {len(days)} maps")
    sec = time.time() - t_wall
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.DataFrame({"date": fdates, "design_row": np.arange(S0, n_rows), "true_adj": yc[W:]})
    for j, mname in enumerate(MODELS):
        f[mname] = preds[:, j]
    f.to_parquet(OUT / f"forecasts{sfx(a.entry)}.parquet", index=False)
    pd.DataFrame(days).to_parquet(OUT / f"factor_days{sfx(a.entry)}.parquet", index=False)
    # the five weekday dummies sum to the intercept on the 13 regular-hours bars: every centred Gram has exactly one
    # null direction, handled by the min-norm solve as in Table 1's OLS-HAR (rank = 11 + k)
    rank_ok = bool((ranks == np.array([len(ix) - 1 for ix in ixs])[None, :]).all())
    info = dict(
        entry=a.entry,
        design=str(design),
        design_shape=list(X.shape),
        walk_start_row=S0,
        first=str(fdates[0]),
        last=str(fdates[-1]),
        bars=nt,
        maps=len(days),
        block_columns=p,
        block_series=len(ser_list),
        models=MODELS,
        rank_11_plus_k_every_bar=rank_ok,
        min_rank={mname: int(ranks[:, j].min()) for j, mname in enumerate(MODELS)},
        max_rank={mname: int(ranks[:, j].max()) for j, mname in enumerate(MODELS)},
        wall_sec=round(sec, 1),
        cpu_sec=round(time.process_time() - t_cpu, 1),
        map_sec=round(t_map, 1),
        host=platform.node(),
        python=platform.python_version(),
        numpy=np.__version__,
        threads="1 (OMP / OpenBLAS / MKL_NUM_THREADS=1)",
    )
    (OUT / f"walk{sfx(a.entry)}.json").write_text(json.dumps(info, indent=1))
    say(f"walk done: {json.dumps(info)}")
    return 0


# ============================================================================ gates
def event_days() -> pd.DatetimeIndex:
    from src.data import loading as LD

    r = pd.read_parquet(ROOT / "data" / "releases.parquet", columns=["endbartime", LD.FOMC_RAW_COL])
    t = pd.to_datetime(r.loc[r[LD.FOMC_RAW_COL] > 0, "endbartime"])
    d = pd.DatetimeIndex(t.dt.normalize().unique())
    d = d.union(pd.DatetimeIndex(pd.to_datetime(list(LD.FOMC_MISSING_STATEMENTS))).normalize())
    d = d.difference(pd.DatetimeIndex(sorted(LD.UNSCHEDULED_FOMC_DAYS)))
    return d[(d >= FIRST_DAY) & (d <= pd.Timestamp("2024-04-30"))].sort_values()


def templates() -> pd.DataFrame:
    return pd.read_parquet(
        WS_FCST, columns=["model", "predictor_set", "date", "pred_adj", "true_adj", "B", "plain", "rv_actual"]
    )


def scoring_frame(entry: str) -> tuple[pd.DataFrame, dict]:
    """One row per forecast bar: date, B, rv_actual, the templates' plain forecasts (ols_har, <m>_full, <m>_har) and
    the factor models' plain forecasts."""
    T = templates()
    base = T[(T["model"] == "ols") & (T["predictor_set"] == "baseline")].set_index("date")
    out = pd.DataFrame({"B": base["B"], "rv_actual": base["rv_actual"], "true_adj": base["true_adj"]})
    tag = {"baseline": "har", "all_features": "full"}
    for (m, s), g in T.groupby(["model", "predictor_set"]):
        gi = g.set_index("date")
        assert gi.index.equals(out.index) and np.array_equal(gi["B"].to_numpy(), out["B"].to_numpy()), (m, s)
        out[f"{m}_{tag[s]}"] = gi["plain"].to_numpy()  # Table 1's plain forecast (pred_adj^2 B)
    f = pd.read_parquet(OUT / f"forecasts{sfx(entry)}.parquet").set_index("date")
    assert f.index.equals(out.index), "factor forecasts and Table 1's rows"
    for mname in MODELS:
        out[mname] = f[mname].to_numpy() ** 2 * out["B"].to_numpy()
    return out, dict(f=f, base=base)


def direct_check(X, y, dates, names, Z, f: pd.DataFrame, day: pd.Timestamp, entry: str) -> list[dict]:
    """G2 on one day: the PCA k = 3 and PLS k = 3 forecasts of its first and last bar by direct computation on the
    materialized window."""
    import pool_linear_pretune as PLP

    fd = pd.DatetimeIndex(f["date"]).normalize()
    rows = f["design_row"].to_numpy()[fd == day]
    d0 = int(rows[0])
    zi = [names.index(cn) for cn in Z]
    hi = [names.index(cn) for cn in AA.HAR12]
    Zw = np.asarray(X[d0 - W : d0][:, zi], dtype=np.float64)
    mu, sd = Zw.mean(axis=0), Zw.std(axis=0)
    act = sd > STD_MIN
    if entry == "table1":
        zser = [re.match(r"^adj_(.+)_ma_\d+$", z).group(1) for z in Z]  # type: ignore[union-attr]
        k1 = [Z.index(f"adj_{s}_ma_1") for s in zser]
        _, cnt = PLP.col_mode(Zw)
        act &= (W - cnt)[k1] >= STD_MIN_EVIDENCE
    a = np.flatnonzero(act)
    Zs = (Zw[:, a] - mu[a]) / sd[a]
    del Zw
    lam, V = np.linalg.eigh(Zs.T @ Zs / W)
    V3 = V[:, ::-1][:, :3]
    yw = y[d0 - W : d0]
    # NIPALS PLS1 on the materialized block: T = Zs R
    Xd, yv = Zs.copy(), yw - yw.mean()
    Wl, Pl = [], []
    for _ in range(3):
        w = Xd.T @ yv
        w /= np.linalg.norm(w)
        tt = Xd @ w
        t2 = float(tt @ tt)
        pl = Xd.T @ tt / t2
        yv = yv - float(yv @ tt) / t2 * tt
        Xd -= np.outer(tt, pl)
        Wl.append(w)
        Pl.append(pl)
    del Xd, Zs
    Wm, Pm = np.array(Wl).T, np.array(Pl).T
    Rpls = Wm @ np.linalg.inv(Pm.T @ Wm)
    lam_d = lam[::-1]
    out = []
    for r in (int(rows[0]), int(rows[-1])):
        Zt = (np.asarray(X[r - W : r + 1][:, zi], dtype=np.float64)[:, a] - mu[a]) / sd[a]
        H = np.asarray(X[r - W : r + 1][:, hi], dtype=np.float64)
        yt = y[r - W : r]
        for mname, L in (("pca3", V3), ("pls3", Rpls)):
            F = Zt @ L
            D = np.column_stack([np.ones(W + 1), H, F])
            coef = np.linalg.lstsq(D[:W], yt, rcond=None)[0]
            direct = float(D[W] @ coef)
            walk = float(f.loc[f["design_row"] == r, mname].iloc[0])
            out.append(
                dict(
                    day=str(day.date()),
                    design_row=r,
                    date=str(dates[r - W]),
                    model=mname,
                    walk=walk,
                    direct=direct,
                    rel=abs(walk - direct) / abs(direct),
                    n_active=len(a),
                    lam3_over_lam4=float(lam_d[2] / lam_d[3]),
                )
            )
    return out


def cmd_gates(a) -> int:
    import ablate_score as AS

    rec: dict = {"entry": a.entry, "fail": []}
    design = Path(a.design).resolve()
    meta, X, y, dates = load_design(design)
    names = meta["names"]
    gs = AA.group_series()
    Z = block_names(names, gs)
    f = pd.read_parquet(OUT / f"forecasts{sfx(a.entry)}.parquet")
    for dck in (design / "design_check.json", design.parent / "design_check.json"):
        if dck.is_file():
            rec["design_check"] = json.loads(dck.read_text())
            break
    # G1: k = 0 = Table 1's OLS-HAR, every bar
    T = templates()
    ols = T[(T["model"] == "ols") & (T["predictor_set"] == "baseline")].set_index("date")
    fi = f.set_index("date")
    same_rows = bool(fi.index.equals(ols.index))
    rel = np.abs(fi["k0"].to_numpy() - ols["pred_adj"].to_numpy()) / np.abs(ols["pred_adj"].to_numpy())
    rel_y = np.abs(fi["true_adj"].to_numpy() - ols["true_adj"].to_numpy()) / np.abs(ols["true_adj"].to_numpy())
    g1: dict = dict(
        rows=len(fi),
        same_rows=same_rows,
        first=str(fi.index[0]),
        last=str(fi.index[-1]),
        max_rel_pred=float(rel.max()),
        median_rel_pred=float(np.median(rel)),
        max_rel_true_adj=float(rel_y.max()),
        tol=G1_TOL,
    )
    g1["pass"] = bool(same_rows and g1["max_rel_pred"] <= G1_TOL and g1["max_rel_true_adj"] <= 1e-14)
    rec["G1"] = g1
    if not g1["pass"]:
        rec["fail"].append("G1")
    # G2: direct lstsq on materialized scores, six spread-out days
    fday = pd.DatetimeIndex(f["date"]).normalize()
    ud = pd.DatetimeIndex(fday.unique())
    ud = ud[ud >= FIRST_DAY]
    pick = [ud[int(q * (len(ud) - 1))] for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    covid = ud[ud >= pd.Timestamp("2020-03-16")][0]
    pick = sorted(set(pick) | {covid})
    rows: list[dict] = []
    for day in pick:
        say(f"G2: {day.date()}")
        rows += direct_check(X, y, dates, names, Z, f, day, a.entry)
    mx = max(r["rel"] for r in rows)
    fdy = pd.read_parquet(OUT / f"factor_days{sfx(a.entry)}.parquet").set_index("day")["n_active"]
    act_ok = all(int(fdy.loc[pd.Timestamp(r["day"])]) == r["n_active"] for r in rows)
    rec["G2"] = dict(
        days=[str(d.date()) for d in pick],
        checks=rows,
        max_rel=mx,
        tol=G2_TOL,
        n_active_walk_eq_direct=act_ok,
        **{"pass": bool(mx <= G2_TOL and act_ok)},
    )
    if not rec["G2"]["pass"]:
        rec["fail"].append("G2")
    # G3: the scored set and the EVENT days
    fr, _ = scoring_frame(a.entry)
    rv = AS.core_rv()
    days = AS.evaluation_days(rv)
    tcols = [c for c in fr.columns if c.endswith("_har") or c.endswith("_full")]
    panel_t = {c: fr[c] for c in tcols}
    panel = dict(panel_t, **{mname: fr[mname] for mname in MODELS})
    idx_t = AS.scored(panel_t, rv, days)
    idx = AS.scored(panel, rv, days)
    ev = event_days()
    sd_days = pd.DatetimeIndex(idx.normalize())
    bars_per_day = pd.Series(1, index=sd_days).groupby(level=0).size()
    ev_bars = int(sd_days.isin(ev).sum())
    rva = fr["rv_actual"].reindex(idx).to_numpy()
    g3 = dict(
        n_bars=len(idx),
        n_bars_templates=len(idx_t),
        same_as_templates=bool(idx.equals(idx_t)),
        n_days=int(sd_days.nunique()),
        all_days_13_bars=bool((bars_per_day == 13).all()),
        n_event_days=len(ev),
        event_days_scored=int(pd.DatetimeIndex(ev).isin(bars_per_day.index).sum()),
        event_bars=ev_bars,
        ordinary_days=int(sd_days.nunique()) - len(ev),
        rv_actual_eq_core_sumret2=bool(np.array_equal(rva, rv.reindex(idx).to_numpy())),
        event_days=[str(d.date()) for d in ev],
    )
    g3["pass"] = bool(
        g3["n_bars"] == N_SCORED
        and g3["same_as_templates"]
        and g3["n_days"] == N_DAYS
        and g3["all_days_13_bars"]
        and g3["n_event_days"] == N_EVENT
        and g3["event_days_scored"] == N_EVENT
        and g3["event_bars"] == 13 * N_EVENT
    )
    rec["G3"] = g3
    if not g3["pass"]:
        rec["fail"].append("G3")
    rec["pass"] = not rec["fail"]
    (OUT / f"gates{sfx(a.entry)}.json").write_text(json.dumps(rec, indent=1, default=str))
    say(
        f"gates ({a.entry}): G1 {g1['pass']} (max rel {g1['max_rel_pred']:.2e}), G2 {rec['G2']['pass']} (max rel "
        f"{mx:.2e}), G3 {g3['pass']} ({g3['n_bars']} bars, {g3['n_event_days']} EVENT days); fail {rec['fail']}"
    )
    return 0 if rec["pass"] else 3


# ============================================================================ scores
def cmd_score(a) -> int:
    import ablate_score as AS
    from src.evaluation.diebold_mariano import dm_test

    entries = ["table1", "sd"] if (OUT / "forecasts_entry_sd.parquet").is_file() else ["table1"]
    for e in entries:
        g = json.loads((OUT / f"gates{sfx(e)}.json").read_text())
        assert g["pass"], f"gates of entry {e} failed: {g['fail']} -- no scoring"
    rv = AS.core_rv()
    days = AS.evaluation_days(rv)
    ev = event_days()
    rows = []
    reading: dict = {}
    for e in entries:
        fr, _ = scoring_frame(e)
        tcols = [c for c in fr.columns if c.endswith("_har") or c.endswith("_full")]
        idx = AS.scored({c: fr[c] for c in tcols + MODELS}, rv, days)
        assert len(idx) == N_SCORED
        dday = pd.DatetimeIndex(idx.normalize())
        span_ix = {"all": idx, "event": idx[dday.isin(ev)], "ordinary": idx[~dday.isin(ev)]}
        y_rv = fr["rv_actual"]

        def loss(col: str, ix: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
            pr, yy = fr[col].reindex(ix).to_numpy(float), y_rv.reindex(ix).to_numpy(float)
            r = yy / pr
            return r - np.log(r) - 1.0, (yy - pr) ** 2

        for sp, ix in span_ix.items():
            L = {cn: loss(cn, ix) for cn in ["ols_har", "ridge_full", "enet_full", "lasso_full", *MODELS]}
            q = {cn: float(v[0].mean()) for cn, v in L.items()}
            full = "ridge_full" if q["ridge_full"] <= q["enet_full"] else "enet_full"
            q_har, q_full = q["ols_har"], q[full]
            mse_har = float(L["ols_har"][1].mean())

            def dm(c1: str, c2: str, li: int) -> dict:
                return dm_test(AS.daily(L[c1][li], ix), AS.daily(L[c2][li], ix), hac_lag=HAC_LAG)

            for cn in [*MODELS, "ols_har", "ridge_full", "enet_full", "lasso_full"]:
                meth = re.match(r"^(pca|pls|k)(\d+)$", cn)
                row = dict(
                    entry=e,
                    model=cn,
                    method=("har" if cn == "k0" else meth.group(1)) if meth else cn.split("_")[1],
                    k=int(meth.group(2)) if meth else None,
                    span=sp,
                    n_bars=len(ix),
                    n_days=int(pd.DatetimeIndex(ix.normalize()).nunique()),
                    qlike=q[cn],
                    mse_e10=float(L[cn][1].mean()) * 1e10,
                    qlike_pct_vs_har=100.0 * (q[cn] / q_har - 1.0),
                    mse_pct_vs_har=100.0 * (float(L[cn][1].mean()) / mse_har - 1.0),
                    full_ref=full.split("_")[0],
                    R=(q_har - q[cn]) / (q_har - q_full),
                )
                for li, ln in ((0, "qlike"), (1, "mse")):
                    if cn != "ols_har":
                        r1 = dm(cn, "ols_har", li)
                        row[f"dm_har_{ln}"], row[f"p_har_{ln}"] = float(r1["dm"]), float(r1["p"])
                        row[f"diff_har_{ln}"] = float(r1["mean_diff"])
                    if cn != full:
                        r2 = dm(cn, full, li)
                        row[f"dm_full_{ln}"], row[f"p_full_{ln}"] = float(r2["dm"]), float(r2["p"])
                        row[f"diff_full_{ln}"] = float(r2["mean_diff"])
                        row["T_days"] = int(r2["T"])
                rows.append(row)
        sc = pd.DataFrame([r for r in rows if r["entry"] == e])
        o = sc[sc["span"] == "ordinary"].set_index("model")
        small = [f"pca{k}" for k in K_PCA if k <= 3] + [f"pls{k}" for k in K_PLS if k <= 3]
        shared = [
            m_
            for m_ in small
            if o.loc[m_, "R"] >= 0.75 and not (o.loc[m_, "p_full_qlike"] < 0.05 and o.loc[m_, "diff_full_qlike"] > 0)
        ]
        r10 = float(o.loc["pca10", "R"])
        full_beats_10 = bool(o.loc["pca10", "p_full_qlike"] < 0.05 and o.loc["pca10", "diff_full_qlike"] > 0)
        dense = bool(r10 < 0.5 and full_beats_10)
        word = "SHARED FACTOR" if shared else ("DENSE BEYOND A FEW FACTORS" if dense else "INTERMEDIATE")
        reading[e] = dict(
            reading=word,
            shared_factor_models=shared,
            shared_condition_met=bool(shared),
            dense_condition_met=dense,
            R10=r10,
            full_beats_pca10_significantly=full_beats_10,
            full_ref_ordinary=str(o.loc["k0", "full_ref"]),
            R_ordinary={m_: float(o.loc[m_, "R"]) for m_ in MODELS},
            best_small_k=max(small, key=lambda m_: float(o.loc[m_, "R"])),
        )
    sc_all = pd.DataFrame(rows)
    sc_all.to_csv(OUT / "scores.csv", index=False, float_format="%.17g")
    desc = descriptive(entries, days)
    (OUT / "reading.json").write_text(json.dumps(dict(reading=reading, descriptive=desc), indent=1, default=str))
    summary(sc_all, reading, desc, entries)
    say(f"score: {reading['table1']['reading']}; R (ordinary) {reading['table1']['R_ordinary']}")
    return 0


def descriptive(entries: list[str], days: pd.DatetimeIndex) -> dict:
    out: dict = {}
    for e in entries:
        fd = pd.read_parquet(OUT / f"factor_days{sfx(e)}.parquet")
        fd = fd[pd.DatetimeIndex(fd["day"]).isin(days)]
        lam = fd[[f"lam{j}" for j in range(1, NPC + 1)]].to_numpy()
        share = lam / fd["n_active"].to_numpy()[:, None]
        cum = np.cumsum(share, axis=1)
        out[e] = dict(
            n_days=len(fd),
            n_active_min=int(fd["n_active"].min()),
            n_active_max=int(fd["n_active"].max()),
            share_pc={str(j + 1): float(share[:, j].mean()) for j in range(5)},
            share_cum={str(k): float(cum[:, k - 1].mean()) for k in K_PCA},
            share_cum3_min=float(cum[:, 2].min()),
            share_cum3_max=float(cum[:, 2].max()),
            lam_mean={str(j + 1): float(lam[:, j].mean()) for j in range(5)},
            n_cols={g: float(fd[f"n_{g}"].mean()) for g in GROUPS7},
            pc1_abs={g: float(fd[f"pc1_abs_{g}"].mean()) for g in GROUPS7},
            pc1_sq={g: float(fd[f"pc1_sq_{g}"].mean()) for g in GROUPS7},
            pls1_abs={g: float(fd[f"pls1_abs_{g}"].mean()) for g in GROUPS7},
            pls1_sq={g: float(fd[f"pls1_sq_{g}"].mean()) for g in GROUPS7},
        )
    return out


def summary(sc: pd.DataFrame, reading: dict, desc: dict, entries: list[str]) -> None:
    g = json.loads((OUT / "gates.json").read_text())
    wk = json.loads((OUT / "walk.json").read_text())
    dck = g.get("design_check", {})
    r = reading["table1"]
    d = desc["table1"]
    L = [
        "# Factor-augmented HAR: shared factor or dense? (RULE.txt section 4)",
        "",
        "Rule: `results/channels_2026-10-26/RULE.txt` section 4 (committed 86ee222 before any scoring).  Code:",
        "`experiments/factor_har_2026_10_26.py` (design, walk, gates, score).  Generated by its score mode.",
        "",
        f"## Reading (ORDINARY days): **{r['reading']}**",
        "",
        f"- SHARED FACTOR condition (some k <= 3, PCA or PLS, R_k >= 0.75 and not significantly worse than the full set):"
        f" {'met by ' + ', '.join(r['shared_factor_models']) if r['shared_factor_models'] else 'not met'}; best k <= 3:"
        f" {r['best_small_k']} (R = {r['R_ordinary'][r['best_small_k']]:.3f}).",
        f"- DENSE condition (R_10 < 0.5 and the full set significantly better than HAR + 10 components):"
        f" R_10 = {r['R10']:.3f}, full set significantly better: {r['full_beats_pca10_significantly']}"
        f" -> {'met' if r['dense_condition_met'] else 'not met'}.",
        f"- L_full on ORDINARY days: Table 1's full-set {r['full_ref_ordinary']} (the lower of ridge and elastic net).",
        "",
        "## Scores",
        "",
        "QLIKE change in percent against Table 1's OLS-HAR, DM (HLN, daily means, lag 8) against the OLS-HAR and against"
        " the full set (negative: the row's loss is lower), R_k = (L_HAR - L_k) / (L_HAR - L_full).",
        "",
    ]
    for sp in SPANS:
        s = sc[(sc["entry"] == "table1") & (sc["span"] == sp)].set_index("model")
        L += [
            f"### {sp} ({int(s['n_days'].iloc[0])} days, {int(s['n_bars'].iloc[0])} bars; full = {s['full_ref'].iloc[0]})",
            "",
            "| model | QLIKE | % vs HAR | DM vs HAR | p | DM vs full | p | R_k | MSE % vs HAR | DM MSE vs HAR | p |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m_ in [*MODELS, "ridge_full", "enet_full", "lasso_full"]:
            x = s.loc[m_]

            def f_(v, nd=2):
                return "--" if pd.isna(v) else f"{v:.{nd}f}"

            L.append(
                f"| {m_} | {x['qlike']:.5f} | {x['qlike_pct_vs_har']:.2f} | {f_(x.get('dm_har_qlike'))} |"
                f" {f_(x.get('p_har_qlike'), 3)} | {f_(x.get('dm_full_qlike'))} | {f_(x.get('p_full_qlike'), 3)} |"
                f" {x['R']:.3f} | {x['mse_pct_vs_har']:.2f} | {f_(x.get('dm_har_mse'))} | {f_(x.get('p_har_mse'), 3)} |"
            )
        L.append("")
    L += [
        "## Descriptive (2,579 scored days; the day's map)",
        "",
        f"- Active block columns a day: {d['n_active_min']} .. {d['n_active_max']} of 258.",
        "- Mean share of the block's variance (eigenvalue / active columns): "
        + ", ".join(f"PC{k} {100 * v:.1f}%" for k, v in d["share_pc"].items())
        + "; cumulative "
        + ", ".join(f"1-{k} {100 * v:.1f}%" for k, v in d["share_cum"].items())
        + f" (components 1-3: {100 * d['share_cum3_min']:.1f}% .. {100 * d['share_cum3_max']:.1f}% across days).",
        "",
        "| group | columns (mean) | PC1 sum abs loading | PC1 share of squared loadings | PLS1 sum abs weight |"
        " PLS1 share of squared weights |",
        "|---|---|---|---|---|---|",
    ]
    for gr in GROUPS7:
        L.append(
            f"| {AA.GROUP_LABEL[gr]} | {d['n_cols'][gr]:.1f} | {d['pc1_abs'][gr]:.3f} | {100 * d['pc1_sq'][gr]:.1f}% |"
            f" {d['pls1_abs'][gr]:.3f} | {100 * d['pls1_sq'][gr]:.1f}% |"
        )
    L += ["", "Loadings: the unit-norm eigenvector of the standardized block's correlation matrix (PC1) and the"]
    L += [
        "unit-norm first SIMPLS weight vector (PLS1, proportional to the columns' window correlations with the target),"
    ]
    L += ["summed by predictor group (experiments/ablate_arms.py GROUPS), averaged over days.", ""]
    if "sd" in reading:
        rs = reading["sd"]
        L += [
            "## Sensitivity: entry rule s.d. > 1e-12 only (no evidence mask)",
            "",
            f"Reading {rs['reading']}; R (ordinary) "
            + ", ".join(f"{m_} {v:.3f}" for m_, v in rs["R_ordinary"].items() if m_ != "k0")
            + ".",
            "",
        ]
    L += [
        "## Gates (gates.json)",
        "",
        f"- G1 {'PASS' if g['G1']['pass'] else 'FAIL'}: k = 0 vs Table 1's OLS-HAR, {g['G1']['rows']:,} bars, max relative"
        f" difference {g['G1']['max_rel_pred']:.2e} (tolerance 1e-10); target max relative {g['G1']['max_rel_true_adj']:.1e}.",
        f"- G2 {'PASS' if g['G2']['pass'] else 'FAIL'}: PCA 3 and PLS 3 vs direct lstsq on materialized scores,"
        f" {len(g['G2']['checks'])} checks on {', '.join(g['G2']['days'])}; max relative difference"
        f" {g['G2']['max_rel']:.2e} (tolerance 1e-8).",
        f"- G3 {'PASS' if g['G3']['pass'] else 'FAIL'}: {g['G3']['n_bars']:,} scored bars on {g['G3']['n_days']:,} days"
        f" (= Table 1's), {g['G3']['n_event_days']} EVENT days ({g['G3']['event_bars']} bars), all scored;"
        f" {g['G3']['ordinary_days']:,} ORDINARY days.",
        "",
        "## Provenance",
        "",
        f"- Design: Table 1's D13 rebuilt on the laptop ({dck.get('design', '?')}) by"
        f" `{dck.get('command', '?')}` with run_info.json's runner_cfg (the w3000 caches are not on the laptop).",
        f"- md5 against the CARC patched build (results/fomcfix_2026-10-26/gates/G3.json, md5_new): {dck.get('md5_equal')}"
        f"; the merged matrix was not written (disk full), its md5 streamed in merge's column order (column order ="
        f" Table 1's 564 columns: {dck.get('merged_order_eq_table1_columns')}). The 12 HAR-set columns are md5-equal to"
        f" Table 1's baseline design X ({dck.get('har12_md5_eq_carc_baseline_X')}); y, b_oos and the dates are equal."
        " Some other column differs from the CARC build in at least one bit (no per-column reference is on the laptop;"
        " the ablation run saw the same kind of last-bit libm difference for sumret4, <= 2.8e-14). G1 bounds the effect"
        " on the HAR part.",
        f"- Walk: {wk['bars']:,} bars from row {wk['walk_start_row']} ({wk['first']}) to {wk['last']}, {wk['maps']:,} daily"
        f" maps; {wk['wall_sec']:.0f} s wall on {wk['host']} (numpy {wk['numpy']}, one thread); rank 11 + k at every"
        f" bar (the weekday dummies sum to the intercept; min-norm as Table 1): {wk['rank_11_plus_k_every_bar']}.",
        "- Templates: results/window_sweep_2014start_fomcfix/w3000/forecasts.parquet (OLS-HAR; full-set ridge, elastic"
        " net, lasso on the patched design).",
        "- Recorded by the RULE: Table 1's full sets are penalized and include availability averages; HAR + k"
        " components is OLS on value averages only. The comparison is of forecast accuracy, not a nesting.",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("design", "walk", "gates", "score"))
    ap.add_argument("--out", default="")
    ap.add_argument("--design", default="")
    ap.add_argument("--entry", default="table1", choices=("table1", "sd"))
    a = ap.parse_args()
    return {"design": cmd_design, "walk": cmd_walk, "gates": cmd_gates, "score": cmd_score}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
