"""Gates for specs/causal_tune_lstm_intraday.py (the intraday-sequence LSTM of the 16:00 bar).

  identity      per bucket: the target, stamps and training window the spec's model
                receives through the executor (_backtest_and_save, whole series) equal --
                sha256 of the bytes -- those of the per-bar LINEAR spec's own run_executor
                invocation (experiments/gate_lstm.capture_linear); the two invocations'
                keyword arguments agree (all but method_name / fit_predict / hyperparams /
                output_file); the target equals the step matrix's adj_RV at the 16:00 bars;
                the stored per-bar LSTM / ridge results (when on disk) carry the same stamps
                and target
  seq_end       per bucket, every series row: the last bar its sequence reads is the panel
                row just before its 16:00 bar, stamped earlier than it; the share of rows
                whose last bar is the same day's 15:30 bar, and the largest number of 16:00
                bars inside an N_MAX sequence (<= EMBARGO)
  mask_flatten  live_feasible, the windows of the first tuning points and N in SEQ_LENS:
                window_keep over the window rows' sequences flattened steps x rows equals
                window_keep over their covered bars (the spec's computation)
  determinism   the walk U (OOS rows U_LO .. U_HI of live_feasible; tuning points 0 and
                TUNE_PER, refits on both sides) run twice: forecasts (both rules, every seed),
                kept columns and every tuning record bit-identical
  pool          the same walk in one process (in-process map) == pooled, bit for bit
  chunk         U as three chunks [U_LO, TUNE_PER), [TUNE_PER, MID), [MID, U_HI) -- the third
                starts inside a tuning period and recomputes its tuning point -- and again
                with the tuning cache (the middle chunk writes it, the third must read it,
                REQUIRE_CACHE=1): concatenated = U bit for bit
  causal_target the target x50 from OOS row R_PERT on (the series targets and the adj_RV
                step column from that session's 16:00 bar on): every forecast of rows
                <= R_PERT and both tuning records bit-identical; row R_PERT + 1 moves (power)
  causal_bar    every step input of every bar from row R_PERT's 16:00 bar on -> 50 x + 1
                (the 15:30-16:00 bar of that session first), target with it: the forecast
                issued at 15:30 for row R_PERT and every earlier one bit-identical; row
                R_PERT + 1 moves
  mask          a synthetic step matrix with no constant and no duplicated column in any
                window: WINDOW_MASK=1 == WINDOW_MASK=0 bit for bit, every kept count = p
  e2e_chunk     (--e2e) end to end through the spec's run cell (the executor, the results
                tables) and experiments/reduce_lstm_intraday_chunks.py at the campaign
                window W = 2000: E2E_ROWS rows unchunked vs three chunks (the last one reading
                the tuning cache of a STAGE=tune run): results / qsel / traces / npz equal

determinism / pool / chunk / causal_* run with the code-path budget FAST (2 seeds, <= 3
epochs) and the reduced window W_G on the real live_feasible step matrix: the walk is
parametrised by W, the campaign's W = 2000 is exercised by identity and e2e_chunk.

NO CHAIN LOAD: every data prep reads the bar-keyed files of data/ only (gate_lstm.bar_data).
Usage: python experiments/gate_lstm_intraday.py [--gates ...] [--buckets ...] [--pool 4] [--e2e]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments", ROOT / "specs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(ROOT)

import gate_lstm as GL  # noqa: E402

SPEC = ROOT / "specs" / "causal_tune_lstm_intraday.py"
SEG, TW = "bar1600", 2000
TUNE_PER = 250  # asserted against the spec
W_G = 300  # the code-path gates' window (fit block 150, embargo 25, validation 125)
U_LO, U_HI = TUNE_PER - 6, TUNE_PER + 6  # tuning points 0 and TUNE_PER in force
MID = TUNE_PER + 3  # a chunk start inside a tuning period
R_PERT = TUNE_PER + 2
FAST = {"N_SEEDS": "2", "MAX_EPOCHS": "3", "PATIENCE": "1"}
E2E_ROWS = 6
REC_KEYS = ("val_mse", "val_qlike", "seed_val_mse", "epochs", "keep_mask")
OUT_KEYS = (
    "preds",
    "preds_q",
    "preds_seeds",
    "preds_q_seeds",
    "kept",
    "kept_n",
    "kept_q",
    "kept_n_q",
    "q_refit",
)
LSTM_DEDUP = ROOT / "results" / "linear_subsection_lstm_dedup"
LINEAR_DEDUP = ROOT / "results" / "linear_subsection_dedup"


def exec_spec(bucket: str, extra: dict | None = None) -> dict:
    for k in list(os.environ):
        if k.startswith("HPC_KW_"):
            os.environ.pop(k)
    os.environ.update(
        HPC_KW_EXOG_BUCKET=bucket,
        HPC_KW_SEGMENT=SEG,
        HPC_KW_TRAIN_WIN=str(TW),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_DATA_PATH=str(GL.BAR_DATA),
        **{f"HPC_KW_{k}": v for k, v in (extra or {}).items()},
    )
    ns: dict = {"__file__": str(SPEC), "__name__": "lstmi_setup"}
    exec(compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec"), ns)
    assert ns["TUNE_PER"] == TUNE_PER, ns["TUNE_PER"]
    return ns


def same_walk(a: dict, b: dict, rows: slice | None = None) -> tuple[bool, str]:
    why = []
    for k in OUT_KEYS:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if rows is not None:
            x, y = x[rows], y[rows]
        if x.shape != y.shape or not np.array_equal(x, y):
            why.append(k)
    for R in sorted(set(a["recs"]) & set(b["recs"])):
        ra, rb = a["recs"][R], b["recs"][R]
        for k in REC_KEYS:
            if not np.array_equal(np.asarray(ra[k]), np.asarray(rb[k])):
                why.append(f"tune {R} {k}")
        for k in ("pick_mse", "pick_qlike", "split"):
            if tuple(np.atleast_1d(ra[k])) != tuple(np.atleast_1d(rb[k])):
                why.append(f"tune {R} {k}")
    return not why, "; ".join(why)


def concat(parts: list[dict]) -> dict:
    out = {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in OUT_KEYS}
    out["recs"] = {}
    for p in parts:
        out["recs"].update(p["recs"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gates",
        default="identity,seq_end,mask_flatten,determinism,pool,chunk,causal_target,causal_bar,mask",
    )
    ap.add_argument("--buckets", default="live_feasible,all_features,baseline")
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument(
        "--e2e", action="store_true", help="also the end-to-end chunk gate (W = 2000)"
    )
    ap.add_argument(
        "--scratch", default="results/linear_subsection_lstm_intraday/gates/scratch"
    )
    ap.add_argument("--out", default="results/linear_subsection_lstm_intraday/gates")
    a = ap.parse_args()
    gates = {g.strip() for g in a.gates.split(",") if g.strip()}
    buckets = [b.strip() for b in a.buckets.split(",") if b.strip()]
    scratch, out = ROOT / a.scratch, ROOT / a.out
    scratch.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    tag = "_".join(sorted(gates | ({"e2e"} if a.e2e else set())))[:60]

    def emit(row: dict) -> None:
        rows.append(row)
        print(
            ("PASS " if row["ok"] else "FAIL ")
            + json.dumps({k: v for k, v in row.items() if k != "ok"}, default=str),
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out / f"gate_lstm_intraday_{tag}.csv", index=False)

    GL.BAR_DATA = GL.bar_data(scratch)
    C_live: dict | None = None
    need_live = bool(
        gates
        & {
            "mask_flatten",
            "determinism",
            "pool",
            "chunk",
            "causal_target",
            "causal_bar",
        }
    )

    # ---------------------------------------------------------------- identity / seq_end
    for bucket in buckets:
        if not gates & {"identity", "seq_end"} and not (
            need_live and bucket == "live_feasible" and C_live is None
        ):
            continue
        t0 = time.time()
        ns = exec_spec(bucket, FAST)
        C = ns["prepare"]()
        if bucket == "live_feasible":
            C_live = {k: C[k] for k in ("P", "pos", "y", "step_names", "bar_stamps")}
        if "identity" in gates:
            from src.backtest.executor import _backtest_and_save

            lin, lin_kw = GL.capture_linear(bucket, scratch)
            my_kw = ns["executor_kwargs"](None, "")
            my_kw["data_path"] = lin_kw.get(
                "data_path"
            )  # both were redirected to BAR_DATA
            kw_diff = sorted(
                k
                for k in set(lin_kw) | set(my_kw)
                if k not in GL.SKIP_KW and lin_kw.get(k) != my_kw.get(k)
            )
            got: dict = {}
            d = scratch / f"cap_lstmi_{bucket}"
            _backtest_and_save(
                C["seg_df"],
                C["design"],
                GL.grab_into(got),
                {},
                C["train_win"],
                1,
                0,
                -1,
                0,
                str(d / f"results_{SEG}.csv"),
                True,
                "divide",
            )
            res = pd.read_csv(d / f"results_{SEG}.csv")
            mine = {
                "y": got["y"],
                "dates": res["date"].astype(str).to_numpy().astype("U19"),
                "true_adj": res["true_adj"].to_numpy(float),
            }
            hashes = {
                k: (GL.sha(lin[k]), GL.sha(mine[k])) for k in ("y", "dates", "true_adj")
            }
            y_is_P = bool(np.array_equal(got["y"], C["y"]))
            stored = {}
            for label, f in (
                (
                    "lstm_dedup",
                    LSTM_DEDUP
                    / bucket
                    / SEG
                    / "lstm"
                    / f"tw{TW}"
                    / "causal_tune_lstm"
                    / "lstm"
                    / bucket
                    / f"results_{SEG}.csv",
                ),
                (
                    "ridge_dedup",
                    LINEAR_DEDUP / bucket / "ridge" / f"tw{TW}" / f"results_{SEG}.csv",
                ),
                (
                    "ridge_hoffman2",
                    GL.LINEAR_STORED
                    / bucket
                    / "ridge"
                    / f"tw{TW}"
                    / f"results_{SEG}.csv",
                ),
            ):
                if f.is_file():
                    st = pd.read_csv(f)
                    same = len(st) == len(mine["dates"]) and bool(
                        (st["date"].astype(str).to_numpy() == mine["dates"]).all()
                    )
                    stored[label] = bool(
                        same
                        and np.array_equal(
                            st["true_adj"].to_numpy(float), mine["true_adj"]
                        )
                    )
            ok = (
                not kw_diff
                and all(x == z for x, z in hashes.values())
                and int(lin["W"]) == int(got["W"]) == TW
                and y_is_P
                and all(stored.values())
            )
            emit(
                {
                    "gate": "identity",
                    "bucket": bucket,
                    "ok": ok,
                    "kwargs_differ": kw_diff,
                    "n_series": len(C["pos"]),
                    "n_oos": len(mine["dates"]),
                    "first": str(mine["dates"][0]),
                    "last": str(mine["dates"][-1]),
                    "n_bars": len(C["P"]),
                    "n_step_features": len(C["step_names"]),
                    "step_names": ";".join(C["step_names"]),
                    "target_is_P_adj_RV": y_is_P,
                    "stored_same_stamps_and_target": stored,
                    **{
                        f"sha_{k}": v[1]
                        + ("" if v[0] == v[1] else f" != linear {v[0]}")
                        for k, v in hashes.items()
                    },
                    "sec": round(time.time() - t0),
                }
            )
        if "seq_end" in gates:
            pos, st = C["pos"], pd.DatetimeIndex(C["bar_stamps"])
            last, tgt = st[pos - 1], st[pos]
            same_day_1530 = (last.normalize() == tgt.normalize()) & (
                last.strftime("%H:%M") == "15:30"
            )
            inside = ns["sessions_in_sequence"](pos, ns["N_MAX"])
            ok = bool(
                (last < tgt).all()
                and (tgt.strftime("%H:%M") == "16:00").all()
                and inside.max() <= ns["EMBARGO"]
            )
            emit(
                {
                    "gate": "seq_end",
                    "bucket": bucket,
                    "ok": ok,
                    "n_series": len(pos),
                    "last_bar_before_target": bool((last < tgt).all()),
                    "share_last_bar_same_day_1530": round(
                        float(same_day_1530.mean()), 5
                    ),
                    "n_last_bar_not_same_day_1530": int((~same_day_1530).sum()),
                    "max_1600_bars_inside_Nmax_sequence": int(inside.max()),
                    "embargo": ns["EMBARGO"],
                    **{
                        f"share_sequence_inside_one_session_N{N}": round(
                            float((ns["sessions_in_sequence"](pos, N) == 0).mean()), 4
                        )
                        for N in ns["SEQ_LENS"]
                    },
                }
            )
        del C

    # ---------------------------------------------------------------- mask_flatten (real windows)
    if "mask_flatten" in gates:
        assert C_live is not None
        import causal_tune_lstm_intraday_jobs as J
        from src.models.window_mask import window_keep

        P, pos = C_live["P"], C_live["pos"]
        bad, checked, counts = [], 0, {}
        for R in (0, TUNE_PER, 2 * TUNE_PER, 1000, 1400):
            if TW + R >= len(pos):
                continue
            ends = pos[R : TW + R]
            for N in (13, 48, 96):
                flat = P[(ends[:, None] + np.arange(-N, 0)[None, :]).ravel()]
                k_flat = window_keep(flat)
                cov = J.covered(
                    ends - (pos[R] - 96), N, pos[TW + R - 1] - (pos[R] - 96)
                )
                k_cov = window_keep(P[pos[R] - 96 : pos[TW + R - 1]][cov])
                checked += 1
                counts[f"R{R}_N{N}"] = len(k_cov)
                if not np.array_equal(k_flat, k_cov):
                    bad.append(f"R{R} N{N}")
        emit(
            {
                "gate": "mask_flatten",
                "bucket": "live_feasible",
                "ok": not bad,
                "why": ";".join(bad),
                "windows": checked,
                "kept": counts,
                "p": P.shape[1],
            }
        )

    # ---------------------------------------------------------------- walks on real live_feasible data
    if gates & {"determinism", "pool", "chunk", "causal_target", "causal_bar"}:
        assert C_live is not None
        P, pos, y = C_live["P"], C_live["pos"], C_live["y"]
        ns = exec_spec("live_feasible", FAST | {"WINDOW_MASK": "1"})
        t = time.time()
        A = ns["walk"](P, pos, y, W_G, U_LO, U_HI, n_workers=a.pool)
        sec_a = time.time() - t
        base_info = {
            "W": W_G,
            "rows": [U_LO, U_HI],
            "budget": FAST,
            "picks": {
                R: [
                    ns["CANDIDATES"][int(r["pick_mse"])],
                    ns["CANDIDATES"][int(r["pick_qlike"])],
                ]
                for R, r in A["recs"].items()
            },
            "kept_n": [int(A["kept_n"].min()), int(A["kept_n"].max())],
            "q_refit_rows": int(A["q_refit"].sum()),
        }
        if "determinism" in gates:
            B = ns["walk"](P, pos, y, W_G, U_LO, U_HI, n_workers=a.pool)
            ok, why = same_walk(A, B)
            emit(
                {
                    "gate": "determinism",
                    "bucket": "live_feasible",
                    "ok": ok,
                    "why": why,
                    "sec": round(sec_a),
                }
                | base_info
            )
        if "pool" in gates:
            t = time.time()
            S = ns["walk"](P, pos, y, W_G, U_LO, U_HI, n_workers=1)
            ok, why = same_walk(A, S)
            emit(
                {
                    "gate": "pool",
                    "bucket": "live_feasible",
                    "ok": ok,
                    "why": why,
                    "pool": a.pool,
                    "sec_serial": round(time.time() - t),
                    "sec_pooled": round(sec_a),
                }
            )
        if "chunk" in gates:
            parts = [
                ns["walk"](P, pos, y, W_G, lo, hi, n_workers=a.pool)
                for lo, hi in ((U_LO, TUNE_PER), (TUNE_PER, MID), (MID, U_HI))
            ]
            ok1, why1 = same_walk(A, concat(parts))
            cache = scratch / "tune_cache_gate"
            shutil.rmtree(cache, ignore_errors=True)
            ns_w = exec_spec(
                "live_feasible", FAST | {"WINDOW_MASK": "1", "TUNE_CACHE": str(cache)}
            )
            p2 = ns_w["walk"](
                P, pos, y, W_G, TUNE_PER, MID, n_workers=a.pool
            )  # writes the cache of TUNE_PER
            ns_r = exec_spec(
                "live_feasible",
                FAST
                | {"WINDOW_MASK": "1", "TUNE_CACHE": str(cache), "REQUIRE_CACHE": "1"},
            )
            p3 = ns_r["walk"](P, pos, y, W_G, MID, U_HI, n_workers=a.pool)
            from_cache = bool(p3["recs"][TUNE_PER]["from_cache"])
            ok2, why2 = same_walk(A, concat([parts[0], p2, p3]))
            emit(
                {
                    "gate": "chunk",
                    "bucket": "live_feasible",
                    "ok": ok1 and ok2 and from_cache,
                    "why": f"plain: {why1}; cached: {why2}",
                    "chunks": [[U_LO, TUNE_PER], [TUNE_PER, MID], [MID, U_HI]],
                    "third_chunk_read_cache": from_cache,
                }
            )
        for g in ("causal_target", "causal_bar"):
            if g not in gates:
                continue
            s = W_G + R_PERT  # series row of OOS row R_PERT
            P2 = P.copy()
            if g == "causal_target":
                P2[pos[s] :, 0] *= 50.0
            else:
                P2[pos[s] :, :] = P2[pos[s] :, :] * 50.0 + 1.0
            y2 = P2[pos, 0].copy()
            assert np.array_equal(y2[:s], y[:s]) and not np.array_equal(y2[s:], y[s:])
            X = ns["walk"](P2, pos, y2, W_G, U_LO, U_HI, n_workers=a.pool)
            keep_rows = slice(0, R_PERT - U_LO + 1)  # OOS rows U_LO .. R_PERT
            ok, why = same_walk(A, X, keep_rows)
            moved = float(
                abs(X["preds"][R_PERT - U_LO + 1] - A["preds"][R_PERT - U_LO + 1])
            )
            emit(
                {
                    "gate": g,
                    "bucket": "live_feasible",
                    "ok": ok and moved > 0,
                    "why": why,
                    "perturbed_from_oos_row": R_PERT,
                    "perturbed_from_bar": str(C_live["bar_stamps"][pos[s]]),
                    "unchanged_rows": [U_LO, R_PERT],
                    "tuning_points_compared": sorted(A["recs"]),
                    "next_row_abs_move": moved,
                }
            )

    # ---------------------------------------------------------------- mask (synthetic full-rank design)
    if "mask" in gates:
        rng = np.random.default_rng(20260929)
        n_rows = W_G + U_HI + 5
        per = 48
        n_bars = 96 + per * n_rows
        Psyn = rng.standard_normal((n_bars, 6))
        Psyn[:, 0] = (
            np.abs(Psyn[:, 0]) + 0.5
        )  # a positive "target" column (QLIKE needs y > 0)
        pos_s = 96 + per * np.arange(n_rows) + per - 1
        y_s = Psyn[pos_s, 0].copy()
        ns0 = exec_spec("baseline", FAST | {"WINDOW_MASK": "0"})
        ns1 = exec_spec("baseline", FAST | {"WINDOW_MASK": "1"})
        M0 = ns0["walk"](Psyn, pos_s, y_s, W_G, U_LO, U_HI, n_workers=a.pool)
        M1 = ns1["walk"](Psyn, pos_s, y_s, W_G, U_LO, U_HI, n_workers=a.pool)
        ok, why = same_walk(M0, M1)
        full = bool(
            (M1["kept_n"] == 6).all()
            and all(r["keep_mask"].all() for r in M1["recs"].values())
        )
        emit(
            {
                "gate": "mask",
                "bucket": "synthetic",
                "ok": ok and full,
                "why": why,
                "p": 6,
                "all_kept": full,
            }
        )

    # ---------------------------------------------------------------- end to end (W = 2000)
    if a.e2e:
        e2e(scratch, a.pool, emit)
    tab = pd.DataFrame(rows)
    print(tab[["gate", "bucket", "ok"]].to_string(index=False))
    if len(tab) and not tab["ok"].all():
        sys.exit("a gate failed")


def run_cell(out: Path, start: int, end: int, cpus: int, extra: dict) -> None:
    if (out / "DONE").is_file():
        print(f"reusing {out} (DONE)")
        return
    env = {k: v for k, v in os.environ.items() if not k.startswith("HPC_KW_")}
    env.update(
        HPC_KW_EXOG_BUCKET="live_feasible",
        HPC_KW_SEGMENT=SEG,
        HPC_KW_TRAIN_WIN=str(TW),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_START=str(start),
        HPC_KW_END=str(end),
        HPC_KW_HALO=str(TW if start else 0),
        HPC_KW_DATA_PATH=str(GL.BAR_DATA),
        HPC_RESULT_DIR=str(out),
        SLURM_CPUS_PER_TASK=str(cpus),
        PYTHONUNBUFFERED="1",
        **{f"HPC_KW_{k}": v for k, v in (FAST | extra).items()},
    )
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(
            [sys.executable, "-u", str(SPEC)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    assert rc == 0, (
        f"spec run failed ({out / 'run.log'}): {(out / 'run.log').read_text(encoding='utf-8')[-1500:]}"
    )
    (out / "DONE").touch()


def e2e(scratch: Path, cpus: int, emit) -> None:
    """U = OOS rows [0, E2E_ROWS) at W = 2000 through the run cell, vs chunks [0, 2), [2, 4)
    (a STAGE=tune run writes no rows), [4, E2E_ROWS) reading the tuning cache, merged by the
    reducer."""
    t = time.time()
    root = scratch / "e2e"
    cache = root / "cache"
    u_dir = root / "U"
    run_cell(u_dir, TW, TW + E2E_ROWS, cpus, {})
    arm = root / "live_feasible" / SEG / "lstm_intraday" / f"tw{TW}"
    run_cell(
        root / "stage_tune",
        TW,
        TW + 1,
        cpus,
        {"STAGE": "tune", "TUNE_CACHE": str(cache)},
    )
    lines = []
    for k, (s, e) in enumerate(((0, 2), (2, 4), (4, E2E_ROWS))):
        run_cell(
            arm / "chunks" / f"c{k}",
            TW + s,
            TW + e,
            cpus,
            {"TUNE_CACHE": str(cache), "REQUIRE_CACHE": "1"},
        )
        lines.append(
            f"live_feasible lstm_intraday {TW} {SEG} {k} {TW + s} {TW + e} {TW}"
        )
    tasks = root / "tasks.txt"
    tasks.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "reduce_lstm_intraday_chunks.py"),
            "--root",
            str(root),
            "--tasks",
            str(tasks),
        ]
    ).returncode
    inner = Path("causal_tune_lstm_intraday") / "lstm_intraday" / "live_feasible"
    bad = [] if rc == 0 else [f"reducer exit {rc}"]
    if rc == 0:
        rt = "round_trip"
        for f, cols in (
            (f"results_{SEG}.csv", ["date", "true_adj", "pred_adj", "true_raw"]),
            (f"results_qsel_{SEG}.csv", ["date", "pred_adj"]),
            (
                f"refit_trace_{SEG}.csv",
                [
                    "oos_row",
                    "date",
                    "cand_idx",
                    "cand_idx_q",
                    "epochs",
                    "kept_n",
                    "kept_n_q",
                ],
            ),
            (
                f"tune_trace_{SEG}.csv",
                [
                    "tune_row",
                    "rule",
                    "cand_idx",
                    "epochs",
                    "val_mse",
                    "val_qlike",
                    "kept_n_N13",
                    "kept_n_N96",
                ],
            ),
            (
                f"tune_candidates_{SEG}.csv",
                ["tune_row", "cand_idx", "epochs", "val_mse", "val_qlike"],
            ),
        ):
            mu = pd.read_csv(u_dir / inner / f, float_precision=rt)
            mc = pd.read_csv(arm / inner / f, float_precision=rt)
            for c in cols:
                if not mu[c].equals(mc[c]):
                    bad.append(f"{f}:{c}")
        with (
            np.load(u_dir / inner / f"lstm_intraday_{SEG}.npz") as zu,
            np.load(arm / inner / f"lstm_intraday_{SEG}.npz") as zc,
        ):
            for key in (
                "pred_adj",
                "pred_adj_qsel",
                "pred_adj_seeds",
                "true_adj",
                "kept",
                "kept_n",
                "tune_row",
                "cand_epochs",
                "cand_val_mse",
                "cand_val_qlike",
                "tune_kept",
            ):
                if not np.array_equal(zu[key], zc[key]):
                    bad.append(f"npz:{key}")
    log = (arm / "chunks" / "c0" / "run.log").read_text(encoding="utf-8")
    emit(
        {
            "gate": "e2e_chunk",
            "bucket": "live_feasible",
            "ok": not bad and "read from" in log,
            "why": "; ".join(bad),
            "W": TW,
            "rows": E2E_ROWS,
            "chunks": 3,
            "chunk0_read_cache": "read from" in log,
            "sec": round(time.time() - t),
            "budget": FAST,
        }
    )


if __name__ == "__main__":
    main()
