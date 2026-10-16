"""Sequential gate for the dev-span tree menu (2026-10-06): tree_menu_dev's OWN screen, verbatim.

experiments/tree_menu_dev_rf.py's gate re-scores stored menu entries with its pool screen (single-
threaded fits in forked processes).  On CARC epyc-7513 nodes XGBoost segfaults while building its
DMatrix inside such a forked worker (canary job 12777632; LightGBM and the random forest are
unaffected), so the XGBoost entries cannot be re-scored that way.  This script re-scores them the way
the study itself scored them: tree_menu_dev._screen_score(X, y, family, params, n_threads) in one
process, n_threads = SLURM_CPUS_PER_TASK (the menu workers ran 8), on the dev matrices of
tree_menu_dev_rf.py (sha256-checked), loaded fully into memory as the study's _dev_matrices did.

Writes results/rf_devmenu_2026-10-06/_work/<tag>/seqgate.json; --append-summary (local, after
`tree_menu_dev_rf.py freeze`) appends a section built from every _work/*/seqgate.json to SUMMARY.md.
Usage:  python experiments/tree_menu_dev_rf_seqgate.py --tag seq8 [--entries lgbm_opt_00,xgb_opt_06]
        python experiments/tree_menu_dev_rf_seqgate.py --append-summary
"""

import _bootstrap  # noqa: F401  (sys.path: repo root + experiments/)

import argparse
import json
import os
import socket
import time

import numpy as np

import tree_menu_dev as TMD
import tree_menu_dev_rf as M


def append_summary() -> None:
    import glob

    paths = sorted(glob.glob(os.path.join(M.WORK, "*", "seqgate.json")))
    L = ["## 3b. Sequential gate (tree_menu_dev._screen_score itself)", ""]
    L.append(
        "XGBoost segfaults while building its DMatrix inside the pool's forked workers on CARC "
        "epyc-7513 nodes (the `*_xgb` gate above), so the stored entries were also re-scored by the "
        "study's own screen function in one process with multi-threaded fits, as the menu workers did "
        "(`experiments/tree_menu_dev_rf_seqgate.py`)."
    )
    L.append("")
    if not paths:
        L += ["No seqgate.json found.", ""]
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            g = json.load(fh)
        part = " (PARTIAL: the run did not finish)" if g.get("partial") else ""
        L.append(
            f"`{p.replace(os.sep, '/')}`: {g['host']} [{g['cpu']}], {g['mode']}, versions "
            f"{json.dumps(g['versions'])}{part}:"
        )
        L.append("")
        L.append(
            "| entry | stored dev screen MSE | re-run | rel diff | pass | seconds |"
        )
        L.append("|---|---|---|---|---|---|")
        for k, v in g["entries"].items():
            L.append(
                f"| {k} | {v['stored']!r} | {v['rerun']!r} | {v['rel_diff']:.2e} | {v['pass']} | "
                f"{v['wall_sec']:.0f} |"
            )
        L.append("")
    with open(
        os.path.join(M.OUT, "SUMMARY.md"), "a", encoding="utf-8", newline=""
    ) as fh:
        fh.write("".join(line + chr(10) for line in L))
    print(f"appended {len(paths)} sequential gate run(s) to {M.OUT}/SUMMARY.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--tag", default="seq8")
    ap.add_argument("--entries", default="lgbm_opt_00,xgb_opt_06,xgb_opt_00")
    ap.add_argument("--append-summary", action="store_true")
    a = ap.parse_args()
    if a.append_summary:
        append_summary()
        return
    man = M._check_dev(M.DEV_DIR)
    X = np.ascontiguousarray(
        np.load(os.path.join(M.DEV_DIR, "X.npy")), dtype=np.float64
    )
    y = np.asarray(np.load(os.path.join(M.DEV_DIR, "y.npy")), dtype=np.float64)
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    out: dict = {
        "tag": a.tag,
        "host": socket.gethostname(),
        "cpu": M._cpu_model(),
        "procs": threads,
        "mode": f"sequential: tree_menu_dev._screen_score, {threads} threads a fit",
        "versions": M._versions(),
        "dev_end_row": man["dev_end_row"],
        "entries": {},
    }
    entries = tuple(a.entries.split(","))
    ok = True
    d = os.path.join(M.WORK, a.tag)
    os.makedirs(d, exist_ok=True)
    for name in entries:
        e = M.STORED[name]
        if TMD._sha(e["family"], e["params"]) != e["sha"]:
            raise SystemExit(f"{name}: params do not hash to the stored sha {e['sha']}")
        t = time.time()
        value = TMD._screen_score(X, y, e["family"], e["params"], threads)
        rel = abs(value - e["dev_screen_mse"]) / e["dev_screen_mse"]
        passed = bool(rel < M.GATE_RTOL)
        if name in M.GATE_REQUIRED:
            ok &= passed
        out["entries"][name] = {
            "family": e["family"],
            "sha": e["sha"],
            "stored": e["dev_screen_mse"],
            "rerun": value,
            "abs_diff": value - e["dev_screen_mse"],
            "rel_diff": rel,
            "pass": passed,
            "required": name in M.GATE_REQUIRED,
            "wall_sec": time.time() - t,
        }
        print(
            f"seqgate {name}: stored {e['dev_screen_mse']!r} rerun {value!r} rel {rel:.2e} "
            f"{'PASS' if passed else 'FAIL'} ({time.time() - t:.0f}s)",
            flush=True,
        )
        with open(os.path.join(d, "seqgate.json"), "w", encoding="utf-8") as fh:
            json.dump({**out, "pass": None, "partial": True}, fh, indent=1)
    out["pass"] = ok if any(n in M.GATE_REQUIRED for n in entries) else None
    with open(os.path.join(d, "seqgate.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
