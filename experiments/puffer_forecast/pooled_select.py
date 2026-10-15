"""Pooled walk (pooled_walk.py): pick each block's configuration and write its monthly refit jobs.

Stdlib only (the compute node's Python 3.6).  Selection as the per-bar walk (phase2_select.py): among the tune
grid's configurations (wd part of the configuration when the grid carries it), the lowest MSE of the seed-averaged
validation forecast (configurations in sorted order; ties: the first; non-finite skipped).  Each block refits at
its origins (the plan's slice rows), on the trailing W bars, each seed for its own best epoch, forecasting every
bar up to the next refit (fc_n = next - origin).

    python3 pooled_select.py --plan pwalk_X/plan.json --k K --tune-out RUN/blockK/tune --data-root data \
        --out RUN/blockK/jobs_refit.txt --selection RUN/blockK/selection.json [--save-first-origin]
"""

import argparse
import array
import csv
import json
import math
import re
import sys

ID_RE = re.compile(
    r"^ptune_(baseline|all_features)_L(\d+)_H(\d+)_lr([0-9.]+)(?:_wd([0-9.e+-]+))?_s(\d+)$"
)
CFG_RE = re.compile(r"^L(\d+)_H(\d+)_lr([0-9.]+)(?:_wd([0-9.e+-]+))?$")


def cfg_name(L, H, lr, wd):
    return "L%s_H%s_lr%s" % (L, H, lr) + ("" if wd is None else "_wd%s" % wd)


def load_y(path):
    values = array.array("f")
    values.frombytes(open(path, "rb").read())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--tune-out", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--save-first-origin", action="store_true")
    a = ap.parse_args()
    plan = json.loads(open(a.plan).read())
    block = [b for b in plan["blocks"] if b["k"] == a.k][0]
    seeds = [int(s) for s in plan["seeds"]]
    epochs, ran = {}, {}
    with open(a.tune_out + "/jobs_out.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            m = ID_RE.match(r["id"])
            if m is None:
                continue
            d, L, H, lr, wd, _ = m.groups()
            epochs[r["id"]] = int(r["best_epoch"])
            if r["status"] != "ok":
                raise SystemExit("tune job %s status %s" % (r["id"], r["status"]))
            ran.setdefault(d, set()).add(cfg_name(L, H, lr, wd))
    fc = {}
    with open(a.tune_out + "/preds.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] != "val":
                continue
            m = ID_RE.match(r["id"])
            if m is None:
                continue
            d, L, H, lr, wd, seed = m.groups()
            fc.setdefault(d, {}).setdefault(cfg_name(L, H, lr, wd), {}).setdefault(
                int(r["row"]), {}
            )[int(seed)] = float(r["f"])
    sel, lines = {}, []
    for d in plan["designs"]:
        if set(fc.get(d, {})) != ran.get(d, set()) or not ran.get(d):
            raise SystemExit(
                "design %s: configs with validation rows != configs run" % d
            )
        y = load_y("%s/pooled_walk_%s/y.f32" % (a.data_root, d))
        best = None
        for cfg in sorted(fc[d]):
            sse, n = 0.0, 0
            for row, by_seed in fc[d][cfg].items():
                if len(by_seed) != len(seeds):
                    raise SystemExit(
                        "%s %s row %s: %s seeds" % (d, cfg, row, len(by_seed))
                    )
                e = sum(by_seed.values()) / float(len(seeds)) - float(y[row])
                sse += e * e
                n += 1
            mse = sse / n
            if not math.isfinite(mse):
                continue
            if best is None or mse < best[0]:
                best = (mse, cfg, n)
        if best is None:
            raise SystemExit("design %s: no finite validation MSE" % d)
        mse, cfg, n_val = best
        L, H, lr, wd = CFG_RE.match(cfg).groups()
        if wd is None:
            wd = plan["wd_fixed"]
        per_seed = {}
        for seed in seeds:
            tid = "ptune_%s_%s_s%s" % (d, cfg, seed)
            if epochs.get(tid, 0) < 1:
                raise SystemExit("missing best_epoch for %s" % tid)
            per_seed[str(seed)] = epochs[tid]
            for i, (o, nx) in enumerate(zip(block["origins"], block["next"])):
                base = dict(
                    id="prefit_%s_r%s_s%s" % (d, o, seed),
                    data="data/pooled_walk_%s" % d,
                    mode="refit",
                    w0=o - plan["W"],
                    w1=o,
                    L=L,
                    H=H,
                    lr=lr,
                    seed=seed,
                    epochs=epochs[tid],
                    fc_n=nx - o,
                    wd=wd,
                )
                for k, v in plan["common"].items():
                    base.setdefault(k, v)
                if a.save_first_origin and i == 0:
                    base["save_params"] = 1
                lines.append(" ".join("%s=%s" % (k, v) for k, v in base.items()))
        sel[d] = dict(cfg=cfg, wd=wd, val_mse=mse, n_val=n_val, best_epoch=per_seed)
        print("block %s %s %s val_mse %.6f n_val %s" % (a.k, d, cfg, mse, n_val))
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(a.selection, "w") as fh:
        json.dump(sel, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote %s refit jobs" % len(lines))


if __name__ == "__main__":
    main()
