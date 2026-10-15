"""Pick the Phase 2 config for one block and write its cold refit jobs.

Stdlib only, and valid on the Hoffman2 compute node's Python 3.6. The rule is
the Phase 1 rule: among the 12 configs, the lowest MSE of the seed-averaged
validation forecast. Each refit then uses that config and that seed's own
best_epoch from the tune jobs_out.csv.

    python3 phase2_select.py --blocks blocks.json --k 0 \
        --tune-out RUN/block0/tune --data-root data --out jobs_refit.txt \
        --selection selection.json
"""

import argparse
import array
import csv
import json
import math
import re
import sys

# _wd<x> (2026-10-14, the weight-decay study): wd is part of the configuration when the tune grid carries it
ID_RE = re.compile(
    r"^tune_(baseline|all_features)_L(\d+)_H(\d+)_lr([0-9.]+)(?:_wd([0-9.e+-]+))?_s(\d+)$"
)


def cfg_name(L, H, lr, wd):
    return "L%s_H%s_lr%s" % (L, H, lr) + ("" if wd is None else "_wd%s" % wd)


def kv(**kwargs):
    return " ".join("%s=%s" % (key, value) for key, value in kwargs.items())


def load_y(path):
    raw = open(path, "rb").read()
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def best_epochs(path):
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out[row["id"]] = int(row["best_epoch"])
    return out


def tune_configs(path):
    """design -> the set of configs the tune jobs_out.csv ran (every tune job writes a row)."""
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            match = ID_RE.match(row["id"])
            if match is not None:
                design, L, H, lr, wd, _ = match.groups()
                out.setdefault(design, set()).add(cfg_name(L, H, lr, wd))
    return out


def val_forecasts(path):
    """design -> cfg -> row -> seed -> f, for validation rows of tune jobs."""
    acc = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "val":
                continue
            match = ID_RE.match(row["id"])
            if match is None:
                continue
            design, L, H, lr, wd, seed = match.groups()
            cfg = cfg_name(L, H, lr, wd)
            acc.setdefault(design, {}).setdefault(cfg, {}).setdefault(
                int(row["row"]), {}
            )[int(seed)] = float(row["f"])
    return acc


def choose(design, forecasts, y, seeds):
    """Lowest seed-averaged validation MSE; configs in sorted order (ties: the first), non-finite skipped."""
    best = None
    for cfg in sorted(forecasts):
        rows = forecasts[cfg]
        if not rows:
            raise SystemExit("design %s cfg %s has no validation rows" % (design, cfg))
        sse = 0.0
        n = 0
        for row_index, by_seed in rows.items():
            if len(by_seed) != len(seeds):
                raise SystemExit(
                    "design %s cfg %s row %s has %s seeds, expected %s"
                    % (design, cfg, row_index, len(by_seed), len(seeds))
                )
            f = sum(by_seed.values()) / float(len(seeds))
            err = f - float(y[row_index])
            sse += err * err
            n += 1
        mse = sse / float(n)
        if not math.isfinite(mse):
            print(
                "design %s cfg %s: non-finite validation MSE, skipped" % (design, cfg)
            )
            continue
        if best is None or mse < best[0]:
            best = (mse, cfg, n)
    if best is None:
        raise SystemExit("design %s: no config with a finite validation MSE" % design)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--tune-out", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection", required=True)
    # save_params=1 on the block's first origin's refits (every seed and design): weights at the best epoch
    parser.add_argument("--save-first-origin", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(open(args.blocks).read())
    block = None
    for item in manifest["blocks"]:
        if item["k"] == args.k:
            block = item
    if block is None:
        raise SystemExit("no block %s" % args.k)
    seeds = [int(s) for s in manifest["seeds"]]
    common = dict(manifest["common"])
    common["fc_n"] = 1
    epochs = best_epochs(args.tune_out + "/jobs_out.csv")
    forecasts = val_forecasts(args.tune_out + "/preds.csv")
    ran = tune_configs(args.tune_out + "/jobs_out.csv")

    selection = {}
    lines = []
    for design in manifest["designs"]:
        y = load_y(args.data_root + "/" + design + "/y.f32")
        if set(forecasts.get(design, {})) != ran.get(design, set()) or not ran.get(
            design
        ):
            raise SystemExit(
                "design %s: configs with validation rows %s != configs run %s"
                % (
                    design,
                    sorted(forecasts.get(design, {})),
                    sorted(ran.get(design, set())),
                )
            )
        mse, cfg, n_val = choose(design, forecasts[design], y, seeds)
        match = re.match(r"L(\d+)_H(\d+)_lr([0-9.]+)(?:_wd([0-9.e+-]+))?$", cfg)
        L, H, lr, wd = match.groups()
        per_seed = {}
        for seed in seeds:
            tid = "tune_%s_%s_s%s" % (design, cfg, seed)
            if tid not in epochs or epochs[tid] < 1:
                raise SystemExit("missing best_epoch for %s" % tid)
            per_seed[str(seed)] = epochs[tid]
            for origin in block["origins"]:
                rid = "refit_%s_r%s_s%s" % (design, origin, seed)
                base = dict(
                    data="data/%s" % design,
                    w0=origin - manifest["W"],
                    w1=origin,
                    L=L,
                    H=H,
                    lr=lr,
                    seed=seed,
                    epochs=epochs[tid],
                )
                base.update(common)
                if (
                    wd is not None
                ):  # the selected configuration's wd (over a common one)
                    base["wd"] = wd
                if args.save_first_origin and origin == block["origins"][0]:
                    base["save_params"] = 1
                lines.append(kv(id=rid, mode="refit", loss="mse", **base))
        selection[design] = {
            "cfg": cfg,
            "val_mse": mse,
            "n_val": n_val,
            "best_epoch": per_seed,
        }
        print(
            "block %s %s %s val_mse %.6f n_val %s" % (args.k, design, cfg, mse, n_val)
        )

    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(args.selection, "w") as handle:
        json.dump(selection, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print("wrote %s refit jobs" % len(lines))


if __name__ == "__main__":
    main()
