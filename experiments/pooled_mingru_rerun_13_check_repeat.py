"""Check (2026-10-22): the same trainer jobs run twice on CARC P100s give the same outputs.  Pairs of run directories
(A B): every job id both hold -- best_epoch and best_val_mse of jobs_out.csv, and every (id, kind, row) forecast f of
preds.csv -- compared as printed strings (bit for bit at the trainer's print precision) and numerically.

    python3 experiments/pooled_mingru_rerun_13_check_repeat.py A/tune B/tune [A2/refit B2/refit ...]
(stdlib only)
"""

import csv
import sys


def load(d):
    jo = {r["id"]: r for r in csv.DictReader(open(d + "/jobs_out.csv"))}
    pr = {}
    for r in csv.DictReader(open(d + "/preds.csv")):
        pr[(r["id"], r["kind"], r["row"])] = r["f"]
    return jo, pr


args = sys.argv[1:]
assert args and len(args) % 2 == 0, "pairs of directories"
ok_all = True
for a, b in zip(args[0::2], args[1::2]):
    ja, pa = load(a)
    jb, pb = load(b)
    ids = sorted(set(ja) & set(jb))
    ep = sum(ja[i]["best_epoch"] != jb[i]["best_epoch"] for i in ids)
    bv = sum(ja[i]["best_val_mse"] != jb[i]["best_val_mse"] for i in ids)
    keys = [k for k in pa if k[0] in set(ids)]
    miss = sum(k not in pb for k in keys)
    neq = [k for k in keys if k in pb and pa[k] != pb[k]]
    mx = max(
        [abs(float(pa[k]) - float(pb[k])) / max(abs(float(pb[k])), 1e-30) for k in neq]
        or [0.0]
    )
    ok = ep == 0 and bv == 0 and miss == 0 and not neq
    ok_all &= ok
    print(
        f"{a} vs {b}: {len(ids)} common jobs; best_epoch differs {ep}, best_val differs {bv}; {len(keys)} forecasts, "
        f"{miss} missing, {len(neq)} differ (max rel {mx:.2e}) -> {'IDENTICAL' if ok else 'DIFFERENT'}"
    )
print("REPEAT_IDENTICAL" if ok_all else "REPEAT_DIFFERENT")
