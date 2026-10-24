"""Split the not-yet-started walk files into ~8 h pieces (whole bar-origin blocks inside one original file, so each
piece stages its original file's init sub-directory), plus a few debug pieces under 1 h.  Local; writes the piece
files into <out>/<window>/pieces/ and a manifest pieces_<window>_<set>.csv."""

import math
import sys
from pathlib import Path

import pandas as pd

SP = Path(sys.argv[1])
OUT = SP / "pieces_out"
SEEDS = 5
# arm: (window dir, set, original files to replace, type, s a refit on that type, target h, nice, debug pieces)
ARMS = [
    ("3000", "baseline", list(range(11, 54)), "P100", 42.3, 8.0, 0, 0),
    ("3000", "all_features", list(range(0, 6)), "L40S", 4.55 / 0.9, 8.0, 0, 0),
    ("3500", "baseline", list(range(0, 8)), "P100", 6.57, 8.0, 100, 5),
    ("3500", "all_features", list(range(0, 4)), "L40S", 1.56 / 0.9, 8.0, 100, 0),
]
DEBUG_ORIGINS = 70  # 350 refits x 6.57 s = 38 min on a P100 (debug limit 60 min)


def tlim(est_s: float) -> int:
    return int(math.ceil(est_s * 1.3 / 60 + 30))


for win, s, files, gpu, sref, target, nice, ndebug in ARMS:
    od = OUT / win / "pieces"
    od.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in files:
        src = SP / f"orig{win}" / f"v2w_{s}.{i:03d}.txt"
        lines = src.read_text().splitlines()
        assert len(lines) % SEEDS == 0
        origins = [
            int(ln.split()[0].split("_r")[1].split("_s")[0]) for ln in lines[::SEEDS]
        ]
        blocks = [lines[k : k + SEEDS] for k in range(0, len(lines), SEEDS)]
        start = 0
        k = 0
        if ndebug and i == files[0]:
            for _ in range(ndebug):
                sel = blocks[start : start + DEBUG_ORIGINS]
                name = f"v2w_{s}.{i:03d}.d{k:02d}.txt"
                (od / name).write_text("\n".join(ln for b in sel for ln in b) + "\n")
                est = len(sel) * SEEDS * sref
                rows.append(
                    dict(
                        piece=f"pieces/{name}",
                        orig=f"v2w_{s}.{i:03d}.txt",
                        initsub=f"w{i:03d}",
                        o0=origins[start],
                        o1=origins[start + len(sel) - 1] + 1,
                        n_jobs=len(sel) * SEEDS,
                        est_s=round(est),
                        tlim_min=60,
                        gpu=gpu,
                        partition="debug",
                        nice=nice,
                    )
                )
                start += len(sel)
                k += 1
        rest = len(blocks) - start
        n = max(1, math.ceil(rest * SEEDS * sref / 3600 / target))
        cuts = [start + round(rest * j / n) for j in range(n + 1)]
        for j in range(n):
            sel = blocks[cuts[j] : cuts[j + 1]]
            name = f"v2w_{s}.{i:03d}.p{j:02d}.txt"
            (od / name).write_text("\n".join(ln for b in sel for ln in b) + "\n")
            est = len(sel) * SEEDS * sref
            rows.append(
                dict(
                    piece=f"pieces/{name}",
                    orig=f"v2w_{s}.{i:03d}.txt",
                    initsub=f"w{i:03d}",
                    o0=origins[cuts[j]],
                    o1=origins[cuts[j + 1] - 1] + 1,
                    n_jobs=len(sel) * SEEDS,
                    est_s=round(est),
                    tlim_min=tlim(est),
                    gpu=gpu,
                    partition="gpu",
                    nice=nice,
                )
            )
        # every origin of the original file in exactly one piece
        got = sum(r["n_jobs"] for r in rows if r["orig"] == f"v2w_{s}.{i:03d}.txt")
        assert got == len(lines), (s, i, got, len(lines))
    m = pd.DataFrame(rows)
    m.to_csv(OUT / win / f"pieces_{s}.csv", index=False, lineterminator="\n")
    print(
        win,
        s,
        gpu,
        len(m),
        "pieces",
        f"est {m.est_s.sum() / 3600:.1f} {gpu}-h",
        f"piece est {m[m.partition == 'gpu'].est_s.min() / 3600:.1f}-{m[m.partition == 'gpu'].est_s.max() / 3600:.1f} h",
        f"tlim {m.tlim_min.min()}-{m.tlim_min.max()} min",
    )
