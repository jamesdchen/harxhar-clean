"""Bundle the native PufferLib-5.0 MinGRU walk-forward for the Colab runner (colab_runner.ipynb, 2026-10-13).

The Colab notebook replaces the cluster job (cluster/puffer_forecast_phase2b_carc.sbatch, ..._phase2_carc_anygpu
.sbatch) when no cluster GPU is free: it builds the same native source against PufferLib 5.0 at the pinned
commit (src/ unchanged) and runs the same blocks (tune -> phase2_select.py -> refit). This script packs what
that job reads from $HOME/puffer_forecast on the cluster:

    native/forecast_train.cu, native/puffer_substrate.cuh
    phase2_select.py, dates.csv
    inputs/<PHASE>/   blocks.json, tune_block*.txt (and maskcheck.txt for phase2b)
    data/<design>/    baseline, all_features
    MANIFEST.md5, bundle.json

    python experiments/puffer_forecast/colab_bundle.py [PHASE]        (default phase2c)

Writes results/puffer_forecast_2026-10-12/_work/colab/puffer_colab_<PHASE>.zip; upload it to
MyDrive/puffer_forecast/ and run the notebook. Stdlib only.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments" / "puffer_forecast"
WORK = REPO / "results" / "puffer_forecast_2026-10-12" / "_work"
OUT = WORK / "colab"
COMMIT = "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"
DESIGNS = ("baseline", "all_features")


def members(phase: str) -> list[tuple[Path, str]]:
    """(source file, path inside the bundle) for every file the cluster job reads."""
    inputs = WORK / phase
    if not (inputs / "blocks.json").exists():
        sys.exit(
            f"{inputs} has no blocks.json: run experiments/puffer_forecast/{phase}.py first"
        )
    files = [
        (HERE / "native" / "forecast_train.cu", "native/forecast_train.cu"),
        (HERE / "native" / "puffer_substrate.cuh", "native/puffer_substrate.cuh"),
        (HERE / "phase2_select.py", "phase2_select.py"),
        (
            WORK / "dates.csv",
            "dates.csv",
        ),  # row -> 16:00 session, for the notebook's manual training
    ]
    files += [
        (p, f"inputs/{phase}/{p.name}") for p in sorted(inputs.iterdir()) if p.is_file()
    ]
    for d in DESIGNS:
        files += [
            (p, f"data/{d}/{p.name}")
            for p in sorted((WORK / "data" / d).iterdir())
            if p.is_file()
        ]
    return files


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "phase2c"
    files = members(phase)
    md5 = {arc: hashlib.md5(src.read_bytes()).hexdigest() for src, arc in files}
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    info = {
        "phase": phase,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "repo_head": head,
        "puffer_commit": COMMIT,
        "forecast_train_md5": md5["native/forecast_train.cu"],
        "n_files": len(files),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"puffer_colab_{phase}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in files:
            z.write(src, arc)
        z.writestr("MANIFEST.md5", "".join(f"{h}  {arc}\n" for arc, h in md5.items()))
        z.writestr("bundle.json", json.dumps(info, indent=1) + "\n")
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
