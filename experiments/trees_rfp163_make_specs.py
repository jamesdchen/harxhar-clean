"""Generate specs/causal_tune_trees_<token>.py: the per-bar tree spec with a dev-span random forest as its SHIPPED forest.

rfp163 (2026-10-07) is the worked instance: the tuned rungs (Optuna, specs/causal_tune_trees_optuna_rfp.py; random
search, specs/causal_tune_trees_tuned_rfp.py) re-run the random forest with the trial-163 dev-span configuration
(cluster/rfprev163_rf_config.json, the 400-trial search's leader at the time) as the shipped forest instead of
sklearn's defaults.  The generator is PARAMETERIZED by the configuration file and the key token (--config, --token;
the final forest's run uses token rf400), the defaults being the rfp163 instance.  Both tuned specs take the shipped
parameters from causal_tune_trees_tuned_jobs.tree_setup(), which EXECUTES the source at
causal_tune_trees_tuned_jobs.TREE_SPEC (specs/causal_tune_trees.py by default); the wrappers point TREE_SPEC at the
file written here (HPC_KW_TREE_SPEC).

The written file is specs/causal_tune_trees.py with EXACTLY TWO lines replaced, by the substitution that
specs/causal_tune_trees_rfdev.py (the untuned dev-forest wrapper) applies before it execs that spec:

  RF_PARAMS: dict = {"n_estimators": 100}  # the sklearn default, stated
    -> RF_PARAMS: dict = <rfdev.rf_params(<config>)!r>  # <token>: ...
  "rf": "no precedent: sklearn RandomForestRegressor defaults",
    -> "rf": <the provenance string rfdev's main() builds for that configuration path>,

so the code is the code the untuned wrapper runs with HPC_KW_RF_CONFIG=<config> (only the comment after RF_PARAMS
differs: "# <token>: ..." instead of "# rfdev: <path>"; asserted below).  For rfp163: RF_PARAMS = {'n_estimators': 355,
'max_depth': 14, 'min_samples_leaf': 2, 'max_features': 0.7356..., 'max_samples': 0.8810..., 'bootstrap': True} --
rfdev.rf_params scales the pooled-window leaf minimum to the per-bar window, max(1, round(27 * TRAIN_WIN / 24000)) = 2
at TRAIN_WIN = 2000.  That scaling is BAKED INTO the written file, so the file is valid for TRAIN_WIN = 2000 only (this
script refuses another TRAIN_WIN in its environment; the Optuna wrapper refuses to run at another TRAIN_WIN).

The written file is standalone: `python specs/causal_tune_trees_<token>.py` (env axes of specs/causal_tune_trees.py,
MODEL=rf) is the untuned per-bar forest with that configuration, the same run as specs/causal_tune_trees_rfdev.py
with HPC_KW_RF_CONFIG=<config>.

Each substitution is asserted to happen exactly once; the file is written with LF line endings.

Run from the repo root:
  python experiments/trees_rfp163_make_specs.py [--config cluster/rfprev163_rf_config.json] [--token rfp163]
         [--out specs/causal_tune_trees_<token>.py]                   write the spec
  ... --check    exit 1 unless the file on disk equals what would be written (the ship script and the cluster tasks
                 run this)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "causal_tune_trees.py"
RFDEV = ROOT / "specs" / "causal_tune_trees_rfdev.py"
CONFIG = "cluster/rfprev163_rf_config.json"  # the default instance (repo-relative, as the cluster jobs name it)
TOKEN = "rfp163"
TRAIN_WIN = 2000  # the per-bar window the leaf minimum is scaled to (baked into the written file)


def out_path(token: str) -> Path:
    return ROOT / "specs" / f"causal_tune_trees_{token}.py"


def repo_relative(path: str) -> str:
    """The configuration path as the jobs name it (relative to the repo root, forward slashes)."""
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(ROOT)
        except ValueError:
            raise SystemExit(
                f"--config {path} lies outside the repo; copy it under cluster/"
            ) from None
    rel = p.as_posix()
    if not (ROOT / rel).is_file():
        raise SystemExit(f"--config {rel} not found under {ROOT}")
    return rel


def load_rfdev():
    """specs/causal_tune_trees_rfdev.py as a module (its main() is not run)."""
    spec = importlib.util.spec_from_file_location("causal_tune_trees_rfdev", RFDEV)
    assert spec is not None and spec.loader is not None, RFDEV
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def provenance(rfdev, path: str, params: dict) -> str:
    """The provenance string specs/causal_tune_trees_rfdev.py main() builds (same expression)."""
    with open(ROOT / path, encoding="utf-8") as f:
        cfg = json.load(f)
    pooled = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    prov = f"dev-span configuration {path} ({cfg.get('name', '?')} sha {cfg.get('sha', '?')})"
    if "min_samples_leaf" in pooled and pooled.get("min_samples_leaf") != params.get(
        "min_samples_leaf"
    ):
        prov += (
            f"; min_samples_leaf scaled {pooled['min_samples_leaf']} -> {params['min_samples_leaf']} "
            f"for the {rfdev._train_win()}-row per-bar window"
        )
    return prov


def rf_params(config: str = CONFIG) -> dict:
    """rfdev.rf_params of the configuration (repo-relative path), at TRAIN_WIN 2000."""
    tw = os.environ.get("HPC_KW_TRAIN_WIN", os.environ.get("TRAIN_WIN", str(TRAIN_WIN)))
    if int(tw) != TRAIN_WIN:
        raise SystemExit(
            f"TRAIN_WIN={tw} in the environment: the shipped-forest spec is generated for TRAIN_WIN {TRAIN_WIN}"
        )
    rfdev = load_rfdev()
    cwd = os.getcwd()
    os.chdir(ROOT)  # rf_params opens the configuration by its repo-relative path
    try:
        return rfdev.rf_params(config)
    finally:
        os.chdir(cwd)


def generate(config: str = CONFIG, token: str = TOKEN) -> tuple[str, dict]:
    if not re.fullmatch(r"[a-z][a-z0-9]*", token):
        raise SystemExit(
            f"--token {token!r}: lower-case letters and digits only (it names files, roots and tables)"
        )
    config = repo_relative(config)
    rfdev = load_rfdev()
    params = rf_params(config)
    if "max_depth" not in params:
        raise SystemExit(
            f"{config}: no max_depth (the Optuna wrapper re-centres the depth axis on it)"
        )
    prov = provenance(rfdev, config, params)
    src = SPEC.read_text(
        encoding="utf-8"
    )  # universal newlines: a CRLF worktree copy reads as LF
    for line in (rfdev.RF_LINE, rfdev.PROV_LINE):
        n = src.count(line)
        if n != 1:
            raise SystemExit(f"{SPEC}: {line!r} occurs {n} times, not once")
    with open(ROOT / config, encoding="utf-8") as f:
        name = json.load(f).get("name", "?")
    new_rf = (
        f"RF_PARAMS: dict = {params!r}  # {token}: {config} ({name}; min_samples_leaf scaled for TRAIN_WIN "
        f"{TRAIN_WIN}), written by experiments/trees_rfp163_make_specs.py"
    )
    new_prov = f'"rf": {prov!r},'
    out = src.replace(rfdev.RF_LINE, new_rf).replace(rfdev.PROV_LINE, new_prov)
    # the code is rfdev's: its substituted source differs from this one in the RF_PARAMS comment only
    ref = src.replace(
        rfdev.RF_LINE, f"RF_PARAMS: dict = {params!r}  # rfdev: {config}"
    ).replace(rfdev.PROV_LINE, f'"rf": {prov!r},')
    a, b = out.splitlines(), ref.splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) == len(src.splitlines()), (len(a), len(b))
    assert (
        len(diff) == 1 and a[diff[0]].split("  #")[0] == b[diff[0]].split("  #")[0]
    ), diff
    changed = [i for i, (x, y) in enumerate(zip(a, src.splitlines())) if x != y]
    assert len(changed) == 2, f"{len(changed)} lines differ from {SPEC.name}, not 2"
    return out, params


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=CONFIG,
        help="the dev-span configuration JSON (repo-relative)",
    )
    ap.add_argument(
        "--token",
        default=TOKEN,
        help="the key token: specs/causal_tune_trees_<token>.py",
    )
    ap.add_argument(
        "--out", default="", help="default specs/causal_tune_trees_<token>.py"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare with the file on disk, write nothing",
    )
    a = ap.parse_args()
    out = Path(a.out).resolve() if a.out else out_path(a.token)
    text, params = generate(a.config, a.token)
    rel = out.relative_to(ROOT) if ROOT in out.parents else out
    if a.check:
        ok = out.is_file() and out.read_bytes() == text.encode("utf-8")
        print(
            f"{rel}: {'up to date' if ok else 'DIFFERS from the generated text (or missing)'}"
        )
        return 0 if ok else 1
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {rel}: RF_PARAMS = {params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
