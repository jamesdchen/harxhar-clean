"""Generate specs/causal_tune_trees_<token>.py: the per-bar tree spec with a dev-span XGBoost menu entry as its SHIPPED XGBoost.

User decision 2026-10-07 (writeup/MASTER_TABLE_REVIEW_2026-10-01.md item 76): the per-bar XGBoost takes the TOP entry of
the 1998-2002 dev-span tree menu, xgb_opt_00 (tree_expert_10, sha 88bec9f385c7, dev MSE 0.107894), instead of
xgb_opt_06 (tree_expert_16, the 7th-ranked entry, hand-picked with no recorded basis after post-2002 scoring) -- the
rule LightGBM (lgbm_opt_00) and the random forest (rf_opt_00) follow.  The XGBoost twin of
experiments/trees_rfp163_make_specs.py (the random forest's generator), PARAMETERIZED by the configuration file and
the key token:

  --config cluster/xgbtop_xgb_config.json  --token xgbtop   (default) xgb_opt_00: the new shipped XGBoost
  --config cluster/xgbtop_shipped_xgb.json --token xgb06    xgb_opt_06: the shipped XGBoost, for the reproduction
                                                             gate (experiments/trees_xgbtop_gate.py)

The written file is specs/causal_tune_trees.py with EXACTLY TWO substitutions:

  the XGB_PARAMS block (the 12-line dict literal "XGB_PARAMS: dict = {" .. "}")
    -> XGB_PARAMS: dict = <the configuration's "params", key order kept>  # <token>: <config> (...)
  "xgb": "tree_expert_16 / xgb_opt_06 sha 195b74503d9d (chunk npz meta)",
    -> "xgb": "<arm> / <name> sha <sha> (<config>; bank chunk npz meta)",

The parameters are the POOLED-window values (the bank's 24,000-bar window, the configuration's provenance.window);
the spec's own code below the block scales min_child_weight to the per-bar window exactly as it scales the shipped
config's (XGB_PARAMS["min_child_weight"] = pooled * TRAIN_WIN / 24000: 50.20 -> 4.183 for xgb_opt_00, 24.86 -> 2.072
for xgb_opt_06 at TRAIN_WIN 2000), so the leaf minimum is "scaled like the shipped config's" by the same line, and the
file is valid for any TRAIN_WIN (no baked scaling, unlike the forest's generator).  Every other line -- design, window
mask, refit / importance cadence, the model constructor XGBRegressor(**XGB_PARAMS, n_jobs = SLURM_CPUS_PER_TASK,
random_state 42, verbosity 0), TreeSHAP, outputs -- is the shipped spec's.

Checks (each substitution exactly once; the written XGB_PARAMS literal evaluates to the configuration's params; the
configuration's sha re-derived by experiments/tree_menu_dev.py's rule; for the xgb06 configuration the dict equals
the shipped spec's XGB_PARAMS literal, so the gate file differs from the shipped spec in the two lines' text only).
The file is written with LF line endings and is standalone: `python specs/causal_tune_trees_<token>.py` (env axes of
specs/causal_tune_trees.py, MODEL=xgb) is the per-bar XGBoost with that configuration.

expected_params(config, train_win) = the per-bar parameters the spec records in its npz meta "params" (the pack
scripts compare them before DONE).

Run from the repo root:
  python experiments/trees_xgbtop_make_specs.py [--config cluster/xgbtop_xgb_config.json] [--token xgbtop]   write
  ... --check     exit 1 unless the file on disk equals what would be written (the cluster tasks run this)
  ... --params [--train-win 2000]   print expected_params as JSON
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "causal_tune_trees.py"
CONFIG = "cluster/xgbtop_xgb_config.json"
TOKEN = "xgbtop"
GATE_CONFIG = "cluster/xgbtop_shipped_xgb.json"
GATE_TOKEN = "xgb06"
XGB_HEAD = "XGB_PARAMS: dict = {\n"
PROV_LINE = '"xgb": "tree_expert_16 / xgb_opt_06 sha 195b74503d9d (chunk npz meta)",'
POOLED_WINDOW_ROWS = (
    24000  # the spec's constant: the pooled window the dev-span menu tuned on
)
SPEC_OWNED = ("n_jobs", "random_state", "verbosity")


def out_path(token: str) -> Path:
    return ROOT / "specs" / f"causal_tune_trees_{token}.py"


def repo_relative(path: str) -> str:
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


def menu_sha(family: str, params: dict) -> str:
    """experiments/tree_menu_dev.py's _sha (the menu's entry hash), the same expression (that module imports the
    experiments bootstrap, so it is restated here rather than imported on the cluster)."""
    blob = json.dumps({"family": family, **params}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def load_config(config: str) -> dict:
    with open(ROOT / config, encoding="utf-8") as f:
        cfg = json.load(f)
    p = cfg.get("params")
    if not isinstance(p, dict) or not p:
        raise SystemExit(f"{config}: no params object")
    if cfg.get("family") != "xgb":
        raise SystemExit(f"{config}: family {cfg.get('family')!r}, not xgb")
    bad = [k for k in SPEC_OWNED if k in p]
    if bad:
        raise SystemExit(f"{config}: {bad} are set by the spec, not the configuration")
    window = (cfg.get("provenance") or {}).get("window")
    if int(window or 0) != POOLED_WINDOW_ROWS:
        raise SystemExit(
            f"{config}: provenance.window {window}; the spec scales min_child_weight from {POOLED_WINDOW_ROWS}"
        )
    for k in (
        "max_depth",
        "learning_rate",
        "n_estimators",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    ):
        if k not in p:
            raise SystemExit(
                f"{config}: params lack {k} (the tuned grids / Optuna spaces are centred on it)"
            )
    if menu_sha("xgb", p) != cfg.get("sha"):
        raise SystemExit(
            f"{config}: params hash to {menu_sha('xgb', p)}, not the recorded sha {cfg.get('sha')}"
        )
    import xgboost as xgb

    xgb.XGBRegressor(**p)  # an unknown keyword fails here, before anything is written
    return cfg


def expected_params(config: str = CONFIG, train_win: int = 2000) -> dict:
    """The per-bar XGB_PARAMS the generated spec builds at TRAIN_WIN (its own scaling expression)."""
    cfg = load_config(repo_relative(config))
    p = dict(cfg["params"])
    _leaf_scale = train_win / POOLED_WINDOW_ROWS
    p["min_child_weight"] = p["min_child_weight"] * _leaf_scale
    return p


def xgb_block(src: str) -> tuple[int, int]:
    """[start, end) of the shipped XGB_PARAMS block in the spec's source (exactly one)."""
    n = src.count(XGB_HEAD)
    if n != 1:
        raise SystemExit(f"{SPEC}: {XGB_HEAD!r} occurs {n} times, not once")
    a = src.index(XGB_HEAD)
    b = src.index("\n}\n", a) + 3
    return a, b


def generate(config: str = CONFIG, token: str = TOKEN) -> tuple[str, dict]:
    if not re.fullmatch(r"[a-z][a-z0-9]*", token):
        raise SystemExit(
            f"--token {token!r}: lower-case letters and digits only (it names files, roots and tables)"
        )
    config = repo_relative(config)
    cfg = load_config(config)
    params = dict(cfg["params"])
    src = SPEC.read_text(
        encoding="utf-8"
    )  # universal newlines: a CRLF worktree copy reads as LF
    a, b = xgb_block(src)
    shipped = ast.literal_eval(src[a:b].split("=", 1)[1].strip())
    if src.count(PROV_LINE) != 1:
        raise SystemExit(
            f"{SPEC}: {PROV_LINE!r} occurs {src.count(PROV_LINE)} times, not once"
        )
    new_block = (
        f"XGB_PARAMS: dict = {params!r}  # {token}: {config} ({cfg['name']} sha {cfg['sha']}; pooled-window values, "
        f"min_child_weight scaled below as the shipped config's), written by experiments/trees_xgbtop_make_specs.py\n"
    )
    new_prov = f'"xgb": "{cfg.get("arm", "?")} / {cfg["name"]} sha {cfg["sha"]} ({config}; bank chunk npz meta)",'
    out = src[:a] + new_block + src[b:]
    out = out.replace(PROV_LINE, new_prov)
    # checks: the literal evaluates to the configuration's params (key order kept); two places changed
    got = ast.literal_eval(new_block.split("=", 1)[1].split("  #", 1)[0].strip())
    assert got == params and list(got) == list(params), (got, params)
    if (
        cfg["sha"] == "195b74503d9d"
    ):  # the shipped config: the same dict as the spec's literal
        assert got == shipped and list(got) == list(shipped), (got, shipped)
    s_lines, o_lines = src.splitlines(), out.splitlines()
    assert len(s_lines) - len(o_lines) == (src[a:b].count("\n") - 1), (
        len(s_lines),
        len(o_lines),
    )
    assert out.count(new_prov) == 1 and PROV_LINE not in out and XGB_HEAD not in out
    return out, params


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=CONFIG, help="the configuration JSON (repo-relative)"
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
    ap.add_argument(
        "--params",
        action="store_true",
        help="print the per-bar parameters (expected_params) as JSON",
    )
    ap.add_argument("--train-win", type=int, default=2000)
    a = ap.parse_args()
    if a.params:
        print(json.dumps(expected_params(a.config, a.train_win)))
        return 0
    out = Path(a.out).resolve() if a.out else out_path(a.token)
    text, params = generate(a.config, a.token)
    rel = out.relative_to(ROOT) if ROOT in out.parents else out
    if a.check:
        ok = out.is_file() and out.read_bytes() == text.encode("utf-8")
        print(
            f"{rel}: {'up to date' if ok else 'DIFFERS from the generated text (or missing)'}"
        )
        return 0 if ok else 1
    out.write_bytes(text.encode("utf-8"))
    print(
        f"wrote {rel}: XGB_PARAMS = {params} (pooled; per-bar at TRAIN_WIN 2000: {expected_params(a.config)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
