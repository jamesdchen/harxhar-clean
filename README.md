# harxhar — pre-hpc-agent-consumption snapshot

Realized-volatility forecasting experiment. Notebook-authored, runnable today via `run.py`; ready to be consumed by `hpc-agent` for chunked HPC execution.

## Layout

```
src/
├── data/               Iterate on data source / NaN policy
│   └── loading.py        load_raw_data, parse_exog_cols
├── features/           Iterate on lag scales / scaler variants
│   ├── transforms.py     diurnal_adjust, winsorize, generate_har_features
│   └── scaling.py        RollingRobustScaler, rolling_robust_scale
├── models/             Iterate on a model in isolation
│   ├── ridge.py          Ridge (closed-form, refit every step)
│   ├── xgboost.py        XGBoost
│   ├── lightgbm.py       LightGBM
│   ├── random_forest.py  Random forest
│   ├── pcr.py            PCA + Ridge
│   ├── baseline.py       Naive HAR-MA(125)
│   ├── patchts.py        PatchTST transformer (GPU)
│   └── ae_ridge.py       Autoencoder + Ridge (GPU)
├── backtest/           Wire data → features → model → eval
│   ├── executor.py       run_executor() — ML scaffold
│   ├── dl_executor.py    DL scaffold
│   ├── multi_stage.py    MultiStageBacktest (baseline + residual feature + regressor)
│   └── tune_tree.py      Optuna loop
└── evaluation/         Scoring + strategy
    ├── metrics.py        Duan smearing, QLIKE / MSE / MAE
    └── strategy.py       PnL evaluation

notebooks/              Scratchpad mirroring src/ (data / features / models / …)
analysis/               Standalone analysis scripts (self-bootstrapping sys.path)
experiments/            One-off campaign + worker scripts, formerly at the root
jobs/                   Cluster submission
├── slurm/                *.sbatch, *.slurm
├── sge/                  *.sge
└── submit/               submit_*.sh drivers
configs/                One YAML per experiment
data/                   Input parquets (30-min bar data)
results/                Tracked summary CSVs (outputs land here)
writeup/                LaTeX paper + figures
run.py                  Entry point: python run.py --config configs/<name>.yaml
```

### Layout change, 2026-08-04

67 one-off scripts and 68 cluster job scripts used to sit at the repository
root, which had grown to 144 files. They now live under `experiments/` and
`jobs/`. `run.py` stays at the root, because it is the documented entry point.

Being at the root meant the interpreter put the root on `sys.path`
automatically, so `import src.…` worked and so did importing a sibling script
by bare name. Neither survives the move, so:

- `experiments/_bootstrap.py` restores both. The 57 moved scripts that need it
  import it first, which makes them work from any working directory.
- Job scripts under `jobs/` also export
  `PYTHONPATH="$PWD:$PWD/experiments"` after their `cd`, so the guarantee does
  not rest on the shim alone.
- Notebooks add `str(REPO)+'/experiments'` alongside their existing `REPO`
  path insert.

To submit a cluster job the path now carries the directory —
`sbatch jobs/slurm/<name>.sbatch`, `qsub jobs/sge/<name>.sge`,
`bash jobs/submit/<name>.sh`. Nothing else about the workflow changes.

Newer additions:

- `src/features/spectral_embedding.py` — graph-Laplacian eigenmaps of temporal residual windows.
- `src/models/knn.py` — generic kNN regressor (Gaussian / uniform / distance weighting).
- `src/backtest/multi_stage.py` — generic harness for baseline + feature-on-residuals + regressor pipelines.
- `notebooks/features/spectral_embedding.ipynb` — exploration of the embedding on real harxhar residuals (2005-2024).

## Usage

```bash
pip install -r requirements.txt

# Single-model configs
python run.py --config configs/ridge.yaml
python run.py --config configs/xgboost.yaml --override train_window=750 seed=7
python run.py --config configs/knn.yaml
python run.py --config configs/spectral_knn.yaml
```

Every YAML has a top-level `model: <name>` field naming a module under
`src/models/`. `run.py` imports that module and calls its `run(**params)` (ML
/ composed models) or `compute(args)` (DL). Simple models (`ridge`, `xgboost`,
`lightgbm`, `random_forest`, `knn`) and composed models (`spectral_knn`) use
the same dispatch — they all configure the same `MultiStageBacktest` harness;
only their residualizer / feature / regressor stages differ.

`--override key=value ...` patches the config from the command line; values
are parsed as YAML scalars.

## hpc-agent consumption

This snapshot is the state *before* `hpc-agent` consumes it. After consumption,
`hpc-agent` will add:

- `.hpc/axes.yaml` — classified data axis (bounded-halo with `train_window * 48`
  look-back for the `ml_*` executors)
- `.hpc/tasks.py`, `.hpc/cli.py` — generated chunk plan + dispatcher
- A vendored `hpc_agent` wheel (optional, for cluster pin)

`src/models/<x>.py` runs as plain Python today (whole-series, no chunking). When
`hpc-agent` re-injects its runtime, the same modules become chunk-aware via
`@register_run` + `current_slice()` without changing the model code.
