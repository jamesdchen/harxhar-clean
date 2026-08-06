# GPU on CARC — enablement runbook (2026-06-29)

How to run the torch DL path (`RegimeMoE`, `ae_ridge`, `patchts`) on a CARC GPU. Written while
the CPU MoE chain runs; **the `harxhar` env's torch must not be touched until that finishes** (the
in-flight jobs import `torch 2.12.1+cpu`; replacing it mid-run breaks the later rungs).

## TL;DR — do you even want GPU here?
**Not for the current cascade.** The `resid_regime` stage refits a *fresh* small MoE per cadence
block (~400×), each on a few-thousand-row h16-19 subset. That is the "thousands of tiny sliding-window
refits" pattern that the tree-tuning work already found **GPU loses on** (per-fit host↔device transfer
dominates; the fit is sub-second on CPU). GPU pays off only if the training pattern changes:
- a **single amortized fit** over all blocks (block/time as a feature) instead of per-block refits;
- **large experts** (deep/wide, many regimes) where one fit saturates a GPU;
- the **auction-imbalance experts** once that data is acquired (bigger model, richer state).

So: keep CPU for the per-block cascade; reach for GPU when you redesign toward a single big fit.

## Recon facts (verified 2026-06-29)
| partition | limit | GPUs/node | notes |
|---|---|---|---|
| `gpu` | 2-00:00:00 | `a100:2`, `a40:2`, `l40s:3`, `v100:2`, `p100:2` | main GPU partition; many p100 idle |
| `debug` | 1:00:00 | `a40:2`, `p100:2` | quick GPU smoke tests |
- CUDA modules: `cuda/12.4.0`, `cuda/12.6.3`. Account `pollok_1603`, QOS `normal` (no partition restriction in the assoc → `gpu` submit expected to work; confirm with one debug job).

## Step 1 — a SEPARATE CUDA torch env (don't disturb `harxhar`)
The pip CUDA wheel **bundles its own CUDA runtime**, so `module load cuda` is not required for torch
itself — only a compatible NVIDIA driver on the node (a100/a40/l40s/v100 all fine).
```bash
module load conda
source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda create -y -p /home1/jc_905/.conda/envs/harxhar_gpu --clone harxhar   # numpy/pandas/sklearn/interpret/xgboost
/home1/jc_905/.conda/envs/harxhar_gpu/bin/python -m pip uninstall -y torch   # remove the +cpu clone
/home1/jc_905/.conda/envs/harxhar_gpu/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```
(Alternatively, after the chain finishes, just swap `harxhar`'s torch to `+cu124` — but a clone keeps a
known-good CPU env for the cascade.)

## Step 2 — GPU sbatch directives
Add to any dig sbatch (e.g. a `moe_dig_gpu.sbatch`):
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1        # or a40:1 / v100:1 / p100:1 ; debug: --partition=debug --gres=gpu:p100:1 --time=1:00:00
#SBATCH --account=pollok_1603
#SBATCH --cpus-per-task=4
PY=/home1/jc_905/.conda/envs/harxhar_gpu/bin/python
```
**No code change needed**: `RegimeMoE.__init__` already auto-selects `cuda` when
`torch.cuda.is_available()`. On a GPU node it uses the GPU; on CPU nodes it falls back.

> Note the 100-job / 2000-core cap is shared; the `gpu` partition also has its own GRES limits. A
> 90-task GPU array would need 90 GPUs — **don't**. For GPU, switch the design to *few large jobs*,
> not a 90-way array of tiny fits (which is the whole reason CPU is right for the current cascade).

## Step 3 — verify (one debug job)
```bash
srun --partition=debug --gres=gpu:p100:1 --account=pollok_1603 --time=0:10:00 \
  /home1/jc_905/.conda/envs/harxhar_gpu/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expect `2.x.y+cu124 True <GPU name>`. Then a `RegimeMoE.fit` on mock data should report `device=cuda`.

## When GPU becomes the right call (the design pivot)
The DL handoff (`mechanism_and_data_to_buy_2026-06-28.md` §8) says state-space nonlinearity, not path.
If the MoE earns its keep at small scale (CPU), the scale-up that justifies GPU is a **single
walk-forward-consistent fit** — e.g. pretrain a shared gate+expert representation across all blocks
with time as a conditioning feature, fine-tuning per cadence — replacing the 400 independent refits.
That is also the form that ingests the auction-imbalance feed cleanly (gate = regime, experts = the
new microstructure state). Until then, GPU is latent capacity, not a bottleneck.
