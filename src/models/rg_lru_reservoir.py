"""Reservoir RG-LRU — FROZEN real-gated recurrence as a nonlinear temporal feature bank, read out by
a closed-form rank-1 rolling ridge. The unification of the two tracks: gated multi-timescale sequence
memory (RG-LRU) fit EXACTLY, with NO SGD.

Why frozen gates are still expressive (echo-state theory): freezing ``W_r, W_i, Λ`` fixes the gate
FUNCTION, not the input-dependence — ``a_t = exp(-C·softplus(Λ)·σ(W_r x_t))`` still varies with x_t,
and the ``d`` channels give ``d`` distinct random input-modulated decays. ESNs with a LINEAR readout
and untrained recurrent weights are universal approximators of causal fading-memory filters
(Grigoryeva & Ortega 2018; Gonon & Ortega 2020) — so with enough width the frozen reservoir + ridge
readout spans the same function class as a trained recurrence. What freezing costs is width-efficiency,
recovered without gradients by scaling ``d``, tuning ``input_scale`` / the Λ decay distribution, or a
STRUCTURED (timescale-aligned) gate init.

Pipeline:
  1. FROZEN reservoir: stream the RG-LRU recurrence over the standardized feature series ONCE (jit
     scan, seconds) -> per-layer hidden states H_t (nonlinear multi-timescale temporal features).
  2. Design D_t = [ standardized feature_row_t  ⊕  reservoir states H_t ]  (raw block included so the
     readout is >= the linear feature-ridge: it can zero the reservoir columns).
  3. Readout: rank-1 rolling ridge over D (``RollingLeastSquares`` via the shared ``_roll_predict``
     core) — walk-forward, EXACT, O(p²)/bar, splittable across cores/jobs exactly like ``dlinear_feat``.

Fit target = ``adj_RV`` (LS + Duan, same convention as the Ridge base). No SGD, O(1)-streaming states,
rank-1-exact readout — the best fit for CPU + the walk-forward + the RollingLeastSquares infra.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from src.models.dlinear_feat import _roll_predict
from src.models.rg_lru import RGLRU

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CACHE_ROOT_DEFAULT = "results/covid_imp_rank"

DEFAULT = {
    "bucket": "all_buckets",
    "d_model": 256,  # reservoir width per layer (bigger -> more coverage, ridge readout absorbs)
    "n_layers": 2,
    "input_scale": 1.0,  # frozen input-projection gain (a reservoir hyperparameter; tune, no SGD)
    "train_window": 24000,  # W: rolling readout window (bars)
    "alpha": 1.0,  # ridge penalty on the readout
    "refit": 480,  # readout solve cadence
    "fit_intercept": True,
    "reservoir_seed": 0,  # frozen-gate RNG seed (determinism; sweep for a small reservoir ensemble)
    "gates": "random",  # random | structured (HAR-timescale decays) | trained (amortized SGD-once)
}

# truncated-BPTT config for the amortized (trained-gates) reservoir; overridable via args.
_TRAIN_CFG = {
    "seq_len": 192,
    "burn_in": 48,
    "epochs": 8,
    "batch_size": 32,
    "lr": 3e-3,
    "weight_decay": 1e-4,
}
C_LOG_TIMESCALE = (
    8.0 * 0.88
)  # C_LOG · σ(bias=2): the effective decay rate at init gate r≈0.88


def _har_timescale_log_lambda(d: int) -> torch.Tensor:
    """Structured decay bank (idea 1): channels round-robin over HAR half-lives {1,5,25,125,625,3125}
    bars. softplus(Λ) is set so a ≈ 0.5^(1/hl) at the init recurrence gate r≈σ(2)=0.88 — i.e. each
    channel forgets on a HAR timescale instead of a random one."""
    hl = np.array([1, 5, 25, 125, 625, 3125], dtype=np.float64)[np.arange(d) % 6]
    sp = np.maximum(-np.log(0.5) / (hl * C_LOG_TIMESCALE), 1e-6)  # target softplus(Λ)
    return torch.log(torch.expm1(torch.tensor(sp, dtype=torch.float32)))


class Reservoir(torch.nn.Module):
    """Stacked RG-LRU layers with ALL parameters frozen — a deterministic nonlinear temporal feature
    map. ``states`` streams the recurrence over the series and returns the concatenated per-layer
    hidden states (O(1) memory/step; state carried across chunks)."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        input_scale: float,
        seed: int,
        gates: str = "random",
    ):
        super().__init__()
        # Seed the GLOBAL torch RNG: nn.Linear inside RGLRU inits from it, so this is what makes the
        # frozen reservoir fully determined by `seed` (a per-Generator proj seed would NOT fix the
        # RGLRU gate weights). log_lambda / LayerNorm init deterministically (no RNG draw).
        torch.manual_seed(int(seed))
        self.proj = torch.nn.Linear(d_in, d_model)
        with torch.no_grad():
            self.proj.weight.mul_(
                float(input_scale)
            )  # input-scaling reservoir hyperparameter
            self.proj.bias.zero_()
        self.layers = torch.nn.ModuleList(RGLRU(d_model) for _ in range(n_layers))
        self.norms = torch.nn.ModuleList(
            torch.nn.LayerNorm(d_model) for _ in range(n_layers)
        )
        if (
            gates == "structured"
        ):  # idea 1: replace the random decay bank with HAR timescales
            lam = _har_timescale_log_lambda(d_model)
            with torch.no_grad():
                for layer in self.layers:
                    layer.log_lambda.copy_(lam)  # type: ignore[operator]  # nn.ModuleList -> Module attr
        for p in self.parameters():
            p.requires_grad_(False)
        self.out_dim = n_layers * d_model

    @torch.no_grad()
    def states(self, X: np.ndarray, device, chunk: int = 8192) -> np.ndarray:
        N = X.shape[0]
        H = np.empty((N, self.out_dim), dtype=np.float32)
        carry: list[torch.Tensor] | None = None
        for lo in range(0, N, chunk):
            hi = min(lo + chunk, N)
            h = self.proj(
                torch.as_tensor(X[lo:hi][None], dtype=torch.float32, device=device)
            )
            per_layer, new_carry = [], []
            for li, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                h0 = carry[li] if carry is not None else None
                seq, hT = layer(norm(h), h0)
                h = h + seq
                per_layer.append(seq)
                new_carry.append(hT.detach())
            carry = new_carry
            H[lo:hi] = torch.cat(per_layer, dim=-1).squeeze(0).cpu().numpy()
        return H


def train_reservoir(X, y, d_in, d_model, n_layers, device, seed, train_cfg=None):
    """Amortized (trained-gates) reservoir: SGD-fit an RG-LRU ONCE on the standardized training arrays,
    then LIFT the learned gate weights into a frozen ``Reservoir``. The one-time SGD orients the gates
    (width-efficient -> small state) while inference stays the exact rank-1 rolling-ridge readout."""
    from src.models.rg_lru import RGLRUForecaster, _fit

    torch.manual_seed(int(seed))
    fc = RGLRUForecaster(d_in, d_model, n_layers).to(device)
    _fit(fc, X, y, {**_TRAIN_CFG, **(train_cfg or {})}, device)
    res = Reservoir(d_in, d_model, n_layers, 1.0, seed).to(device)
    res.proj.load_state_dict(fc.proj.state_dict())
    for rl, fl in zip(res.layers, fc.layers):
        rl.load_state_dict(fl.state_dict())
    for rn, fn in zip(res.norms, fc.norms):
        rn.load_state_dict(fn.state_dict())
    for p in res.parameters():
        p.requires_grad_(False)
    return res


# ── Readout: rank-1 rolling ridge over the (in-memory or mmap'd) design ───────────────────────


def _predict_block_design(
    design_path: str,
    y_path: str,
    W: int,
    a: int,
    b: int,
    alpha: float,
    refit: int,
    fit_intercept: bool,
):
    """Chunk worker: mmap ONLY the ``[a-W, b)`` slice of the scratch design + targets (no big pickling)."""
    off = a - W
    D = np.ascontiguousarray(np.load(design_path, mmap_mode="r")[off:b])
    y = np.asarray(np.load(y_path, mmap_mode="r")[off:b], dtype=np.float64)
    return _roll_predict(
        D, y, off, W, a, b, alpha, refit, fit_intercept, 0
    )  # t_min=0: reservoir valid ∀t


def _readout(
    D: np.ndarray,
    y: np.ndarray,
    W: int,
    alpha: float,
    refit: int,
    fit_intercept: bool,
    oos_lo: int,
    oos_hi: int,
    n_jobs: int,
    scratch_dir: str,
) -> np.ndarray:
    if n_jobs <= 1:
        _, preds = _roll_predict(
            D, y, 0, W, oos_lo, oos_hi, alpha, refit, fit_intercept, 0
        )
        return preds
    # Parallel: persist design + targets to scratch memmaps; workers mmap their own slice.
    dpath = os.path.join(scratch_dir, "_reservoir_design.npy")
    ypath = os.path.join(scratch_dir, "_reservoir_y.npy")
    np.save(dpath, D)
    np.save(ypath, np.asarray(y, dtype=np.float64))
    raw = np.linspace(oos_lo, oos_hi, n_jobs + 1)
    edges = [oos_lo]
    for e in raw[1:-1]:
        s = int(np.ceil(e / refit) * refit)
        if edges[-1] < s < oos_hi:
            edges.append(s)
    edges.append(oos_hi)
    blocks = [(edges[j], edges[j + 1]) for j in range(len(edges) - 1)]
    logger.info(f"readout chunk-parallel: {len(blocks)} blocks over {n_jobs} procs")
    preds = np.empty(oos_hi - oos_lo, dtype=np.float64)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [
            ex.submit(
                _predict_block_design,
                dpath,
                ypath,
                W,
                a,
                b,
                alpha,
                refit,
                fit_intercept,
            )
            for a, b in blocks
        ]
        for f in futs:
            a, blk = f.result()
            preds[a - oos_lo : a - oos_lo + len(blk)] = blk
    return preds


def compute(args) -> None:
    from src.evaluation.metrics import (
        build_results_dataframe,
        calculate_metrics,
        save_chunk_reduce,
    )

    def _p(key: str):
        v = getattr(args, key, None)
        return DEFAULT[key] if v is None else v

    cache_root = getattr(args, "cache_root", None) or CACHE_ROOT_DEFAULT
    bucket = str(_p("bucket"))
    W = int(_p("train_window"))
    alpha, refit = float(_p("alpha")), int(_p("refit"))
    fit_intercept = bool(_p("fit_intercept"))
    n_jobs = int(getattr(args, "n_jobs", None) or 1)
    device = torch.device("cpu")

    b = f"{cache_root}/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.asarray(np.load(f"{b}/y.npy"), dtype=np.float64)
    base = np.asarray(np.load(f"{b}/base.npy"), dtype=np.float64)
    N, d_in = X.shape

    # Causal standardization from the training region (features already causal/rank-space).
    mu, sd = X[:W].mean(0), X[:W].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = ((X - mu) / sd).astype(np.float32)

    d_model, n_layers, seed, gates = (
        int(_p("d_model")),
        int(_p("n_layers")),
        int(_p("reservoir_seed")),
        str(_p("gates")),
    )
    if (
        gates == "trained"
    ):  # amortized: SGD the gates once on the standardized train region, then freeze
        ymu, ysd = float(y[:W].mean()), float(y[:W].std() or 1.0)
        res = train_reservoir(
            Xs[:W],
            ((y[:W] - ymu) / ysd).astype(np.float32),
            d_in,
            d_model,
            n_layers,
            device,
            seed,
        )
    else:  # random | structured
        res = Reservoir(
            d_in, d_model, n_layers, float(_p("input_scale")), seed, gates=gates
        ).to(device)
    logger.info(
        f"reservoir gates={gates} d_in={d_in} d_model={d_model} layers={n_layers} out_dim={res.out_dim} seed={seed}"
    )
    t0 = time.time()
    H = res.states(Xs, device)  # (N, out_dim) frozen nonlinear temporal features
    logger.info(f"reservoir states {N} bars in {time.time() - t0:.1f}s")

    # Design = [standardized features ⊕ reservoir states], columns standardized (causal) so one ridge
    # alpha treats them comparably; readout >= linear feature-ridge (can zero the reservoir block).
    D = np.ascontiguousarray(np.hstack([Xs.astype(np.float64), H.astype(np.float64)]))
    dmu, dsd = D[:W].mean(0), D[:W].std(0)
    dsd = np.where(dsd > 0, dsd, 1.0)
    D = (D - dmu) / dsd

    oos_hi = N if getattr(args, "end", -1) in (None, -1) else min(int(args.end), N)
    logger.info(
        f"readout rank-1 ridge p={D.shape[1]} W={W} alpha={alpha} refit={refit} OOS=[{W},{oos_hi}) n_jobs={n_jobs}"
    )
    t1 = time.time()
    with tempfile.TemporaryDirectory() as scratch:
        preds = _readout(
            D, y, W, alpha, refit, fit_intercept, W, oos_hi, n_jobs, scratch
        )
    logger.info(f"readout in {time.time() - t1:.1f}s ({len(preds)} preds)")

    y_oos, base_oos = y[W:oos_hi], base[W:oos_hi]
    horizon = int(getattr(args, "horizon", 1) or 1)
    df = build_results_dataframe(preds, y_oos, np.arange(W, oos_hi), base_oos, horizon)
    out = args.output_file
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    save_chunk_reduce(df, out)
    metrics = calculate_metrics(df)
    with open(os.path.join(os.path.dirname(out) or ".", "metrics.json"), "w") as f:
        json.dump(metrics, f)
    logger.info(f"Saved {len(df)} rows -> {out}  QLIKE={metrics.get('qlike')}")
