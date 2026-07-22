"""RG-LRU (Real-Gated Linear Recurrent Unit, Griffin/Hawk) forecaster — CPU-optimized for the
walk-forward RV backtest. Replaces the torch DLinear as the *input-gated sequence* model.

It is LINEAR IN THE STATE (hence "linear recurrent unit" and hence the scan applies), but the
transition/input coefficients are INPUT-DEPENDENT gates (sigmoids of x_t) — so the input→output map
is nonlinear even though the state recurrence is affine. LEARNING the gate parameters is non-convex,
so this module trains them by SGD. Input-adaptivity does NOT by itself require SGD, though: FREEZING
the gates (reservoir / echo-state) turns the hidden states into fixed nonlinear temporal features and
leaves only a LINEAR readout — a closed-form ridge, rank-1 updatable via ``RollingLeastSquares`` with
no SGD (the same exact/streaming track as ``bucket_ridge``). This file is the learned-gate variant.

Core recurrence (De et al. 2024, "Griffin"): a DIAGONAL, real-gated linear recurrence — a strict
generalization of HAR, where each channel has its own INPUT-DEPENDENT decay (learnable multi-
timescale, regime-adaptive memory) instead of HAR's fixed daily/weekly/monthly box averages::

    r_t = σ(W_r x_t + b_r)                        # recurrence gate      (B,T,d)
    i_t = σ(W_i x_t + b_i)                        # input gate
    a_t = exp(-C · softplus(Λ) ⊙ r_t)            # per-channel decay in (0,1); Λ learnable log-rate
    x̃_t = √(1 - a_t²) ⊙ (i_t ⊙ x_t)             # variance-preserving input normalization
    h_t = a_t ⊙ h_{t-1} + x̃_t                    # diagonal elementwise recurrence  (O(d) state)
    ŷ_t = head(h_t)

Mapping the three cited efficiency tricks to THIS use case (CPU-only, walk-forward):
  * Diagonal elementwise recurrence (O(d), O(1) streaming) — trick #3, the core. THE win here: it
    lets us TRAIN ONCE then STREAM the recurrence across the whole OOS in a single pass (no per-bar
    / per-window retrain, no explicit lag window — the recurrence *is* the learned temporal filter).
  * Associative parallel scan (O(log L)) — trick #1 — is a GPU result; on CPU it buys nothing, and a
    division-based cumulative scan is UNSTABLE for small gates (a_t can be ~e⁻⁸, so 1/∏a overflows).
    We use a numerically-stable SEQUENTIAL scan, `torch.jit.script`-compiled to C speed.
  * Rank-1 outer-product + Sherman-Morrison (M_t = M_{t-1}+v_t k_tᵀ) — trick #2 — is a DIFFERENT
    STATE PARAMETERIZATION within the same linear-recurrence family: a d×d MATRIX memory (linear /
    gated-linear attention, GLA/DeltaNet). RG-LRU chose the DIAGONAL (vector) state, so there is no
    accumulating matrix to invert; and the Sherman-Morrison INVERSE part is a least-squares FITTING
    trick (recursive least squares / delta rule), which needs a closed-form solve. It therefore lives
    in `bucket_ridge` / `RollingLeastSquares` (the ungated linear track, or a FROZEN-gate reservoir),
    not in this SGD-fit learned-gate variant. Diagonal is the right complexity for a scalar target.

Fit target = `adj_RV` by (masked) MSE — the same LS retransform convention as the Ridge base — then
Duan-smeared to raw RV so QLIKE lands against the HAR floor. Device-agnostic (runs on CPU torch).
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

C_LOG = 8.0  # Griffin's decay temperature: a_t = exp(-C·softplus(Λ)·r_t)
CACHE_ROOT_DEFAULT = "results/covid_imp_rank"

DEFAULT = {
    "bucket": "all_buckets",
    "d_model": 64,
    "n_layers": 2,
    "train_window": 24000,  # bars used to fit the model once
    "seq_len": 192,  # truncated-BPTT window
    "burn_in": 48,  # leading steps excluded from the loss (state warmup)
    "epochs": 8,
    "batch_size": 32,
    "lr": 3e-3,
    "weight_decay": 1e-4,
}


# ── Stable sequential scan (jit-compiled: C-speed, no division, differentiable) ──────────────


@torch.jit.script
def _scan_seq(
    a: torch.Tensor, bx: torch.Tensor, h0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """h_t = a_t ⊙ h_{t-1} + bx_t, sequential over T (stable for any a_t∈(0,1); no ∏/÷ underflow).
    a, bx: (B,T,d); h0: (B,d). Returns (h_seq (B,T,d), h_T (B,d)). jit.script removes Python overhead
    so the single full-OOS streaming pass (T~2.4e5) is seconds, not minutes."""
    B, T, d = a.shape
    h = h0
    outs = torch.empty_like(a)
    for t in range(T):
        h = a[:, t] * h + bx[:, t]
        outs[:, t] = h
    return outs, h


# ── RG-LRU layer ─────────────────────────────────────────────────────────────────────────────


class RGLRU(nn.Module):
    """One diagonal real-gated linear recurrent layer."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.w_r = nn.Linear(d, d)
        self.w_i = nn.Linear(d, d)
        # Λ init so the decay a spans slow→fast timescales at r=1 (a ∈ ~[0.5, 0.999]): softplus(Λ)
        # from ~9e-4 to ~0.09. Uniform-in-a init, following the Griffin LRU initialization.
        with torch.no_grad():
            a_lo, a_hi = 0.5, 0.999
            u = torch.linspace(a_lo, a_hi, d)
            sp = -torch.log(u) / C_LOG  # target softplus(Λ)
            self.log_lambda = nn.Parameter(torch.log(torch.expm1(sp.clamp(min=1e-6))))
        self.w_r.bias.data.fill_(
            2.0
        )  # start gates open (r≈0.88 → long memory) for stable warmup

    def forward(
        self, x: torch.Tensor, h0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, d = x.shape
        if h0 is None:
            h0 = x.new_zeros(B, d)
        r = torch.sigmoid(self.w_r(x))
        i = torch.sigmoid(self.w_i(x))
        a = torch.exp(-C_LOG * F.softplus(self.log_lambda) * r)  # (B,T,d) ∈ (0,1)
        a2 = (a * a).clamp(max=1.0 - 1e-6)
        bx = torch.sqrt(1.0 - a2) * (i * x)
        return _scan_seq(a, bx, h0)


class RGLRUForecaster(nn.Module):
    """Input proj → stacked RG-LRU layers (residual + LayerNorm) → scalar head. ``carry`` holds the
    per-layer hidden state so inference can STREAM across chunks/the whole OOS with O(1) state."""

    def __init__(self, d_in: int, d_model: int, n_layers: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList(RGLRU(d_model) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.head = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, carry: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.proj(x)
        new_carry: list[torch.Tensor] = []
        for li, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h0 = carry[li] if carry is not None else None
            seq, hT = layer(norm(h), h0)
            h = h + seq  # residual
            new_carry.append(hT)
        return self.head(h).squeeze(-1), new_carry  # (B,T)


# ── Train-once + stream ──────────────────────────────────────────────────────────────────────


def _fit(model, X, y, cfg, device):
    """Truncated-BPTT fit on the training region (sampled length-``seq_len`` windows); masked-MSE
    on ``adj`` after a ``burn_in`` state warmup. One global fit — no per-window retrain."""
    seq_len, burn = int(cfg["seq_len"]), int(cfg["burn_in"])
    bs, n_train = int(cfg["batch_size"]), X.shape[0]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    starts_all = np.arange(0, n_train - seq_len)
    steps = max(1, len(starts_all) // bs)
    model.train()
    for ep in range(int(cfg["epochs"])):
        perm = np.random.permutation(starts_all)
        tot = 0.0
        for k in range(steps):
            s = perm[k * bs : (k + 1) * bs]
            idx = torch.as_tensor(
                s[:, None] + np.arange(seq_len)[None, :], device=device
            )
            xb, yb = Xt[idx], yt[idx]
            pred, _ = model(xb)
            loss = F.mse_loss(pred[:, burn:], yb[:, burn:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        logger.info(f"  epoch {ep + 1}/{cfg['epochs']} mse={tot / steps:.5f}")


@torch.no_grad()
def _stream(model, X, device, chunk=4096):
    """Single causal forward over the FULL series, carrying state across chunks (O(1) memory/step).
    Returns ŷ_t for every bar. This is the whole OOS inference — one pass, no windowing/retrain."""
    model.eval()
    N = X.shape[0]
    preds = np.empty(N, dtype=np.float64)
    carry: list[torch.Tensor] | None = None
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        xb = torch.as_tensor(X[lo:hi][None, :, :], dtype=torch.float32, device=device)
        p, carry = model(xb, carry)
        carry = [c.detach() for c in carry]
        preds[lo:hi] = p.squeeze(0).cpu().numpy()
    return preds


def _load_bucket(cache_root: str, bucket: str):
    b = f"{cache_root}/{bucket}"
    feats = json.load(open(f"{b}/meta.json"))["feats"]
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.asarray(np.load(f"{b}/y.npy"), dtype=np.float64)
    base = np.asarray(np.load(f"{b}/base.npy"), dtype=np.float64)
    return X, y, base, feats


def compute(args) -> None:
    from src.evaluation.metrics import (
        build_results_dataframe,
        calculate_metrics,
        save_chunk_reduce,
    )

    def _p(key: str):
        v = getattr(args, key, None)
        return DEFAULT[key] if v is None else v

    torch.manual_seed(int(getattr(args, "seed", 42) or 42))
    np.random.seed(int(getattr(args, "seed", 42) or 42))
    cache_root = getattr(args, "cache_root", None) or CACHE_ROOT_DEFAULT
    bucket = str(_p("bucket"))
    W = int(_p("train_window"))
    device = torch.device(
        "cuda:0"
        if (getattr(args, "gpu_count", 0) and torch.cuda.is_available())
        else "cpu"
    )
    if device.type == "cpu" and getattr(args, "cpu_threads", None):
        torch.set_num_threads(int(args.cpu_threads))

    logger.info(f"Loading bucket {bucket} from {cache_root}")
    X, y, base, feats = _load_bucket(cache_root, bucket)
    N, d_in = X.shape

    # Causal standardization from the training region only (features are already causal/rank-space).
    mu, sd = X[:W].mean(0), X[:W].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = ((X - mu) / sd).astype(np.float32)
    ymu, ysd = float(y[:W].mean()), float(y[:W].std() or 1.0)
    ys = ((y - ymu) / ysd).astype(np.float32)

    cfg = {k: _p(k) for k in DEFAULT}
    model = RGLRUForecaster(d_in, int(cfg["d_model"]), int(cfg["n_layers"])).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"RG-LRU d_in={d_in} d_model={cfg['d_model']} layers={cfg['n_layers']} params={n_params} device={device}"
    )

    t0 = time.time()
    _fit(model, Xs[:W], ys[:W], cfg, device)
    logger.info(f"fit ({cfg['epochs']} epochs) in {time.time() - t0:.1f}s")
    t1 = time.time()
    preds_std = _stream(model, Xs, device)  # full-series streaming pass
    logger.info(f"stream {N} bars in {time.time() - t1:.1f}s")
    preds_adj = preds_std * ysd + ymu  # back to adj scale

    oos_lo = W
    oos_hi = N if getattr(args, "end", -1) in (None, -1) else min(int(args.end), N)
    preds = preds_adj[oos_lo:oos_hi]
    y_oos, base_oos = y[oos_lo:oos_hi], base[oos_lo:oos_hi]
    horizon = int(getattr(args, "horizon", 1) or 1)
    df = build_results_dataframe(
        preds, y_oos, np.arange(oos_lo, oos_hi), base_oos, horizon
    )

    out = args.output_file
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    save_chunk_reduce(df, out)
    metrics = calculate_metrics(df)
    with open(os.path.join(os.path.dirname(out) or ".", "metrics.json"), "w") as f:
        json.dump(metrics, f)
    logger.info(f"Saved {len(df)} rows -> {out}  QLIKE={metrics.get('qlike')}")
