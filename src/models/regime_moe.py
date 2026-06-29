"""Interpretable regime mixture-of-experts — the Hero-B regime-stage learner (DL).

Drops into the ``resid_regime`` cascade (``resid_amortized.preds_chunk``) as a third
``REGIME_MODEL`` option beside ``ebm`` / ``xgb``: a learned soft-tree GATE (the
automated analog of the hand-found ``hour`` regime axis) routing per-regime
NAM/EBM-equivalent EXPERTS, with an optional high-order diagnostic MLP head.

Architecture (after `regime_moe_model.py`):
    state row x  -> SoftTreeGate -> soft regime weights g(x)
                 -> NAMExpert_k(x) (grouped-Conv additive shape fns, <=2-way)
    y_resid = sum_k g_k(x) * e_k(x)   [+ per-expert high-order MLP probe]

The module exposes :class:`RegimeMoE`, a sklearn-contract adapter (``.fit(X, y)`` /
``.predict(X)``) so ``_tree_factory("moe", cfg)`` can construct it exactly like the
EBM/XGB regime learners. Cascade-specific adaptations over the bare scaffold:

* refit per cadence block on the few-thousand-row h16-19 leftover -> small nets,
  early stopping, GPU when present;
* no validation split exists in the cascade -> carve a TEMPORAL tail of the train
  rows for early stopping (respects walk-forward; the recent tail best proxies the
  OOS block that follows);
* gate hyperplanes + grouped-Conv NAM need standardized inputs -> standardize on
  train stats (the survivors mix unpenalized-HAR + scaled exog + regime_extra);
* restore best-val weights (the bare trainer early-stops but never checkpoints).

Validation ladder (the reference's own three runs, mapped to the deployed floor):
    depth=0  single global NAM  -> can any DL match the EBM regime (0.12033)?
    depth>=1 soft-tree gate     -> does a learned partition beat the clock-gate?
    +use_diagnostic_mlp         -> is >=3-way structure the missing gap?
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── 1. Differentiable soft decision-tree gate (the learned regime axis) ──────
class SoftTreeGate(nn.Module):
    """Hierarchical soft partition. Each internal node is a linear hyperplane
    ``P(left) = sigmoid(W x + b)``; leaf weight = product of branch probabilities
    along its root->leaf path. Post-fit the node weights are the human-readable
    regime hyperplanes (the automated corr-by-hour localization).

    General depth: reduces bit-for-bit to the depth-1/2 scaffold but supports any
    depth so the partition can scale "only if it earns it".
    """

    def __init__(self, input_dim: int, depth: int = 1):
        super().__init__()
        if depth < 1:
            raise ValueError("SoftTreeGate needs depth >= 1 (depth 0 = no gate)")
        self.depth = depth
        self.num_leaves = 2**depth
        self.num_nodes = 2**depth - 1
        # one hyperplane per internal node; node ordering is heap order (root=0)
        self.decision_nodes = nn.Linear(input_dim, self.num_nodes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decisions = torch.sigmoid(self.decision_nodes(x))  # [B, num_nodes] = P(left)
        # Walk the tree level by level: each frontier entry carries (path prob, node idx).
        leaf_probs = [(torch.ones_like(decisions[:, 0]), 0)]
        for _ in range(self.depth):
            nxt: list[tuple[torch.Tensor, int]] = []
            for prob, node in leaf_probs:
                p_left = decisions[:, node]
                nxt.append((prob * p_left, 2 * node + 1))  # left child
                nxt.append((prob * (1.0 - p_left), 2 * node + 2))  # right child
            leaf_probs = nxt
        p_leaves = torch.stack([p for p, _ in leaf_probs], dim=1)  # [B, num_leaves]

        # Load-balancing aux loss: penalize uneven leaf utilization (anti-collapse).
        mean_routing = p_leaves.mean(dim=0)
        var_routing = p_leaves.var(dim=0)
        load_balance = (var_routing / (mean_routing**2 + 1e-8)).mean()
        return p_leaves, load_balance


# ── 2. Interpretable NAM expert (EBM-equivalent additive shapes, <=2-way) ────
class NAMExpert(nn.Module):
    """Neural additive model. Grouped 1D convolutions learn one isolated shape
    function per feature in a single kernel (no Python feature loop). Optional
    high-order MLP head is the DIAGNOSTIC probe for >=3-way interactions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        dropout: float = 0.3,
        use_high_order_probe: bool = False,
    ):
        super().__init__()
        self.use_high_order_probe = use_high_order_probe
        # isolated per-feature shapes via groups=input_dim (zero cross-talk)
        self.layer1 = nn.Conv1d(input_dim, input_dim * hidden_dim, 1, groups=input_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Conv1d(input_dim * hidden_dim, input_dim, 1, groups=input_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        if use_high_order_probe:
            self.high_order_mlp = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.LayerNorm(32),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x.unsqueeze(2))  # [B, input_dim, 1] -> [B, input_dim*hidden, 1]
        h = self.dropout(self.activation(h))
        contributions = self.layer2(h)  # [B, input_dim, 1]
        out = (self.bias + contributions.sum(dim=1).squeeze(-1)).unsqueeze(1)  # [B, 1]
        if self.use_high_order_probe:
            return out + self.high_order_mlp(x)
        return out


# ── 2b. Factorization-Machine expert (native low-rank PAIRWISE interactions) ──
class FMExpert(nn.Module):
    """Factorization Machine — pairwise interactions baked in NATIVELY via low-rank factors:
        y = b + w·x + 0.5·Σ_f[(Σ_i V_if x_i)² − Σ_i V_if² x_i²]   (the FM trick, O(B·p·rank)).
    The low-rank ``rank`` IS the regularizer (vs enumerating p² pairwise shape fns) — the EBM's winning
    bias (bagged PAIRWISE) as ONE differentiable primitive, with the soft-tree gate on top. The XGB
    depth evidence (best d3, but the pairwise EBM beats it) says pairwise is the right interaction order.
    """

    def __init__(
        self,
        input_dim: int,
        rank: int = 8,
        dropout: float = 0.15,
        use_high_order_probe: bool = False,
        linear: bool = True,
    ):
        super().__init__()
        self.use_high_order_probe = use_high_order_probe
        self.use_linear = linear
        if linear:
            self.linear = nn.Linear(input_dim, 1)  # b + w·x (main-effect part)
        else:
            # pure-pairwise: base (linbest) + global tree already took the main effects; the regime
            # leftover is interactions, so model only bias + pairwise -> focuses capacity, drops p params.
            self.bias = nn.Parameter(torch.zeros(1))
        self.V = nn.Parameter(torch.randn(input_dim, rank) * 0.01)  # low-rank pairwise factors
        self.dropout = nn.Dropout(dropout)
        if use_high_order_probe:
            self.high_order_mlp = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.LayerNorm(32),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        lin = self.linear(x) if self.use_linear else self.bias  # [B,1] or broadcast bias
        xv = x @ self.V  # [B, rank]
        x2v2 = (x * x) @ (self.V * self.V)  # [B, rank]
        pair = 0.5 * (xv.pow(2) - x2v2).sum(dim=1, keepdim=True)  # [B, 1] — all pairwise interactions
        out = lin + pair
        if self.use_high_order_probe:
            return out + self.high_order_mlp(x)
        return out


# ── 3. The orchestrator ──────────────────────────────────────────────────────
class InterpretableRegimeMoE(nn.Module):
    """Learned soft regimes -> interpretable NAM experts -> optional per-regime
    high-order diagnostic. depth=0 degenerates to one global NAM (EBM-equivalent)."""

    def __init__(
        self,
        input_dim: int,
        depth: int = 1,
        hidden_dim: int = 16,
        dropout: float = 0.3,
        use_diagnostic_mlp: bool = False,
        expert: str = "fm",
        rank: int = 8,
        fm_linear: bool = True,
    ):
        super().__init__()
        self.is_baseline = depth == 0
        self.num_experts = 2**depth if depth > 0 else 1

        def _mk_expert():
            if expert == "fm":  # native low-rank pairwise (the EBM-matching interaction order)
                return FMExpert(input_dim, rank, dropout, use_diagnostic_mlp, fm_linear)
            return NAMExpert(input_dim, hidden_dim, dropout, use_diagnostic_mlp)

        self.experts = nn.ModuleList([_mk_expert() for _ in range(self.num_experts)])
        if not self.is_baseline:
            self.gate = SoftTreeGate(input_dim, depth)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = torch.cat([e(x) for e in self.experts], dim=1)  # [B, K]
        if self.is_baseline:
            return expert_outputs, x.new_zeros(())
        routing, aux = self.gate(x)
        y_resid = torch.sum(routing * expert_outputs, dim=-1, keepdim=True)
        return y_resid, aux


# ── 4. sklearn-contract adapter (the _tree_factory("moe") product) ──────────
class RegimeMoE:
    """``.fit(X, y)`` / ``.predict(X)`` wrapper around :class:`InterpretableRegimeMoE`.

    Standardizes inputs (and optionally the residual target) on train stats, carves
    a temporal val tail for early stopping, trains with AdamW + load-balance aux,
    and restores best-val weights. Heavy weight decay + dropout + early stop for the
    few-thousand-row, ~93%-noise h16-19 leftover.
    """

    def __init__(
        self,
        depth: int = 1,
        hidden_dim: int = 16,
        dropout: float = 0.15,
        n_bags: int = 8,
        colsample: float = 0.5,
        expert: str = "fm",
        rank: int = 8,
        fm_linear: bool = True,
        wd_v: float | None = None,
        basis: str = "none",
        n_knots: int = 3,
        epochs: int = 150,
        lr: float = 1e-3,
        weight_decay: float = 1e-1,
        aux_weight: float = 0.5,
        use_diagnostic_mlp: bool = False,
        val_frac: float = 0.2,
        patience: int = 12,
        min_val_rows: int = 64,
        scale_y: bool = True,
        device: str | None = None,
        seed: int = 42,
    ):
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_bags = n_bags
        self.colsample = colsample
        self.expert = expert
        self.rank = rank
        self.fm_linear = fm_linear
        self.wd_v = wd_v
        self.basis = basis
        self.n_knots = n_knots
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.aux_weight = aux_weight
        self.use_diagnostic_mlp = use_diagnostic_mlp
        self.val_frac = val_frac
        self.patience = patience
        self.min_val_rows = min_val_rows
        self.scale_y = scale_y
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = seed

    # standardization helpers: scale from dispersion-carrying obs; constant
    # columns pass through centered (no arbitrary epsilon).
    def _fit_scaler(self, X: np.ndarray) -> None:
        self._x_mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        self._x_std = np.where(std > 0, std, 1.0)
        if self.scale_y:
            self._y_mean = float(self._y.mean())
            ys = float(self._y.std())
            self._y_std = ys if ys > 0 else 1.0
        else:
            self._y_mean, self._y_std = 0.0, 1.0

    def _xz(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    def _basis_expand(self, Xz: np.ndarray) -> np.ndarray:
        """Quantile-knot HINGE basis: per feature [x, relu(x-q1), .., relu(x-qm)]. The linear combo of
        hinges = a piecewise-linear 1D shape; the FM's bilinear of two hinge-bases = a flexible 2D
        interaction -> a differentiable GA2M (EBM-style shapes). Threshold-mixture form; knots are
        data-adaptive quantiles (no magic). basis != "hinge" -> identity."""
        if self.basis != "hinge":
            return Xz
        cols = [Xz]
        for k in range(self.n_knots):
            cols.append(np.maximum(Xz - self._knots[:, k], 0.0))
        return np.concatenate(cols, axis=1).astype(np.float32)

    def _train_one(self, Xtr, ytr, Xva, yva, p, bag_seed, use_val):
        """Train one bag to its best-val weights; returns the model."""
        torch.manual_seed(bag_seed)
        model = InterpretableRegimeMoE(
            input_dim=p,
            depth=self.depth,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            use_diagnostic_mlp=self.use_diagnostic_mlp,
            expert=self.expert,
            rank=self.rank,
            fm_linear=self.fm_linear,
        ).to(self.device)
        # separate (stronger) L2 on the FM factors V than the rest: the pairwise factors are where FM
        # overfitting lives, so uniform decay under-regularizes the interactions. wd_v=None -> uniform.
        wd_v = self.weight_decay if self.wd_v is None else self.wd_v
        v_params = [prm for nm, prm in model.named_parameters() if nm.endswith(".V")]
        other = [prm for nm, prm in model.named_parameters() if not nm.endswith(".V")]
        opt = torch.optim.AdamW(
            [
                {"params": other, "weight_decay": self.weight_decay},
                {"params": v_params, "weight_decay": wd_v},
            ],
            lr=self.lr,
        )
        mse = nn.MSELoss()
        best_val, best_state, wait = float("inf"), None, 0
        for _ in range(self.epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            pred, aux = model(Xtr)
            (mse(pred, ytr) + self.aux_weight * aux).backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                vloss = mse(model(Xva)[0], yva).item()
            if vloss < best_val:
                best_val, wait = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif use_val:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegimeMoE:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._y = np.ascontiguousarray(y, dtype=np.float32).ravel()
        n, p = X.shape
        self._fit_scaler(X)
        Xz = self._xz(X)
        # per-feature INPUT saturation: clamp each standardized feature to its train [min,max] so every
        # shape function goes FLAT outside the observed support — exactly like an EBM bin (the EBM's real
        # inductive bias). Bounded-by-construction at the FUNCTION level (not an aggregate output clamp);
        # this is what stops the unbounded-SiLU extrapolation that blew up the Duan smear (moe_d0=0.152).
        # No-op on the training rows (they define the range); it only bites OOS extremes at predict.
        self._xz_lo = Xz.min(axis=0, keepdims=True)
        self._xz_hi = Xz.max(axis=0, keepdims=True)
        Xz = np.clip(Xz, self._xz_lo, self._xz_hi).astype(np.float32)
        if self.basis == "hinge":  # quantile-knot hinge basis -> nonlinear shapes + 2D interactions
            qs = np.linspace(0.0, 1.0, self.n_knots + 2)[1:-1]  # interior quantiles e.g. [.25,.5,.75]
            self._knots = np.quantile(Xz, qs, axis=0).T  # [p, n_knots], data-adaptive knots
            Xz = self._basis_expand(Xz)
        p = Xz.shape[1]  # basis expansion changes the working feature width
        yz = ((self._y - self._y_mean) / self._y_std).astype(np.float32)

        # temporal val tail (chronological); bootstrap the train pool per bag (bagging =
        # the EBM's variance control a single per-block NAM lacked).
        n_val = int(round(self.val_frac * n))
        use_val = n_val >= self.min_val_rows and (n - n_val) >= self.min_val_rows
        n_tr = (n - n_val) if use_val else n
        dev = self.device
        Xt = torch.as_tensor(Xz, device=dev)
        yt = torch.as_tensor(yz, device=dev).unsqueeze(1)
        Xva = Xt[n_tr:n] if use_val else Xt
        yva = yt[n_tr:n] if use_val else yt

        rng = np.random.default_rng(self.seed)
        n_sub = max(1, int(round(self.colsample * p)))
        self.models_ = []
        for b in range(self.n_bags):
            # bootstrap ROWS + subsample FEATURES per bag -> decorrelated bags (the RF trick), the
            # strongest variance reducer for this small, noise-dominated, high-dim regime sample.
            ridx = torch.as_tensor(rng.integers(0, n_tr, size=n_tr), device=dev)
            fcols = np.sort(rng.choice(p, size=n_sub, replace=False))
            fidx = torch.as_tensor(fcols, device=dev)
            model = self._train_one(
                Xt[ridx][:, fidx], yt[ridx], Xva[:, fidx], yva, n_sub, self.seed + b, use_val
            )
            self.models_.append((model, fidx))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        # per-feature saturation: hold OOS-extreme features at the train boundary -> shape fns saturate
        Xz = np.clip(self._xz(X), self._xz_lo, self._xz_hi).astype(np.float32)
        Xz = self._basis_expand(Xz)  # apply the fitted basis (no-op if basis="none")
        Xt = torch.as_tensor(Xz, device=self.device)
        with torch.no_grad():
            preds = []
            for m, fidx in self.models_:  # each bag uses its own feature subset
                m.eval()
                preds.append(m(Xt[:, fidx])[0])
            pred = torch.stack(preds, 0).mean(0).squeeze(1)
        return (pred.cpu().numpy() * self._y_std + self._y_mean).astype(np.float64)
