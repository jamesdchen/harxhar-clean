"""Rolling causal hyperparameter tuning for per-bar-refit tree regressors.

Library home of the ``RollingTunedTree`` wrapper from the audited
``specs/rolling_tree_tune.py`` and ``specs/causal_tune_tree.py`` experiment
sources. Those specs are hash-bound audit artifacts and deliberately keep
their own inline copies; this module is the reusable, importable form for
new drivers and notebooks. Do not edit the specs to point here.

The wrapper eliminates look-ahead bias from hyperparameter selection: the
tree's ``(max_depth, learning_rate)`` config is re-selected PERIODICALLY
(every ``tune_per`` fits) by a grid search scored on a leakage-clean forward
validation tail carved from the CURRENT training window — the split is
:func:`src.models.reclasso_har.forward_window_split` (fit block, embargo
gap, val tail), so every OOS forecast's hyperparameters were chosen without
seeing its future. The model refit itself stays on the engine's cadence
(every bar under the production tree setting ``refit_frequency=1``). Grid
cells fit at a cheap ``grid_n_estimators`` for grid economy; the argmin
config deploys at the production ``deploy_n_estimators`` for every refit
until the next tuning. Strictly causal: only rows the engine hands in are
ever read.

:class:`src.backtest.multi_stage.MultiStageBacktest` constructs a FRESH
regressor instance every refit, so tuning state cannot live on the
instances. The audited specs hold it at class level (module-global);
here it lives on a :class:`RollingTunedTreeFactory` — a configured,
zero-arg-callable factory that owns the fit counter, the current config
and the selection trace — so several tuners can coexist in one process:

>>> factory = make_rolling_tuned_tree("xgb", [(3, 0.05), (5, 0.1)])
>>> backtest = MultiStageBacktest(
...     residualizer=IdentityResidualizer(),
...     regressor_factory=factory,
...     refit_frequency=1,
... )
>>> preds = backtest.run(X, y, train_win)
>>> factory.trace  # per-tuning selections (evidence)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.models.reclasso_har import forward_window_split

ModelFamily = Literal["xgb", "lgbm"]

#: Grid-cell / deployed configuration: ``(max_depth, learning_rate)``.
TreeConfig = tuple[int, float]

# Defaults = the audited specs/causal_tune_tree.py constants (the linear
# verdict's tuning cadence and split, the rolling_tree_tune n_estimators
# ruling, the production DEFAULT_*_PARAMS n_estimators and seed).
DEFAULT_TUNE_PER = 250
DEFAULT_VAL_TAIL = 125
DEFAULT_EMBARGO = 25
DEFAULT_GRID_N_ESTIMATORS = 300
DEFAULT_DEPLOY_N_ESTIMATORS = 500
DEFAULT_SEED = 42


def _make_tree(
    model: ModelFamily, depth: int, lr: float, n_estimators: int, seed: int
) -> XGBRegressor | LGBMRegressor:
    """Construct one booster of the given family and config.

    Production default threading (``DEFAULT_*_PARAMS`` ``n_jobs=-1``): each
    cluster task owns its cores; cross-task parallelism is the map axes.
    """
    if model == "xgb":
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=depth,
            learning_rate=lr,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    return LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=depth,
        learning_rate=lr,
        n_jobs=-1,
        verbose=-1,
        random_state=seed,
    )


class RollingTunedTree:
    """One fresh-refit regressor tied to a shared :class:`RollingTunedTreeFactory`.

    Sklearn-style ``.fit(X_train, y_train)`` / ``.predict(X_test)`` as
    :class:`~src.backtest.multi_stage.MultiStageBacktest` expects. The engine
    constructs a new instance every refit; all tuning state (fit counter,
    current config, selection trace) lives on the owning factory, which is
    where callers read the evidence trace.

    Parameters
    ----------
    factory
        The configured factory whose tuning state this instance shares.
    """

    def __init__(self, factory: "RollingTunedTreeFactory") -> None:
        self._factory = factory
        self._model: XGBRegressor | LGBMRegressor | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RollingTunedTree":
        """Fit on the engine-supplied trailing window; tune on cadence.

        Every ``tune_per`` fits (counted across instances via the factory),
        :func:`forward_window_split` carves the leakage-clean
        fit-embargo-val split of the current window and each grid cell —
        fit on the fit block at ``grid_n_estimators`` — is MSE-scored on
        the val tail; the argmin becomes the current config. Every fit then
        refits the current config on the FULL window at
        ``deploy_n_estimators``.
        """
        f = self._factory
        if f.n_fits % f.tune_per == 0:
            n = len(X_train)
            fit_lo, fit_hi, val_lo, val_hi = forward_window_split(
                n, n, f.val_tail, f.embargo
            )
            X_fit, y_fit = X_train[fit_lo:fit_hi], y_train[fit_lo:fit_hi]
            X_val, y_val = X_train[val_lo:val_hi], y_train[val_lo:val_hi]
            val_mse = [
                float(
                    np.mean(
                        (
                            _make_tree(f.model, d, lr, f.grid_n_estimators, f.seed)
                            .fit(X_fit, y_fit)
                            .predict(X_val)
                            - y_val
                        )
                        ** 2
                    )
                )
                for (d, lr) in f.grid
            ]
            f.config = f.grid[int(np.argmin(val_mse))]
            f.trace.append(f.config)
        f.n_fits += 1
        assert f.config is not None  # set on the first fit (n_fits == 0 tunes)
        depth, lr = f.config
        self._model = _make_tree(f.model, depth, lr, f.deploy_n_estimators, f.seed).fit(
            X_train, y_train
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict with the deployed booster fit by the last :meth:`fit`."""
        if self._model is None:
            raise RuntimeError("predict() called before fit()")
        return np.asarray(self._model.predict(X_test))


class RollingTunedTreeFactory:
    """Configured, stateful, zero-arg-callable factory of rolling-tuned trees.

    Pass an instance directly as
    ``MultiStageBacktest(regressor_factory=factory, ...)``: each call
    returns a fresh :class:`RollingTunedTree` sharing this factory's tuning
    state, so the periodic re-selection survives the engine's fresh-refit
    protocol without module-global state (several factories can coexist).

    Parameters
    ----------
    model
        Tree family, ``"xgb"`` (:class:`xgboost.XGBRegressor`,
        ``tree_method="hist"``) or ``"lgbm"``
        (:class:`lightgbm.LGBMRegressor`).
    grid
        Candidate ``(max_depth, learning_rate)`` configs searched at each
        tuning.
    tune_per
        Re-select the config every this many fits. The first fit always
        tunes.
    val_tail
        Length of the forward validation tail carved from the training
        window (the ``val_len`` of :func:`forward_window_split`).
    embargo
        Gap between the fit block and the val tail; sized to kill
        rolling-feature fit-to-val bleed (25 covers ``har_ma_25`` in the
        audited specs).
    grid_n_estimators
        ``n_estimators`` for the cheap grid-cell fits.
    deploy_n_estimators
        ``n_estimators`` for the deployed full-window refits.
    seed
        ``random_state`` for every booster constructed.

    Attributes
    ----------
    n_fits : int
        Fits served so far (across all instances).
    config : TreeConfig or None
        Currently deployed ``(max_depth, learning_rate)``; ``None`` until
        the first fit.
    trace : list of TreeConfig
        Selection made at each tuning, in order (evidence).
    """

    def __init__(
        self,
        model: ModelFamily,
        grid: Sequence[TreeConfig],
        *,
        tune_per: int = DEFAULT_TUNE_PER,
        val_tail: int = DEFAULT_VAL_TAIL,
        embargo: int = DEFAULT_EMBARGO,
        grid_n_estimators: int = DEFAULT_GRID_N_ESTIMATORS,
        deploy_n_estimators: int = DEFAULT_DEPLOY_N_ESTIMATORS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if model not in ("xgb", "lgbm"):
            raise ValueError(
                f"unknown model family {model!r}; expected 'xgb' or 'lgbm'"
            )
        if not grid:
            raise ValueError(
                "grid must contain at least one (max_depth, learning_rate)"
            )
        if tune_per <= 0:
            raise ValueError(f"tune_per must be positive, got {tune_per}")
        if val_tail <= 0:
            raise ValueError(f"val_tail must be positive, got {val_tail}")
        if embargo < 0:
            raise ValueError(f"embargo must be non-negative, got {embargo}")
        self.model: ModelFamily = model
        self.grid: list[TreeConfig] = [(int(d), float(lr)) for d, lr in grid]
        self.tune_per = int(tune_per)
        self.val_tail = int(val_tail)
        self.embargo = int(embargo)
        self.grid_n_estimators = int(grid_n_estimators)
        self.deploy_n_estimators = int(deploy_n_estimators)
        self.seed = int(seed)
        self.n_fits: int = 0
        self.config: TreeConfig | None = None
        self.trace: list[TreeConfig] = []

    def reset(self) -> None:
        """Clear the tuning state (fit counter, current config, trace).

        Call between independent backtest arms reusing one factory —
        mirrors ``RollingTunedTree.reset`` in the audited specs.
        """
        self.n_fits = 0
        self.config = None
        self.trace = []

    def __call__(self) -> RollingTunedTree:
        """Return a fresh regressor sharing this factory's tuning state."""
        return RollingTunedTree(self)


def make_rolling_tuned_tree(
    model: ModelFamily,
    grid: Sequence[TreeConfig],
    *,
    tune_per: int = DEFAULT_TUNE_PER,
    val_tail: int = DEFAULT_VAL_TAIL,
    embargo: int = DEFAULT_EMBARGO,
    grid_n_estimators: int = DEFAULT_GRID_N_ESTIMATORS,
    deploy_n_estimators: int = DEFAULT_DEPLOY_N_ESTIMATORS,
    seed: int = DEFAULT_SEED,
) -> RollingTunedTreeFactory:
    """Build a configured :class:`RollingTunedTreeFactory`.

    Convenience constructor; see :class:`RollingTunedTreeFactory` for the
    parameter semantics. The returned factory is passed directly as
    ``MultiStageBacktest(regressor_factory=...)`` and exposes ``trace`` /
    ``config`` / ``n_fits`` / ``reset()``.

    Returns
    -------
    RollingTunedTreeFactory
        The configured zero-arg-callable factory.
    """
    return RollingTunedTreeFactory(
        model,
        grid,
        tune_per=tune_per,
        val_tail=val_tail,
        embargo=embargo,
        grid_n_estimators=grid_n_estimators,
        deploy_n_estimators=deploy_n_estimators,
        seed=seed,
    )
