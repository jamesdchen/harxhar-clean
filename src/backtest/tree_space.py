"""Single source of truth for the (widened) tree-tuning search space.

Used by BOTH the campaign strategy (.hpc/tasks.py, which asks) and the fleet-seed
importer (seed_campaign.py, which attaches these distributions to imported trials),
so the bespoke fleet's trials land in-distribution when warm-starting the campaign's
Optuna study. Widened past the bounds the first sweep hit at the boundary
(min_child→500, colsample→0.1, lr→0.003); n_estimators clamped to ``n_est_cap``.
"""

from __future__ import annotations


def widened_space(model: str, n_est_cap: int = 300) -> dict:
    import optuna.distributions as D

    if model in ("lgbm", "lightgbm"):
        sp = {
            "n_estimators": D.IntDistribution(100, 600, step=50),
            "max_depth": D.IntDistribution(2, 16),
            "learning_rate": D.FloatDistribution(0.003, 0.3, log=True),
            "num_leaves": D.IntDistribution(7, 128),
            "min_child_samples": D.IntDistribution(5, 500, log=True),
            "subsample": D.FloatDistribution(0.3, 1.0),
            "colsample_bytree": D.FloatDistribution(0.1, 1.0),
            "reg_alpha": D.FloatDistribution(1e-9, 100.0, log=True),
            "reg_lambda": D.FloatDistribution(1e-9, 100.0, log=True),
        }
    elif model in ("xgb", "xgboost"):
        sp = {
            "n_estimators": D.IntDistribution(100, 600, step=50),
            "max_depth": D.IntDistribution(2, 16),
            "learning_rate": D.FloatDistribution(0.003, 0.5, log=True),
            "min_child_weight": D.IntDistribution(1, 200, log=True),
            "subsample": D.FloatDistribution(0.3, 1.0),
            "colsample_bytree": D.FloatDistribution(0.1, 1.0),
            "reg_alpha": D.FloatDistribution(1e-9, 100.0, log=True),
            "reg_lambda": D.FloatDistribution(1e-9, 100.0, log=True),
            "gamma": D.FloatDistribution(0.0, 10.0),
        }
    elif model == "ebm":
        # Microsoft EBM (ExplainableBoostingRegressor): a bagged, boosted GAM (additive per-feature
        # shape functions + a few pairwise interactions) — the inductive-bias-correct residualizer
        # for the smooth/additive/bagged residual the tuning kept pointing at. No n_estimators ->
        # return before the n_est_cap clamp.
        return {
            "learning_rate": D.FloatDistribution(0.005, 0.2, log=True),
            "max_leaves": D.IntDistribution(2, 4),
            "interactions": D.IntDistribution(0, 20),
            "max_bins": D.IntDistribution(128, 1024),
            "min_samples_leaf": D.IntDistribution(2, 50, log=True),
            "max_rounds": D.IntDistribution(100, 800, step=100),  # capped for chunked feasibility (~47s/fit at 500x4)
            "outer_bags": D.IntDistribution(2, 4),
        }
    else:
        raise ValueError(f"unknown model: {model!r}")
    d = sp["n_estimators"]
    sp["n_estimators"] = D.IntDistribution(d.low, min(d.high, n_est_cap), step=d.step)
    return sp
