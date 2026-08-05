"""Smoke test for the RegimeMoE regime-stage learner (CPU). Mock data, all rungs."""

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import numpy as np

from src.models.regime_moe import RegimeMoE

rng = np.random.default_rng(0)
X = rng.standard_normal((3000, 24)).astype("float32")
# planted low-order structure + a 2-way interaction + noise
y = (
    0.4 * X[:, 0]
    - 0.3 * X[:, 1]
    + 0.2 * X[:, 2] * (X[:, 3] > 0)
    + 0.1 * rng.standard_normal(3000)
).astype("float32")
var_y = float(np.var(y[:200]))

for depth in (0, 1, 2):
    m = RegimeMoE(depth=depth, epochs=80, hidden_dim=8, seed=0).fit(X, y)
    p = m.predict(X[:200])
    mse = float(np.mean((p - y[:200]) ** 2))
    print(
        f"depth={depth} input_dim={X.shape[1]} pred_shape={tuple(p.shape)} "
        f"finite={bool(np.isfinite(p).all())} train_mse={mse:.5f} var_y={var_y:.5f}"
    )

m = RegimeMoE(depth=1, epochs=80, hidden_dim=8, use_diagnostic_mlp=True, seed=0).fit(X, y)
print("diag_probe finite=", bool(np.isfinite(m.predict(X[:200])).all()))
print("SMOKE_OK")
