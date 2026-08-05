# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
import numpy as np


class BlockRidge:
    """Sliding-window ridge with block gram updates, one solve per decision.

    A = [1 | X] with the intercept column unpenalized — identical optimum to
    ``Ridge(alpha, fit_intercept=True)``. Sliding the window by k rows costs
    O(k p^2) BLAS-3 (vs k rank-1 Sherman-Morrison updates); ``coef`` is one
    O(p^3) solve. Exact rebuild every ``rebuild_every`` slides bounds float
    drift, mirroring RollingLeastSquares' reinit contract.
    """

    def __init__(self, alpha: float, rebuild_every: int = 250) -> None:
        self.alpha = float(alpha)
        self.rebuild_every = int(rebuild_every)
        self._n_upd = 0

    @staticmethod
    def _aug(Xw):
        Xw = np.asarray(Xw, dtype=np.float64)
        return np.hstack([np.ones((len(Xw), 1)), Xw])

    def build(self, Xw, yw) -> None:
        A = self._aug(Xw)
        self.G = A.T @ A
        self.c = A.T @ np.asarray(yw, dtype=np.float64)
        self._n_upd = 0

    def needs_rebuild(self) -> bool:
        return self._n_upd >= self.rebuild_every

    def slide(self, X_in, y_in, X_out, y_out) -> None:
        Ai, Ao = self._aug(X_in), self._aug(X_out)
        self.G += Ai.T @ Ai - Ao.T @ Ao
        self.c += Ai.T @ np.asarray(y_in, dtype=np.float64) - Ao.T @ np.asarray(
            y_out, dtype=np.float64
        )
        self._n_upd += 1

    def coef(self, alpha=None) -> np.ndarray:
        a = self.alpha if alpha is None else alpha
        if np.isscalar(a):
            R = self.G + float(a) * np.eye(self.G.shape[0])
            R[0, 0] -= float(a)
        else:  # per-feature penalty vector (intercept always unpenalized)
            pen = np.concatenate([[0.0], np.asarray(a, dtype=np.float64)])
            R = self.G + np.diag(pen)
        return np.linalg.solve(R, self.c)
