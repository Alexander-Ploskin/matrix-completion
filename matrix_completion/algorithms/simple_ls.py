# algorithms/simple_ls.py
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from matrix_completion.matrix_completion import MatrixCompletion
from matrix_completion.dataset import Matrix
from matrix_completion.logger import Logger
from matrix_completion.utils.metrics import (
    calculate_relative_error,
    calculate_relative_residual,
)


class SimpleLS(MatrixCompletion):
    """
    ALS / LS matrix completion on observed entries.

    Solves (approximately):
        min_{U,V} 0.5 || P_Omega(UV^T - Y) ||_F^2 + 0.5*lambda_reg (||U||_F^2 + ||V||_F^2)

    Uses lstsq to avoid singular normal-equation solves. [web:149]
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.lambda_reg = float(params.get("lambda_reg", 0.0))
        self.random_state = params.get("random_state", None)
        self.log_every = int(params.get("log_every", 1))  # 1 => log each iter

    @staticmethod
    def _ridge_lstsq(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
        """
        Solve min_x ||A x - b||^2 + lam ||x||^2 via augmented least squares:
            [A        ] x ~= [b]
            [sqrt(l) I]      [0]
        """
        if lam <= 0:
            return np.linalg.lstsq(A, b, rcond=None)[0]

        r = A.shape[1]
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(r, dtype=A.dtype)])
        b_aug = np.concatenate([b, np.zeros(r, dtype=b.dtype)])
        return np.linalg.lstsq(A_aug, b_aug, rcond=None)[0]

    def complete_matrix(self, matrix: Matrix, logger: Logger) -> np.ndarray:
        # Optional compatibility if some code still expects model.iters_info
        self.iters_info = logger.records

        logger.start()

        Y = matrix.M_noisy if matrix.M_noisy is not None else (matrix.M_true * matrix.mask)
        Omega = matrix.mask

        m, n = matrix.M_true.shape
        r = self.rank
        lam = self.lambda_reg

        rng = np.random.default_rng(self.random_state)
        U = rng.standard_normal((m, r))
        V = rng.standard_normal((n, r))

        # Precompute observed indices to avoid repeated boolean scans
        row_obs = [np.flatnonzero(Omega[i, :]) for i in range(m)]
        col_obs = [np.flatnonzero(Omega[:, j]) for j in range(n)]

        for it in tqdm(range(self.num_iters), total=self.num_iters):
            # ---- Update U ----
            for i in range(m):
                idx = row_obs[i]
                if idx.size == 0:
                    continue

                V_i = V[idx, :]           # (#obs_i, r)
                y_i = Y[i, idx].reshape(-1)

                U[i, :] = self._ridge_lstsq(V_i, y_i, lam)

            # ---- Update V ----
            for j in range(n):
                idx = col_obs[j]
                if idx.size == 0:
                    continue

                U_j = U[idx, :]           # (#obs_j, r)
                y_j = Y[idx, j].reshape(-1)

                V[j, :] = self._ridge_lstsq(U_j, y_j, lam)

            # ---- Metrics / logging (no early stopping) ----
            if (it % self.log_every) == 0:
                X = U @ V.T
                R = Omega * (X - Y)

                cost = 0.5 * float(np.sum(R * R))
                if lam > 0:
                    cost += 0.5 * lam * (float(np.sum(U * U)) + float(np.sum(V * V)))

                logger.log(
                    iter=it,
                    cost=cost,
                    grad_norm=np.nan,
                    dir_norm=np.nan,
                    rel_error=float(calculate_relative_error(X, matrix.M_true)),
                    rel_residual=float(calculate_relative_residual(X, Y, Omega)),
                )

        return U @ V.T
