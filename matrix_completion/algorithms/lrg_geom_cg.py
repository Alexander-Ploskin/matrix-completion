import cupy as np
from tqdm import tqdm

from matrix_completion.matrix_completion import MatrixCompletion
from matrix_completion.dataset import Matrix
from matrix_completion.logger import Logger
from matrix_completion.utils.metrics import (
    calculate_relative_error,
    calculate_relative_residual
)


class LRGeomCG(MatrixCompletion):
    def __init__(self, params: dict):
        """
        params_str:
        {
            "num_iters": 100, - Maximum number of iterations
            "tol": 1e-4, - Convergence criteria
        }
        """
        super().__init__(params)
        self.singular_values_eps = params.get('singular_values_eps', 1e-6)
    
    def complete_matrix(self, matrix: Matrix, logger: Logger):
        """
        Described in Algorithm 1 in original paper.
        """

        logger.start()

        A = matrix.M_true
        omega_mask = matrix.mask
        
        m = A.shape[0]
        n = A.shape[1]

        k = self.rank

        A_omega = p_omega(A, omega_mask)

        # initialize 0 iteration
        X_prev, (U_prev, _, V_prev) = self._get_initial_approximation(m, n, k)
        X_omega_prev = p_omega(X_prev, omega_mask)
        R = X_omega_prev - A_omega

        grad_prev = self._riemannian_grad(U_prev, V_prev, R)
        dir_prev = grad_prev

        # Initialize 1st iteration
        X, (U, S, V) = self._get_initial_approximation(m, n, k)
        X_omega = p_omega(X, omega_mask)

        for i in tqdm(range(self.num_iters), total=self.num_iters):
            R = X_omega - A_omega

            grad = self._riemannian_grad(U, V, R)
            if grad.norm <= self.tol:
                break
            
            dir = self._conjugate_direction(
                U=U, V=V, grad=grad,
                grad_prev=grad_prev,
                dir_prev=dir_prev,
            )
            dir_norm = float(dir.norm)
            
            X_prev, U_prev, V_prev = X, U, V
            t_opt = self._initial_guess(U, V, -R, dir, omega_mask)
            X, U, S, V = self._retraction(U, S, V, dir * t_opt*0.5, k)
            X_omega = p_omega(X, omega_mask)

            rel_error = float(calculate_relative_error(X.get(), A))
            rel_residual = float(calculate_relative_residual(X.get(), A, omega_mask))

            logger.log(
                iter=i,
                cost=np.nan,
                grad_norm=float(grad.norm),
                dir_norm=dir_norm,
                rel_error=rel_error,
                rel_residual=rel_residual,
            )

        return X

    def _get_initial_approximation(self, m, n, k):
        X_L = np.random.randn(m, k)
        X_R = np.random.randn(n, k)
        X = X_L @ X_R.T

        return X, truncated_svd(X, k)
    
    def _riemannian_grad(self, U, V, R):
        """
        Described in Algorithm 2 in original paper.
        """
        Ru = R.T @ U
        Rv = R @ V

        M = U.T @ Rv

        Up = Rv - U @ M
        Vp = Ru - V @ M.T

        return TangentVector(
            U=U, V=V, M=M,
            Up=Up, Vp=Vp,
        )

    def _vector_transport(self, U_plus, V_plus, v):
        """
        Algorithm 3
        """
        Av = v.V.T @ V_plus
        Au = v.U.T @ U_plus

        Bv = v.Vp.T @ V_plus
        Bu = v.Up.T @ U_plus

        M1_plus = Au.T @ v.M @ Av
        U1_plus = v.U @ (v.M @ Av)
        V1_plus = v.V @ (v.M.T @ Au)

        M2_plus = Bu.T @ Av
        U2_plus = v.Up @ Av
        V2_plus = v.V @ Bu

        M3_plus = Au.T @ Bv
        U3_plus = v.U @ Bv
        V3_plus = v.Vp @ Au

        M_plus = M1_plus + M2_plus + M3_plus

        Up_plus = U1_plus + U2_plus + U3_plus
        Up_plus = Up_plus - U_plus @ (U_plus.T @ Up_plus)

        Vp_plus = V1_plus + V2_plus + V3_plus
        Vp_plus = Vp_plus - V_plus @ (V_plus.T @ Vp_plus)

        return TangentVector(
            U=U_plus, V=V_plus, M=M_plus, Up=Up_plus, Vp=Vp_plus
        )

    def _conjugate_direction(self, U, V, grad, grad_prev, dir_prev):
        """
        Algorithm 4
        """
        grad_prev_tr = self._vector_transport(U_plus=U, V_plus=V, v=grad_prev)
        dir_prev_tr = self._vector_transport(U_plus=U, V_plus=V, v=dir_prev)

        delta = grad - grad_prev_tr
        beta = max(0, delta.scalar_product(grad) / grad_prev.norm**2)
        dir = dir_prev_tr * beta - grad

        alpha = dir.scalar_product(grad) / (dir.norm * grad.norm)
        if alpha <= 0.1:
            return grad

        return dir

    def _initial_guess(self, U, V, R, dir, omega_mask):
        """
        Algorithm 5
        """
        U_ext = np.concatenate([U @ dir.M + dir.Up, U], axis=1)
        V_ext = np.concatenate([V, dir.Vp], axis=1)

        N = p_omega(U_ext @ V_ext.T, omega_mask)
        t_opt = np.trace(N.T @ R) / np.trace(N.T @ N)

        return t_opt

    def _retraction(self, U, S, V, scaled_dir, k):
        """
        Algorithm 6
        """
        Qu, Ru = np.linalg.qr(scaled_dir.Up)
        Qv, Rv = np.linalg.qr(scaled_dir.Vp)

        row1 = np.concatenate([S + scaled_dir.M, Rv.T], axis=1)
        row2 = np.concatenate([Ru, np.zeros((Ru.shape[0], row1.shape[1] - Ru.shape[1]))], axis=1)
        s = np.concatenate([row1, row2], axis=0)

        Us, Ss, Vs = truncated_svd(s, k)

        S_plus = Ss + self.singular_values_eps
        U_plus = np.concatenate([U, Qu], axis=1) @ Us
        V_plus = np.concatenate([V, Qv], axis=1) @ Vs

        return U_plus @ S_plus @ V_plus.T, U_plus, S_plus, V_plus


class TangentVector:
    def __init__(self, U, V, M, Up, Vp, nu=None):
        assert np.allclose(Up.T @ U, 0), "Matrix does not belong to Tangent Space"
        assert np.allclose(Vp.T @ V, 0), "Matrix does not belong to Tangent Space"

        self.U = U
        self.V = V

        self.M = M
        self.Up = Up
        self.Vp = Vp

        self.nu = U @ M @ V.T + Up @ V.T + U @ Vp.T if nu is None else nu
        self.norm = np.linalg.norm(self.nu)

    def scalar_product(self, other):
        return np.trace(self.nu.T @ other.nu)

    def __mul__(self, scalar):
        return TangentVector(
            U=self.U, V=self.V,
            M=self.M * scalar,
            Up=self.Up * scalar,
            Vp=self.Vp * scalar,
            nu=self.nu * scalar
        )
    def __truediv__(self, scalar):
        return self * (1.0 / scalar)

    def __neg__(self,):
        return self * -1

    def __add__(self, other):
        assert np.allclose(self.U, other.U), "Matrix does not belong to Tangent Space"
        assert np.allclose(self.V, other.V), "Matrix does not belong to Tangent Space"
        return TangentVector(
            U=self.U, V=self.V,
            M=self.M + other.M,
            Up=self.Up + other.Up,
            Vp=self.Vp + other.Vp,
        )

    def __sub__(self, other):
        assert np.allclose(self.U, other.U)
        assert np.allclose(self.V, other.V)
        return self + other * -1

    def __matmul__(self, other):
        assert np.allclose(self.U, other.U)
        assert np.allclose(self.V, other.V)
        pass

def p_omega(X, omega_mask):
    matrix = np.zeros(X.shape)
    matrix[omega_mask] = X[omega_mask]

    return matrix

def truncated_svd(X, k):
    """
    Calculates truncated SVD
    """
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    return U[:, :k], np.diag(S[:k]), Vh.T[:, :k]
