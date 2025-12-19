import numpy as np
from dataclasses import dataclass


@dataclass
class Matrix:
    M_true: np.ndarray
    mask: np.ndarray
    M_noisy: np.ndarray | None


class MatrixGenerator:
    @staticmethod
    def get_matrix(
        m=None,
        n=None,
        k=10,
        random_state=None,
        missing_fraction=0.1,
        noise_level=0.1,
    ) -> Matrix:
        """
        Returns a matrix object based on the given parameters.
        If a dataset name is provided, it returns a DatasetMatrix.
        Otherwise, it returns a RandomMatrix.
        """

        # Ensure dimensions m and n are provided for RandomMatrix
        if m is None or n is None:
            raise ValueError(
                "For RandomMatrix, both 'm' (rows) and 'n' (columns) must be specified."
            )

        rm = RandomMatrix(
            m=m,
            n=n,
            k=k,
            random_state=random_state,
            missing_fraction=missing_fraction,
            noise_level=noise_level,
        )
        return rm.generate()


class RandomMatrix:
    def __init__(self, m, n, k, random_state=None, missing_fraction=0.1, noise_level=0.0):
        """
        Initializes the RandomMatrix generator.

        Parameters:
        - m: Number of rows.
        - n: Number of columns.
        - k: Rank of the matrix.
        - random_state: Seed for reproducibility.
        - missing_fraction: Fraction of entries to remove (set to 0).
        - noise_level: Standard deviation of Gaussian noise (set to 0 for no noise).
        """
        super().__init__()
        self.m = m  # Number of rows
        self.n = n  # Number of columns
        self.rank = k  # Rank of the matrix
        self.random_state = random_state  # Random state for reproducibility
        self.missing_fraction = missing_fraction  # Fraction of missing entries
        self.noise_level = noise_level  # Noise level for Gaussian noise

    def generate(self) -> Matrix:
        """
        Generates a random low-rank matrix with missing values and optional Gaussian noise.

        Returns:
        - M_true: Low-rank matrix (m x n).
        - M_missing: Matrix with missing values (m x n).
        - M_noisy (optional): Matrix with Gaussian noise added to non-missing entries (m x n).
                             Returned only if noise_level > 0.
        """
        # Step 1: Create a low-rank matrix
        if self.random_state is not None:
            np.random.seed(self.random_state)

        U = np.random.randn(self.m, self.rank)
        V = np.random.randn(self.n, self.rank)
        M_true = U @ V.T

        # Step 2: Remove entries randomly to create missing values
        M_missing = M_true.copy()

        total_entries = self.m * self.n
        missing_entries = int(total_entries * self.missing_fraction)

        missing_indices = np.random.choice(
            total_entries, missing_entries, replace=False
        )
        M_missing.flat[missing_indices] = 0
        mask = M_missing != 0  # Mask indicating non-missing entries

        # Step 3: Add Gaussian noise if specified
        if self.noise_level > 0:
            noise = self.noise_level * np.random.randn(self.m, self.n)
            M_noisy = M_missing.copy()
            M_noisy[mask] += noise[mask]
            return Matrix(
                M_true=M_true,
                mask=mask,
                M_noisy=M_noisy
            )

        return Matrix(
            M_true=M_true,
            mask=mask,
            M_noisy=None
        )
