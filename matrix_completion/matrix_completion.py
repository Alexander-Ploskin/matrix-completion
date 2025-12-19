from abc import ABC, abstractmethod
import numpy as np

from matrix_completion.dataset import Matrix
from matrix_completion.logger import Logger


class MatrixCompletion(ABC):
    def __init__(self, params: dict) -> None:
        self.num_iters = params["num_iters"]
        self.tol = params["tol"]
        self.rank = params["rank"]

    @abstractmethod
    def complete_matrix(self, matrix: Matrix, logger: Logger) -> np.ndarray:
        pass
