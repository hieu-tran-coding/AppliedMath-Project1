"""Verify Ax = b using NumPy"""

import numpy as np


def verify_solution(A, x, b):
    """
    Verify solution using NumPy

    Returns:
        True if Ax ≈ b
        False otherwise
    """

    A = np.array(A, dtype=float)
    x = np.array(x, dtype=float)
    b = np.array(b, dtype=float)

    return np.allclose(A @ x, b)