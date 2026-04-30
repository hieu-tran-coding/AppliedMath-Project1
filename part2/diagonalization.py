"""
diagonalization.py
==================
Diagonalization implemented from scratch
"""

# Import basic matrix operations and QR iteration algorithm from decomposition.py
from decomposition import (
    mat_shape, mat_diag, mat_mul, mat_sub, mat_frobenius_norm,
    print_matrix, _qr_iteration, _sort_eigen, _list2d
)

# =============================================================================
# PART 6: DIAGONALIZATION  A = P D P⁻¹
# =============================================================================

def _mat_inv_2x2(A: list[list[float]]) -> list[list[float]]:
    """Inverse of a 2x2 matrix (internal use)."""
    det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    if abs(det) < 1e-12:
        raise ValueError("Singular matrix, not invertible.")
    return [[A[1][1]/det, -A[0][1]/det],
            [-A[1][0]/det, A[0][0]/det]]

def _mat_inv_gauss_jordan(A: list[list[float]]) -> list[list[float]]:
    """
    Compute the inverse matrix using the Gauss-Jordan method.
    [A | I] → [I | A⁻¹]
    """
    n, _ = mat_shape(A)
    # Create the augmented matrix [A | I]
    Aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]

    for col in range(n):
        # Find the maximum pivot (partial pivoting)
        max_row = max(range(col, n), key=lambda r: abs(Aug[r][col]))
        Aug[col], Aug[max_row] = Aug[max_row], Aug[col]

        pivot = Aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError(f"Singular matrix at column {col}.")

        # Normalize the pivot row
        Aug[col] = [x / pivot for x in Aug[col]]

        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = Aug[row][col]
            Aug[row] = [Aug[row][j] - factor * Aug[col][j]
                        for j in range(2 * n)]

    # Extract the inverse part
    return [Aug[i][n:] for i in range(n)]

def diagonalize(A: list[list[float]], max_iter: int = 1000, tol: float = 1e-10):
    """
    Diagonalize a symmetric matrix: A = P @ D @ P⁻¹

    Parameters:
        A : symmetric matrix n x n

    Returns:
        P    : matrix of eigenvectors (each COLUMN is an eigenvector)
        D    : diagonal matrix containing eigenvalues
        Pinv : P⁻¹
        eigenvalues : list[float]
    """
    n, _ = mat_shape(A)

    eigenvalues, P = _qr_iteration(A, max_iter=max_iter, tol=tol)
    eigenvalues, P = _sort_eigen(eigenvalues, P)

    D    = mat_diag(eigenvalues)
    Pinv = _mat_inv_gauss_jordan(P)

    return P, D, Pinv, eigenvalues

# =============================================================================
# DEMO TEST CASE (DIAGONALIZATION)
# =============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Test case 5: Diagonalization
    # ------------------------------------------------------------------
    print("\n--- Test 5: Symmetric Matrix Diagonalization ---")
    A5 = _list2d([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
    ])
    print_matrix(A5, "A5")

    P, D, Pinv, eigs = diagonalize(A5)
    print(f"\n  Eigenvalues: {[round(e, 6) for e in eigs]}")
    print_matrix(D, "D")

    # Verification: P @ D @ P⁻¹ ≈ A
    A5_rec = mat_mul(mat_mul(P, D), Pinv)
    err5   = mat_frobenius_norm(mat_sub(A5, A5_rec))
    print(f"\n  ||A - PDP⁻¹||_F = {err5:.2e}  →  {'✓ OK' if err5 < 1e-5 else '✗ FAIL'}")

    print("\n" + "=" * 60)
    print("  All test cases completed!")
    print("=" * 60)