"""
decomposition.py
================
SVD Decomposition (Singular Value Decomposition) implemented from scratch.

"""

import math

# =============================================================================
# PART 0: BASIC MATRIX OPERATIONS
# =============================================================================

def mat_zeros(m: int, n: int) -> list[list[float]]:
    """Create an m x n zero matrix."""
    return [[0.0] * n for _ in range(m)]

def mat_identity(n: int) -> list[list[float]]:
    """Create an n x n identity matrix."""
    I = mat_zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def mat_copy(A: list[list[float]]) -> list[list[float]]:
    """Copy a matrix."""
    return [row[:] for row in A]

def mat_shape(A: list[list[float]]) -> tuple[int, int]:
    """Return (rows, columns) of a matrix."""
    return len(A), len(A[0])

def mat_transpose(A: list[list[float]]) -> list[list[float]]:
    """Transpose matrix A."""
    m, n = mat_shape(A)
    T = mat_zeros(n, m)
    for i in range(m):
        for j in range(n):
            T[j][i] = A[i][j]
    return T

def mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices A (m x k) and B (k x n)."""
    m, k = mat_shape(A)
    k2, n = mat_shape(B)
    assert k == k2, f"Size mismatch: ({m}x{k}) @ ({k2}x{n})"
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C

def mat_sub(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Subtract two matrices of the same size."""
    m, n = mat_shape(A)
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
    return C

def mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Add two matrices of the same size."""
    m, n = mat_shape(A)
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C

def mat_scale(A: list[list[float]], c: float) -> list[list[float]]:
    """Multiply a matrix by a scalar c."""
    m, n = mat_shape(A)
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] * c
    return C

def mat_frobenius_norm(A: list[list[float]]) -> float:
    """Frobenius norm of a matrix."""
    m, n = mat_shape(A)
    s = 0.0
    for i in range(m):
        for j in range(n):
            s += A[i][j] ** 2
    return math.sqrt(s)

def vec_dot(u: list[float], v: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(u, v))

def vec_norm(v: list[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(vec_dot(v, v))

def vec_scale(v: list[float], c: float) -> list[float]:
    """Multiply a vector by a scalar."""
    return [x * c for x in v]

def vec_sub(u: list[float], v: list[float]) -> list[float]:
    """Subtract two vectors."""
    return [a - b for a, b in zip(u, v)]

def mat_col(A: list[list[float]], j: int) -> list[float]:
    """Extract column j of a matrix as a vector."""
    return [A[i][j] for i in range(len(A))]

def mat_row(A: list[list[float]], i: int) -> list[float]:
    """Extract row i of a matrix as a vector."""
    return A[i][:]

def mat_set_col(A: list[list[float]], j: int, v: list[float]) -> None:
    """Set column j of a matrix to vector v (in-place)."""
    for i in range(len(v)):
        A[i][j] = v[i]

def mat_diag(v: list[float]) -> list[list[float]]:
    """Create a diagonal matrix from vector v."""
    n = len(v)
    D = mat_zeros(n, n)
    for i in range(n):
        D[i][i] = v[i]
    return D

def print_matrix(A: list[list[float]], name: str = "A", decimals: int = 6) -> None:
    """Print the matrix to the console."""
    m, n = mat_shape(A)
    print(f"{name} ({m}x{n}):")
    for row in A:
        print("  [" + "  ".join(f"{x:+.{decimals}f}" for x in row) + "]")


# =============================================================================
# PART 1: QR DECOMPOSITION USING GRAM-SCHMIDT (internal use for QR iteration)
# =============================================================================

def qr_gram_schmidt(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    QR decomposition using the classical Gram-Schmidt method.
    A = Q @ R
    - Q: orthonormal matrix (orthogonal columns, unit norm)
    - R: upper triangular matrix

    Parameters:
        A: input matrix m x n (m >= n)

    Returns:
        Q (m x n), R (n x n)
    """
    m, n = mat_shape(A)
    Q = mat_zeros(m, n)
    R = mat_zeros(n, n)

    for j in range(n):
        # Extract column j of A
        v = mat_col(A, j)

        # Subtract projection onto previous columns of Q
        for i in range(j):
            q_i = mat_col(Q, i)
            R[i][j] = vec_dot(q_i, v)
            proj = vec_scale(q_i, R[i][j])
            v = vec_sub(v, proj)

        # Normalize
        norm_v = vec_norm(v)
        if norm_v < 1e-12:
            # Linearly dependent column — fill with zero vector
            R[j][j] = 0.0
        else:
            R[j][j] = norm_v
            v = vec_scale(v, 1.0 / norm_v)

        mat_set_col(Q, j, v)

    return Q, R


# =============================================================================
# PART 2: QR ITERATION ALGORITHM — FIND EIGENVALUES AND EIGENVECTORS
# =============================================================================

def _qr_iteration(A: list[list[float]], max_iter: int = 1000, tol: float = 1e-10):
    """
    QR Iteration algorithm to find eigenvalues and eigenvectors
    of a symmetric matrix.

    Idea:
        A₀ = A
        Aₖ = Qₖ Rₖ  (QR decomposition)
        A_{k+1} = Rₖ Qₖ
        → Aₖ converges to a diagonal matrix containing eigenvalues

    Eigenvectors are accumulated via: V = Q₀ Q₁ Q₂ ...

    Parameters:
        A        : symmetric matrix n x n
        max_iter : maximum number of iterations
        tol      : convergence threshold (based on sum of squared sub-diagonal elements)

    Returns:
        eigenvalues  : list[float]
        eigenvectors : list[list[float]]  — each COLUMN is an eigenvector
    """
    n, _ = mat_shape(A)
    Ak = mat_copy(A)
    V  = mat_identity(n)   # accumulate eigenvectors

    for iteration in range(max_iter):
        Q, R = qr_gram_schmidt(Ak)
        Ak = mat_mul(R, Q)          # A_{k+1} = R_k Q_k
        V  = mat_mul(V, Q)          # accumulate eigenvectors

        # Check convergence: sum of squares of sub-diagonal elements
        off_diag = 0.0
        for i in range(n):
            for j in range(i):
                off_diag += Ak[i][j] ** 2
        if math.sqrt(off_diag) < tol:
            break

    eigenvalues = [Ak[i][i] for i in range(n)]
    return eigenvalues, V


# =============================================================================
# PART 3: SORT EIGENVALUES AND EIGENVECTORS IN DESCENDING ORDER
# =============================================================================

def _sort_eigen(eigenvalues: list[float], eigenvectors: list[list[float]]):
    """
    Sort eigenvalues in descending order and sort corresponding eigenvector columns.

    Returns:
        sorted_values  : list[float]
        sorted_vectors : list[list[float]]
    """
    n = len(eigenvalues)
    order = sorted(range(n), key=lambda i: eigenvalues[i], reverse=True)

    sorted_values  = [eigenvalues[i] for i in order]
    m = len(eigenvectors)
    sorted_vectors = mat_zeros(m, n)
    for new_j, old_j in enumerate(order):
        col = mat_col(eigenvectors, old_j)
        mat_set_col(sorted_vectors, new_j, col)

    return sorted_values, sorted_vectors


# =============================================================================
# PART 4: MAIN SVD DECOMPOSITION
# =============================================================================

def svd(A: list[list[float]], max_iter: int = 1000, tol: float = 1e-10):
    """
    SVD decomposition from scratch: A = U @ Sigma @ V^T

    Algorithm:
        1. Compute B = AᵀA  (symmetric, positive semi-definite matrix)
        2. Use QR iteration to find eigenvalues λᵢ and eigenvectors V of B
        3. Singular values: σᵢ = √λᵢ
        4. Left singular vectors: uᵢ = A @ vᵢ / σᵢ

    Parameters:
        A        : input matrix m x n
        max_iter : maximum iterations for QR iteration
        tol      : convergence threshold

    Returns:
        U     : matrix m x m (left singular vectors)
        sigma : list of singular values (descending)
        Vt    : matrix n x n = V^T (right singular vectors)
    """
    m, n = mat_shape(A)

    # -----------------------------------------------------------------------
    # Step 1: Compute B = AᵀA
    # -----------------------------------------------------------------------
    At = mat_transpose(A)
    B  = mat_mul(At, A)          # n x n, symmetric, positive semi-definite

    # -----------------------------------------------------------------------
    # Step 2: QR iteration on B → eigenvalues of AᵀA + eigenvectors V
    # -----------------------------------------------------------------------
    eigenvalues_B, V = _qr_iteration(B, max_iter=max_iter, tol=tol)

    # Clamp to 0 to avoid negative square roots due to rounding errors
    eigenvalues_B = [max(0.0, lam) for lam in eigenvalues_B]

    # -----------------------------------------------------------------------
    # Step 3: Sort in descending order
    # -----------------------------------------------------------------------
    eigenvalues_B, V = _sort_eigen(eigenvalues_B, V)

    # -----------------------------------------------------------------------
    # Step 4: Compute singular values σᵢ = √λᵢ
    # -----------------------------------------------------------------------
    sigma = [math.sqrt(lam) for lam in eigenvalues_B]

    # -----------------------------------------------------------------------
    # Step 5: Compute U — left singular vectors
    #   uᵢ = A @ vᵢ / σᵢ  (for σᵢ > 0)
    #   For σᵢ ≈ 0: supplement with arbitrary orthogonal vectors (Gram-Schmidt on U)
    # -----------------------------------------------------------------------
    rank = sum(1 for s in sigma if s > tol)

    U = mat_zeros(m, m)

    # Compute u for non-zero singular values
    for j in range(rank):
        vj  = mat_col(V, j)
        Avj = [sum(A[i][k] * vj[k] for k in range(n)) for i in range(m)]
        uj  = vec_scale(Avj, 1.0 / sigma[j])
        mat_set_col(U, j, uj)

    # Gram-Schmidt to complete U with remaining columns (null space of Aᵀ)
    if rank < m:
        # Initialize remaining columns with standard basis vectors e_j
        candidate_idx = 0
        for j in range(rank, m):
            while candidate_idx < m:
                e = [0.0] * m
                e[candidate_idx] = 1.0
                candidate_idx += 1

                # Subtract projection onto existing columns in U
                v = e[:]
                for k in range(j):
                    uk = mat_col(U, k)
                    proj = vec_dot(uk, v)
                    v = vec_sub(v, vec_scale(uk, proj))

                nv = vec_norm(v)
                if nv > 1e-10:
                    mat_set_col(U, j, vec_scale(v, 1.0 / nv))
                    break

    # -----------------------------------------------------------------------
    # Step 6: Prepare Vt = V^T and full Sigma m x n
    # -----------------------------------------------------------------------
    Vt    = mat_transpose(V)
    p     = min(m, n)
    sigma = sigma[:p]   # keep only p singular values

    return U, sigma, Vt


# =============================================================================
# PART 5: LOW-RANK APPROXIMATION
# =============================================================================

def low_rank_approximation(U, sigma, Vt, k: int) -> list[list[float]]:
    """
    Compute rank-k approximation of matrix A:
        Aₖ = Σᵢ₌₁ᵏ σᵢ · uᵢ · vᵢᵀ

    This is the best rank-k approximation under the Frobenius norm (Eckart-Young Theorem).

    Parameters:
        U     : left singular vectors (m x m)
        sigma : list of singular values
        Vt    : right singular vectors^T (n x n)
        k     : approximation rank

    Returns:
        Ak : rank-k approximation matrix (m x n)
    """
    m = len(U)
    n = len(Vt[0])
    Ak = mat_zeros(m, n)

    k = min(k, len(sigma))
    for i in range(k):
        ui = mat_col(U, i)           # column vector m x 1
        vi = mat_row(Vt, i)          # row vector 1 x n  (= vᵢᵀ)

        # Outer product: uᵢ @ vᵢᵀ  (rank-1 matrix)
        for r in range(m):
            for c in range(n):
                Ak[r][c] += sigma[i] * ui[r] * vi[c]

    return Ak


# =============================================================================
# PART 7: VERIFICATION (uses numpy only for verifying)
# =============================================================================

def verify_svd(A_orig: list[list[float]], U, sigma, Vt, tol: float = 1e-6) -> bool:
    """
    Verify SVD result by reconstructing A from U, Sigma, Vt
    and comparing relative error.

    Do not use numpy — compute purely using self-implemented matrix operations.

    Returns:
        True if ||A - U Σ Vᵀ||_F / ||A||_F < tol
    """
    m, n = mat_shape(A_orig)
    p = len(sigma)

    # Reconstruct A_approx = U @ Sigma_full @ Vt
    Sigma_full = mat_zeros(m, n)
    for i in range(p):
        Sigma_full[i][i] = sigma[i]

    A_rec = mat_mul(mat_mul(U, Sigma_full), Vt)

    err   = mat_frobenius_norm(mat_sub(A_orig, A_rec))
    scale = mat_frobenius_norm(A_orig)

    rel_err = err / (scale + 1e-15)
    print(f"  ||A - UΣVᵀ||_F / ||A||_F = {rel_err:.2e}  →  {'✓ OK' if rel_err < tol else '✗ FAIL'}")
    return rel_err < tol


def verify_orthogonal(M: list[list[float]], name: str = "M", tol: float = 1e-6) -> bool:
    """
    Check orthogonality: M^T @ M ≈ I
    """
    n, _ = mat_shape(M)
    Mt  = mat_transpose(M)
    MtM = mat_mul(Mt, M)
    I   = mat_identity(n)

    err = mat_frobenius_norm(mat_sub(MtM, I))
    print(f"  ||{name}ᵀ{name} - I||_F = {err:.2e}  →  {'✓ OK' if err < tol else '✗ FAIL'}")
    return err < tol


def verify_svd_with_numpy(A_list: list[list[float]], U, sigma, Vt):
    """
    Additional verification using numpy (for comparison only, NOT used for the algorithm).
    """
    try:
        import numpy as np
        A_np  = np.array(A_list)
        U_np, s_np, Vt_np = np.linalg.svd(A_np)

        print("\n  [NumPy reference]")
        print(f"  Singular values (numpy): {s_np.round(6).tolist()}")
        print(f"  Singular values (ours) : {[round(s, 6) for s in sigma]}")

        # Compare singular values (already sorted descending)
        p = min(len(sigma), len(s_np))
        max_diff = max(abs(sigma[i] - s_np[i]) for i in range(p))
        print(f"  Max |σᵢ - σᵢ_numpy| = {max_diff:.2e}  →  {'✓ OK' if max_diff < 1e-5 else '✗ FAIL'}")
    except ImportError:
        print("  NumPy is not available, skipping numpy verification step.")


# =============================================================================
# PART 8: DEMO (CONTAINS SVD & LOW-RANK APPROXIMATION ONLY)
# =============================================================================

def _list2d(data):
    """Convert list of list to floats."""
    return [[float(x) for x in row] for row in data]

if __name__ == "__main__":
    print("=" * 60)
    print("  DEMO: SVD Decomposition From Scratch")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test case 1: 3x2 matrix
    # ------------------------------------------------------------------
    print("\n--- Test 1: 3x2 matrix ---")
    A1 = _list2d([
        [1, 2],
        [3, 4],
        [5, 6],
    ])
    print_matrix(A1, "A1")

    U1, s1, Vt1 = svd(A1)
    print(f"\n  Singular values: {[round(s, 6) for s in s1]}")
    print_matrix(U1, "U1")
    print_matrix(mat_transpose(Vt1), "V1")
    print("\nVerification:")
    verify_svd(A1, U1, s1, Vt1)
    verify_orthogonal(U1, "U1")
    verify_orthogonal(mat_transpose(Vt1), "V1")
    verify_svd_with_numpy(A1, U1, s1, Vt1)

    # ------------------------------------------------------------------
    # Test case 2: 3x3 square matrix
    # ------------------------------------------------------------------
    print("\n--- Test 2: 3x3 square matrix ---")
    A2 = _list2d([
        [4, 0, 0],
        [3, 2, 0],
        [1, 1, 3],
    ])
    print_matrix(A2, "A2")

    U2, s2, Vt2 = svd(A2)
    print(f"\n  Singular values: {[round(s, 6) for s in s2]}")
    print("\nVerification:")
    verify_svd(A2, U2, s2, Vt2)
    verify_orthogonal(U2, "U2")
    verify_svd_with_numpy(A2, U2, s2, Vt2)

    # ------------------------------------------------------------------
    # Test case 3: Special matrix — has singular value = 0 (rank-deficient)
    # ------------------------------------------------------------------
    print("\n--- Test 3: Rank-deficient matrix ---")
    A3 = _list2d([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],   # row 3 = row 1 + row 2 * 2 - ...  (linearly dependent)
    ])
    print_matrix(A3, "A3")

    U3, s3, Vt3 = svd(A3)
    print(f"\n  Singular values: {[round(s, 6) for s in s3]}")
    print("\nVerification:")
    verify_svd(A3, U3, s3, Vt3)
    verify_svd_with_numpy(A3, U3, s3, Vt3)

    # ------------------------------------------------------------------
    # Test case 4: Low-rank Approximation
    # ------------------------------------------------------------------
    print("\n--- Test 4: Low-rank Approximation ---")
    A4 = _list2d([
        [3, 2, 2],
        [2, 3, -2],
    ])
    U4, s4, Vt4 = svd(A4)
    print(f"  Singular values: {[round(s, 6) for s in s4]}")

    for k in range(1, len(s4) + 1):
        Ak = low_rank_approximation(U4, s4, Vt4, k)
        err = mat_frobenius_norm(mat_sub(A4, Ak))
        print(f"  k={k}: ||A - A_{k}||_F = {err:.6f}")