"""
decomposition.py
================
Phân rã SVD (Singular Value Decomposition) cài đặt từ đầu (from scratch).

"""

import math

# =============================================================================
# PHẦN 0: CÁC PHÉP TOÁN MA TRẬN CƠ BẢN
# =============================================================================

def mat_zeros(m: int, n: int) -> list[list[float]]:
    """Tạo ma trận 0 kích thước m x n."""
    return [[0.0] * n for _ in range(m)]

def mat_identity(n: int) -> list[list[float]]:
    """Tạo ma trận đơn vị n x n."""
    I = mat_zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def mat_copy(A: list[list[float]]) -> list[list[float]]:
    """Sao chép ma trận."""
    return [row[:] for row in A]

def mat_shape(A: list[list[float]]) -> tuple[int, int]:
    """Trả về (số hàng, số cột) của ma trận."""
    return len(A), len(A[0])

def mat_transpose(A: list[list[float]]) -> list[list[float]]:
    """Chuyển vị ma trận A."""
    m, n = mat_shape(A)
    T = mat_zeros(n, m)
    for i in range(m):
        for j in range(n):
            T[j][i] = A[i][j]
    return T

def mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Nhân hai ma trận A (m x k) và B (k x n)."""
    m, k = mat_shape(A)
    k2, n = mat_shape(B)
    assert k == k2, f"Kích thước không khớp: ({m}x{k}) @ ({k2}x{n})"
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C

def mat_sub(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Trừ hai ma trận cùng kích thước."""
    m, n = mat_shape(A)
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
    return C

def mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Cộng hai ma trận cùng kích thước."""
    m, n = mat_shape(A)
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C

def mat_scale(A: list[list[float]], c: float) -> list[list[float]]:
    """Nhân ma trận với hằng số c."""
    m, n = mat_shape(A)
    C = mat_zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] * c
    return C

def mat_frobenius_norm(A: list[list[float]]) -> float:
    """Chuẩn Frobenius của ma trận."""
    m, n = mat_shape(A)
    s = 0.0
    for i in range(m):
        for j in range(n):
            s += A[i][j] ** 2
    return math.sqrt(s)

def vec_dot(u: list[float], v: list[float]) -> float:
    """Tích vô hướng của hai vector."""
    return sum(x * y for x, y in zip(u, v))

def vec_norm(v: list[float]) -> float:
    """Chuẩn Euclidean của vector."""
    return math.sqrt(vec_dot(v, v))

def vec_scale(v: list[float], c: float) -> list[float]:
    """Nhân vector với hằng số."""
    return [x * c for x in v]

def vec_sub(u: list[float], v: list[float]) -> list[float]:
    """Trừ hai vector."""
    return [a - b for a, b in zip(u, v)]

def mat_col(A: list[list[float]], j: int) -> list[float]:
    """Lấy cột j của ma trận dưới dạng vector."""
    return [A[i][j] for i in range(len(A))]

def mat_row(A: list[list[float]], i: int) -> list[float]:
    """Lấy hàng i của ma trận dưới dạng vector."""
    return A[i][:]

def mat_set_col(A: list[list[float]], j: int, v: list[float]) -> None:
    """Gán cột j của ma trận bằng vector v (in-place)."""
    for i in range(len(v)):
        A[i][j] = v[i]

def mat_diag(v: list[float]) -> list[list[float]]:
    """Tạo ma trận đường chéo từ vector v."""
    n = len(v)
    D = mat_zeros(n, n)
    for i in range(n):
        D[i][i] = v[i]
    return D

def print_matrix(A: list[list[float]], name: str = "A", decimals: int = 6) -> None:
    """In ma trận ra màn hình."""
    m, n = mat_shape(A)
    print(f"{name} ({m}x{n}):")
    for row in A:
        print("  [" + "  ".join(f"{x:+.{decimals}f}" for x in row) + "]")


# =============================================================================
# PHẦN 1: PHÂN RÃ QR BẰNG GRAM-SCHMIDT (dùng nội bộ cho QR iteration)
# =============================================================================

def qr_gram_schmidt(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Phân rã QR bằng phương pháp Gram-Schmidt cổ điển.
    A = Q @ R
    - Q: ma trận trực chuẩn (cột trực giao, chuẩn bằng 1)
    - R: ma trận tam giác trên

    Tham số:
        A: ma trận đầu vào m x n (m >= n)

    Trả về:
        Q (m x n), R (n x n)
    """
    m, n = mat_shape(A)
    Q = mat_zeros(m, n)
    R = mat_zeros(n, n)

    for j in range(n):
        # Lấy cột j của A
        v = mat_col(A, j)

        # Trừ đi hình chiếu lên các cột trước đó của Q
        for i in range(j):
            q_i = mat_col(Q, i)
            R[i][j] = vec_dot(q_i, v)
            proj = vec_scale(q_i, R[i][j])
            v = vec_sub(v, proj)

        # Chuẩn hóa
        norm_v = vec_norm(v)
        if norm_v < 1e-12:
            # Cột phụ thuộc tuyến tính — điền vector 0
            R[j][j] = 0.0
        else:
            R[j][j] = norm_v
            v = vec_scale(v, 1.0 / norm_v)

        mat_set_col(Q, j, v)

    return Q, R


# =============================================================================
# PHẦN 2: THUẬT TOÁN QR LẶP — TÌM EIGENVALUES VÀ EIGENVECTORS
# =============================================================================

def _qr_iteration(A: list[list[float]], max_iter: int = 1000, tol: float = 1e-10):
    """
    Thuật toán QR lặp (QR Algorithm) để tìm eigenvalues và eigenvectors
    của ma trận đối xứng (symmetric matrix).

    Ý tưởng:
        A₀ = A
        Aₖ = Qₖ Rₖ  (phân rã QR)
        A_{k+1} = Rₖ Qₖ
        → Aₖ hội tụ về ma trận đường chéo chứa eigenvalues

    Eigenvectors được tích lũy qua: V = Q₀ Q₁ Q₂ ...

    Tham số:
        A        : ma trận đối xứng n x n
        max_iter : số vòng lặp tối đa
        tol      : ngưỡng hội tụ (dựa trên tổng bình phương phần tử dưới đường chéo)

    Trả về:
        eigenvalues  : list[float]
        eigenvectors : list[list[float]]  — mỗi CỘT là một eigenvector
    """
    n, _ = mat_shape(A)
    Ak = mat_copy(A)
    V  = mat_identity(n)   # tích lũy eigenvectors

    for iteration in range(max_iter):
        Q, R = qr_gram_schmidt(Ak)
        Ak = mat_mul(R, Q)          # A_{k+1} = R_k Q_k
        V  = mat_mul(V, Q)          # tích lũy eigenvectors

        # Kiểm tra hội tụ: tổng bình phương phần tử dưới đường chéo chính
        off_diag = 0.0
        for i in range(n):
            for j in range(i):
                off_diag += Ak[i][j] ** 2
        if math.sqrt(off_diag) < tol:
            break

    eigenvalues = [Ak[i][i] for i in range(n)]
    return eigenvalues, V


# =============================================================================
# PHẦN 3: SẮP XẾP EIGENVALUES VÀ EIGENVECTORS GIẢM DẦN
# =============================================================================

def _sort_eigen(eigenvalues: list[float], eigenvectors: list[list[float]]):
    """
    Sắp xếp eigenvalues giảm dần và sắp xếp cột eigenvectors tương ứng.

    Trả về:
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
# PHẦN 4: PHÂN RÃ SVD CHÍNH
# =============================================================================

def svd(A: list[list[float]], max_iter: int = 1000, tol: float = 1e-10):
    """
    Phân rã SVD từ đầu (from scratch): A = U @ Sigma @ V^T

    Thuật toán:
        1. Tính B = AᵀA  (ma trận đối xứng, semi-definite dương)
        2. Dùng QR iteration tìm eigenvalues λᵢ và eigenvectors V của B
        3. Singular values: σᵢ = √λᵢ
        4. Left singular vectors: uᵢ = A @ vᵢ / σᵢ

    Tham số:
        A        : ma trận đầu vào m x n
        max_iter : số vòng lặp tối đa cho QR iteration
        tol      : ngưỡng hội tụ

    Trả về:
        U     : ma trận m x m (left singular vectors)
        sigma : list singular values (giảm dần)
        Vt    : ma trận n x n = V^T (right singular vectors)
    """
    m, n = mat_shape(A)

    # -----------------------------------------------------------------------
    # Bước 1: Tính B = AᵀA
    # -----------------------------------------------------------------------
    At = mat_transpose(A)
    B  = mat_mul(At, A)          # n x n, đối xứng, semi-definite dương

    # -----------------------------------------------------------------------
    # Bước 2: QR iteration trên B → eigenvalues của AᵀA + eigenvectors V
    # -----------------------------------------------------------------------
    eigenvalues_B, V = _qr_iteration(B, max_iter=max_iter, tol=tol)

    # Kẹp về 0 để tránh căn âm do sai số làm tròn
    eigenvalues_B = [max(0.0, lam) for lam in eigenvalues_B]

    # -----------------------------------------------------------------------
    # Bước 3: Sắp xếp giảm dần
    # -----------------------------------------------------------------------
    eigenvalues_B, V = _sort_eigen(eigenvalues_B, V)

    # -----------------------------------------------------------------------
    # Bước 4: Tính singular values σᵢ = √λᵢ
    # -----------------------------------------------------------------------
    sigma = [math.sqrt(lam) for lam in eigenvalues_B]

    # -----------------------------------------------------------------------
    # Bước 5: Tính U — left singular vectors
    #   uᵢ = A @ vᵢ / σᵢ  (với σᵢ > 0)
    #   Với σᵢ ≈ 0: bổ sung vector trực giao tùy ý (Gram-Schmidt trên U)
    # -----------------------------------------------------------------------
    rank = sum(1 for s in sigma if s > tol)

    U = mat_zeros(m, m)

    # Tính u cho các singular value khác 0
    for j in range(rank):
        vj  = mat_col(V, j)
        Avj = [sum(A[i][k] * vj[k] for k in range(n)) for i in range(m)]
        uj  = vec_scale(Avj, 1.0 / sigma[j])
        mat_set_col(U, j, uj)

    # Gram-Schmidt để hoàn thiện U với các cột còn lại (null space của Aᵀ)
    if rank < m:
        # Khởi tạo các cột còn lại bằng vector chuẩn e_j
        candidate_idx = 0
        for j in range(rank, m):
            while candidate_idx < m:
                e = [0.0] * m
                e[candidate_idx] = 1.0
                candidate_idx += 1

                # Trừ hình chiếu lên các cột đã có trong U
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
    # Bước 6: Chuẩn bị Vt = V^T và Sigma đầy đủ m x n
    # -----------------------------------------------------------------------
    Vt    = mat_transpose(V)
    p     = min(m, n)
    sigma = sigma[:p]   # chỉ giữ p singular values

    return U, sigma, Vt


# =============================================================================
# PHẦN 5: XẤP XỈ HẠNG THẤP (Low-rank Approximation)
# =============================================================================

def low_rank_approximation(U, sigma, Vt, k: int) -> list[list[float]]:
    """
    Tính xấp xỉ hạng k của ma trận A:
        Aₖ = Σᵢ₌₁ᵏ σᵢ · uᵢ · vᵢᵀ

    Đây là xấp xỉ tốt nhất hạng k theo chuẩn Frobenius (Định lý Eckart-Young).

    Tham số:
        U     : left singular vectors (m x m)
        sigma : list singular values
        Vt    : right singular vectors^T (n x n)
        k     : hạng xấp xỉ

    Trả về:
        Ak : ma trận xấp xỉ hạng k (m x n)
    """
    m = len(U)
    n = len(Vt[0])
    Ak = mat_zeros(m, n)

    k = min(k, len(sigma))
    for i in range(k):
        ui = mat_col(U, i)           # vector cột m x 1
        vi = mat_row(Vt, i)          # vector hàng 1 x n  (= vᵢᵀ)

        # Outer product: uᵢ @ vᵢᵀ  (rank-1 matrix)
        for r in range(m):
            for c in range(n):
                Ak[r][c] += sigma[i] * ui[r] * vi[c]

    return Ak


# =============================================================================
# PHẦN 7: KIỂM CHỨNG (dùng numpy chỉ để verify)
# =============================================================================

def verify_svd(A_orig: list[list[float]], U, sigma, Vt, tol: float = 1e-6) -> bool:
    """
    Kiểm chứng kết quả SVD bằng cách tái tạo A từ U, Sigma, Vt
    và so sánh sai số tương đối.

    Không dùng numpy — tính thuần túy bằng phép toán ma trận tự cài đặt.

    Trả về:
        True nếu ||A - U Σ Vᵀ||_F / ||A||_F < tol
    """
    m, n = mat_shape(A_orig)
    p = len(sigma)

    # Tái tạo A_approx = U @ Sigma_full @ Vt
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
    Kiểm tra tính trực giao: M^T @ M ≈ I
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
    Kiểm chứng bổ sung bằng numpy (chỉ để so sánh, KHÔNG dùng cho thuật toán).
    """
    try:
        import numpy as np
        A_np  = np.array(A_list)
        U_np, s_np, Vt_np = np.linalg.svd(A_np)

        print("\n  [NumPy reference]")
        print(f"  Singular values (numpy): {s_np.round(6).tolist()}")
        print(f"  Singular values (ours) : {[round(s, 6) for s in sigma]}")

        # So sánh singular values (thứ tự đã giảm dần)
        p = min(len(sigma), len(s_np))
        max_diff = max(abs(sigma[i] - s_np[i]) for i in range(p))
        print(f"  Max |σᵢ - σᵢ_numpy| = {max_diff:.2e}  →  {'✓ OK' if max_diff < 1e-5 else '✗ FAIL'}")
    except ImportError:
        print("  NumPy không khả dụng, bỏ qua bước kiểm chứng numpy.")


# =============================================================================
# PHẦN 8: DEMO (CHỈ CHỨA SVD & XẤP XỈ HẠNG THẤP)
# =============================================================================

def _list2d(data):
    """Chuyển list of list về float."""
    return [[float(x) for x in row] for row in data]

if __name__ == "__main__":
    print("=" * 60)
    print("  DEMO: Phân Rã SVD From Scratch")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test case 1: Ma trận 3x2
    # ------------------------------------------------------------------
    print("\n--- Test 1: Ma trận 3x2 ---")
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
    print("\nKiểm chứng:")
    verify_svd(A1, U1, s1, Vt1)
    verify_orthogonal(U1, "U1")
    verify_orthogonal(mat_transpose(Vt1), "V1")
    verify_svd_with_numpy(A1, U1, s1, Vt1)

    # ------------------------------------------------------------------
    # Test case 2: Ma trận vuông 3x3
    # ------------------------------------------------------------------
    print("\n--- Test 2: Ma trận vuông 3x3 ---")
    A2 = _list2d([
        [4, 0, 0],
        [3, 2, 0],
        [1, 1, 3],
    ])
    print_matrix(A2, "A2")

    U2, s2, Vt2 = svd(A2)
    print(f"\n  Singular values: {[round(s, 6) for s in s2]}")
    print("\nKiểm chứng:")
    verify_svd(A2, U2, s2, Vt2)
    verify_orthogonal(U2, "U2")
    verify_svd_with_numpy(A2, U2, s2, Vt2)

    # ------------------------------------------------------------------
    # Test case 3: Ma trận đặc biệt — có singular value = 0 (hạng thấp)
    # ------------------------------------------------------------------
    print("\n--- Test 3: Ma trận hạng thấp (rank-deficient) ---")
    A3 = _list2d([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],   # hàng 3 = hàng 1 + hàng 2 * 2 - ...  (phụ thuộc tuyến tính)
    ])
    print_matrix(A3, "A3")

    U3, s3, Vt3 = svd(A3)
    print(f"\n  Singular values: {[round(s, 6) for s in s3]}")
    print("\nKiểm chứng:")
    verify_svd(A3, U3, s3, Vt3)
    verify_svd_with_numpy(A3, U3, s3, Vt3)

    # ------------------------------------------------------------------
    # Test case 4: Xấp xỉ hạng thấp
    # ------------------------------------------------------------------
    print("\n--- Test 4: Xấp xỉ hạng thấp (Low-rank Approximation) ---")
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