from manim import *
import numpy as np

A = [[2.0, 1.0], [0.0, 3.0]]

def to_3d(v):
    return [v[0], v[1], 0]

def create_matrix_text(matrix):
    rows = []
    for row in matrix:
        row_text = Text("  ".join(str(x) for x in row)).scale(0.6)
        rows.append(row_text)
    return VGroup(*rows).arrange(DOWN)

def create_caption(text):
    return Text(text).scale(0.45).to_edge(DOWN)

class MatrixDecompositionScene(Scene):
    def construct(self):
        self.show_introduction()
        self.show_svd()
        self.show_diag()

    def show_introduction(self):
        title = Text("Phan ra ma tran: SVD va Cheo hoa").scale(0.7).to_edge(UP)

        matrix_A = create_matrix_text([[2, 1], [0, 3]])
        label = Text("Ma tran A").scale(0.6).next_to(matrix_A, UP)

        self.play(Write(title))
        self.play(Write(label), FadeIn(matrix_A))
        self.wait(2)
        self.play(FadeOut(title, matrix_A, label))

    def show_svd(self):
        title = Text("SVD: A = U Σ V^T").scale(0.7).to_edge(UP)
        self.play(Write(title))

        axes = Axes(x_range=[-4,4], y_range=[-4,4])
        circle = Circle(radius=1, color=BLUE)
        caption = create_caption("Hinh tron don vi ban dau")

        self.play(Create(axes), Create(circle), Write(caption))
        self.wait()

        U, s, Vt = np.linalg.svd(A)
        Sigma = np.diag(s)

        v1 = Arrow(ORIGIN, to_3d(Vt.T[:,0]), color=ORANGE)
        v2 = Arrow(ORIGIN, to_3d(Vt.T[:,1]), color=ORANGE)

        self.play(Create(v1), Create(v2))
        self.wait()

        self.play(Transform(circle, circle.copy().apply_matrix(Vt.tolist())))
        caption_vt = create_caption("Quay theo V^T")
        self.play(ReplacementTransform(caption, caption_vt))
        caption = caption_vt
        self.wait()

        self.play(Transform(circle, circle.copy().apply_matrix(Sigma.tolist())))
        caption_sigma = create_caption("Co gian theo Sigma")
        self.play(ReplacementTransform(caption, caption_sigma))
        caption = caption_sigma
        self.wait()

        self.play(Transform(circle, circle.copy().apply_matrix(U.tolist())))
        caption_u = create_caption("Quay theo U")
        self.play(ReplacementTransform(caption, caption_u))
        caption = caption_u
        self.wait()

        u1 = Arrow(ORIGIN, to_3d(U[:,0]), color=GREEN)
        u2 = Arrow(ORIGIN, to_3d(U[:,1]), color=GREEN)

        self.play(Create(u1), Create(u2))
        self.wait()

        self.play(FadeOut(circle, axes, v1, v2, u1, u2, caption, title))

    def show_diag(self):
        title = Text("Cheo hoa: A = P D P^-1").scale(0.7).to_edge(UP)
        self.play(Write(title))

        axes = Axes(x_range=[-4,4], y_range=[-4,4])
        square = Square(side_length=1.5, color=BLUE)
        caption = create_caption("Hinh vuong ban dau")

        self.play(Create(axes), Create(square), Write(caption))
        self.wait()

        eigenvalues, P = np.linalg.eig(A)
        D = np.diag(eigenvalues)
        P_inv = np.linalg.inv(P)

        e1 = Arrow(ORIGIN, to_3d(P[:,0]), color=YELLOW)
        e2 = Arrow(ORIGIN, to_3d(P[:,1]), color=YELLOW)

        self.play(Create(e1), Create(e2))
        self.wait()

        self.play(Transform(square, square.copy().apply_matrix(P_inv.tolist())))
        caption_basis = create_caption("Chuyen sang co so rieng")
        self.play(ReplacementTransform(caption, caption_basis))
        caption = caption_basis
        self.wait()

        self.play(Transform(square, square.copy().apply_matrix(D.tolist())))
        caption_diag = create_caption("Nhan theo ma tran duong cheo D")
        self.play(ReplacementTransform(caption, caption_diag))
        caption = caption_diag
        self.wait()

        self.play(Transform(square, square.copy().apply_matrix(P.tolist())))
        caption_back = create_caption("Quay ve he co so ban dau")
        self.play(ReplacementTransform(caption, caption_back))
        caption = caption_back
        self.wait()

        self.play(FadeOut(square, axes, e1, e2, caption, title))