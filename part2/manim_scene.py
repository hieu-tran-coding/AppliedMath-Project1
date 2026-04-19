"""
manim_scene.py
==============
SVD Visualization -- Structured Presentation

Structure:
  1. Title
  2. SVD Theory  -- explain each decomposition step
  3. Example 1   -- A = [[2,1],[1,2]]  step-by-step decomposition
  4. Example 2   -- A = [[1,1],[0,1]]  step-by-step decomposition
  5. Geometric meaning analysis for both examples
  6. Summary

Course : MTH00051 -- Applied Mathematics and Statistics
School : FIT -- HCMUS

HOW TO RUN:
    manim -pqh manim_scene.py FullPresentation      # full video
    manim -pql manim_scene.py FullPresentation      # quick preview
    manim -pql manim_scene.py TitleScene
    manim -pql manim_scene.py SVDTheoryScene
    manim -pql manim_scene.py Example1Scene
    manim -pql manim_scene.py Example2Scene
    manim -pql manim_scene.py GeometricMeaningScene
    manim -pql manim_scene.py SummaryScene
"""

from manim import *
import numpy as np

# ==============================================================
# COLORS
# ==============================================================
C_TITLE  = YELLOW_B
C_A      = BLUE_B
C_U      = GREEN_C
C_SIG    = ORANGE
C_V      = PURPLE_B
C_ATA    = TEAL_C
C_EIGVAL = GOLD_C
C_EIGVEC = PINK
C_STEP   = YELLOW_C
C_INFO   = GREY_B
C_CIRC   = WHITE
C_ELLIP  = RED_C
C_E1     = RED_C
C_E2     = BLUE_C
C_CHECK  = GREEN_C

# ==============================================================
# TIMING
# ==============================================================
WS  = 1.2   # short wait
WM  = 2.0   # medium wait
WL  = 3.0   # long wait
AT  = 1.4   # normal anim
ATS = 2.0   # slow anim
ATF = 0.8   # fast anim

GEO_SCALE = 0.68
GEO_SHIFT = RIGHT * 3.0
MAT_SHIFT = LEFT  * 3.2


# ==============================================================
# UTILITIES
# ==============================================================

def fv(x, d=3):
    """Format float as short string."""
    v = float(x)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.{d}f}"


def fv2(x):
    """Format float with 2 decimal places."""
    return fv(x, d=2)


def mat_mob(A, color=WHITE, sc=0.52):
    """Matrix Mobject from 2D list or numpy array."""
    if isinstance(A, np.ndarray):
        A = A.tolist()
    rows = [[fv(x) for x in row] for row in A]
    return Matrix(rows).set_color(color).scale(sc)


def diag_mob(vals, color=C_SIG, sc=0.52):
    """Diagonal matrix Mobject."""
    n    = len(vals)
    rows = [[fv(vals[i]) if i == j else "0" for j in range(n)] for i in range(n)]
    return Matrix(rows).set_color(color).scale(sc)


def to_screen(xy):
    """Math coords -> Manim screen coords (geometric panel)."""
    return GEO_SCALE * np.array([xy[0], xy[1], 0.0]) + np.array([GEO_SHIFT[0], GEO_SHIFT[1], 0.0])


def unit_circle_image(M, color=WHITE):
    """Parametric curve: image of unit circle under M."""
    M_arr = np.array(M, dtype=float)
    def f(t):
        v = M_arr @ np.array([np.cos(t), np.sin(t)])
        return to_screen(v)
    return ParametricFunction(f, t_range=[0.0, TAU], color=color, stroke_width=2.8)


def geo_arrow(tip_xy, color=WHITE):
    """Arrow from origin to tip in geometric panel."""
    return Arrow(to_screen([0, 0]), to_screen(tip_xy), color=color,
                 buff=0, stroke_width=3.0, max_tip_length_to_length_ratio=0.20)


def geo_plane():
    """NumberPlane for the right geometric panel."""
    return NumberPlane(
        x_range=[-4, 4, 1], y_range=[-3, 3, 1],
        background_line_style={"stroke_color": GREY_D,
                               "stroke_opacity": 0.4, "stroke_width": 0.7},
        axis_config={"stroke_color": GREY_B, "stroke_width": 1.1,
                     "include_ticks": False},
    ).scale(GEO_SCALE).shift(GEO_SHIFT)


def banner(text, color=C_TITLE, fs=30):
    """Section header pinned to top."""
    return Text(text, font_size=fs, color=color).to_edge(UP, buff=0.20)


def step_label(num, text, color=C_STEP):
    """Numbered step label."""
    return VGroup(
        Text(f"Step {num}:", font_size=21, color=color, weight=BOLD),
        Text(text,           font_size=20, color=WHITE),
    ).arrange(RIGHT, buff=0.18)


def highlight_box(mob, color):
    """Surrounding rectangle highlight."""
    return SurroundingRectangle(mob, color=color, buff=0.12,
                                corner_radius=0.08, stroke_width=1.5)


# ==============================================================
# SCENE 1 -- TITLE
# ==============================================================

def run_title(sc):
    title = Text("Singular Value Decomposition",
                 font_size=48, color=C_TITLE, weight=BOLD)
    sub   = Text("A step-by-step walkthrough with two examples",
                 font_size=24, color=WHITE)
    VGroup(title, sub).arrange(DOWN, buff=0.45).move_to(UP * 0.8)

    sep = Line(5.5 * LEFT, 5.5 * RIGHT, stroke_width=1.2, color=C_TITLE)
    sep.next_to(sub, DOWN, buff=0.45)

    formula = VGroup(
        MathTex(r"A",      color=C_A,   font_size=52),
        MathTex(r"=",                   font_size=52),
        MathTex(r"U",      color=C_U,   font_size=52),
        MathTex(r"\Sigma", color=C_SIG, font_size=52),
        MathTex(r"V^\top", color=C_V,   font_size=52),
    ).arrange(RIGHT, buff=0.28).next_to(sep, DOWN, buff=0.55)

    footer = Text("MTH00051 -- FIT HCMUS",
                  font_size=18, color=GREY_C).to_edge(DOWN, buff=0.3)

    sc.play(Write(title), run_time=1.8)
    sc.play(FadeIn(sub, shift=DOWN * 0.12))
    sc.play(Create(sep))
    sc.play(LaggedStart(*[Write(p) for p in formula], lag_ratio=0.3), run_time=1.8)
    sc.play(FadeIn(footer))
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.8)
    sc.wait(0.3)


# ==============================================================
# SCENE 2 -- SVD THEORY (step-by-step explanation)
# ==============================================================

def run_svd_theory(sc):
    """
    Explain the 3 computational steps of SVD:
      Step 1 -- Compute A^T A, find its eigenvalues and eigenvectors -> V
      Step 2 -- Singular values sigma_i = sqrt(lambda_i) -> Sigma
      Step 3 -- Compute left singular vectors u_i = A v_i / sigma_i -> U
    """

    hdr = banner("SVD -- Theory and Computation Steps")
    sc.play(Write(hdr))
    sc.wait(WS)

    # ---- Overview formula ----
    overview = VGroup(
        Text("Goal: decompose any matrix", font_size=22, color=C_INFO),
        VGroup(
            MathTex(r"A",      color=C_A,   font_size=48),
            MathTex(r"=",                   font_size=48),
            MathTex(r"U",      color=C_U,   font_size=48),
            MathTex(r"\Sigma", color=C_SIG, font_size=48),
            MathTex(r"V^\top", color=C_V,   font_size=48),
        ).arrange(RIGHT, buff=0.25),
    ).arrange(DOWN, buff=0.22)
    overview.next_to(hdr, DOWN, buff=0.45)
    sc.play(FadeIn(overview[0]))
    sc.play(LaggedStart(*[Write(p) for p in overview[1]], lag_ratio=0.3), run_time=1.8)
    sc.wait(WM)

    # ---- Step 1 ----
    s1_title = step_label(1, "Compute A^T A  -- find eigenvalues and eigenvectors")
    s1_title.next_to(overview, DOWN, buff=0.55)

    s1_body = VGroup(
        MathTex(r"B = A^\top A", color=C_ATA, font_size=26),
        Text("B is symmetric positive semi-definite", font_size=18, color=C_INFO),
        MathTex(r"B\,\mathbf{v}_i = \lambda_i\,\mathbf{v}_i,\quad \lambda_i \ge 0",
                color=C_EIGVAL, font_size=26),
        Text("Columns of V = eigenvectors of A^T A  (orthonormal)",
             font_size=18, color=C_V),
    ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
    s1_body.next_to(s1_title, DOWN, buff=0.22)
    box1 = highlight_box(s1_body, C_ATA)

    sc.play(Write(s1_title), run_time=AT)
    sc.play(FadeIn(s1_body), Create(box1), run_time=AT)
    sc.wait(WM)

    # ---- Step 2 ----
    s2_title = step_label(2, "Compute singular values  ->  build Sigma")
    s2_title.next_to(s1_body, DOWN, buff=0.45)

    s2_body = VGroup(
        MathTex(r"\sigma_i = \sqrt{\lambda_i}", color=C_SIG, font_size=28),
        Text("Sort descending:  sigma_1 >= sigma_2 >= ... >= 0",
             font_size=18, color=C_INFO),
        MathTex(r"\Sigma = \mathrm{diag}(\sigma_1,\,\sigma_2,\,\ldots)",
                color=C_SIG, font_size=24),
        Text("sigma_i tells how much A stretches in direction v_i",
             font_size=18, color=C_SIG),
    ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    s2_body.next_to(s2_title, DOWN, buff=0.18)
    box2 = highlight_box(s2_body, C_SIG)

    sc.play(Write(s2_title), run_time=AT)
    sc.play(FadeIn(s2_body), Create(box2), run_time=AT)
    sc.wait(WM)

    sc.play(FadeOut(*sc.mobjects), run_time=0.7)
    sc.wait(0.2)

    # ---- Step 3 (new slide) ----
    hdr2 = banner("SVD -- Theory and Computation Steps (cont.)")
    sc.play(Write(hdr2))

    s3_title = step_label(3, "Compute left singular vectors  ->  build U")
    s3_title.next_to(hdr2, DOWN, buff=0.55)

    s3_body = VGroup(
        MathTex(r"\mathbf{u}_i = \frac{A\,\mathbf{v}_i}{\sigma_i}",
                color=C_U, font_size=32),
        Text("u_i is the unit vector in the direction A sends v_i",
             font_size=18, color=C_INFO),
        MathTex(r"U = [\,\mathbf{u}_1 \;\; \mathbf{u}_2 \;\; \cdots\,]",
                color=C_U, font_size=26),
        Text("U is orthogonal:  U^T U = I", font_size=18, color=C_U),
    ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
    s3_body.next_to(s3_title, DOWN, buff=0.22)
    box3 = highlight_box(s3_body, C_U)

    sc.play(Write(s3_title), run_time=AT)
    sc.play(FadeIn(s3_body), Create(box3), run_time=AT)
    sc.wait(WM)

    # ---- Verification ----
    verify_title = Text("Verification:", font_size=22, color=C_STEP, weight=BOLD)
    verify_body  = VGroup(
        MathTex(r"A = U\,\Sigma\,V^\top", font_size=30, color=WHITE),
        MathTex(r"U^\top U = I,\quad V^\top V = I", font_size=22, color=C_INFO),
        MathTex(r"\mathrm{rank}(A) = \text{number of nonzero singular values}",
                font_size=22, color=C_EIGVAL),
    ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
    VGroup(verify_title, verify_body).arrange(DOWN, buff=0.18, aligned_edge=LEFT).next_to(
        s3_body, DOWN, buff=0.50)
    vbox = highlight_box(VGroup(verify_title, verify_body), C_CHECK)

    sc.play(Write(verify_title))
    sc.play(FadeIn(verify_body), Create(vbox), run_time=AT)
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.8)
    sc.wait(0.3)


# ==============================================================
# EXAMPLE HELPER -- shared step-by-step decomposition
# ==============================================================

def run_example_decomposition(sc, A, ex_num, title_str, note_str=""):
    """
    Show the full SVD decomposition of A step by step.
    Uses numpy ONLY for computing values to display -- not for the algorithm logic shown.

    Layout: all text/matrices on screen, one step at a time.
    """
    # Compute via numpy (for display values only)
    AtA              = A.T @ A
    eigvals, V_mat   = np.linalg.eigh(AtA)
    order            = np.argsort(-eigvals)
    eigvals          = eigvals[order]
    V_mat            = V_mat[:, order]
    sigmas           = np.sqrt(np.maximum(eigvals, 0.0))
    U_mat, s_np, Vt_np = np.linalg.svd(A)
    # Use numpy svd for U for numerical stability in display
    sigma_list       = list(s_np)

    # ================================================================
    # SLIDE A -- introduce the matrix
    # ================================================================
    hdr = banner(f"Example {ex_num}: {title_str}")
    sc.play(Write(hdr))

    A_lbl = MathTex(r"A = ", color=C_A, font_size=40)
    A_mob = mat_mob(A, color=C_A, sc=0.65)
    A_grp = VGroup(A_lbl, A_mob).arrange(RIGHT, buff=0.18).move_to(ORIGIN + UP * 0.3)
    sc.play(Write(A_lbl), Write(A_mob), run_time=AT)
    sc.wait(WS)

    anchor = A_grp   # goal will be placed below this
    if note_str:
        note = Text(note_str, font_size=20, color=C_INFO)
        note.next_to(A_grp, DOWN, buff=0.40)
        sc.play(FadeIn(note))
        sc.wait(WS)
        anchor = note   # push goal below the note

    goal = VGroup(
        Text("We want to find:", font_size=20, color=C_INFO),
        VGroup(
            MathTex(r"U",      color=C_U,   font_size=36),
            MathTex(r"\Sigma", color=C_SIG, font_size=36),
            MathTex(r"V^\top", color=C_V,   font_size=36),
        ).arrange(RIGHT, buff=0.2),
        Text("such that  A = U Sigma V^T", font_size=20, color=C_INFO),
    ).arrange(DOWN, buff=0.18)
    goal.next_to(anchor, DOWN, buff=0.45)
    sc.play(FadeIn(goal), run_time=AT)
    sc.wait(WM)
    sc.play(FadeOut(*sc.mobjects), run_time=0.7)
    sc.wait(0.2)

    # ================================================================
    # SLIDE B -- Step 1: compute A^T A and its eigenstuff
    # ================================================================
    hdr = banner(f"Example {ex_num} -- Step 1: Compute A^T A")
    sc.play(Write(hdr))

    # Show A again (small, top-left)
    A_small = VGroup(MathTex(r"A = ", color=C_A, font_size=26),
                     mat_mob(A, color=C_A, sc=0.44)).arrange(RIGHT, buff=0.1)
    A_small.to_edge(LEFT, buff=0.5).to_edge(UP, buff=0.9)
    sc.play(FadeIn(A_small))

    # A^T A formula and result
    ata_formula = VGroup(
        MathTex(r"A^\top A = ", color=C_ATA, font_size=30),
        MathTex(r"?", color=C_ATA, font_size=30),
    ).arrange(RIGHT, buff=0.1).next_to(hdr, DOWN, buff=0.55)
    sc.play(Write(ata_formula), run_time=AT)
    sc.wait(WS)

    ata_result = VGroup(
        MathTex(r"A^\top A = ", color=C_ATA, font_size=28),
        mat_mob(AtA, color=C_ATA, sc=0.55),
    ).arrange(RIGHT, buff=0.12)
    ata_result.move_to(ata_formula)
    sc.play(Transform(ata_formula, ata_result), run_time=AT)
    sc.wait(WS)

    # Eigenvalue equation
    eig_intro = Text("Find eigenvalues and eigenvectors of A^T A:",
                     font_size=20, color=C_INFO)
    eig_eq    = MathTex(r"A^\top A\,\mathbf{v} = \lambda\,\mathbf{v}",
                        color=C_EIGVAL, font_size=28)
    VGroup(eig_intro, eig_eq).arrange(DOWN, buff=0.2).next_to(ata_result, DOWN, buff=0.4)
    sc.play(FadeIn(eig_intro), run_time=ATF)
    sc.play(Write(eig_eq), run_time=AT)
    sc.wait(WS)

    # Show eigenvalues
    ev_items = VGroup(*[
        MathTex(fr"\lambda_{i+1} = {fv2(eigvals[i])}", color=C_EIGVAL, font_size=26)
        for i in range(len(eigvals))
    ]).arrange(RIGHT, buff=0.6)
    ev_items.next_to(eig_eq, DOWN, buff=0.30)
    ev_box = highlight_box(ev_items, C_EIGVAL)

    ev_lbl = Text("Eigenvalues:", font_size=20, color=C_EIGVAL)
    ev_lbl.next_to(ev_items, LEFT, buff=0.2)
    sc.play(FadeIn(ev_lbl), FadeIn(ev_items), Create(ev_box), run_time=AT)
    sc.wait(WS)

    # Show eigenvectors (columns of V)
    vc_items = VGroup(*[
        VGroup(
            MathTex(fr"\mathbf{{v}}_{i+1} = ", color=C_V, font_size=22),
            mat_mob(V_mat[:, i].reshape(-1, 1), color=C_V, sc=0.48),
        ).arrange(RIGHT, buff=0.08)
        for i in range(V_mat.shape[1])
    ]).arrange(RIGHT, buff=0.5)
    vc_items.next_to(ev_items, DOWN, buff=0.32)

    vc_lbl = Text("Eigenvectors:", font_size=20, color=C_V)
    vc_lbl.next_to(vc_items, LEFT, buff=0.15)
    sc.play(FadeIn(vc_lbl), FadeIn(vc_items), run_time=AT)
    sc.wait(WS)

    # V matrix
    V_show = VGroup(
        MathTex(r"V = ", color=C_V, font_size=26),
        mat_mob(V_mat, color=C_V, sc=0.52),
        MathTex(r"\Rightarrow\;V^\top = ", color=C_V, font_size=26),
        mat_mob(V_mat.T, color=C_V, sc=0.52),
    ).arrange(RIGHT, buff=0.15)
    V_show.next_to(vc_items, DOWN, buff=0.30)
    sc.play(FadeIn(V_show), run_time=AT)
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.7)
    sc.wait(0.2)

    # ================================================================
    # SLIDE C -- Step 2: singular values -> Sigma
    # ================================================================
    hdr = banner(f"Example {ex_num} -- Step 2: Singular Values")
    sc.play(Write(hdr))

    s2_formula = VGroup(
        MathTex(r"\sigma_i = \sqrt{\lambda_i}", color=C_SIG, font_size=32),
        Text("Take square root of each eigenvalue of A^T A", font_size=19, color=C_INFO),
    ).arrange(DOWN, buff=0.2).next_to(hdr, DOWN, buff=0.5)
    sc.play(Write(s2_formula[0]))
    sc.play(FadeIn(s2_formula[1]))
    sc.wait(WS)

    # Arrow: lambda -> sigma
    conv_items = VGroup(*[
        VGroup(
            MathTex(fr"\lambda_{i+1} = {fv2(eigvals[i])}", color=C_EIGVAL, font_size=24),
            MathTex(r"\;\longrightarrow\;", font_size=24),
            MathTex(fr"\sigma_{i+1} = \sqrt{{{fv2(eigvals[i])}}} = {fv2(sigma_list[i])}",
                    color=C_SIG, font_size=24),
        ).arrange(RIGHT, buff=0.15)
        for i in range(len(sigma_list))
    ]).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
    conv_items.next_to(s2_formula, DOWN, buff=0.42)

    for item in conv_items:
        sc.play(FadeIn(item), run_time=AT)
        sc.wait(0.4)
    sc.wait(WS)

    # Build Sigma matrix
    Sigma_lbl = Text("Build the diagonal matrix Sigma:", font_size=20, color=C_SIG)
    Sigma_lbl.next_to(conv_items, DOWN, buff=0.38)
    Sigma_show = VGroup(
        MathTex(r"\Sigma = ", color=C_SIG, font_size=30),
        diag_mob(sigma_list, color=C_SIG, sc=0.58),
    ).arrange(RIGHT, buff=0.12)
    Sigma_show.next_to(Sigma_lbl, DOWN, buff=0.22)
    sig_box = highlight_box(Sigma_show, C_SIG)

    sc.play(FadeIn(Sigma_lbl))
    sc.play(FadeIn(Sigma_show), Create(sig_box), run_time=AT)
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.7)
    sc.wait(0.2)

    # ================================================================
    # SLIDE D -- Step 3: left singular vectors -> U
    # ================================================================
    hdr = banner(f"Example {ex_num} -- Step 3: Left Singular Vectors")
    sc.play(Write(hdr))

    s3_formula = VGroup(
        MathTex(r"\mathbf{u}_i = \frac{A\,\mathbf{v}_i}{\sigma_i}",
                color=C_U, font_size=36),
        Text("Normalize A*v_i to get each column of U", font_size=19, color=C_INFO),
    ).arrange(DOWN, buff=0.2).next_to(hdr, DOWN, buff=0.5)
    sc.play(Write(s3_formula[0]))
    sc.play(FadeIn(s3_formula[1]))
    sc.wait(WS)

    # Compute u_i for each i
    for i in range(V_mat.shape[1]):
        vi    = V_mat[:, i]
        si    = sigma_list[i]
        Avi   = A @ vi
        ui    = Avi / si if si > 1e-12 else np.zeros_like(Avi)

        row = VGroup(
            MathTex(fr"\mathbf{{u}}_{i+1}", color=C_U, font_size=22),
            MathTex(r"=\frac{A\,\mathbf{v}_" + str(i+1) + r"}{\sigma_" + str(i+1) + "}",
                    color=C_U, font_size=22),
            MathTex(r"=\frac{1}{" + fv2(si) + r"}", color=C_SIG, font_size=22),
            mat_mob(Avi.reshape(-1, 1), color=C_A, sc=0.42),
            MathTex(r"=", font_size=22),
            mat_mob(ui.reshape(-1, 1), color=C_U, sc=0.44),
        ).arrange(RIGHT, buff=0.12)

        if i == 0:
            row.next_to(s3_formula, DOWN, buff=0.42)
        else:
            row.next_to(prev_row, DOWN, buff=0.30)

        sc.play(FadeIn(row), run_time=AT)
        sc.wait(WS)
        prev_row = row

    # Build U
    U_lbl = Text("Assemble U from columns:", font_size=20, color=C_U)
    U_lbl.next_to(prev_row, DOWN, buff=0.38)
    U_show = VGroup(
        MathTex(r"U = ", color=C_U, font_size=30),
        mat_mob(U_mat, color=C_U, sc=0.58),
    ).arrange(RIGHT, buff=0.12)
    U_show.next_to(U_lbl, DOWN, buff=0.20)
    u_box = highlight_box(U_show, C_U)

    sc.play(FadeIn(U_lbl))
    sc.play(FadeIn(U_show), Create(u_box), run_time=AT)
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.7)
    sc.wait(0.2)

    # ================================================================
    # SLIDE E -- Full decomposition result
    # ================================================================
    hdr = banner(f"Example {ex_num} -- Full SVD Result")
    sc.play(Write(hdr))

    # Title
    result_title = Text("Putting it all together:", font_size=22, color=C_STEP)
    result_title.next_to(hdr, DOWN, buff=0.45)
    sc.play(FadeIn(result_title))

    # A = U Sigma V^T with colored labels
    eq_lbl = VGroup(
        MathTex(r"A", color=C_A, font_size=36),
        MathTex(r"=", font_size=36),
        MathTex(r"U", color=C_U, font_size=36),
        MathTex(r"\Sigma", color=C_SIG, font_size=36),
        MathTex(r"V^\top", color=C_V, font_size=36),
    ).arrange(RIGHT, buff=0.2).next_to(result_title, DOWN, buff=0.28)
    sc.play(LaggedStart(*[Write(p) for p in eq_lbl], lag_ratio=0.3), run_time=1.5)
    sc.wait(WS)

    # Show actual matrices side by side
    A_part = VGroup(MathTex(r"A = ", color=C_A, font_size=24),
                    mat_mob(A, color=C_A, sc=0.50)).arrange(RIGHT, buff=0.1)
    eq_sym  = MathTex(r"=", font_size=28)
    U_part  = VGroup(MathTex(r"U = ", color=C_U, font_size=24),
                     mat_mob(U_mat, color=C_U, sc=0.50)).arrange(RIGHT, buff=0.08)
    S_part  = VGroup(MathTex(r"\Sigma = ", color=C_SIG, font_size=24),
                     diag_mob(sigma_list, color=C_SIG, sc=0.50)).arrange(RIGHT, buff=0.08)
    Vt_part = VGroup(MathTex(r"V^\top = ", color=C_V, font_size=24),
                     mat_mob(V_mat.T, color=C_V, sc=0.50)).arrange(RIGHT, buff=0.08)

    matrices_row = VGroup(A_part, eq_sym, U_part, S_part, Vt_part).arrange(RIGHT, buff=0.18)
    matrices_row.next_to(eq_lbl, DOWN, buff=0.35)
    sc.play(FadeIn(matrices_row), run_time=AT)
    sc.wait(WM)

    # Verification
    verify = VGroup(
        MathTex(r"U^\top U = I\;\checkmark", color=C_CHECK, font_size=22),
        MathTex(r"V^\top V = I\;\checkmark", color=C_CHECK, font_size=22),
        MathTex(r"\|A - U\Sigma V^\top\|_F \approx 0\;\checkmark",
                color=C_CHECK, font_size=22),
    ).arrange(RIGHT, buff=0.5)
    verify.next_to(matrices_row, DOWN, buff=0.40)
    v_box = highlight_box(verify, C_CHECK)

    sc.play(FadeIn(verify), Create(v_box), run_time=AT)
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.8)
    sc.wait(0.3)


# ==============================================================
# SCENE 5 -- GEOMETRIC MEANING (analysis after both examples)
# ==============================================================

def run_geometric_meaning(sc, A1, A2, ex1_title, ex2_title):
    """
    Analyze the geometric meaning of SVD for both examples.
    Show: unit circle -> V^T -> Sigma V^T -> U Sigma V^T  for each example.
    Then compare and explain what sigma_1, sigma_2 represent.
    """

    U1, s1, Vt1 = np.linalg.svd(A1)
    U2, s2, Vt2 = np.linalg.svd(A2)
    S1 = np.diag(s1)
    S2 = np.diag(s2)

    # ================================================================
    # SLIDE A -- Explain the geometric interpretation
    # ================================================================
    hdr = banner("Geometric Meaning of SVD")
    sc.play(Write(hdr))

    interp = VGroup(
        Text("Every matrix A acts on the unit circle in 3 stages:",
             font_size=21, color=C_INFO),
        VGroup(
            VGroup(MathTex(r"V^\top:", color=C_V, font_size=24),
                   Text("Rotate into the right coordinates", font_size=19)).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(r"\Sigma:", color=C_SIG, font_size=24),
                   Text("Scale along each axis (by sigma_i)", font_size=19)).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(r"U:", color=C_U, font_size=24),
                   Text("Rotate into the output space", font_size=19)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT),
        MathTex(r"\text{Unit circle} \;\xrightarrow{V^\top}\; "
                r"\text{circle} \;\xrightarrow{\Sigma}\; "
                r"\text{ellipse} \;\xrightarrow{U}\; "
                r"\text{final ellipse}",
                font_size=22, color=C_STEP),
    ).arrange(DOWN, buff=0.30, aligned_edge=LEFT)
    interp.next_to(hdr, DOWN, buff=0.50)

    sc.play(FadeIn(interp[0]))
    sc.wait(WS)
    for item in interp[1]:
        sc.play(FadeIn(item), run_time=ATF)
        sc.wait(0.4)
    sc.play(Write(interp[2]), run_time=AT)
    sc.wait(WM)

    sigma_meaning = VGroup(
        Text("The singular values give the semi-axes of the output ellipse:",
             font_size=20, color=C_INFO),
        MathTex(r"\sigma_1 = \text{length of major axis},\quad "
                r"\sigma_2 = \text{length of minor axis}",
                color=C_SIG, font_size=22),
    ).arrange(DOWN, buff=0.18)
    sigma_meaning.next_to(interp, DOWN, buff=0.38)
    sig_box = highlight_box(sigma_meaning, C_SIG)

    sc.play(FadeIn(sigma_meaning), Create(sig_box), run_time=AT)
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.7)
    sc.wait(0.2)

    # ================================================================
    # SLIDE B -- Example 1 geometric walkthrough
    # ================================================================
    hdr = banner(f"Geometric Meaning -- Example 1: {ex1_title}")
    sc.play(Write(hdr))

    # Left panel: sigma info
    left_info = VGroup(
        Text(f"Example 1:  A = {ex1_title}", font_size=18, color=C_A),
        MathTex(fr"\sigma_1 = {fv2(s1[0])}", color=C_SIG, font_size=26),
        MathTex(fr"\sigma_2 = {fv2(s1[1])}", color=C_SIG, font_size=26),
        Text("Major axis length = sigma_1", font_size=17, color=C_INFO),
        Text("Minor axis length = sigma_2", font_size=17, color=C_INFO),
    ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
    left_info.move_to(MAT_SHIFT)
    sc.play(FadeIn(left_info), run_time=AT)

    # Right panel: animated transform
    plane1 = geo_plane()
    sc.play(Create(plane1), run_time=AT)

    circ = unit_circle_image(np.eye(2), color=C_CIRC)
    sc.play(Create(circ), run_time=AT)

    # Single persistent label -- always Transform this one object so old text disappears
    def make_step_lbl(text, ref_plane):
        t = Text(text, font_size=18, color=C_STEP)
        t.next_to(ref_plane, DOWN, buff=0.15)
        return t

    step_lbl1 = make_step_lbl("Unit circle  (before any transform)", plane1)
    sc.play(FadeIn(step_lbl1), run_time=ATF)
    sc.wait(WS)

    c1 = unit_circle_image(Vt1, color=C_V)
    sc.play(
        Transform(circ, c1),
        Transform(step_lbl1, make_step_lbl("After V^T  (rotation -- still a circle)", plane1)),
        run_time=ATS)
    sc.wait(WM)

    c2 = unit_circle_image(S1 @ Vt1, color=C_SIG)
    sc.play(
        Transform(circ, c2),
        Transform(step_lbl1, make_step_lbl("After Sigma V^T  (scale -> ellipse)", plane1)),
        run_time=ATS)
    sc.wait(WM)

    c3 = unit_circle_image(U1 @ S1 @ Vt1, color=C_ELLIP)
    sc.play(
        Transform(circ, c3),
        Transform(step_lbl1, make_step_lbl(
            "After U Sigma V^T  (final rotation -- image of A)", plane1)),
        run_time=ATS)

    # Mark the two semi-axes
    ax1 = geo_arrow(U1[:, 0] * s1[0], color=C_SIG)
    ax2 = geo_arrow(U1[:, 1] * s1[1], color=C_SIG)
    ax1_lbl = MathTex(fr"\sigma_1={fv2(s1[0])}", color=C_SIG, font_size=18)
    ax2_lbl = MathTex(fr"\sigma_2={fv2(s1[1])}", color=C_SIG, font_size=18)
    ax1_lbl.move_to(to_screen(U1[:, 0] * s1[0] * 1.25))
    ax2_lbl.move_to(to_screen(U1[:, 1] * s1[1] * 1.30))

    sc.play(Create(ax1), Create(ax2), FadeIn(ax1_lbl), FadeIn(ax2_lbl), run_time=AT)
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.7)
    sc.wait(0.2)

    # ================================================================
    # SLIDE C -- Example 2 geometric walkthrough
    # ================================================================
    hdr = banner(f"Geometric Meaning -- Example 2: {ex2_title}")
    sc.play(Write(hdr))

    left_info2 = VGroup(
        Text(f"Example 2:  A = {ex2_title}", font_size=18, color=C_A),
        MathTex(fr"\sigma_1 = {fv2(s2[0])}", color=C_SIG, font_size=26),
        MathTex(fr"\sigma_2 = {fv2(s2[1])}", color=C_SIG, font_size=26),
        Text("Major axis length = sigma_1", font_size=17, color=C_INFO),
        Text("Minor axis length = sigma_2", font_size=17, color=C_INFO),
    ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
    left_info2.move_to(MAT_SHIFT)
    sc.play(FadeIn(left_info2), run_time=AT)

    plane2 = geo_plane()
    sc.play(Create(plane2), run_time=AT)

    circ2 = unit_circle_image(np.eye(2), color=C_CIRC)
    sc.play(Create(circ2), run_time=AT)

    step_lbl2 = make_step_lbl("Unit circle  (before any transform)", plane2)
    sc.play(FadeIn(step_lbl2), run_time=ATF)
    sc.wait(WS)

    d1 = unit_circle_image(Vt2, color=C_V)
    sc.play(
        Transform(circ2, d1),
        Transform(step_lbl2, make_step_lbl("After V^T  (rotation -- still a circle)", plane2)),
        run_time=ATS)
    sc.wait(WM)

    d2 = unit_circle_image(S2 @ Vt2, color=C_SIG)
    sc.play(
        Transform(circ2, d2),
        Transform(step_lbl2, make_step_lbl("After Sigma V^T  (scale -> ellipse)", plane2)),
        run_time=ATS)
    sc.wait(WM)

    d3 = unit_circle_image(U2 @ S2 @ Vt2, color=C_ELLIP)
    sc.play(
        Transform(circ2, d3),
        Transform(step_lbl2, make_step_lbl(
            "After U Sigma V^T  (final rotation -- image of A)", plane2)),
        run_time=ATS)

    bx1 = geo_arrow(U2[:, 0] * s2[0], color=C_SIG)
    bx2 = geo_arrow(U2[:, 1] * s2[1], color=C_SIG)
    bx1_lbl = MathTex(fr"\sigma_1={fv2(s2[0])}", color=C_SIG, font_size=18)
    bx2_lbl = MathTex(fr"\sigma_2={fv2(s2[1])}", color=C_SIG, font_size=18)
    bx1_lbl.move_to(to_screen(U2[:, 0] * s2[0] * 1.25))
    bx2_lbl.move_to(to_screen(U2[:, 1] * s2[1] * 1.30))

    sc.play(Create(bx1), Create(bx2), FadeIn(bx1_lbl), FadeIn(bx2_lbl), run_time=AT)
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.7)
    sc.wait(0.2)

    # ================================================================
    # SLIDE D -- Comparison and interpretation
    # ================================================================
    hdr = banner("Geometric Meaning -- Comparison")
    sc.play(Write(hdr))

    cmp_title = Text("Comparing the two examples:", font_size=22, color=C_STEP)
    cmp_title.next_to(hdr, DOWN, buff=0.45)
    sc.play(FadeIn(cmp_title))

    row1 = VGroup(
        Text(f"Ex 1  ({ex1_title}):", font_size=20, color=C_A),
        MathTex(fr"\sigma_1={fv2(s1[0])},\;\sigma_2={fv2(s1[1])}",
                color=C_SIG, font_size=22),
        Text(f"ratio = {s1[0]/s1[1]:.2f}  (condition number)",
             font_size=18, color=C_EIGVAL),
    ).arrange(RIGHT, buff=0.35)

    row2 = VGroup(
        Text(f"Ex 2  ({ex2_title}):", font_size=20, color=C_A),
        MathTex(fr"\sigma_1={fv2(s2[0])},\;\sigma_2={fv2(s2[1])}",
                color=C_SIG, font_size=22),
        Text(f"ratio = {s2[0]/s2[1]:.2f}  (condition number)",
             font_size=18, color=C_EIGVAL),
    ).arrange(RIGHT, buff=0.35)

    cmp_table = VGroup(row1, row2).arrange(DOWN, buff=0.30, aligned_edge=LEFT)
    cmp_table.next_to(cmp_title, DOWN, buff=0.38)
    sc.play(FadeIn(row1), run_time=AT); sc.wait(0.5)
    sc.play(FadeIn(row2), run_time=AT); sc.wait(WS)

    # Key takeaways
    takeaway_title = Text("Key takeaways:", font_size=21, color=C_STEP, weight=BOLD)
    takeaways = VGroup(
        Text("sigma_1 / sigma_2 = condition number  (measures distortion)",
             font_size=18, color=C_INFO),
        Text("Large ratio -> ill-conditioned matrix (nearly singular)",
             font_size=18, color=C_INFO),
        Text("sigma_i = 0 -> rank drops by 1  (A is rank-deficient)",
             font_size=18, color=C_INFO),
        Text("Best rank-k approx: keep only the k largest singular values",
             font_size=18, color=C_INFO),
    ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
    VGroup(takeaway_title, takeaways).arrange(DOWN, buff=0.18, aligned_edge=LEFT).next_to(
        cmp_table, DOWN, buff=0.45)
    t_box = highlight_box(VGroup(takeaway_title, takeaways), C_STEP)

    sc.play(Write(takeaway_title))
    for t in takeaways:
        sc.play(FadeIn(t), run_time=ATF)
        sc.wait(0.3)
    sc.play(Create(t_box))
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=0.8)
    sc.wait(0.3)


# ==============================================================
# SCENE 6 -- SUMMARY
# ==============================================================

def run_summary(sc):
    hdr = banner("Summary")
    sc.play(Write(hdr))

    steps = VGroup(
        VGroup(
            Text("Step 1:", font_size=22, color=C_STEP, weight=BOLD),
            Text("Compute A^T A,  find eigenvalues lambda_i and eigenvectors -> V",
                 font_size=19),
        ).arrange(RIGHT, buff=0.2),
        VGroup(
            Text("Step 2:", font_size=22, color=C_STEP, weight=BOLD),
            Text("sigma_i = sqrt(lambda_i)  ->  build Sigma",
                 font_size=19),
        ).arrange(RIGHT, buff=0.2),
        VGroup(
            Text("Step 3:", font_size=22, color=C_STEP, weight=BOLD),
            Text("u_i = A v_i / sigma_i  ->  build U",
                 font_size=19),
        ).arrange(RIGHT, buff=0.2),
    ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
    steps.next_to(hdr, DOWN, buff=0.50)

    sc.play(LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=0.4), run_time=AT)
    sc.wait(WS)

    final_eq = VGroup(
        MathTex(r"A", color=C_A, font_size=44),
        MathTex(r"=", font_size=44),
        MathTex(r"U", color=C_U, font_size=44),
        MathTex(r"\Sigma", color=C_SIG, font_size=44),
        MathTex(r"V^\top", color=C_V, font_size=44),
    ).arrange(RIGHT, buff=0.22)
    final_eq.next_to(steps, DOWN, buff=0.50)
    eq_box = highlight_box(final_eq, C_TITLE)
    sc.play(LaggedStart(*[Write(p) for p in final_eq], lag_ratio=0.3), run_time=1.5)
    sc.play(Create(eq_box))
    sc.wait(WS)

    geo_note = VGroup(
        Text("Geometric meaning:", font_size=20, color=C_STEP),
        MathTex(r"V^\top", color=C_V, font_size=24),
        Text("rotate ->", font_size=18),
        MathTex(r"\Sigma", color=C_SIG, font_size=24),
        Text("scale ->", font_size=18),
        MathTex(r"U", color=C_U, font_size=24),
        Text("rotate", font_size=18),
    ).arrange(RIGHT, buff=0.18)
    geo_note.next_to(final_eq, DOWN, buff=0.40)
    sc.play(FadeIn(geo_note), run_time=AT)
    sc.wait(WS)

    footer = Text("End -- MTH00051  FIT HCMUS",
                  font_size=22, color=C_TITLE).to_edge(DOWN, buff=0.4)
    sc.play(FadeIn(footer))
    sc.wait(WL)
    sc.play(FadeOut(*sc.mobjects), run_time=1.0)


# ==============================================================
# EXAMPLE DATA
# ==============================================================

A1 = np.array([[2.0, 1.0],
                [1.0, 2.0]])   # symmetric SPD -- clean eigenvalues

A2 = np.array([[1.0, 1.0],
                [0.0, 1.0]])   # shear matrix -- interesting distortion


# ==============================================================
# INDIVIDUAL SCENE CLASSES
# ==============================================================

class TitleScene(Scene):
    def construct(self): run_title(self)

class SVDTheoryScene(Scene):
    def construct(self): run_svd_theory(self)

class Example1Scene(Scene):
    def construct(self):
        run_example_decomposition(
            self, A1, ex_num=1,
            title_str="Symmetric SPD  A = [[2,1],[1,2]]",
            note_str="Symmetric matrix -- eigenvalues are real, eigenvectors are orthogonal")

class Example2Scene(Scene):
    def construct(self):
        run_example_decomposition(
            self, A2, ex_num=2,
            title_str="Shear Matrix  A = [[1,1],[0,1]]",
            note_str="Shear matrix -- singular values follow the golden ratio!")

class GeometricMeaningScene(Scene):
    def construct(self):
        run_geometric_meaning(
            self, A1, A2,
            ex1_title="Symmetric SPD",
            ex2_title="Shear Matrix")

class SummaryScene(Scene):
    def construct(self): run_summary(self)


# ==============================================================
# FULL PRESENTATION
# ==============================================================

class demo_video(Scene):
    """
    Complete video in one pass.

    Sections:
      1. Title
      2. SVD Theory  (step-by-step explanation)
      3. Example 1   (step-by-step decomposition)
      4. Example 2   (step-by-step decomposition)
      5. Geometric meaning analysis for both examples
      6. Summary
    """
    def construct(self):

        # 1. Title
        run_title(self)

        # 2. Theory
        run_svd_theory(self)

        # 3. Example 1 -- symmetric SPD
        run_example_decomposition(
            self, A1, ex_num=1,
            title_str="Symmetric SPD  A = [[2,1],[1,2]]",
            note_str="Symmetric matrix -- eigenvalues are real, eigenvectors are orthogonal")

        # 4. Example 2 -- shear
        run_example_decomposition(
            self, A2, ex_num=2,
            title_str="Shear Matrix  A = [[1,1],[0,1]]",
            note_str="Shear matrix -- singular values follow the golden ratio!")

        # 5. Geometric meaning
        run_geometric_meaning(
            self, A1, A2,
            ex1_title="Symmetric SPD",
            ex2_title="Shear Matrix")

        # 6. Summary
        run_summary(self)