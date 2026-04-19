# Đồ Án 1: Ma Trận và Cơ Sở Tính Toán Khoa Học

Đồ án này tập trung vào việc cài đặt các thuật toán Đại số tuyến tính cốt lõi hoàn toàn từ đầu (**from scratch**) bằng Python, đồng thời phân tích, đánh giá hiệu năng (thời gian thực thi, tính ổn định số học) của chúng so với các thư viện chuyên dụng như NumPy/SciPy. Ngoài ra, đồ án còn ứng dụng thư viện Manim để trực quan hóa hình học các phép biến đổi ma trận.

## Cấu trúc thư mục (Project Structure)

Dự án được chia thành 3 phần chính tương ứng với yêu cầu của đồ án:

```text
├── part1/                  # PHẦN 1: Các phép toán cơ bản của ma trận
│   ├── determinant.py      # Tính định thức (Determinant) qua khử Gauss
│   ├── gaussian.py         # Thuật toán khử Gauss (Partial Pivoting)
│   ├── inverse.py          # Tính ma trận nghịch đảo
│   ├── rank_basis.py       # Tìm hạng (Rank) và cơ sở không gian nghiệm
│   └── part1_demo.ipynb    # Notebook demo các chức năng của Phần 1
│
├── part2/                  # PHẦN 2: Phân rã ma trận và Trực quan hóa
│   ├── decomposition.py    # Thuật toán phân rã SVD (Singular Value Decomposition)
│   ├── diagonalization.py  # Thuật toán chéo hóa ma trận đối xứng (QR Iteration)
│   ├── manim_scene.py      # Kịch bản Manim trực quan hóa hình học của SVD
│   └── demo_video.mp4      # Video kết quả xuất ra từ Manim
│
├── part3/                  # PHẦN 3: Giải hệ phương trình và Phân tích hiệu năng
│   ├── solvers.py          # Cài đặt 3 phương pháp giải: Gauss, SVD, Gauss-Seidel
│   ├── benchmark.py        # Script chạy thực nghiệm đo thời gian & sai số
│   ├── analysis.ipynb      # Báo cáo thực nghiệm (Đồ thị Log-Log & Nhận xét)
│   └── ...                 # Các file phụ trợ copy từ phần trước
│
├── requirements.txt        # Danh sách các thư viện cần thiết
└── README.md               # File giới thiệu dự án này
```

## Hướng dẫn Cài đặt (Installation)

Đảm bảo máy tính của bạn đã cài đặt Python (phiên bản >= 3.8). Khuyến nghị sử dụng môi trường ảo (Virtual Environment).

**Bước 1: Clone repository về máy**
```bash
git clone [https://github.com/hieu-tran-coding/AppliedMath-Project1.git](https://github.com/hieu-tran-coding/AppliedMath-Project1.git)
cd AppliedMath-Project1
```

**Bước 2: Cài đặt các thư viện Python**
```bash
pip install -r requirements.txt
```

*Lưu ý cho Phần 2 (Manim):* Để render được video toán học, thư viện `manim` yêu cầu hệ thống của bạn phải có sẵn **FFmpeg** (và LaTeX nếu bạn render công thức phức tạp). Hãy tham khảo [tài liệu cài đặt chính thức của Manim](https://docs.manim.community/en/stable/installation.html) để biết thêm chi tiết.

## Hướng dẫn Sử dụng (Usage)

### Phần 1 & Phần 3: Chạy Demo và Đọc Báo cáo
Các phần này đã được viết sẵn dưới dạng Jupyter Notebook để dễ dàng trình bày code, đồ thị và lời giải thích.
- Mở Terminal/Command Prompt.
- Chạy lệnh: `jupyter notebook`
- Trên trình duyệt, mở file `part1/part1_demo.ipynb` (Demo thuật toán cơ bản) hoặc `part3/analysis.ipynb` (Báo cáo đánh giá hiệu năng).

*Hoặc chạy trực tiếp script đo lường:*
```bash
python part3/benchmark.py
```

### Phần 2: Render Video Trực Quan Hóa (SVD)
Để tạo ra video trực quan hóa phép biến đổi hình học **Rotate - Scale - Rotate** của SVD, bạn di chuyển vào thư mục `part2` và chạy lệnh Manim sau:
```bash
cd part2
manim -pql manim_scene.py FullPresentation
```
- Cờ `-pql` dùng để render nhanh ở chất lượng thấp (Low Quality) nhằm mục đích preview. 
- Để render chất lượng cao (1080p 60fps), hãy đổi thành `-pqh`.

## Thành viên nhóm (Contributors)

* [Vũ Hoàng Long] - [23120058]
* [Bạch Trương Tấn Phát] - [23120316]
* [Bạch Trương Tấn Phát] - [23120316]
* [Bạch Trương Tấn Phát] - [23120316]
* [Bạch Trương Tấn Phát] - [23120316]

---
*Đồ án thuộc môn học Cơ sở Tính toán Khoa học / Toán Ứng Dụng.*
```
