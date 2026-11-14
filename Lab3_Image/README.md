# LAB 3 – K-Means Image Segmentation

## 1. Giới thiệu

Dự án sử dụng Python để phân đoạn hình ảnh bằng thuật toán K-Means. Mỗi pixel được biểu diễn bằng một vector gồm ba giá trị màu trong không gian RGB. Các pixel có màu sắc gần nhau được gom vào cùng một cụm.

## 2. Mục tiêu

- Tìm hiểu cách hoạt động của thuật toán K-Means.
- Đọc và tiền xử lý hình ảnh bằng Python.
- Chuyển các pixel thành vector màu RGB.
- Phân cụm pixel và tạo hình ảnh sau phân đoạn.
- So sánh hình ảnh gốc với hình ảnh đã phân đoạn.

## 3. Công nghệ sử dụng

- Python
- NumPy
- OpenCV
- Matplotlib
- Pillow (hỗ trợ đọc ảnh khi cần)

## 4. Cấu trúc thư mục

```text
Lab3_Image/
├── data/
│   └── raw/
│       └── OIP.webp
├── doc/
│   ├── description.md
│   └── report.md
├── figures/
│   └── segmentation_k4.png
├── results/
│   └── segmented_k4.png
├── src/
│   ├── image_loader.py
│   ├── src/eval.py
│   ├── preprocessing.py
│   ├── kmeans.py
│   ├── segmentation.py
│   └── visualization.py
├── tests/
│   └── test_kmeans.py
├── run_pipeline.py
├── Demo.py
├── requirements.txt
└── README.md
```

## 5. Thông số thực hiện
Đầu tiên, chương trình đọc hình ảnh OIP.webp từ thư mục data/raw. Sau đó, ảnh được chuyển sang không gian màu RGB và thu nhỏ để giảm thời gian xử lý.

Tiếp theo, mỗi pixel được chuyển thành một vector gồm 3 giá trị màu R, G, B. Chương trình sử dụng thuật toán K-Means trong file src/kmeans.py để chia các pixel thành 4 cụm.

Sau khi phân cụm, mỗi pixel sẽ được thay bằng màu đại diện của cụm tương ứng. Cuối cùng, các pixel được ghép lại thành hình ảnh mới

- Ảnh đầu vào: `data/raw/OIP.webp`
- Không gian màu: RGB
- Số cụm: K = 4
- Kích thước cạnh dài tối đa sau khi thu nhỏ: 200 pixel
- Số vòng lặp tối đa: 100
- Sai số hội tụ: 0.0001
- Seed khởi tạo: 42

Trong chương trình, thuật toán K-Means được cài đặt trong file `src/kmeans.py`.

## 6. Cài đặt thư viện

Mở Terminal tại thư mục `Lab3_Image`, sau đó chạy:

```bash
pip install -r requirements.txt
```

## 7. Cách chạy chương trình

Đảm bảo ảnh `OIP.webp` nằm trong thư mục `data/raw/`. Tại Terminal, chạy lệnh:

```bash
python run_pipeline.py
```

## 8. Kết quả

Sau khi chạy thành công, chương trình tạo ra:

- `results/segmented_k4.png`: Hình ảnh sau khi phân đoạn.
- `figures/segmentation_k4.png`: Hình so sánh ảnh gốc và ảnh sau phân đoạn.

Nếu không tìm thấy ảnh đầu vào, hãy kiểm tra lại tên file và đường dẫn `data/raw/OIP.webp`.

## 9. Kết luận

Sau khi thực hiện bài tập, em hiểu được cách sử dụng thuật toán K-Means để phân đoạn hình ảnh. Việc chuyển các pixel thành vector RGB giúp chương trình có thể phân nhóm các pixel dựa trên màu sắc. Kết quả sau khi phân đoạn có ít màu hơn ảnh ban đầu và các vùng màu được thể hiện rõ hơn.
