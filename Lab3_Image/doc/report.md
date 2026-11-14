# BÁO CÁO THỰC HÀNH 3
## PHÂN ĐOẠN ẢNH BẰNG THUẬT TOÁN K-MEANS

## 1. Mục tiêu bài thực hành

Bài thực hành này sử dụng thuật toán K-Means để phân đoạn một hình ảnh thành K nhóm màu. Mỗi pixel được xem như một điểm dữ liệu, có các đặc trưng là giá trị màu RGB. Thuật toán sẽ gom những pixel có màu gần nhau vào cùng một cụm, sau đó thay màu của từng pixel bằng màu đại diện của cụm đó.

Qua bài này, em tìm hiểu cách đọc ảnh bằng Python, tiền xử lý dữ liệu ảnh, cài đặt thuật toán K-Means và quan sát sự thay đổi của ảnh sau khi phân cụm.

## 2. Công cụ và thư viện sử dụng

- **Python:** ngôn ngữ lập trình chính.
- **OpenCV (cv2):** đọc ảnh, thay đổi kích thước và chuyển đổi không gian màu.
- **Pillow (PIL):** hỗ trợ đọc ảnh khi OpenCV không đọc được.
- **NumPy:** biểu diễn ảnh dưới dạng mảng và thực hiện các phép tính.
- **Matplotlib:** hiển thị và lưu ảnh so sánh.
- **pytest:** chạy các kiểm thử cho thuật toán và kết quả phân đoạn.

## 3. Dữ liệu đầu vào

Ảnh đầu vào được đặt tại `data/raw/OIP.webp`. Đây là ảnh phong cảnh mùa đông có ngôi nhà, cây thông và người tuyết. Chương trình đọc ảnh ở không gian màu RGB.

Trong file `run_pipeline.py`, các thông số đang sử dụng là:

| Thông số | Giá trị |
|---|---|
| Đường dẫn ảnh | `data/raw/OIP.webp` |
| Không gian màu | RGB |
| Số cụm K | 4 |
| Kích thước cạnh dài tối đa | 200 pixel |
| Số vòng lặp tối đa | 100 |
| Sai số hội tụ | `1e-4` |
| Seed khởi tạo | 42 |

## 4. Các bước thực hiện

### Bước 1. Đọc ảnh

Hàm `load_image()` trong `src/image_loader.py` dùng OpenCV để đọc ảnh. Nếu OpenCV không đọc được thì chương trình thử dùng PIL. Sau đó ảnh được chuyển sang không gian màu đã chọn (RGB, HSV hoặc LAB). Trong lần chạy này, chương trình sử dụng RGB.

### Bước 2. Tiền xử lý ảnh

Trong `src/preprocessing.py`, hàm `resize_image()` thu nhỏ ảnh nhưng vẫn giữ nguyên tỉ lệ. Cạnh dài nhất được giới hạn ở 200 pixel để giảm số lượng điểm dữ liệu và giúp K-Means chạy nhanh hơn.

Tiếp theo, hàm `flatten()` chuyển ảnh từ dạng `(chiều cao, chiều rộng, số kênh màu)` thành một mảng hai chiều. Mỗi hàng của mảng tương ứng với một pixel, gồm ba giá trị màu R, G và B.

Sau khi thu nhỏ, ảnh có kích thước **200 × 113 pixel**, tương đương **22.600 pixel** được đưa vào phân cụm.

### Bước 3. Phân cụm bằng K-Means

Thuật toán được tự cài đặt trong `src/kmeans.py`, không gọi trực tiếp K-Means từ thư viện scikit-learn. Các bước chính:

1. Chọn ngẫu nhiên K pixel làm tâm cụm ban đầu (centroid).
2. Tính khoảng cách Euclid bình phương từ từng pixel đến các tâm cụm.
3. Gán pixel vào cụm có tâm gần nhất.
4. Tính lại tâm cụm bằng giá trị trung bình của các pixel thuộc cụm đó.
5. Lặp lại việc gán cụm và cập nhật tâm cho đến khi tâm cụm thay đổi rất ít hoặc đạt số vòng lặp tối đa.

Nếu có cụm không chứa pixel nào, chương trình chọn lại một pixel ngẫu nhiên làm tâm cụm để tiếp tục chạy. Trong bài này, K được đặt bằng 4 nên ảnh được gom thành tối đa bốn nhóm màu đại diện.

### Bước 4. Tạo ảnh sau phân đoạn

Hàm `segment()` trong `src/segmentation.py` lấy màu tâm cụm thay cho màu ban đầu của từng pixel. Sau đó mảng pixel được đưa trở lại kích thước ảnh ban đầu sau tiền xử lý.

### Bước 5. Hiển thị và lưu kết quả

Hàm `show()` trong `src/visualization.py` đặt ảnh gốc và ảnh sau phân đoạn cạnh nhau để dễ so sánh. Chương trình lưu ảnh so sánh tại `figures/segmentation_k4.png` và ảnh kết quả tại `results/segmented_k4.png`.

## 5. Kết quả thực nghiệm

Ảnh sau khi phân đoạn vẫn giữ được những vùng chính như mái nhà, cây thông, người tuyết và nền tuyết. Tuy nhiên, số lượng màu đã giảm đáng kể, các vùng có màu gần nhau được thể hiện bằng cùng một màu đại diện. Vì vậy ảnh kết quả trông đơn giản và có dạng các mảng màu rõ hơn so với ảnh ban đầu.

Với K = 4, chương trình sử dụng tối đa bốn màu đại diện cho các cụm. Việc giảm số màu giúp thể hiện được ý tưởng phân đoạn ảnh, nhưng một số chi tiết nhỏ và chuyển màu nhẹ trong ảnh gốc không còn rõ như ban đầu.

**Hình 1.** So sánh ảnh gốc và ảnh sau phân đoạn: `figures/segmentation_k4.png`.

**Hình 2.** Ảnh kết quả sau phân đoạn: `results/segmented_k4.png`.

## 6. Đánh giá và kiểm thử

Ngoài việc quan sát ảnh, dự án có sử dụng phương pháp Elbow để tham khảo số cụm K. Hàm `inertia()` tính tổng bình phương khoảng cách từ mỗi pixel đến tâm cụm tương ứng (Inertia/WCSS). Hàm `elbow()` chạy K-Means với nhiều giá trị K và vẽ biểu đồ Inertia theo K.

Quan sát biểu đồ `figures/elbow.png`, Inertia giảm khá nhanh khi tăng K từ 2 lên khoảng 5, sau đó mức giảm chậm dần. Điều này cho thấy vùng K khoảng 4–5 có thể được cân nhắc. Trong pipeline chính, em chọn K = 4 để tạo ảnh phân đoạn và thuận tiện quan sát kết quả. Elbow chỉ là cách tham khảo, không phải lúc nào cũng xác định được một giá trị K duy nhất.

Dự án cũng có các kiểm thử trong `tests/test_kmeans.py`, gồm kiểm tra kích thước đầu ra, khả năng tách hai nhóm màu rõ ràng, khôi phục đúng kích thước ảnh và xu hướng Inertia giảm khi K tăng. Khi chạy `pytest -q`, kết quả là **4 kiểm thử đều đạt**.

## 7. Cách chạy chương trình

Mở terminal tại thư mục gốc của dự án, cài các thư viện cần thiết rồi chạy:

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Để chạy phần đánh giá Elbow:

```bash
python -m src.eval
```

Để chạy kiểm thử:

```bash
pytest -q
```

## 8. Kết luận

Qua bài thực hành, em đã thực hiện được các bước cơ bản của bài toán phân đoạn ảnh bằng K-Means: đọc ảnh, thu nhỏ ảnh, chuyển mỗi pixel thành vector màu, phân cụm và tạo lại ảnh từ các tâm cụm. Kết quả với K = 4 cho thấy ảnh được rút gọn còn các mảng màu đại diện nhưng vẫn nhận ra được những đối tượng chính.

Bài làm giúp em hiểu rõ hơn cách K-Means hoạt động trên dữ liệu ảnh. Nếu phát triển thêm, có thể thử các giá trị K khác nhau, thay đổi không gian màu RGB sang HSV hoặc LAB và so sánh kết quả để tìm cách phân đoạn phù hợp hơn.

