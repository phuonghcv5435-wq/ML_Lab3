 # PRACTICE EXERCISE 3: K-MEANS IMAGE SEGMENTATION
**Topic:** Unsupervised Learning - Image Segmentation (Phân đoạn hình ảnh)

## 1. WHAT – Chúng ta sẽ làm gì?

Image Segmentation (Phân đoạn hình ảnh) là quá trình chia một hình ảnh thành nhiều vùng hoặc nhóm nhỏ dựa trên những đặc điểm tương đồng của các pixel.

Trong bài thực hành này, nhóm sử dụng thuật toán K-Means để phân đoạn hình ảnh dựa trên giá trị màu RGB. Mỗi pixel trong ảnh được xem như một điểm dữ liệu gồm ba giá trị màu: Red (R), Green (G) và Blue (B).

Các pixel có màu sắc gần giống nhau sẽ được gom vào cùng một cụm. Sau khi phân cụm, mỗi pixel được thay thế bằng màu đại diện của cụm đó để tạo ra hình ảnh mới.

**Mục tiêu:**
- Tìm hiểu nguyên lý hoạt động của thuật toán K-Means.
- Đọc và xử lý hình ảnh bằng Python.
- Chuyển các pixel thành vector màu RGB.
- Áp dụng K-Means để phân nhóm các pixel.
- Tạo hình ảnh sau khi phân đoạn.
- So sánh hình ảnh gốc với hình ảnh đã phân đoạn.

**Input:**
- Hình ảnh đầu vào: OIP.webp.
- Dữ liệu màu RGB của từng pixel trong hình ảnh.

**Output:**
- Hình ảnh sau khi phân đoạn bằng K-Means.
- Hình ảnh so sánh giữa ảnh gốc và ảnh sau phân đoạn.
- Các pixel được thay thế bằng màu đại diện của từng cụm.

**Phạm vi bài toán:**

Bài thực hành tập trung vào phân đoạn hình ảnh dựa trên màu sắc của pixel bằng thuật toán K-Means. Chương trình sử dụng không gian màu RGB và chia hình ảnh thành 4 cụm màu.

## 2. WHY – Tại sao thực hiện bài toán này?

Một hình ảnh thường chứa rất nhiều pixel với các màu sắc khác nhau. Việc xử lý từng pixel riêng lẻ có thể làm cho dữ liệu trở nên phức tạp.

Phân đoạn hình ảnh giúp:
- Gom các pixel có màu sắc gần giống nhau vào cùng một nhóm.
- Giảm số lượng màu sắc trong hình ảnh.
- Giúp hình ảnh đơn giản hơn và dễ quan sát hơn.
- Tìm hiểu cách áp dụng thuật toán phân cụm vào xử lý hình ảnh.

**Lý do sử dụng Machine Learning**

Machine Learning có thể hỗ trợ máy tính tìm ra những nhóm dữ liệu có đặc điểm tương đồng mà không cần con người phân loại từng pixel trước.

**Tại sao lựa chọn Unsupervised Learning?**

Bài toán không có sẵn nhãn cho từng pixel trong hình ảnh. Vì vậy, nhóm sử dụng phương pháp Unsupervised Learning để thuật toán tự tìm kiếm các nhóm màu dựa trên dữ liệu RGB.

K-Means phù hợp với bài thực hành vì thuật toán có thể chia dữ liệu thành số cụm được xác định trước.

## 3. WHEN – Khi nào thực hiện?

Bài toán được thực hiện khi cần xử lý và phân chia hình ảnh thành các nhóm màu khác nhau.

Trong phạm vi bài thực hành Lab 3, nhóm sử dụng một hình ảnh có sẵn để tìm hiểu cách hoạt động của K-Means và quan sát sự thay đổi của hình ảnh sau khi phân cụm.

## 4. WHERE – Thực hiện ở đâu?

Bài thực hành được triển khai bằng ngôn ngữ Python trong môi trường lập trình trên máy tính.

Hình ảnh đầu vào được lưu trong thư mục `data/raw/`. Các file code xử lý được đặt trong thư mục `src/`.

| Thành phần | Công cụ sử dụng |
|---|---|
| Ngôn ngữ lập trình | Python |
| Xử lý dữ liệu | NumPy |
| Đọc và xử lý hình ảnh | OpenCV |
| Hỗ trợ đọc ảnh | Pillow |
| Hiển thị hình ảnh | Matplotlib |
| Kiểm thử chương trình | Pytest |

## 5. WHICH – Sử dụng dữ liệu và thuật toán nào?

### 5.1. Dữ liệu đầu vào

Bài thực hành sử dụng hình ảnh `OIP.webp` được lưu trong thư mục `data/raw/`.

Thông tin xử lý hình ảnh:

| Thành phần | Mô tả |
|---|---|
| Ảnh đầu vào | OIP.webp |
| Không gian màu | RGB |
| Số kênh màu | 3 kênh R, G, B |
| Kích thước cạnh dài tối đa | 200 pixel |
| Dữ liệu sau xử lý | Vector màu RGB của từng pixel |

### 5.2. Thuật toán K-Means

Nhóm sử dụng thuật toán K-Means để phân nhóm các pixel dựa trên giá trị màu RGB.

Trong bài thực hành, thuật toán được tự cài đặt trong file `src/kmeans.py` bằng thư viện NumPy.

Một số thông số của thuật toán:

| Thông số | Giá trị |
|---|---|
| Số cụm K | 4 |
| Số vòng lặp tối đa | 100 |
| Sai số hội tụ (Tolerance) | 0.0001 |
| Seed khởi tạo | 42 |
| Khoảng cách | Euclidean Distance |

K-Means thực hiện việc chọn các tâm cụm ban đầu, tính khoảng cách từ pixel đến các tâm cụm, gán pixel vào cụm gần nhất và cập nhật lại tâm cụm. Quá trình tiếp tục cho đến khi đạt điều kiện hội tụ hoặc hết số vòng lặp.

## 6. HOW – Chúng ta thực hiện như thế nào?

Quy trình thực hiện bài toán gồm các bước sau:

**1. Image Loading – Đọc hình ảnh**

- Đọc hình ảnh `OIP.webp` từ thư mục `data/raw/`.
- Sử dụng OpenCV để đọc ảnh.
- Nếu OpenCV không đọc được ảnh, chương trình sử dụng Pillow để hỗ trợ.
- Chuyển hình ảnh sang không gian màu RGB.

**2. Data Preprocessing – Tiền xử lý dữ liệu**

- Kiểm tra hình ảnh đầu vào.
- Thu nhỏ hình ảnh với kích thước cạnh dài tối đa là 200 pixel để giảm thời gian xử lý.
- Chuyển hình ảnh từ dạng mảng 3 chiều thành mảng 2 chiều.
- Mỗi pixel được biểu diễn bằng một vector gồm ba giá trị màu RGB.

**3. Model Building – Xây dựng mô hình**

- Sử dụng thuật toán K-Means được cài đặt trong file `src/kmeans.py`.
- Khởi tạo 4 tâm cụm ban đầu.
- Tính khoảng cách giữa các pixel và tâm cụm.
- Gán từng pixel vào cụm gần nhất.
- Cập nhật lại tâm cụm dựa trên giá trị trung bình của các pixel trong cụm.
- Lặp lại quá trình cho đến khi đạt điều kiện hội tụ hoặc đủ số vòng lặp.

**4. Image Segmentation – Phân đoạn hình ảnh**

- Lấy màu đại diện của từng cụm.
- Thay thế màu của mỗi pixel bằng màu của tâm cụm tương ứng.
- Ghép các pixel lại thành hình ảnh mới có kích thước ban đầu sau khi thu nhỏ.

**5. Visualization – Trực quan hóa kết quả**

- Hiển thị hình ảnh gốc.
- Hiển thị hình ảnh sau khi phân đoạn.
- Đặt hai hình ảnh cạnh nhau để dễ dàng so sánh.
- Lưu hình ảnh so sánh vào thư mục `figures/`.

**6. Save Results – Lưu kết quả**

Sau khi chương trình chạy xong, kết quả được lưu vào các thư mục:

- `results/segmented_k4.png`: Hình ảnh sau khi phân đoạn.
- `figures/segmentation_k4.png`: Hình ảnh so sánh giữa ảnh gốc và ảnh sau phân đoạn.

## 7. SUMMARY – Tổng kết bài toán

| Nội dung | Mô tả |
|---|---|
| Problem | Image Segmentation |
| Machine Learning Type | Unsupervised Learning |
| Task | Clustering |
| Input | Hình ảnh OIP.webp |
| Color Space | RGB |
| Main Algorithm | K-Means |
| Number of Clusters | K = 4 |
| Main Libraries | NumPy, OpenCV, Matplotlib, Pillow |
| Output | Hình ảnh sau khi phân đoạn |
| Visualization | So sánh ảnh gốc và ảnh kết quả |

Thông qua bài thực hành, nhóm tìm hiểu được cách áp dụng thuật toán K-Means vào bài toán phân đoạn hình ảnh. Nhóm thực hiện các bước từ đọc ảnh, tiền xử lý, chuyển pixel thành vector RGB, phân cụm đến tạo và hiển thị hình ảnh kết quả.

Qua đó, nhóm hiểu rõ hơn về cách hoạt động của thuật toán K-Means và cách áp dụng phương pháp học không giám sát vào xử lý hình ảnh.
