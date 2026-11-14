# MÔ TẢ DỰ ÁN
## PHÂN ĐOẠN ẢNH BẰNG THUẬT TOÁN K-MEANS

### 1. Giới thiệu dự án

Dự án sử dụng ngôn ngữ lập trình Python để thực hiện phân đoạn hình ảnh bằng thuật toán K-Means. Chương trình xem mỗi pixel trong ảnh là một điểm dữ liệu với ba giá trị màu RGB. Các pixel có màu sắc gần giống nhau sẽ được gom vào cùng một nhóm.

### 2. Mục tiêu dự án

- Tìm hiểu nguyên lý hoạt động của thuật toán K-Means.
- Áp dụng thuật toán vào xử lý hình ảnh.
- Phân nhóm pixel dựa trên giá trị màu RGB.
- Giảm số lượng màu sắc trong ảnh.
- So sánh hình ảnh trước và sau khi phân đoạn.

### 3. Công cụ sử dụng

- Python: Ngôn ngữ lập trình chính.
- NumPy: Xử lý dữ liệu ảnh dạng mảng.
- OpenCV: Đọc và xử lý hình ảnh.
- Matplotlib: Hiển thị và lưu kết quả.
- Pillow: Hỗ trợ đọc hình ảnh.
- Pytest: Kiểm tra chương trình.

### 4. Chức năng chính

- Đọc ảnh đầu vào từ thư mục dữ liệu.
- Thay đổi kích thước ảnh.
- Chuyển ảnh thành các vector màu RGB.
- Phân nhóm pixel bằng thuật toán K-Means.
- Tạo ảnh mới sau khi phân đoạn.
- Hiển thị và so sánh ảnh gốc với ảnh kết quả.
- Tạo biểu đồ Elbow để tham khảo số cụm phù hợp.

### 5. Thông số thực hiện

- Không gian màu: RGB.
- Số cụm K: 4.
- Kích thước cạnh dài tối đa: 200 pixel.
- Số vòng lặp tối đa: 100.
- Sai số hội tụ: 0.0001.
- Seed khởi tạo: 42.

### 6. Kết quả đạt được

Sau khi chạy chương trình, hình ảnh được chia thành tối đa 4 nhóm màu đại diện. Các pixel có màu sắc tương tự nhau được gom vào cùng một nhóm. Kết quả giúp hình ảnh đơn giản hơn nhưng vẫn giữ được những vùng và đối tượng chính.

### 7. Kết luận

Qua dự án, em hiểu thêm về cách hoạt động của thuật toán K-Means và cách áp dụng thuật toán vào bài toán phân đoạn ảnh bằng Python. Đồng thời, em cũng thực hành được các bước đọc ảnh, xử lý dữ liệu, phân cụm pixel và hiển thị kết quả.
