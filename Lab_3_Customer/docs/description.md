# PRACTICE EXERCISE 3: CUSTOMER SEGMENTATION
**Topic:** Unsupervised Learning - Customer Segmentation (Phân khúc khách hàng)

## 1. WHAT – Chúng ta sẽ làm gì?
Customer Segmentation (Phân khúc khách hàng): chia tập khách hàng thành nhiều nhóm nhỏ dựa trên những đặc điểm hoặc hành vi tương đồng.

Trong thực tế, mỗi khách hàng có những đặc điểm khác nhau về độ tuổi, thu nhập và mức độ chi tiêu. Nếu doanh nghiệp chỉ xem xét toàn bộ khách hàng như một nhóm duy nhất thì sẽ khó nhận biết sự khác biệt giữa các nhóm khách hàng cũng như xây dựng những chiến lược kinh doanh phù hợp.

Vì vậy, bài thực hành này tập trung xây dựng mô hình Machine Learning nhằm phân cụm khách hàng dựa trên các đặc điểm có trong bộ dữ liệu Mall Customers Dataset.

Bài toán sử dụng phương pháp Unsupervised Learning (Học không giám sát), trong đó dữ liệu ban đầu không có nhãn phân loại. Mô hình sẽ tự tìm kiếm những nhóm khách hàng có đặc điểm tương đồng dựa trên dữ liệu đầu vào.

**Mục tiêu:**
- Phân tích và khám phá các đặc điểm của tập dữ liệu khách hàng.
- Xác định những thuộc tính phù hợp để thực hiện phân cụm.
- Áp dụng các thuật toán Clustering để chia khách hàng thành những nhóm khác nhau.
- Đánh giá chất lượng phân cụm bằng các phương pháp phù hợp.
- Trực quan hóa kết quả để quan sát và phân tích đặc điểm của từng nhóm khách hàng.

**Input:** Thông tin của từng khách hàng, bao gồm:
- Gender: Giới tính.
- Age: Độ tuổi.
- Annual Income (k$): Thu nhập hằng năm.
- Spending Score (1-100): Điểm chi tiêu.
**Output:** 
- Nhãn cụm (Cluster Label) được gán cho từng khách hàng.
- Số lượng khách hàng thuộc mỗi cụm.
- Đặc điểm nổi bật của từng nhóm khách hàng.
- Biểu đồ trực quan thể hiện sự phân bố các cụm.

**Phạm vi bài toán:**
Bài thực hành tập trung vào việc khám phá cấu trúc dữ liệu và phân nhóm khách hàng dựa trên các thuộc tính sẵn có.

Bài toán không nhằm dự đoán doanh thu hoặc xác định trước một loại khách hàng cụ thể. Thay vào đó, mô hình sẽ tự phát hiện các nhóm khách hàng dựa trên mức độ tương đồng giữa các đặc trưng.

## 2. WHY – Tại sao thực hiện bài toán này?
Trong hoạt động kinh doanh, mỗi khách hàng có độ tuổi, thu nhập và thói quen chi tiêu khác nhau. Nếu chỉ xem xét toàn bộ khách hàng như một nhóm chung, doanh nghiệp sẽ khó nhận biết sự khác biệt giữa các nhóm đối tượng.

Vì vậy, phân cụm khách hàng giúp:
- Nhận diện các nhóm khách hàng có đặc điểm tương đồng.
- Hiểu rõ hơn về hành vi và mức độ chi tiêu của khách hàng.
- Hỗ trợ doanh nghiệp xây dựng chiến lược marketing phù hợp cho từng nhóm.
- Hỗ trợ đưa ra các quyết định kinh doanh dựa trên dữ liệu.

**Lý do sử dụng Machine Learning**
Khi số lượng khách hàng lớn, việc phân nhóm hoàn toàn bằng phương pháp thủ công sẽ tốn nhiều thời gian và khó đảm bảo tính nhất quán.

Machine Learning giúp tự động phân tích dữ liệu, tìm kiếm những điểm tương đồng và phát hiện cấu trúc nhóm mà con người có thể khó nhận ra khi quan sát dữ liệu ban đầu.

**Tại sao lựa chọn Unsupervised Learning**
Bộ dữ liệu Mall Customers không cung cấp sẵn nhãn phân loại khách hàng.
Do đó, không thể trực tiếp sử dụng các thuật toán Classification theo cách thông thường vì chưa có biến mục tiêu (Target Label) để mô hình học.

Unsupervised Learning phù hợp với bài toán này vì:
- Không yêu cầu dữ liệu phải có nhãn từ trước.
- Có khả năng khám phá cấu trúc tiềm ẩn trong dữ liệu.
- Có thể tự động nhóm những đối tượng có đặc điểm tương đồng.
- Phù hợp với mục tiêu khám phá và phân khúc khách hàng.

## 3. WHEN – Khi nào thực hiện?

Bài toán được thực hiện trong giai đoạn phân tích dữ liệu khách hàng, khi doanh nghiệp đã có dữ liệu thu thập từ hoạt động kinh doanh nhưng chưa có nhãn phân loại khách hàng cụ thể.

Trong phạm vi bài thực hành, quá trình phân cụm được thực hiện trên dataset có sẵn, nhằm khám phá cấu trúc dữ liệu và tìm ra các nhóm khách hàng tiềm năng.

## 4. WHERE – Thực hiện ở đâu?

Bài toán được triển khai trên môi trường Python bằng Visual Studio Code và Jupyter Notebook.

Nguồn dữ liệu sử dụng là Mall Customers Dataset, chứa thông tin về khách hàng của một trung tâm thương mại.

| Thành phần | Công cụ sử dụng |
|---|---|
| Ngôn ngữ lập trình | Python |
| Môi trường phát triển | Visual Studio Code |
| Môi trường thực thi | Jupyter Notebook |
| Xử lý dữ liệu | Pandas, NumPy |
| Trực quan hóa dữ liệu | Matplotlib, Seaborn |
| Xây dựng mô hình | Scikit-learn |

## 5. WHICH – Sử dụng dữ liệu và thuật toán nào?

### 5.1. Dataset

Bài thực hành sử dụng bộ dữ liệu Mall Customers Dataset.

Đây là bộ dữ liệu thường được sử dụng trong các bài toán phân khúc khách hàng, chứa thông tin cơ bản về khách hàng tại một trung tâm thương mại.

Bộ dữ liệu cung cấp những đặc điểm có thể sử dụng để phân tích sự khác biệt giữa các khách hàng, đặc biệt là thu nhập và mức độ chi tiêu.

| Thuộc tính | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| CustomerID | Integer | Mã định danh khách hàng |
| Gender | Categorical | Giới tính khách hàng |
| Age | Integer | Độ tuổi khách hàng |
| Annual Income (k$) | Integer | Thu nhập hằng năm, đơn vị nghìn USD |
| Spending Score (1-100) | Integer | Điểm chi tiêu từ 1 đến 100 |

Trong đó, `CustomerID` chỉ dùng để định danh, không đưa vào tính toán khoảng cách khi phân cụm.

### 5.2. Thuật toán

Dự kiến thử nghiệm và so sánh ba thuật toán:

| Thuật toán | Đặc điểm |
|---|---|
| K-Means | Phân cụm dựa trên tâm cụm |
| DBSCAN | Phân cụm dựa trên mật độ dữ liệu |
| Hierarchical Clustering | Phân cụm theo cấu trúc phân cấp |

Trong bài thực hành, nhóm sẽ sử dụng K-Means làm thuật toán chính để tìm hiểu quy trình phân cụm và xác định số lượng cụm phù hợp.

Bên cạnh đó, DBSCAN và Hierarchical Clustering được xem xét để so sánh cách tiếp cận cũng như đặc điểm kết quả.

Nhóm không mặc định thuật toán nào tốt nhất. Việc lựa chọn mô hình phù hợp sẽ dựa trên kết quả thực nghiệm, khả năng phân tách các cụm và các chỉ số đánh giá.

## 6. HOW – Chúng ta thực hiện như thế nào?

Quy trình thực hiện bài toán gồm các giai đoạn:

1. **Data Collection:** Thu thập và đọc Mall Customers Dataset.
- Sử dụng bộ dữ liệu Mall Customers Dataset.
- Đọc dữ liệu vào môi trường Python bằng thư viện Pandas.
- Kiểm tra số lượng dòng, số lượng cột và cấu trúc dữ liệu.
- Xác định ý nghĩa của từng thuộc tính.

2. **EDA:** Khám phá dữ liệu
- Kiểm tra kiểu dữ liệu của từng thuộc tính.
- Kiểm tra giá trị thiếu (Missing Values).
- Kiểm tra dữ liệu trùng lặp (Duplicate Values).
- Thống kê các giá trị cơ bản như trung bình, độ lệch chuẩn, giá trị nhỏ nhất và lớn nhất.
- Quan sát phân phối dữ liệu.
- Phân tích mối quan hệ giữa các thuộc tính.
- Phát hiện những giá trị bất thường nếu có.

Mục đích của giai đoạn này là xác định chất lượng dữ liệu và hỗ trợ lựa chọn các bước tiền xử lý phù hợp.

3. **Data Preprocessing:** Tiền xử lý dữ liệu
- Xử lý dữ liệu thiếu: Kiểm tra và lựa chọn phương pháp xử lý phù hợp nếu dữ liệu xuất hiện giá trị thiếu.
-  Xử lý dữ liệu trùng lặp: Kiểm tra các bản ghi trùng lặp và loại bỏ nếu cần thiết.
- Lựa chọn đặc trưng: Loại bỏ những thuộc tính không mang ý nghĩa phân cụm như CustomerID.
- Mã hóa dữ liệu phân loại: Nếu sử dụng Gender, cần chuyển đổi dữ liệu dạng chữ thành dạng số bằng phương pháp phù hợp.
- huẩn hóa dữ liệu: Các thuộc tính có thể có thang đo và độ lớn khác nhau. Vì vậy, cần xem xét sử dụng những phương pháp như StandardScaler hoặc MinMaxScaler. Việc chuẩn hóa giúp tránh trường hợp một thuộc tính có giá trị lớn chi phối quá trình tính khoảng cách.

4. **Model Building:** Xây dựng và thực nghiệm các thuật toán K-Means, DBSCAN và Hierarchical Clustering.
5. **Model Evaluation:** Sử dụng Elbow Method, Silhouette Score và trực quan hóa để đánh giá, so sánh kết quả.
6. **Visualization:** Trực quan hóa kết quả
Sử dụng Matplotlib và Seaborn để trực quan hóa dữ liệu và kết quả phân cụm.

Các biểu đồ dự kiến:
- Biểu đồ phân phối độ tuổi khách hàng.
- Biểu đồ phân phối thu nhập hằng năm.
- Biểu đồ phân phối điểm chi tiêu.
- Biểu đồ Elbow Method.
- Biểu đồ Silhouette Score (nếu cần).
- Scatter Plot thể hiện các cụm khách hàng bằng những màu sắc khác nhau.
- Biểu đồ so sánh đặc điểm giữa các cụm.

7. **Result Analysis:** Phân tích kết quả
Sau khi hoàn thành quá trình phân cụm, nhóm tiến hành phân tích đặc điểm của từng nhóm khách hàng.

Một số nội dung cần xem xét:
- Số lượng khách hàng trong mỗi cụm.
- Độ tuổi trung bình của từng cụm.
- Thu nhập trung bình của từng cụm.
- Điểm chi tiêu trung bình của từng cụm.
- Sự khác biệt giữa các nhóm khách hàng.
- Ý nghĩa thực tế của từng nhóm.

Từ đó, nhóm có thể đưa ra những nhận xét về hành vi và đặc điểm khách hàng dựa trên kết quả mô hình.


## 7. SUMMARY – Tổng kết bài toán

| Nội dung | Mô tả |
|---|---|
| Problem | Customer Segmentation |
| Machine Learning Type | Unsupervised Learning |
| Task | Clustering |
| Dataset | Mall Customers Dataset |
| Main Features | Age, Annual Income, Spending Score |
| Main Algorithm | K-Means |
| Comparison Algorithms | DBSCAN, Hierarchical Clustering |
| Evaluation | Elbow Method, Silhouette Score |
| Visualization | Matplotlib, Seaborn |
| Expected Result | Các nhóm khách hàng có đặc điểm tương đồng |

Thông qua bài thực hành, nhóm sẽ tìm hiểu quy trình xây dựng một bài toán Unsupervised Learning từ bước khám phá dữ liệu, tiền xử lý, xây dựng mô hình đến đánh giá và phân tích kết quả.

Bài toán Customer Segmentation giúp nhóm hiểu rõ hơn cách các thuật toán Clustering hoạt động và cách ứng dụng Machine Learning để khám phá những mẫu dữ liệu có ý nghĩa trong thực tế.