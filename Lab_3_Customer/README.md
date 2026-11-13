# PRACTICE EXERCISE 3: CUSTOMER SEGMENTATION

## 1. Giới thiệu

Bài thực hành sử dụng các thuật toán Machine Learning không giám sát (Unsupervised Learning) để phân khúc khách hàng dựa trên những đặc điểm tương đồng.

Dự án sử dụng bộ dữ liệu **Mall Customers Dataset** và thực nghiệm ba thuật toán:
- K-Means
- DBSCAN
- Hierarchical Clustering

Mục tiêu là khám phá các nhóm khách hàng, đánh giá kết quả phân cụm và phân tích đặc điểm của từng nhóm.

## 2. Dataset

- Tên dataset: Mall Customers Dataset
- Số lượng mẫu: 200 khách hàng
- Số lượng thuộc tính: 5

Các thuộc tính gồm:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

Trong quá trình xây dựng mô hình, hai đặc trưng được lựa chọn là `Annual Income (k$)` và `Spending Score (1-100)`.

## 3. Công nghệ sử dụng

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook
- Visual Studio Code

## 4. Cấu trúc project

```text
Lab_3_Customer/
│
├── data/
│   └── raw/
│
├── docs/
│   └── description.md
│
├── models/
│
├── notebooks/
│   └── practice3_customer_segmentation.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
├── src/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Các bước thực hiện

1. Thu thập và đọc dữ liệu.
2. Khám phá và phân tích dữ liệu (EDA).
3. Tiền xử lý và chuẩn hóa dữ liệu.
4. Xây dựng mô hình phân cụm.
5. Đánh giá và so sánh các thuật toán.
6. Trực quan hóa và phân tích kết quả.

## 6. Kết quả thực nghiệm

Dự án thực hiện và so sánh ba thuật toán phân cụm bằng Elbow Method, Silhouette Score và biểu đồ trực quan.

| Algorithm | Number of Clusters | Silhouette Score |
|---|---:|---:|
| K-Means | 5 | 0.5547 |
| DBSCAN | 2 | 0.3876 |
| Hierarchical | 5 | 0.5538 |

K-Means được lựa chọn làm mô hình chính để phân tích 5 nhóm khách hàng trong bài thực hành.

## 7. Hướng dẫn chạy project

### Bước 1: Clone project

```bash
git clone <repository-url>
cd Lab_3_Customer
```

### Bước 2: Tạo môi trường ảo

```bash
python -m venv .venv
```

### Bước 3: Kích hoạt môi trường ảo

Windows:

```bash
.venv\Scripts\activate
```

### Bước 4: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 5: Chạy notebook

Mở Visual Studio Code, chọn notebook:

`notebooks/practice3_customer_segmentation.ipynb`

Chọn kernel `.venv` và chạy lần lượt các cell.

## 8. Tài liệu

Xem thêm nội dung mô tả bài toán tại:

`docs/description.md`
