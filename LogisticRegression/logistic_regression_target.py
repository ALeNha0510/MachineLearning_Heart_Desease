import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
data = pd.read_csv('D:\\NHA\\NHA\\DaLatUni\\DaLatUni\\MachineLearning\\2115244_BtNhom\\data\\heart.csv')

print("5 hàng đầu tiên của tập dữ liệu:")
print(data.head())

print("\nThống kê tổng quan:")
print(data.describe())

print("\nThông tin về tập dữ liệu:")
print(data.info())

print("\nGiá trị thiếu trong mỗi cột:")
print(data.isnull().sum())

# Phân tích dữ liệu thăm dò
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ma trận tương quan của các đặc trưng")
plt.show()

# Lựa chọn đặc trưng dựa trên tương quan với mục tiêu
threshold = 0.3
correlation_matrix = data.corr()
high_corr_features = correlation_matrix.index[abs(correlation_matrix["target"]) > threshold].tolist()
high_corr_features.remove("target")

print("Các đặc trưng được chọn dựa trên tương quan với mục tiêu:")
print(high_corr_features)

# Tạo một DataFrame mới chỉ với các đặc trưng được chọn
X_selected = data[high_corr_features]
y = data["target"]

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Tách dữ liệu Train-Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

# Tính thời gian huấn luyện mô hình
start_train = time.time()
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
end_train = time.time()

# Dự đoán mô hình
start_predict = time.time()
y_pred = logreg.predict(X_test)
end_predict = time.time()

# Tính thời gian chạy
train_time = end_train - start_train
predict_time = end_predict - start_predict

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Độ chính xác: {accuracy:.4f}")
print("\nMa trận nhầm lẫn:")
print(conf_matrix)
print("\nBáo cáo phân loại:")
print(class_report)

# In thời gian chạy
print(f"\nThời gian huấn luyện mô hình: {train_time:.4f} giây")
print(f"Thời gian dự đoán: {predict_time:.4f} giây")

# Biểu diễn ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Dự đoán Âm tính", "Dự đoán Dương tính"],
            yticklabels=["Thực tế Âm tính", "Thực tế Dương tính"])
plt.xlabel("Nhãn Dự đoán")
plt.ylabel("Nhãn Thực tế")
plt.title("Biểu đồ nhiệt của Ma trận nhầm lẫn")
plt.show()
