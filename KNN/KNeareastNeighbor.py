import time
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('D:\\NHA\\NHA\\DaLatUni\\DaLatUni\\MachineLearning\\2115244_BtNhom\\data\\heart.csv')

# Xử lý các giá trị bị thiếu bằng cách thay thế với giá trị trung bình
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

X = data_scaled[:, :-1]  # Tất cả các cột trừ cột 'target'
y = data_imputed[:, -1]  # Chỉ cột 'target'

n = len(y)
k = int(np.sqrt(n))  # Giá trị K ban đầu

# Tìm kiếm giá trị K tối ưu dựa trên cross-validation
best_k = k
best_score = 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    mean_score = scores.mean()
    if mean_score > best_score:
        best_k = k
        best_score = mean_score

print(f'Best K: {best_k} with cross-validation score: {best_score}')

# Bước 3: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Bước 4: Xây dựng và huấn luyện mô hình KNN với K tốt nhất
knn = KNeighborsClassifier(n_neighbors=best_k)

# Bắt đầu đo thời gian huấn luyện
start_train = time.time()
knn.fit(X_train, y_train)
end_train = time.time()

# Tính thời gian huấn luyện
train_time = end_train - start_train

# Bước 5: Dự đoán và đo thời gian dự đoán
start_predict = time.time()
y_pred = knn.predict(X_test)
end_predict = time.time()

# Tính thời gian dự đoán
predict_time = end_predict - start_predict

# Tính toán và hiển thị độ chính xác và các chỉ số khác
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print("Classification Report:\n", classification_report(y_test, y_pred))

# In thời gian huấn luyện và thời gian dự đoán
print(f'Thời gian huấn luyện mô hình: {train_time:.4f} giây')
print(f'Thời gian dự đoán: {predict_time:.4f} giây')
