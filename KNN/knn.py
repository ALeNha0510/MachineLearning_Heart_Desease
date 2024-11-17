import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Đọc dữ liệu
df = pd.read_csv('D:\\NHA\\NHA\\DaLatUni\\DaLatUni\\MachineLearning\\2115244_BtNhom\\data\\heart.csv')
df.head()

# Kiểm tra thông tin chung và xử lý dữ liệu bị thiếu
df.describe()
df.info()
df.isnull().sum()

# Vẽ ma trận tương quan giữa các đặc trưng
corrmat = df.corr()
features_corr = corrmat.index
plt.figure(figsize=(10, 15))
sns.heatmap(df[features_corr].corr(), annot=True, cmap='RdYlGn')
plt.title("Ma trận tương quan giữa các đặc trưng")  # Tiêu đề cho ma trận tương quan
plt.show()

# Vẽ histogram cho các cột dữ liệu
df.hist(bins=15, figsize=(10, 15), color='blue')
plt.suptitle("Biểu đồ histogram cho từng đặc trưng", fontsize=10)  # Tiêu đề cho toàn bộ biểu đồ histogram
plt.show()

# Tiền xử lý dữ liệu
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Chuẩn hóa các đặc trưng số
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

# Tách dữ liệu thành X và y
X = dataset.drop(['target'], axis=1)
y = dataset['target']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

# Mô hình Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print("================================================")
print("Kết quả mô hình Logistic Regression:")
print(f"Độ chính xác: {lr_accuracy* 100:.2f}%")
print(f"Độ chính xác (Precision): {lr_precision* 100:.2f}%")
print(f"Độ nhạy (Recall): {lr_recall* 100:.2f}%")
print(f"F1 Score: {lr_f1* 100:.2f}%")

# Mô hình Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_score = nb.score(X_test, y_test)
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)

print("================================================")
print("Kết quả mô hình Naive Bayes:")
print(f"Độ chính xác: {nb_accuracy* 100:.2f}%")
print(f"Độ chính xác (Precision): {nb_precision* 100:.2f}%")
print(f"Độ nhạy (Recall): {nb_recall* 100:.2f}%")
print(f"F1 Score: {nb_f1* 100:.2f}%")

# Mô hình KNN
knn = KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto')
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

print("================================================")
print("Kết quả mô hình KNN:")
print(f"Độ chính xác: {knn_accuracy* 100:.2f}%")
print(f"Độ chính xác (Precision): {knn_precision* 100:.2f}%")
print(f"Độ nhạy (Recall): {knn_recall* 100:.2f}%")
print(f"F1 Score: {knn_f1* 100:.2f}%")

# Vẽ ma trận nhầm lẫn cho mỗi mô hình
def plot_confusion_matrix(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  # Tính toán ma trận nhầm lẫn
    cm_df = pd.DataFrame(cm, columns=['Dự đoán: 0', 'Dự đoán: 1'], index=['Thực tế: 0', 'Thực tế: 1'])
    
    # Vẽ ma trận nhầm lẫn bằng heatmap của seaborn
    plt.figure(figsize=(7, 5))  # Thiết lập kích thước biểu đồ
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14}, linewidths=0.5)
    plt.title(f'Ma trận nhầm lẫn của {model_name}')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    plt.show()

# Vẽ ma trận nhầm lẫn cho từng mô hình
plot_confusion_matrix("Logistic Regression", y_test, lr_pred)
plot_confusion_matrix("Naive Bayes", y_test, nb_pred)
plot_confusion_matrix("KNN", y_test, knn_pred)

# Vẽ so sánh độ chính xác giữa các mô hình
model_names = ['Logistic Regression', 'Naive Bayes', 'KNN']
accuracy_scores = [lr_accuracy, nb_accuracy, knn_accuracy]
colors = ['paleturquoise', 'lightblue', 'skyblue']
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color=colors)
plt.xlabel('Các mô hình học máy')
plt.ylabel('Độ chính xác')
plt.title('So sánh độ chính xác giữa các mô hình')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Tạo dữ liệu cho các metric của mỗi mô hình
model_metrics = {
    'Metric': ['Độ chính xác', 'Độ chính xác (Precision)', 'Độ nhạy (Recall)', 'F1 Score'],
    'Logistic Regression': [lr_accuracy, lr_precision, lr_recall, lr_f1],
    'Naive Bayes': [nb_accuracy, nb_precision, nb_recall, nb_f1],
    'KNN': [knn_accuracy, knn_precision, knn_recall, knn_f1]
}

# Chuyển dữ liệu vào DataFrame để dễ dàng vẽ biểu đồ
metrics_df = pd.DataFrame(model_metrics)

# Vẽ biểu đồ so sánh các metric giữa các mô hình
metrics_df.set_index('Metric').plot(kind='bar', figsize=(12, 8), color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title("So sánh các Metric của Logistic Regression, Naive Bayes và KNN")
plt.xlabel("Các Metric")
plt.ylabel("Điểm số")
plt.xticks(rotation=0)  # Giữ các nhãn trên trục x không xoay
plt.legend(title="Các mô hình")
plt.tight_layout()
plt.show()
