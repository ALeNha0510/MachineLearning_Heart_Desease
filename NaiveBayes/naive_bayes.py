import csv
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

with open('D:\\NHA\\NHA\\DaLatUni\\DaLatUni\\MachineLearning\\2115244_BtNhom\\data\\heart.csv', 'r') as ifile:
    spamreader = csv.reader(ifile)
    X = []
    Y = []
    i = 0
    for row in spamreader:
        if i > 0:  # Bỏ qua dòng tiêu đề
            Y.append(row[-1])  # Cột 'target'
            X.append(row[:-1])  # Các thuộc tính đầu vào
        i += 1

# Chuyển đổi dữ liệu thành mảng NumPy
X = np.array(X).astype(float)
Y = np.array(Y).astype(float)

# Đường dẫn để lưu kết quả huấn luyện
output_dir = 'D:\\NHA\\NHA\\DaLatUni\\DaLatUni\\MachineLearning\\2115244_BtNhom\\NaiveBayes\\Results'
os.makedirs(output_dir, exist_ok=True)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Hàm huấn luyện mô hình và lưu kết quả
def train_and_save_results(classifier, X, Y, file_number, random_state=None):       
    print(f"Đang huấn luyện với random_state = {random_state}...")
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra với random_state khác nhau
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=random_state)
    
    # Đo thời gian huấn luyện
    start_train = time.time()
    classifier.fit(X_train, Y_train)
    end_train = time.time()
    train_time = end_train - start_train
    
    # Đo thời gian dự đoán
    start_predict = time.time()
    test_predictions = classifier.predict(X_test)
    end_predict = time.time()
    predict_time = end_predict - start_predict
    
    # Đánh giá mô hình
    accuracy = metrics.accuracy_score(Y_test, test_predictions)
    report = classification_report(Y_test, test_predictions, target_names=['Không có bệnh', 'Có bệnh'], digits=3)
    
    print(f"Test accuracy for file train_{file_number}: {accuracy}")
    print(report)
    print(f"Thời gian huấn luyện: {train_time:.4f} giây")
    print(f"Thời gian dự đoán: {predict_time:.4f} giây")
    
    # Lưu kết quả dự đoán vào file CSV
    output_file = os.path.join(output_dir, f'train_{file_number}.csv')
    with open(output_file, 'w', newline='', encoding='utf-8') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(["Dữ liệu mẫu", "Kết quả dự đoán"])
        
        for x, prediction in zip(X_test, test_predictions):
            diagnosis = "Có bệnh" if prediction == 1 else "Không có bệnh"
            writer.writerow([x.tolist(), diagnosis])

# Kiểm tra số lượng file đã tồn tại
existing_files = [f for f in os.listdir(output_dir) if f.startswith('train_') and f.endswith('.csv')]
file_number = len(existing_files) + 1  # Xác định số file tiếp theo cần tạo

# Tiến hành huấn luyện và lưu kết quả với số file đã xác định
train_and_save_results(GaussianNB(), X, Y, file_number, random_state=42)
