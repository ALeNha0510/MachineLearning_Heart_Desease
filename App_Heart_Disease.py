import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Đọc dữ liệu từ file heart.csv
df = pd.read_csv('D:\\NHA\\NHA\\DaLatUni\\DaLatUni\\MachineLearning\\2115244_BtNhom\\data\\heart.csv')

# Tách dữ liệu đặc trưng và nhãn (target)
X = df.drop('target', axis=1)
y = df['target']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tạo các mô hình học máy
logreg_model = LogisticRegression()
nb_model = GaussianNB()
knn_model = KNeighborsClassifier()

# Hàm huấn luyện mô hình
def train_models():
    logreg_model.fit(X_scaled, y)
    nb_model.fit(X_scaled, y)
    knn_model.fit(X_scaled, y)

# Huấn luyện các mô hình
train_models()

# Hàm dự đoán bệnh tim mạch
def predict_disease(features, model_type):
    # Chuẩn hóa dữ liệu đầu vào
    features_df = pd.DataFrame([features], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    features_scaled = scaler.transform(features_df)

    # Dự đoán với mô hình chọn
    if model_type == 'Hồi quy Logistic':
        prediction = logreg_model.predict(features_scaled)
    elif model_type == 'Naive Bayes':
        prediction = nb_model.predict(features_scaled)
    else:  # KNN
        # Sử dụng features_scaled thay vì features
        prediction = knn_model.predict(features_scaled)
        prob = knn_model.predict_proba(features_scaled)  # Dự đoán xác suất
        return prediction[0], prob[0]  # trả về giá trị xác suất cho lớp 1

    # Tính xác suất (chỉ áp dụng với Logistic Regression và Naive Bayes)
    if model_type in ['Hồi quy Logistic', 'Naive Bayes']:
        prob = logreg_model.predict_proba(features_scaled)[:, 1] if model_type == 'Hồi quy Logistic' else nb_model.predict_proba(features_scaled)[:, 1]
        prob = prob[0] * 100  # Chuyển sang phần trăm
    
    return prediction[0], prob


# Hàm xử lý khi nhấn nút gửi
# Hàm xử lý khi nhấn nút gửi
def on_submit():
    try:
        # Lấy giá trị từ các trường nhập liệu
        features = [
            float(entry_age.get()), 
            float(entry_sex.get()), 
            float(entry_cp.get()), 
            float(entry_trestbps.get()), 
            float(entry_chol.get()), 
            float(entry_fbs.get()), 
            float(entry_restecg.get()), 
            float(entry_thalach.get()), 
            float(entry_exang.get()), 
            float(entry_oldpeak.get()),
            float(entry_slope.get()),
            float(entry_ca.get()),
            float(entry_thal.get())
        ]
        
        # Lấy giá trị thuật toán chọn
        model_type = combo_model.get()

        # Dự đoán
        prediction, prob = predict_disease(features, model_type)
        
        if isinstance(prob, np.ndarray):  # Nếu prob là mảng numpy
            prob = prob[1] * 100  # Lấy xác suất của lớp 1 (nếu mô hình nhị phân)

        # Hiển thị kết quả
        result = f"Xác suất mắc bệnh tim mạch: {prob:.2f}%\n"
        if prediction == 1:
            result += "Bạn có thể mắc bệnh tim mạch."
        else:
            result += "Bạn không mắc bệnh tim mạch."
        
        messagebox.showinfo("Kết quả dự đoán", result)
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập đúng tất cả các trường.")

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Chẩn đoán bệnh tim mạch")

# Nhập các trường thông tin
tk.Label(root, text="Tuổi (Age):").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Giới tính (Sex): 1 là nam, 0 là nữ").grid(row=1, column=0)
entry_sex = tk.Entry(root)
entry_sex.grid(row=1, column=1)

tk.Label(root, text="Loại đau ngực (cp): 0 đến 3").grid(row=2, column=0)
entry_cp = tk.Entry(root)
entry_cp.grid(row=2, column=1)

tk.Label(root, text="Huyết áp (trestbps):").grid(row=3, column=0)
entry_trestbps = tk.Entry(root)
entry_trestbps.grid(row=3, column=1)

tk.Label(root, text="Cholesterol (chol):").grid(row=4, column=0)
entry_chol = tk.Entry(root)
entry_chol.grid(row=4, column=1)

tk.Label(root, text="Đường huyết đói (fbs > 120 mg/dL): 1 là đúng, 0 là sai").grid(row=5, column=0)
entry_fbs = tk.Entry(root)
entry_fbs.grid(row=5, column=1)

tk.Label(root, text="Kết quả điện tâm đồ (restecg): 0 đến 2").grid(row=6, column=0)
entry_restecg = tk.Entry(root)
entry_restecg.grid(row=6, column=1)

tk.Label(root, text="Nhịp tim tối đa (thalach):").grid(row=7, column=0)
entry_thalach = tk.Entry(root)
entry_thalach.grid(row=7, column=1)

tk.Label(root, text="Cơn đau ngực do gắng sức (exang): 1 là có, 0 là không").grid(row=8, column=0)
entry_exang = tk.Entry(root)
entry_exang.grid(row=8, column=1)

tk.Label(root, text="Độ chênh ST (oldpeak):").grid(row=9, column=0)
entry_oldpeak = tk.Entry(root)
entry_oldpeak.grid(row=9, column=1)

tk.Label(root, text="Độ dốc của ST (slope): 0 đến 2").grid(row=10, column=0)
entry_slope = tk.Entry(root)
entry_slope.grid(row=10, column=1)

tk.Label(root, text="Số mạch lớn (ca): 0 đến 4").grid(row=11, column=0)
entry_ca = tk.Entry(root)
entry_ca.grid(row=11, column=1)

tk.Label(root, text="Kết quả kiểm tra thallium (thal): 0 đến 3").grid(row=12, column=0)
entry_thal = tk.Entry(root)
entry_thal.grid(row=12, column=1)

# Chọn thuật toán
tk.Label(root, text="Chọn thuật toán:").grid(row=13, column=0)
from tkinter import ttk

combo_model = ttk.Combobox(root, values=["Hồi quy Logistic", "Naive Bayes", "KNN"])
combo_model.grid(row=13, column=1)
combo_model.set("Hồi quy Logistic")

# Nút gửi
submit_button = tk.Button(root, text="Dự đoán", command=on_submit)
submit_button.grid(row=14, column=0, columnspan=2)

# Chạy giao diện
root.mainloop()
