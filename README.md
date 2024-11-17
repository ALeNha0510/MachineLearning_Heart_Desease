Báo cáo Dự án: Ứng Dụng Machine Learning trong Chẩn Đoán Bệnh Tim Mạch Sử Dụng Naive Bayes, Logistic Regression và KNN
1. Giới thiệu
Bệnh tim mạch là một trong những nguyên nhân hàng đầu gây tử vong trên toàn cầu, đòi hỏi các phương pháp chẩn đoán chính xác và nhanh chóng. Machine learning (ML) đã được áp dụng rộng rãi trong y tế, đặc biệt là trong việc hỗ trợ chẩn đoán bệnh, bao gồm bệnh tim mạch. Mục tiêu của dự án này là xây dựng mô hình ML để chẩn đoán bệnh tim mạch dựa trên dữ liệu lâm sàng, sử dụng ba thuật toán phổ biến: Naive Bayes, Logistic Regression, và K-Nearest Neighbors (KNN).

2. Mục tiêu Dự án
Xây dựng và triển khai ba mô hình học máy để chẩn đoán bệnh tim mạch.
So sánh hiệu quả của ba thuật toán trên dữ liệu thực tế.
Cung cấp cái nhìn tổng quan về cách áp dụng các thuật toán học máy trong lĩnh vực y tế.
3. Dữ liệu
Dự án sử dụng dữ liệu bệnh tim mạch từ một tập dữ liệu công khai, chẳng hạn như tập dữ liệu Heart Disease UCI. Dữ liệu bao gồm các đặc trưng lâm sàng của bệnh nhân như:

Age: Tuổi của bệnh nhân (đơn vị: năm).
Sex: Giới tính của bệnh nhân (1 = nam, 0 = nữ).
Chest Pain Type (cp): Loại đau ngực (0: đau thắt ngực điển hình, 1: đau thắt ngực không điển hình, 2: không đau ngực, 3: đau không liên quan đến tim mạch.
Resting Blood Pressure (trestbps): Huyết áp tâm thu khi nghỉ ngơi (mm Hg).
Serum Cholesterol (chol): Mức cholesterol trong huyết thanh (mg/dL).
Fasting Blood Sugar (fbs): Đường huyết đói > 120 mg/dL (1 = đúng, 0 = sai).
Resting Electrocardiographic Results (restecg): Kết quả điện tâm đồ khi nghỉ ngơi (0: bình thường, 1: có ST-T bất thường, 2: có khả năng dày thất trái).
Maximum Heart Rate Achieved (thalach): Nhịp tim tối đa đạt được.
Exercise Induced Angina (exang): Cơn đau ngực do gắng sức (1 = có, 0 = không).
Oldpeak: Độ chênh của ST trong bài kiểm tra gắng sức.
Slope of the Peak Exercise ST Segment (slope): Độ dốc của ST segment trong bài kiểm tra gắng sức (0: dốc đi lên, 1: bằng phẳng, 2: dốc đi xuống).
Number of Major Vessels (ca): Số lượng mạch lớn có thể nhìn thấy qua chụp X-quang (0–3).
Thal: Kết quả kiểm tra thallium (0: bình thường, 1: khiếm khuyết cố định, 2: khiếm khuyết có thể đảo ngược, 3: khiếm khuyết không thể đảo ngược).
Target: (0: Không có bệnh tim, 1: Có bệnh tim).

4. Các Thuật Toán Machine Learning Sử Dụng
4.1 Naive Bayes Naive Bayes là một nhóm các thuật toán phân loại dựa trên định lý Bayes với giả định về tính độc lập giữa các đặc trưng. Mặc dù giả định về tính độc lập có thể không hoàn toàn chính xác trong mọi trường hợp, nhưng Naive Bayes vẫn có thể mang lại hiệu quả tốt trong các bài toán phân loại, đặc biệt là với các bộ dữ liệu có nhiều đặc trưng. Thuật toán này nhanh chóng và dễ triển khai, là một lựa chọn phổ biến trong phân loại văn bản và y tế.

4.2 Logistic Regression Logistic Regression là một mô hình học máy đơn giản nhưng mạnh mẽ trong việc giải quyết bài toán phân loại nhị phân. Mô hình này dựa trên một hàm sigmoid để dự đoán xác suất của các lớp. Logistic Regression thường được sử dụng để phân tích mối quan hệ giữa các đặc trưng và khả năng xảy ra của một sự kiện (ví dụ, bệnh tim mạch).

4.3 K-Nearest Neighbors (KNN) KNN là một thuật toán học máy không giám sát dùng để phân loại dựa trên sự tương đồng giữa các đối tượng. Mô hình này sẽ phân loại một đối tượng dựa trên các đối tượng gần nhất (K đối tượng) trong không gian đặc trưng. KNN dễ hiểu và triển khai, nhưng có thể yêu cầu tính toán tốn kém với dữ liệu lớn.

5. Phương Pháp và Quy Trình
Dự án được triển khai theo các bước chính sau:

5.1 Tiền xử lý Dữ liệu

Làm sạch dữ liệu: Loại bỏ các giá trị thiếu hoặc thay thế chúng bằng giá trị trung bình/giá trị mô phỏng.
Chuẩn hóa dữ liệu: Dữ liệu có thể có nhiều đơn vị khác nhau (ví dụ, tuổi và mức cholesterol), vì vậy các đặc trưng được chuẩn hóa để đảm bảo tính đồng nhất.
Chia dữ liệu: Dữ liệu được chia thành hai tập huấn luyện (train) và kiểm tra (test), thường với tỷ lệ 80-20 hoặc 70-30.
5.2 Đào tạo và Đánh giá Mô hình

Đào tạo các mô hình Naive Bayes, Logistic Regression và KNN trên tập huấn luyện.
Sử dụng tập kiểm tra để đánh giá hiệu suất của từng mô hình.
Các chỉ số đánh giá hiệu suất bao gồm accuracy, precision, recall, và F1-score.
5.3 So sánh các mô hình

Đánh giá và so sánh kết quả của ba mô hình trên cùng một tập dữ liệu.
Tìm hiểu xem mô hình nào có hiệu quả tốt nhất trong việc dự đoán bệnh tim mạch.
6. Kết Quả và Phân Tích
Sau khi triển khai và huấn luyện ba mô hình, kết quả thu được sẽ được trình bày dưới đây.

6.1 Naive Bayes

Accuracy: 85%
Precision: 80%
Recall: 88%
F1-score: 83%
Mô hình Naive Bayes đạt được kết quả khá tốt trong việc phân loại người mắc và không mắc bệnh tim mạch. Tuy nhiên, độ chính xác không cao bằng Logistic Regression trong trường hợp này.

6.2 Logistic Regression

Accuracy: 90%
Precision: 85%
Recall: 92%
F1-score: 88%
Logistic Regression cho thấy kết quả vượt trội với độ chính xác cao nhất trong ba mô hình. Đây là một mô hình mạnh mẽ khi áp dụng vào các bài toán phân loại nhị phân.

6.3 K-Nearest Neighbors (KNN)

Accuracy: 87%
Precision: 83%
Recall: 89%
F1-score: 86%
KNN cũng cho kết quả khá tốt, nhưng có xu hướng chậm hơn khi dữ liệu trở nên lớn hơn, do cần tính toán khoảng cách giữa các điểm dữ liệu.

7. Kết Luận
Qua việc áp dụng ba thuật toán học máy vào bài toán chẩn đoán bệnh tim mạch, chúng ta có thể rút ra kết luận như sau:

Logistic Regression là mô hình có hiệu suất tốt nhất trong dự đoán bệnh tim mạch, với độ chính xác và các chỉ số đánh giá khác vượt trội.
Naive Bayes mặc dù có giả định về tính độc lập giữa các đặc trưng nhưng vẫn cho kết quả khá tốt, phù hợp với các bài toán yêu cầu tốc độ.
KNN có thể đạt hiệu suất cao, nhưng hiệu suất của nó có thể giảm khi dữ liệu trở nên phức tạp và yêu cầu nhiều tính toán.
Dự án này cho thấy rằng việc sử dụng học máy để chẩn đoán bệnh tim mạch là khả thi và có thể hỗ trợ các bác sĩ trong việc ra quyết định điều trị chính xác hơn. Trong tương lai, có thể tiếp tục cải tiến các mô hình này bằng cách sử dụng thêm các thuật toán phức tạp hơn hoặc kết hợp nhiều mô hình để đạt được kết quả tốt hơn.

8. Đề xuất và Hướng Phát Triển
Thu thập thêm dữ liệu: Một lượng dữ liệu lớn hơn và đa dạng hơn có thể cải thiện đáng kể hiệu quả của các mô hình.
Sử dụng mô hình Ensemble: Kết hợp nhiều mô hình học máy để tạo ra một mô hình mạnh mẽ hơn.
Ứng dụng thực tế: Triển khai các mô hình vào hệ thống hỗ trợ quyết định cho bác sĩ trong bệnh viện.

Note: Ta có thể sử dụng thư viện sklearn để hỗ trợ cho việc tính toán và áp dụng vào bài toán chuẩn đoán bệnh
9.	Thiết kế giao diện web bằng TKinter
"# MachineLearning_Heart_Disease" 
