# HỆ THỐNG NHẬN DIỆN HOA QUẢ
<img src="https://github.com/user-attachments/assets/db53adff-8dd4-4b7b-971d-1b189f31d1be" alt="Đại-Nam-University" width="250"/>
<img src="https://github.com/user-attachments/assets/3ba4abb5-fa53-4c90-9775-7b14cc4c36b6" alt="AIOT-LAB" width="250"/>

📌 Giới thiệu

Dự án này sử dụng mô hình YOLOv8 để nhận diện các loại hoa quả thông qua hình ảnh. Mô hình được huấn luyện trên tập dữ liệu Fruits 360 từ Roboflow và triển khai trên Python. Mục tiêu của dự án là giúp nhận diện chính xác các loại trái cây thông qua camera theo thời gian thực.

🏗️ Hệ thống

📂 Cấu trúc dự án

📦 Fruit_Recognition_Project  
├── 📂 run  
├── 📂 test  
├── 📂 train  
├── 📂 valid  
├── best.pt  
├── data.yaml  
├── main.py  
└── README.md  

⚙️ Yêu cầu hệ thống

💻 Phần mềm

Python 3.x

Google Colab - Huấn luyện mô hình

Thư viện Python: ultralytics, roboflow, OpenCV, NumPy, Matplotlib, torch

🚀 Hướng dẫn cài đặt & chạy

1️⃣ Cài đặt môi trường Python

2️⃣ Tải dataset từ Roboflow

B1: Vào trang web chứa Dataset cần tải trên Roboflow, chọn phiên bản YOLOv8 và sao chép code snippet  
B2: Dán vào trong Google Colab để chạy để tải Dataset về

3️⃣ Huấn luyện mô hình YOLOv8

Mô hình sẽ huấn luyện trên tập dữ liệu với số epochs phù hợp.

Kết quả mô hình được lưu trong thư mục model.

4️⃣ Nhận diện hoa quả

Hệ thống sẽ dự đoán và hiển thị tên hoa quả trên ảnh đầu ra.

📈 Cải tiến trong tương lai

Cải thiện độ chính xác mô hình bằng cách tối ưu hóa hyperparameters.

Tích hợp AI vào ứng dụng di động để nhận diện hoa quả dễ dàng hơn.

Mở rộng dataset với nhiều loại trái cây hơn.

🏆 Người đóng góp

[Trần Quang Lâm]  
[Trần Đoàn Quang Huy]  
[Nguyễn Công Thành]
