import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")  # Đảm bảo đường dẫn đúng với mô hình của bạn

# Mở camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Dự đoán đối tượng trên frame
    results = model(frame)
    
    # Vẽ kết quả lên frame
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box[:6]
            label = f"{model.names[int(class_id)]}: {score:.2f}"
            
            # Vẽ hình chữ nhật và nhãn
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Hiển thị kết quả
    cv2.imshow("Fruit Detection", frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
