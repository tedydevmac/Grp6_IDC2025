import cv2
from ultralytics import YOLO

# Load model (detect task)
model = YOLO("/Users/tedgoh/Grp6_IDC2025/ml/models/bestV11_3_saved_model/bestV11_3_full_integer_quant.tflite", task="detect")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference on CPU
    results = model.predict(frame, imgsz=512, conf=0.75, iou=0.5, device="cpu", show=False)

    # Draw boxes and labels
    annotated = results[0].plot()  

    cv2.imshow("YOLOv11n TFLite (CPU)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
