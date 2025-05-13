import cv2
from ultralytics import YOLO

# Load model (detect task)
model = YOLO("/Volumes/T9/tedgoh/Grp6_IDC2025/ml/models/best2_saved_model/best_full_integer_quant.tflite", task="detect")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference on CPU
    results = model.predict(frame, imgsz=640, conf=0.65, iou=0.4, device="cpu", show=False)

    # Draw boxes and labels
    annotated = results[0].plot()  

    cv2.imshow("YOLOv11n TFLite (CPU)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
