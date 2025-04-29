#!/usr/bin/env python3
import cv2
from ultralytics import YOLO

# ---------------------------
# Load your trained YOLOv8 model
# ---------------------------
# Replace 'path/to/your_model.pt' with the actual path to your trained model file.
model = YOLO("/Users/tedgoh/Grp6_IDC2025/ml/models/best5.pt")
print("YOLOv8 model loaded successfully.")

# ---------------------------
# Initialize the camera
# ---------------------------
cap = cv2.VideoCapture(2)  # 0 is typically the default camera for a laptop
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera is active. Press 'q' to quit.")

# ---------------------------
# Main Loop: Capture, Infer, and Display
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    # Run inference on the current frame with a confidence threshold of 0.5 (adjust as needed)
    results = model.predict(frame, conf=0.9)
    
    # The YOLOv8 results have a built-in plot function that returns an annotated frame.
    # Here we use the first result (assuming one frame per inference).
    annotated_frame = results[0].plot()
    
    # Display the frame with detections.
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    # Exit loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
