import cv2
import numpy as np
import onnxruntime as ort
import time

# Load ONNX model
session = ort.InferenceSession("/home/sst/IDC25G6/Grp6_IDC2025/ml/models/bestV11.onnx")
input_name = session.get_inputs()[0].name

# Model input size (adjust if needed)
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Confidence threshold
CONF_THRESH = 0.65

# Start video capture
cap = cv2.VideoCapture(0)  # 0 = default camera

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img, axis=0)  # [1,3,H,W]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)

    # Inference
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    end = time.time()

    output = outputs[0]  # Assuming output shape is [num_boxes, 6]
    for det in output:
        x1, y1, x2, y2, conf, cls = det
        if conf > CONF_THRESH:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f'{int(cls)}:{conf:.2f}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    fps = 1 / (end - start)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("YOLO ONNX Inference", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()