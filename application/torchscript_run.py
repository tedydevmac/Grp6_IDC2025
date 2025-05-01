import cv2
import numpy as np
import onnxruntime as ort
import time

CONF_THRESH = 0.4
INPUT_SIZE = 640  # For YOLOv11n default

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded, r, dw, dh

def preprocess(frame):
    img, r, dw, dh = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img, r, dw, dh

def postprocess(output, r, dw, dh, frame_shape):
    results = []
    predictions = output[0][0]  # shape: [num_preds, num_attrs]
    for det in predictions:
        x, y, w, h = det[0:4]
        objectness = det[4]
        class_scores = det[5:]
        cls = np.argmax(class_scores)
        conf = objectness * class_scores[cls]

        if conf < CONF_THRESH:
            continue

        # Rescale boxes to original image
        x -= dw
        y -= dh
        x /= r
        y /= r
        w /= r
        h /= r

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        results.append((x1, y1, x2, y2, conf, cls))
    return results

# Load model
session = ort.InferenceSession("/home/sst/IDC25G6/Grp6_IDC2025/ml/models/bestV11_simplified.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Start camera
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor, r, dw, dh = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    detections = postprocess(outputs, r, dw, dh, frame.shape)

    for x1, y1, x2, y2, conf, cls in detections:
        label = f"{int(cls)}:{conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("YOLOv11 Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()