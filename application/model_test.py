import cv2
import numpy as np
import onnxruntime

# -------------------------------
# Configuration and Model Loading
# -------------------------------
img_size = 640          # Input size for the network
conf_threshold = 0.175   # Confidence threshold for detections

# Replace these with your ONNX model filename and your classes
onnx_model_path = "/Users/tedgoh/Grp6_IDC2025/ml/yolov5/runs/train/exp11/weights/best.onnx"
class_names = ["burger"]  # Adjust or extend as needed

# Create an ONNX Runtime session
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("Model input name:", input_name)

# -------------------------------
# Preprocessing Functions
# -------------------------------
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """
    Resize image with unchanged aspect ratio using padding.
    Returns the padded image, scale factor, and top-left padding.
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # resize image
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, left, top

def preprocess(img):
    # Letterbox resize
    img_padded, scale, pad_x, pad_y = letterbox(img, new_shape=img_size)

    # Convert BGR to RGB, transpose, and normalize (if needed).
    # Many YOLO models expect images in RGB with pixel values 0-1.
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0

    # HWC to CHW and add batch dimension
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_transposed, axis=0)

    return img_input, scale, pad_x, pad_y

# -------------------------------
# Postprocessing Function
# -------------------------------
def process_detections(detections, scale, pad_x, pad_y, original_shape):
    """
    Convert detections from network output into bounding boxes in the original image.
    Detections are assumed to be in the format: [x_center, y_center, width, height, conf, class_id]
    Coordinates are relative to the resized image; we need to unpad and rescale them.
    """
    boxes = []
    for det in detections[0]:
        x_center, y_center, width, height, conf, cls = det
        if conf < conf_threshold:
            continue

        # Convert from center coordinates to top-left coordinates in padded image
        x1 = (x_center - width / 2) - pad_x
        y1 = (y_center - height / 2) - pad_y
        x2 = (x_center + width / 2) - pad_x
        y2 = (y_center + height / 2) - pad_y

        # Scale coordinates to original image
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale

        boxes.append({
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'conf': conf,
            'class_id': int(cls)
        })
    return boxes

# -------------------------------
# Inference and Visualization
# -------------------------------
def run_inference(frame):
    original_shape = frame.shape[:2]
    input_tensor, scale, pad_x, pad_y = preprocess(frame)

    # Run inference (input tensor shape: [1, 3, 640, 640])
    outputs = session.run(None, {input_name: input_tensor})
    # Assume the output is a single array with shape [1, N, 6]
    detections = outputs[0]
    boxes = process_detections(detections, scale, pad_x, pad_y, original_shape)
    return boxes

def draw_boxes(frame, boxes):
    for box_info in boxes:
        x1, y1, x2, y2 = box_info['box']
        conf = box_info['conf']
        cls = box_info['class_id']
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# -------------------------------
# Main Script to Capture Video
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = run_inference(frame)
        frame = draw_boxes(frame, boxes)
        cv2.imshow("ONNX YOLOv5 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
