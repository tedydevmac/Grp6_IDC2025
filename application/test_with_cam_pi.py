import cv2
from ultralytics import YOLO
import threading
import queue

# Load model (detect task)
model = YOLO("/Users/tedgoh/Grp6_IDC2025/ml/models/bestV11_4_saved_model/bestV11_4_full_integer_quant.tflite", task="detect")

# Initialize camera with optimized resolution and buffer size for Raspberry Pi
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Directly set width to 320
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Directly set height to 240
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to minimize latency
cap.set(cv2.CAP_PROP_FPS, 15)  # Limit frame rate to 15 FPS for efficiency
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

frame_queue = queue.Queue(maxsize=2)  # Thread-safe queue for frames
stop_event = threading.Event()

# Thread for capturing frames
def capture_frames():
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            # Skip adding frames if the queue is full
            if not frame_queue.full():
                frame_queue.put(frame)

# Thread for processing frames
def process_frames():
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Inference on CPU with optimized settings
            results = model.predict(frame, imgsz=256, conf=0.8, iou=0.4, device="cpu", show=False)  # Reduced imgsz, conf, and iou for speed

            # Draw boxes and labels
            annotated = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv11n TFLite (Raspberry Pi)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

# Start threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
capture_thread.start()
process_thread.start()

# Wait for threads to finish
capture_thread.join()
process_thread.join()

cap.release()
cv2.destroyAllWindows()