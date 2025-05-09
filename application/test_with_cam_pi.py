from ultralytics import YOLO
import cv2
import threading
import time
from queue import Queue

# 1. Configuration
MODEL_PATH = "yolo11n_full_integer_quant.tflite"
CAMERA_ID = 0
FRAME_QUEUE_SIZE = 4
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 2. Load model once
model = YOLO(MODEL_PATH)

# 3. Queues for frames & results
frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
result_queue = Queue(maxsize=FRAME_QUEUE_SIZE)

def capture_thread():
    """ Continuously captures frames from camera and pushes into frame_queue. """
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Drop oldest if queue full to keep latest frames
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()

def inference_thread():
    """ Pulls frames from frame_queue, runs inference, and pushes annotated frames to result_queue. """
    while True:
        frame = frame_queue.get()
        start = time.time()
        # streaming gives a generator but here we know it's one result per frame
        results = model(source=frame, stream=True,
                        conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        for res in results:
            annotated = res.plot()
            fps = 1.0 / (time.time() - start)
            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            # Manage result queue size similarly
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except Queue.Empty:
                    pass
            result_queue.put(annotated)

def display_thread():
    """ Fetches annotated frames from result_queue and displays them. """
    while True:
        frame = result_queue.get()
        cv2.imshow("YOLO11n TFLite (Multithreaded)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 4. Launch threads
    threads = [
        threading.Thread(target=capture_thread, daemon=True),
        threading.Thread(target=inference_thread, daemon=True),
        threading.Thread(target=display_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    # 5. Keep main thread alive until display_thread exits
    threads[2].join()
