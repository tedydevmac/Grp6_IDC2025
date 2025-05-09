import threading
from queue import Queue
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# Load your model
model = YOLO("bestV11_3_full_integer_quant.tflite", task="detect")

class CameraStream:
    def __init__(self, size=(512, 512)):
        self.picam2 = Picamera2()
        # Use video config so no PiCamera2 preview windows pop up
        config = self.picam2.create_video_configuration(
            main={"size": size}
        )
        self.picam2.configure(config)
        self.queue = Queue(maxsize=1)
        self.running = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.picam2.start()
        self.running = True
        self.thread.start()
        return self

    def _capture_loop(self):
        while self.running:
            frame = self.picam2.capture_array()
            try:
                self.queue.put_nowait(frame)  # Automatically discard old frames
            except Queue.Full:
                pass

    def read(self):
        return self.queue.get()

    def stop(self):
        self.running = False
        self.thread.join()
        self.picam2.stop()

# Initialize camera stream
cam = CameraStream(size=(512, 512)).start()

# Create a single, resizable OpenCV window
window_name = "YOLOv11n TFLite (512×512)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

def inference_thread():
    while True:
        frame = cam.read()
        results = model.predict(
            source=frame,
            imgsz=512,
            conf=0.5,
            iou=0.5,
            device="cpu",
            show=False
        )
        annotated = results[0].plot()
        cv2.imshow(window_name, annotated)

# Start the inference thread
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

try:
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()
