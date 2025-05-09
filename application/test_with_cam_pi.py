import threading
from queue import Queue
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# 1️⃣ Initialize your YOLO TFLite model (CPU mode)
model = YOLO(
    "bestV11_3_full_integer_quant.tflite",
    task="detect"
)

# 2️⃣ Camera capture class with hardware resizing + threading
class CameraStream:
    def __init__(self, size=(512, 512)):
        self.picam2 = Picamera2()
        # Configure PiCamera2 to output at model’s input size (512×512)
        config = self.picam2.create_preview_configuration(
            main={"size": size}
        )  # hardware resizing via ISP :contentReference[oaicite:5]{index=5}
        self.picam2.configure(config)
        self.queue = Queue(maxsize=1)
        self.running = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.picam2.start()
        self.running = True
        self.thread.start()   # dedicated capture thread :contentReference[oaicite:6]{index=6}
        return self

    def _capture_loop(self):
        while self.running:
            frame = self.picam2.capture_array()
            # Always keep only the latest frame in the queue
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Queue.Empty:
                    pass
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.running = False
        self.thread.join()
        self.picam2.stop()

# 3️⃣ Start camera stream
cam = CameraStream(size=(512, 512)).start()

try:
    while True:
        # 4️⃣ Grab the latest pre-resized frame
        frame = cam.read()  # shape: (512, 512, 3)
        
        # (Optional) If you ever need to double-check size or crop ROI:
        # if frame.shape[:2] != (512, 512):
        #     frame = cv2.resize(
        #         frame, (512, 512),
        #         interpolation=cv2.INTER_NEAREST
        #     )  # nearest-neighbor fallback :contentReference[oaicite:7]{index=7}

        # 5️⃣ Run inference (Ultralytics handles TFLite + XNNPACK under the hood)
        results = model.predict(
            source=frame,
            imgsz=512,
            conf=0.5,
            iou=0.5,
            device="cpu",
            show=False
        )

        # 6️⃣ Annotate & display
        annotated = results[0].plot()
        cv2.imshow("YOLOv11n TFLite (512×512)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()
