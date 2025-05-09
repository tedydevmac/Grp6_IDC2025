import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# --- Setup TFLite interpreter with multi-threading ---
interpreter = Interpreter(model_path="bestV11_3_full_integer_quant.tflite", num_threads=4)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- CameraThread identical to before, giving 512×512 RGB frames ---
class CameraStream:
    def __init__(self, size=(512,512)):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": size})
        self.picam2.configure(config)
        self.queue = Queue(maxsize=1)
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.running = False

    def start(self):
        self.picam2.start()
        self.running = True
        self.thread.start()
        return self

    def _capture_loop(self):
        while self.running:
            frame = self.picam2.capture_array()  # 512×512×3 RGB uint8
            if not self.queue.empty():
                try: self.queue.get_nowait()
                except: pass
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.running = False
        self.thread.join()
        self.picam2.stop()

cam = CameraStream().start()

# Prepare a single OpenCV window
cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 800, 600)

# --- ThreadPoolExecutor for parallel inference ---
executor = ThreadPoolExecutor(max_workers=2)

def run_inference(frame):
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(frame, axis=0))
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    return boxes, classes, scores

try:
    while True:
        frame = cam.read()  # already right size & format

        # Submit inference task to the executor
        future = executor.submit(run_inference, frame)
        boxes, classes, scores = future.result()

        # --- Draw green boxes ourselves ---
        h, w, _ = frame.shape
        for i, s in enumerate(scores):
            if s < 0.5: continue
            y1, x1, y2, x2 = boxes[i]
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{int(classes[i])}:{s:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()
    executor.shutdown()
