import cv2
import time
import serial
from ultralytics import YOLO
import threading
from collections import deque

# ---------------------------
# Serial (PySerial) Setup
# ---------------------------
try:
    ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)
    time.sleep(2)  # Allow some time to establish connection
    print("Serial connection established.")
except Exception as e:
    print("Serial initialization error:", e)
    ser = None

def send_serial_command(command: str):
    global last_command
    if ser and command != last_command:
        try:
            ser.write(command.encode())
            last_command = command
            print("Sent command:", command)
        except Exception as e:
            print("Error sending command:", e)
    else:
        print("Serial port not available or command unchanged.")

# ---------------------------
# Load YOLOv8 Model
# ---------------------------
model_path = "/home/sst/IDC25G6/Grp6_IDC2025/ml/models/best4_saved_model_512_40epochs/best_full_integer_quant.tflite"

print("Loading YOLO11n model...")
model = YOLO(model_path, task="detect")
print("Model loaded.")

# ---------------------------
# Global Variables for Accumulated Counts
# ---------------------------
accumulated_medical_counts = {"napkin": 0, "syringe": 0, "bandage": 0}
last_command = None
# Track which objects have been detected to avoid counting them multiple times
detected_medical_objects = []

# ---------------------------
# Reset Accumulated Counts
# ---------------------------
def reset_accumulated_counts():
    global accumulated_medical_counts, detected_medical_objects
    accumulated_medical_counts = {"napkin": 0, "syringe": 0, "bandage": 0}
    detected_medical_objects = []  # Also reset the tracking of detected objects

# ---------------------------
# Update Accumulated Counts
# ---------------------------
def update_accumulated_counts(detection_results, accumulated_counts, detected_boxes):
    global detected_medical_objects
    
    # Compare new detections with previously detected objects
    new_detected_objects = []
    for item, count in detection_results.items():
        for i in range(count):
            box = next((b for b, item_name in detected_boxes if item_name == item), None)
            if box:
                # Check if this object has been detected before
                is_new_object = True
                for old_box, old_item in detected_medical_objects:
                    if old_item == item and calculate_iou(box, old_box) > 0.05:  # Higher IoU threshold for same object
                        is_new_object = False
                        break
                
                if is_new_object:
                    new_detected_objects.append((box, item))
                    accumulated_counts[item] += 1
    
    # Update the list of detected objects
    detected_medical_objects.extend(new_detected_objects)

# ---------------------------
# Detection Functions
# ---------------------------
def detect_items(frame, item_classes):
    """
    Detect and count items in the frame using YOLOv8.
    Filters out duplicate detections of the same object.
    item_classes: A dictionary mapping class indices to item names.
    Returns a dictionary with counts for each item and the detected boxes.
    """
    results = model.predict(frame, conf=0.75, imgsz=512)  # Adjust confidence threshold as needed
    item_counts = {item: 0 for item in item_classes.values()}

    detected_boxes = []  # Store processed bounding boxes to avoid duplicates

    for result in results:
        for box in result.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            if cls in item_classes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                box_coords = (x1, y1, x2, y2)

                # Check for duplicate detections
                is_duplicate = any(calculate_iou(box_coords, existing_box) > 0.5 for existing_box, _ in detected_boxes)

                if not is_duplicate:
                    item_name = item_classes[cls]
                    detected_boxes.append((box_coords, item_name))
                    item_counts[item_name] += 1

    return item_counts, detected_boxes

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    box1, box2: Tuples of (x1, y1, x2, y2).
    Returns IoU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# ---------------------------
# Check for Arduino Input
# ---------------------------
def check_arduino_input():
    if ser and ser.in_waiting > 0:
        try:
            input_data = ser.readline().decode('utf-8').strip()
            print("Received from Arduino:", input_data)
            return input_data
        except Exception as e:
            print("Error reading from Arduino:", e)
    return None

# ---------------------------
# Initialize a thread-safe deque for frames
# ---------------------------
frame_queue = deque(maxlen=2)
stop_event = threading.Event()

# Thread for capturing frames with frame skipping
def capture_frames(cap, frame_skip=4):
    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if frame_count % frame_skip == 0:  # Skip frames
                frame_queue.append(frame)

# Thread for processing frames
def process_frames(food_classes, medical_classes):
    global accumulated_medical_counts
    detection_mode = "medical"

    while not stop_event.is_set():
        # Check for Arduino input to set detection mode
        arduino_input = check_arduino_input()
        if arduino_input == "food":
            detection_mode = "food"
            print("Switched to food detection mode.")
        elif arduino_input == "medical":
            detection_mode = "medical"
            print("Switched to medical detection mode.")
        elif arduino_input == "reset":
            print("Resetting counts as per Arduino command...")
            reset_accumulated_counts()
        elif arduino_input == "stop":
            print("Detection functions paused as per Arduino command...")
            detection_mode = None

        # Perform detection based on the current mode
        if detection_mode and frame_queue:
            frame = frame_queue.popleft()

            if detection_mode == "food":
                food_results, _ = detect_items(frame, food_classes)
                for item, count in food_results.items():
                    if count > 0:  # If any food item is detected
                        send_serial_command(f"food_{item}")
                        print(f"Detected and sent food item: {item}")

            elif detection_mode == "medical":
                medical_results, medical_boxes = detect_items(frame, medical_classes)
                update_accumulated_counts(medical_results, accumulated_medical_counts, medical_boxes)
                print("Accumulated medical detection results:", accumulated_medical_counts)

                # Send medical results via serial
                for item, count in accumulated_medical_counts.items():
                    send_serial_command(f"medical_{item}:{count}")

# Main function
if __name__ == '__main__':
    cv2.setUseOptimized(True)  # Enable OpenCV optimization
    cv2.setNumThreads(2)  # Use 2 threads for OpenCV

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Starting video stream. Waiting for Arduino input.")

    food_classes = {4: "sandwich", 1: "burger", 2: "hotdog"}
    medical_classes = {3: "napkin", 5: "syringe", 0: "bandage"}

    # Start threads
    capture_thread = threading.Thread(target=capture_frames, args=(cap, 4))  # Skip every 2nd frame
    process_thread = threading.Thread(target=process_frames, args=(food_classes, medical_classes))
    capture_thread.start()
    process_thread.start()

    # Wait for threads to finish
    capture_thread.join()
    process_thread.join()

    cap.release()