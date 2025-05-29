import cv2
import time
import serial
from ultralytics import YOLO
import threading
from collections import deque
import glob
# ---------------------------
# Serial (PySerial) Setup
# ---------------------------
ports_to_try = [
    '/dev/ttyUSB0',
    '/dev/ttyUSB1',
    '/dev/ttyUSB2',
    '/dev/ttyUSB3',
    '/dev/tty',
    '/dev/tty.usbmodem*',
    '/dev/tty.usbserial*',  
    '/dev/ttyACM0',
    '/dev/ttyACM1',
    '/dev/ttyACM2',
    '/dev/ttyACM3',
    '/dev/ttyAMA0',
    '/dev/ttyAMA1',
    '/dev/ttyAMA2',
    '/dev/ttyAMA3',
    '/dev/ttyS0',
    '/dev/ttyS1',
    '/dev/ttyS2',
    '/dev/ttyS3',
]
ser = None
for port_pattern in ports_to_try:
    matching_ports = glob.glob(port_pattern)
    if matching_ports:
        try:
            ser = serial.Serial(matching_ports[0], baudrate=9600, timeout=1)
            print(f"Connected to Arduino on port: {ser.portstr}")
            break
        except Exception as e:
            print(f"Failed to connect to Arduino on port{matching_ports[0]}: {e}")

def send_serial_command(command: str):
    global last_command
    if ser and command != last_command:
        try:
            ser.write(command.encode())
            ser.flush()  # Ensure data is sent immediately
            last_command = command
            print("Sent to Arduino: ", command)
        except Exception as e:
            print(f"Error sending to Arduino: {e}")
            # Don't let serial errors stop the program
    else:
        if not ser:
            print("Serial port not available.")
        elif command == last_command:
            # Skip sending duplicate commands
            pass

# ---------------------------
# Load YOLO11n Model
# ---------------------------
model_path = "/home/sst/IDC25G6/Grp6_IDC2025/ml/models/best6_saved_model_512_55epochs/best_full_integer_quant.tflite"

print("Loading model...")
model = YOLO(model_path, task="detect")
print("Model loaded.")

# ---------------------------
# Global Variables for Accumulated Counts
# ---------------------------
accumulated_medical_counts = {"napkin": 0, "syringe": 0, "bandage": 0}
last_command = None
# Track which objects have been detected to avoid counting them multiple times
detected_medical_objects = []
# Track the last time we sent medical updates to Arduino
last_medical_update_time = 0

# ---------------------------
# Reset Accumulated Counts
# ---------------------------
def reset_accumulated_counts():
    global accumulated_medical_counts, detected_medical_objects
    accumulated_medical_counts = {"napkin": 0, "syringe": 0, "bandage": 0}
    detected_medical_objects = []  # Also reset the tracking of detected objects

# ---------------------------
# Cleanup function to prevent memory issues
# ---------------------------
def cleanup_detected_objects(max_objects=100):
    global detected_medical_objects
    if len(detected_medical_objects) > max_objects:
        # Keep only the most recent detections
        detected_medical_objects = detected_medical_objects[-max_objects:]

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
                    if old_item == item:
                        # Use both IoU and center distance for better accuracy
                        iou = calculate_iou(box, old_box)
                        center_dist = calculate_center_distance(box, old_box)
                        if iou > 0.15 and center_dist < 50:  # Lowered IoU threshold and added distance check
                            is_new_object = False
                            break
                
                if is_new_object:
                    new_detected_objects.append((box, item))
                    accumulated_counts[item] += 1
    
    # Update the list of detected objects
    detected_medical_objects.extend(new_detected_objects)
    
    # Clean up old detections periodically
    cleanup_detected_objects(max_objects=100)

# ---------------------------
# Detection Functions
# ---------------------------
def detect_items(frame, item_classes):
    """
    Detect and count items in the frame using model.
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

                # Check for duplicate detections using both IoU and center distance
                is_duplicate = any((calculate_iou(box_coords, existing_box) > 0.15 and 
                                  calculate_center_distance(box_coords, existing_box) < 30) 
                                  for existing_box, _ in detected_boxes)

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

def calculate_center_distance(box1, box2):
    """
    Calculate Euclidean distance between centers of two bounding boxes.
    box1, box2: Tuples of (x1, y1, x2, y2).
    Returns distance between centers.
    """
    # Calculate centers
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    
    # Calculate Euclidean distance
    distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
    return distance

# ---------------------------
# Check for Arduino Input
# ---------------------------
def check_arduino_input():
    if ser:
        try:
            if ser.in_waiting > 0:
                input_data = ser.readline().decode('utf-8', errors='replace').strip()
                print("Received from Arduino: ", input_data)
                return input_data
        except Exception as e:
            print("Error reading from Arduino: ", e)
            # Don't let serial errors stop the program
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
    global accumulated_medical_counts, last_medical_update_time
    detection_mode = "food"

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
            detected_boxes = []  # Initialize for visualization

            if detection_mode == "food":
                food_results, detected_boxes = detect_items(frame, food_classes)
                for item, count in food_results.items():
                    if count > 0:  # If any food item is detected
                        send_serial_command(f"food_{item}")
                        print(f"Detected and sent food item: {item}")

            elif detection_mode == "medical":
                medical_results, detected_boxes = detect_items(frame, medical_classes)
                update_accumulated_counts(medical_results, accumulated_medical_counts, detected_boxes)
                print("Accumulated medical detection results: ", accumulated_medical_counts)

                current_time = time.time()
                # Send medical results via serial every 3 seconds
                if current_time - last_medical_update_time >= 3:
                    for item, count in accumulated_medical_counts.items():
                        send_serial_command(f"medical_{item}:{count}\n")
                    last_medical_update_time = current_time

            # Visualization
            vis_frame = visualize_detections(frame, detected_boxes, detection_mode, 
                                           accumulated_medical_counts if detection_mode == "medical" else None)
            cv2.imshow("Object Detection", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    cv2.destroyAllWindows()

# ---------------------------
# Visualization Functions
# ---------------------------
def visualize_detections(frame, detected_boxes, mode, accumulated_counts=None):
    """Draw bounding boxes and labels on frame for visualization"""
    vis_frame = frame.copy()
    
    # Draw bounding boxes and labels
    for box, item_name in detected_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label background
        label = item_name
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_frame, (x1, y1-25), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add mode indicator
    cv2.putText(vis_frame, f"Mode: {mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add accumulated counts for medical mode
    if mode == "medical" and accumulated_counts:
        y_offset = 60
        for item, count in accumulated_counts.items():
            text = f"{item}: {count}"
            cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    return vis_frame

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