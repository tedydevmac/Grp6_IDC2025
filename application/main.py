#!/usr/bin/env python3
import cv2
import time
import serial
from ultralytics import YOLO

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
    if ser:
        try:
            ser.write(command.encode())
            print("Sent command:", command)
        except Exception as e:
            print("Error sending command:", e)
    else:
        print("Serial port not available. Cannot send command.")

# ---------------------------
# Load YOLOv8 Model
# ---------------------------
# Replace this path with the location of your trained YOLOv8 model.
model_path = "path/to/single_model.pt"  # Path to the single YOLOv8 model

print("Loading YOLOv8 model...")
model = YOLO(model_path)
print("Model loaded.")

# ---------------------------
# Global Variables for Accumulated Counts
# ---------------------------
accumulated_medical_counts = {"napkin": 0, "syringe": 0, "bandage": 0}

# ---------------------------
# Reset Accumulated Counts
# ---------------------------
def reset_accumulated_counts():
    global accumulated_medical_counts
    accumulated_medical_counts = {"napkin": 0, "syringe": 0, "bandage": 0}

# ---------------------------
# Update Accumulated Counts
# ---------------------------
def update_accumulated_counts(detection_results, accumulated_counts):
    for item, count in detection_results.items():
        accumulated_counts[item] += count

# ---------------------------
# Detection Functions
# ---------------------------
def detect_items(frame, item_classes):
    """
    Detect and count items in the frame using YOLOv8.
    Filters out duplicate detections of the same object.
    item_classes: A dictionary mapping class indices to item names.
    Returns a dictionary with counts for each item.
    """
    results = model.predict(frame, conf=0.5)  # Adjust confidence threshold as needed
    item_counts = {item: 0 for item in item_classes.values()}

    detected_boxes = []  # Store processed bounding boxes to avoid duplicates

    for result in results:
        for box in result.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            if cls in item_classes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                box_area = (x2 - x1) * (y2 - y1)

                # Check for duplicate detections
                is_duplicate = False
                for existing_box in detected_boxes:
                    iou = calculate_iou((x1, y1, x2, y2), existing_box)
                    if iou > 0.5:  # IoU threshold for duplicate detection
                        is_duplicate = True
                        break

                if not is_duplicate:
                    detected_boxes.append((x1, y1, x2, y2))
                    item_counts[item_classes[cls]] += 1

    return item_counts

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

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# ---------------------------
# Check for Arduino Input
# ---------------------------
def check_arduino_input():
    if (ser and ser.in_waiting > 0):
        try:
            input_data = ser.readline().decode('utf-8').strip()
            print("Received from Arduino:", input_data)
            return input_data
        except Exception as e:
            print("Error reading from Arduino:", e)
    return None

# ---------------------------
# Main Loop: Capture and Process Frames
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)  # Open the default camera; change if necessary.
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting video stream. Waiting for Arduino input.")

    food_classes = {0: "sandwich", 1: "hotdog", 2: "burger"}
    medical_classes = {0: "napkin", 1: "syringe", 2: "bandage"}

    detection_mode = None  # Initialize detection mode

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Frame", frame)

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
            print("Stopping program as per Arduino command...")
            break

        # Perform detection based on the current mode
        if detection_mode == "food":
            food_results = detect_items(frame, food_classes)
            for item, count in food_results.items():
                if count > 0:  # If any food item is detected
                    send_serial_command(f"food_{item}")
                    print(f"Detected and sent food item: {item}")

        elif detection_mode == "medical":
            medical_results = detect_items(frame, medical_classes)
            update_accumulated_counts(medical_results, accumulated_medical_counts)
            print("Accumulated medical detection results:", accumulated_medical_counts)

            # Send medical results via serial
            for item, count in accumulated_medical_counts.items():
                send_serial_command(f"medical_{item}:{count}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
