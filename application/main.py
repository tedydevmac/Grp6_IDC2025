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
# Main Loop: Capture and Process Frames
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)  # Open the default camera; change if necessary.
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting video stream. Press 'f' for food detection, 'm' for medical detection, or 'q' to quit.")

    food_classes = {0: "sandwich", 1: "hotdog", 2: "burger"}
    medical_classes = {0: "gauze", 1: "antiseptic", 2: "bandage"}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            food_results = detect_items(frame, food_classes)
            print("Food detection results:", food_results)
            if food_results["sandwich"] > 0:
                send_serial_command("sandwich")
            elif food_results["hotdog"] > 0:
                send_serial_command("hotdog")
            elif food_results["burger"] > 0:
                send_serial_command("burger")

        elif key == ord('m'):
            medical_results = detect_items(frame, medical_classes)
            print("Medical detection results:", medical_results)
            for item, count in medical_results.items():
                if count < 3:
                    send_serial_command(f"low_{item}")

        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
