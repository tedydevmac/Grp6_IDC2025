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
# Load YOLOv8 Models
# ---------------------------
# Replace these paths with the locations of your trained YOLOv8 models.
food_model_path = "path/to/food_model.pt"      # Model for food items (sandwich, hotdog, burger)
medical_model_path = "path/to/medical_model.pt"  # Model for medical items (gauze, antiseptic, bandage)

print("Loading YOLOv8 food model...")
food_model = YOLO(food_model_path)
print("Food model loaded.")

print("Loading YOLOv8 medical model...")
medical_model = YOLO(medical_model_path)
print("Medical model loaded.")

# ---------------------------
# Detection Functions
# ---------------------------
def detect_food(frame):
    """
    Detect food objects in the frame using YOLOv8.
    Assumes:
      - Class 0: sandwich
      - Class 1: hotdog
      - Class 2: burger
    Returns a dictionary with counts for each food item.
    """
    results = food_model.predict(frame, conf=0.5)  # adjust confidence threshold as needed
    food_counts = {"sandwich": 0, "hotdog": 0, "burger": 0}
    
    # The results object can include one or more prediction sets.
    # Here we assume results[0] gives detections for the frame.
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            # Update counts based on class index
            if cls == 0:
                food_counts["sandwich"] += 1
            elif cls == 1:
                food_counts["hotdog"] += 1
            elif cls == 2:
                food_counts["burger"] += 1

    return food_counts

def detect_medical(frame):
    """
    Detect and count medical items in the frame using YOLOv8.
    Assumes:
      - Class 0: gauze
      - Class 1: antiseptic
      - Class 2: bandage
    Returns a dictionary with counts for each medical item.
    """
    results = medical_model.predict(frame, conf=0.5)  # adjust confidence threshold as needed
    medical_counts = {"gauze": 0, "antiseptic": 0, "bandage": 0}
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            if cls == 0:
                medical_counts["gauze"] += 1
            elif cls == 1:
                medical_counts["antiseptic"] += 1
            elif cls == 2:
                medical_counts["bandage"] += 1

    return medical_counts

# ---------------------------
# Main Loop: Capture and Process Frames
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)  # Open the default camera; change if necessary.
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting video stream. Press 'f' for food detection, 'm' for medical detection, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            food_results = detect_food(frame)
            print("Food detection results:", food_results)
            # Example: send a command based on detected food items
            if food_results["sandwich"] > 0:
                send_serial_command("sandwich")
            elif food_results["hotdog"] > 0:
                send_serial_command("hotdog")
            elif food_results["burger"] > 0:
                send_serial_command("burger")
                
        elif key == ord('m'):
            medical_results = detect_medical(frame)
            print("Medical detection results:", medical_results)
            # Send serial command if count of any item is below threshold (e.g., less than 3)
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
