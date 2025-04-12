#!/usr/bin/env python3
import cv2
import torch
import serial
import time
from torchvision import transforms  # (if you need additional image preprocessing)

# ---------------------------
# Serial (PySerial) Setup
# ---------------------------
# Adjust the serial port as appropriate (e.g., '/dev/ttyUSB0' or '/dev/ttyACM0')
try:
    ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)
    time.sleep(2)  # Allow time for connection to be established
    print("Serial connection established.")
except Exception as e:
    print("Error initializing serial communication:", e)
    ser = None

def send_serial_command(command):
    """
    Send a command (string) over serial to the Arduino.
    """
    if ser:
        try:
            ser.write(command.encode())
            print("Sent command:", command)
        except Exception as e:
            print("Failed to send command:", e)
    else:
        print("Serial port not initialized; cannot send command.")

# ----------------------------------
# Load YOLO models using Torch Hub
# ----------------------------------
# The following uses the ultralytics/yolov5 repository. Ensure your custom models
# are located at the provided paths. You can also convert your models to ncnn later for performance.
#
# Example: If you trained your food detector with three classes (sandwich, hotdog, burger)
# and saved it as "food_model.pt", update the path accordingly.
# Similarly, update "medical_model.pt" for the medical items model.

food_model_path = "path/to/food_model.pt"      # Replace with your actual model path
medical_model_path = "path/to/medical_model.pt"  # Replace with your actual model path

print("Loading food model...")
food_model = torch.hub.load('ultralytics/yolov5', 'custom', path=food_model_path, force_reload=True)
print("Food model loaded.")

print("Loading medical model...")
medical_model = torch.hub.load('ultralytics/yolov5', 'custom', path=medical_model_path, force_reload=True)
print("Medical model loaded.")

# -------------------------------------------------
# Detection functions for food and medical items
# -------------------------------------------------

def detect_food(frame):
    """
    Detect food objects in the frame.
    Assumes:
      - Class 0: sandwich
      - Class 1: hotdog
      - Class 2: burger
    Returns a dictionary with counts for each food item.
    """
    # Run inference
    results = food_model(frame)
    # results.xyxy[0] contains detections (x1, y1, x2, y2, confidence, class)
    detections = results.xyxy[0]
    
    # Initialize counters
    food_counts = {"sandwich": 0, "hotdog": 0, "burger": 0}
    
    for *box, conf, cls in detections:
        class_index = int(cls.item())
        if class_index == 0:
            food_counts["sandwich"] += 1
        elif class_index == 1:
            food_counts["hotdog"] += 1
        elif class_index == 2:
            food_counts["burger"] += 1

    return food_counts

def detect_medical(frame):
    """
    Detect and count medical items in the frame.
    Assumes:
      - Class 0: gauze
      - Class 1: antiseptic cream
      - Class 2: bandage
    Returns a dictionary with counts for each medical item.
    """
    results = medical_model(frame)
    detections = results.xyxy[0]
    
    # Initialize counters
    medical_counts = {"gauze": 0, "antiseptic": 0, "bandage": 0}
    
    for *box, conf, cls in detections:
        class_index = int(cls.item())
        if class_index == 0:
            medical_counts["gauze"] += 1
        elif class_index == 1:
            medical_counts["antiseptic"] += 1
        elif class_index == 2:
            medical_counts["bandage"] += 1

    return medical_counts

# ------------------------------------
# Main Loop: Capture and Process Frames
# ------------------------------------
def main():
    # Open the default camera (or provide an alternative video source)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting main loop. Press 'f' for food detection, 'm' for medical detection, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Optionally, resize or preprocess the frame here if required
        # frame = cv2.resize(frame, (640, 480))
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Use key presses to trigger different detection tasks:
        if key == ord('f'):
            # Perform food object detection
            food_results = detect_food(frame)
            print("Food detection results:", food_results)
            # Example: send a command based on detected food (modify as needed)
            if food_results["sandwich"] > 0:
                send_serial_command("sandwich")
            elif food_results["hotdog"] > 0:
                send_serial_command("hotdog")
            elif food_results["burger"] > 0:
                send_serial_command("burger")
                
        elif key == ord('m'):
            # Perform medical item detection and count
            medical_results = detect_medical(frame)
            print("Medical detection results:", medical_results)
            # Example: if any count is below a threshold (e.g., less than 3), send an alert
            for item, count in medical_results.items():
                if count < 3:
                    send_serial_command(f"low_{item}")
                
        elif key == ord('q'):
            print("Exiting loop.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
