#!/usr/bin/env python3
import cv2
import numpy as np
import serial
import time
import ncnn

# ---------------------------
# Setup Serial Communication
# ---------------------------
try:
    ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)
    time.sleep(2)
    print("Serial port initialized.")
except Exception as e:
    print("Error initializing serial:", e)
    ser = None

def send_serial_command(cmd):
    if ser:
        try:
            ser.write(cmd.encode())
            print("Sent command:", cmd)
        except Exception as e:
            print("Error sending serial command:", e)
    else:
        print("Serial not available. Command not sent.")

# ---------------------------------------------------------
# Load ncnn Models for Food and Medical Object Detection
# ---------------------------------------------------------
# Update these file paths to the location where you saved your converted models.
# Each model requires a .param and .bin file.
food_param_path = "path/to/food_yolov5.param"
food_bin_path   = "path/to/food_yolov5.bin"

medical_param_path = "path/to/medical_yolov5.param"
medical_bin_path   = "path/to/medical_yolov5.bin"

# Initialize the ncnn network for food objects
food_net = ncnn.Net()
ret = food_net.load_param(food_param_path)
if ret != 0:
    print("Failed to load food model param:", ret)
ret = food_net.load_model(food_bin_path)
if ret != 0:
    print("Failed to load food model bin:", ret)
print("Food model loaded via ncnn.")

# Initialize the ncnn network for medical objects
medical_net = ncnn.Net()
ret = medical_net.load_param(medical_param_path)
if ret != 0:
    print("Failed to load medical model param:", ret)
ret = medical_net.load_model(medical_bin_path)
if ret != 0:
    print("Failed to load medical model bin:", ret)
print("Medical model loaded via ncnn.")

# ----------------------------
# Preprocessing Function
# ----------------------------
def letterbox(image, target_size=640):
    height, width = image.shape[:2]
    scale = target_size / max(width, height)
    new_w, new_h = int(width * scale), int(height * scale)
    resized = cv2.resize(image, (new_w, new_h))
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, left, top

# -------------------------------
# Generic ncnn Detection Function
# -------------------------------
def detect_with_ncnn(net, image, target_size=640):
    # Preprocess image: letterbox resize
    processed, scale, pad_x, pad_y = letterbox(image, target_size=target_size)
    
    # Convert processed image to ncnn Mat (assumes image is in BGR format)
    in_mat = ncnn.Mat.from_pixels(processed, ncnn.Mat.PIXEL_BGR, processed.shape[1], processed.shape[0])
    
    # Create an extractor for inference
    ex = net.create_extractor()
    ex.set_light_mode(True)
    
    # 'data' is typically the input blob name; adjust if your model expects a different name.
    ex.input("data", in_mat)
    
    # Extract the output blob; the name "output" is a common default but may differ
    ret, out = ex.extract("output")
    if ret != 0:
        print("Error during inference, ret =", ret)
        return [], scale, pad_x, pad_y

    # Convert the output to a NumPy array for further processing.
    # NOTE: The exact parsing depends on how your YOLO model output is structured.
    # Here we assume each detection consists of 6 values: [x, y, w, h, confidence, class]
    output_size = out.w  # for example, or use out.total() if available
    # For demonstration, assume out is a flat buffer of float32 values:
    detections = np.array(out).reshape(-1, 6)
    return detections, scale, pad_x, pad_y

# ------------------------------------------
# Detection Functions for Specific Tasks
# ------------------------------------------
def detect_food_ncnn(frame):
    detections, scale, pad_x, pad_y = detect_with_ncnn(food_net, frame)
    # Initialize counts for food items (assumed: 0=sandwich, 1=hotdog, 2=burger)
    counts = {"sandwich": 0, "hotdog": 0, "burger": 0}
    for det in detections:
        x, y, w, h, conf, cls = det
        if conf > 0.5:  # confidence threshold
            if int(cls) == 0:
                counts["sandwich"] += 1
            elif int(cls) == 1:
                counts["hotdog"] += 1
            elif int(cls) == 2:
                counts["burger"] += 1
    return counts

def detect_medical_ncnn(frame):
    detections, scale, pad_x, pad_y = detect_with_ncnn(medical_net, frame)
    # Initialize counts for medical items (assumed: 0=gauze, 1=antiseptic, 2=bandage)
    counts = {"gauze": 0, "antiseptic": 0, "bandage": 0}
    for det in detections:
        x, y, w, h, conf, cls = det
        if conf > 0.5:
            if int(cls) == 0:
                counts["gauze"] += 1
            elif int(cls) == 1:
                counts["antiseptic"] += 1
            elif int(cls) == 2:
                counts["bandage"] += 1
    return counts

# -----------------------
# Main Loop for Inference
# -----------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return

    print("Starting main loop with ncnn inference.")
    print("Press 'f' for food detection, 'm' for medical detection, 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            food_results = detect_food_ncnn(frame)
            print("Food detection results:", food_results)
            if food_results["sandwich"] > 0:
                send_serial_command("sandwich")
            elif food_results["hotdog"] > 0:
                send_serial_command("hotdog")
            elif food_results["burger"] > 0:
                send_serial_command("burger")
        elif key == ord('m'):
            medical_results = detect_medical_ncnn(frame)
            print("Medical detection results:", medical_results)
            for item, count in medical_results.items():
                if count < 3:
                    send_serial_command(f"low_{item}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
