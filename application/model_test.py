import cv2
import numpy as np
import ncnn

# Initialize the NCNN Net
net = ncnn.Net()
net.load_param("/home/sst/IDC25G6/Grp6_IDC2025/ml/models/best4.param")  # Update with the path to your .param file
net.load_model("/home/sst/IDC25G6/Grp6_IDC2025/ml/models/best4.bin")   # Update with the path to your .bin file

# Define the input size expected by the YOLOv8 model
input_size = 640  # Update this if your model uses a different input size

# Define a helper function for preprocessing the frame
def preprocess(frame, input_ssize):
    h, w, c = frame.shape
    scale = min(input_size / w, input_size / h)
    resized_w = int(w * scale)
    resized_h = int(h * scale)
    resized = cv2.resize(frame, (resized_w, resized_h))
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[:resized_h, :resized_w, :] = resized
    return canvas, scale, resized_w, resized_h

# Define a helper function to draw bounding boxes
def draw_bboxes(frame, detections, scale, resized_w, resized_h):
    h, w, _ = frame.shape
    for det in detections:
        x_min = int(det[0] / scale)
        y_min = int(det[1] / scale)
        x_max = int(det[2] / scale)
        y_max = int(det[3] / scale)
        confidence = det[4]
        label = int(det[5])
        
        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Add label and confidence
        cv2.putText(
            frame,
            f"Class {label} {confidence:.2f}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    return frame

# Define the main function
def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        input_frame, scale, resized_w, resized_h = preprocess(frame, input_size)
        
        # Convert frame to NCNN Mat
        ncnn_mat = ncnn.Mat.from_pixels(input_frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, input_size, input_size)
        
        # Inference
        ex = net.create_extractor()
        ex.input("images", ncnn_mat)
        ret, out = ex.extract("output0")  # Update "data" and "output" based on your YOLOv8 model's input & output names
        
        # Parse detections
        detections = []
        for i in range(out.h):
            values = out.row(i)
            x_min, y_min, x_max, y_max, confidence, label = values[:6]
            if confidence > 0.05:  # Set confidence threshold
                detections.append([x_min, y_min, x_max, y_max, confidence, label])
        
        # Draw bounding boxes
        frame = draw_bboxes(frame, detections, scale, resized_w, resized_h)
        
        # Display the frame
        cv2.imshow("YOLOv8 NCNN Camera Test", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()