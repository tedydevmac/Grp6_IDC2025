import numpy as np
import ncnn
import torch
import cv2

def draw_bounding_boxes(frame, detections):
    """
    Draw bounding boxes and labels on the frame.
    :param frame: The video frame.
    :param detections: List of detections, each containing (x, y, w, h, confidence).
    """
    for detection in detections:
        x, y, w, h, confidence = detection
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def test_camera_inference():
    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        with ncnn.Net() as net:
            net.load_param("/Users/tedgoh/Grp6_IDC2025/ml/models/best5ncnn/best5.ncnn.param")
            net.load_model("/Users/tedgoh/Grp6_IDC2025/ml/models/best5ncnn/best5.ncnn.bin")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                # Preprocess the frame
                frame_resized = cv2.resize(frame, (640, 640))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                in0 = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                with net.create_extractor() as ex:
                    ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

                    _, out0 = ex.extract("out0")
                    output = torch.from_numpy(np.array(out0)).unsqueeze(0)

                # Parse the output to get bounding boxes and confidence scores
                # Replace this with your model's actual output parsing logic
                detections = [
                    (50, 50, 100, 100, 0.85),  # Example detection: (x, y, w, h, confidence)
                    (200, 150, 120, 120, 0.92)
                ]

                # Clone the original frame to avoid drawing on the same frame repeatedly
                frame_copy = frame.copy()

                # Draw bounding boxes on the cloned frame
                draw_bounding_boxes(frame_copy, detections)

                # Display the frame
                cv2.imshow("Camera Feed", frame_copy)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_inference()