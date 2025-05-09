import cv2
import numpy as np
import tensorflow as tf
import time

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="/Users/tedgoh/testing_ground/bestV11_3_saved_model/bestV11_3_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]
input_dtype = input_details[0]['dtype']

# Setup webcam
cap = cv2.VideoCapture(0)

fps = 0
prev_time = time.time()
conf_threshold = 0.5


while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (input_width, input_height))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    if input_dtype == np.uint8:
        input_data = np.expand_dims(rgb_frame, axis=0).astype(np.uint8)
    elif input_dtype == np.int8:
        input_data = np.expand_dims((rgb_frame.astype(np.float32) - 128) / 128, axis=0).astype(np.int8)
    else:
        input_data = np.expand_dims(rgb_frame.astype(np.float32) / 255.0, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve and process output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Output data shape:", output_data.shape)
    print("Sample output data:", output_data[:5])

    # Use provided label map for class names
    class_names = ['bandage', 'burger', 'hotdog', 'napkin', 'sandwich', 'syringe']

    # YOLOv11-style post-processing for TFLite with 3-scale output
    h_orig, w_orig = frame.shape[:2]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def decode_predictions(output, input_dim, num_classes, anchors, stride, conf_threshold):
        grid_size = int(np.sqrt(output.shape[0] / len(anchors)))
        output = output.reshape((grid_size, grid_size, len(anchors), 5 + num_classes))
        boxes = []
        for y in range(grid_size):
            for x in range(grid_size):
                for a in range(len(anchors)):
                    bx, by, bw, bh, obj = output[y, x, a, :5]
                    scores = output[y, x, a, 5:]
                    class_id = np.argmax(scores)
                    score = sigmoid(obj) * sigmoid(scores[class_id])
                    if score > conf_threshold:
                        anchor_w, anchor_h = anchors[a]
                        cx = (sigmoid(bx) + x) * stride
                        cy = (sigmoid(by) + y) * stride
                        w = np.exp(bw) * anchor_w
                        h = np.exp(bh) * anchor_h
                        x0 = int((cx - w / 2) * w_orig / input_dim)
                        y0 = int((cy - h / 2) * h_orig / input_dim)
                        x1 = int((cx + w / 2) * w_orig / input_dim)
                        y1 = int((cy + h / 2) * h_orig / input_dim)
                        boxes.append((x0, y0, x1, y1, score, class_id))
        return boxes

    # YOLOv11 decoding logic
    anchors = [
        [(10, 13), (16, 30), (33, 23)],   # stride 8
        [(30, 61), (62, 45), (59, 119)],  # stride 16
        [(116, 90), (156, 198), (373, 326)]  # stride 32
    ]
    strides = [8, 16, 32]
    num_classes = len(class_names)
    predictions = output_data[0]  # shape (N, 5376)
    all_boxes = []
    split_size = predictions.shape[1] // 3
    for i, stride in enumerate(strides):
        anchor_set = anchors[i]
        pred = predictions[:, i * split_size:(i + 1) * split_size].flatten()
        boxes = decode_predictions(pred, input_width, num_classes, anchor_set, stride, conf_threshold)
        all_boxes.extend(boxes)

    for x0, y0, x1, y1, score, class_id in all_boxes:
        label_text = f"{class_names[class_id]}: {score:.2f}"
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x0, max(y0 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # FPS calculation
    end_time = time.time()
    fps = 1.0 / (end_time - start_time)
    fps_label = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the result
    cv2.imshow('TFLite Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()