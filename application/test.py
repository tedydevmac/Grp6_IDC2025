import numpy as np
import tensorflow as tf

# Load and prepare the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="/Users/tedgoh/testing_ground/bestV11_3_saved_model/bestV11_3_full_integer_quant.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Generate dummy data matching the model’s input shape/type
shape = input_details[0]['shape']
dtype = input_details[0]['dtype']
dummy_input = (np.random.rand(*shape) * 2 - 1).astype(np.int8)
# Run inference
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

# Fetch the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output shape:", output_data.shape)
print("Sample output:", output_data.flat[:10])