from ultralytics import YOLO  # Import the YOLO class
import os
import subprocess

def convert_pt_to_ncnn(pt_model_path, input_shape, onnx_model_path, ncnn_param_path, ncnn_bin_path):
    """
    Converts a YOLOv8 PyTorch .pt model to NCNN format.

    Args:
        pt_model_path (str): Path to the PyTorch .pt model.
        input_shape (tuple): Input shape for the model (e.g., (1, 3, 224, 224)).
        onnx_model_path (str): Path to save the intermediate ONNX model.
        ncnn_param_path (str): Path to save the NCNN .param file.
        ncnn_bin_path (str): Path to save the NCNN .bin file.
    """
    # Load the YOLOv8 model
    model = YOLO(pt_model_path)

    # Export the model to ONNX format
    model.export(format="onnx", imgsz=input_shape[2:], dynamic=False, simplify=True, opset=11)
    print(f"ONNX model saved to {onnx_model_path}")

    # Convert ONNX to NCNN using ncnnoptimize
    ncnn_tools_dir = os.getenv('NCNN_TOOLS_DIR', '/Users/tedgoh/Grp6_IDC2025/ncnn/build/tools')
    onnx2ncnn = os.path.join(ncnn_tools_dir, '/Users/tedgoh/Grp6_IDC2025/ncnn/build/tools/onnx/onnx2ncnn')
    ncnnoptimize = os.path.join(ncnn_tools_dir, '/Users/tedgoh/Grp6_IDC2025/ncnn/build/tools/ncnnoptimize')

    # Run onnx2ncnn
    subprocess.run([onnx2ncnn, onnx_model_path, ncnn_param_path, ncnn_bin_path], check=True)
    print(f"NCNN param and bin files saved to {ncnn_param_path} and {ncnn_bin_path}")

if __name__ == "__main__":
    convert_pt_to_ncnn(
        "/Users/tedgoh/Grp6_IDC2025/ml/models/best2.pt",
        (1, 3, 384, 640),
        "/Users/tedgoh/Grp6_IDC2025/ml/models/best2.onnx",
        "/Users/tedgoh/Grp6_IDC2025/ml/models/best2.param",
        "/Users/tedgoh/Grp6_IDC2025/ml/models/best2.bin"
    )