import os
from ultralytics import YOLO

def get_model() -> YOLO:
    onnx_file = "yolov8n_int8.onnx"
    if not os.path.exists(onnx_file):
        model = YOLO('yolov8n.pt')
        onnx_file = model.export(format='onnx', quantize=8, simplify=True, data='coco8.yaml')

    return YOLO(onnx_file, task='detect')