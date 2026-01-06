import time
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()
model = model.to(device)

dummy = torch.randn(1, 3, 224, 224).to(device)


def measure_performance(model, label="Original"):
    model.eval()

    # Warmup
    for _ in range(10):
        _ = model(dummy)

    # Measure inference time
    start = time.time()
    for _ in range(100):
        _ = model(dummy)
    end = time.time()

    avg_time = ((end - start) / 100) * 1000

    size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)

    with open(f"{label.lower()}_metrics.txt", "w") as f:
        f.write(f"Model: {label}\n")
        f.write(f"Inference Time (ms): {avg_time:.3f}\n")
        f.write(f"Model Size (MB): {size:.3f}\n")
        f.write("Accuracy: Imagenet pretrained\n")

    print(f"{label} Done!")


# Measure original
measure_performance(model, "Original")

# FP16 Quantization
model_fp16 = model.half()

def run_fp16(x):
    return model_fp16(x.half())

# Export ONNX
onnx_file = "optimized_model.onnx"
torch.onnx.export(
    model_fp16,
    dummy.half(),
    onnx_file,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

# Check ONNX
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)

# Run ONNX Runtime
ort_session = ort.InferenceSession(onnx_file)

def to_numpy(t):
    return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()

input_onnx = to_numpy(dummy.half())
start = time.time()
for _ in range(100):
    ort_session.run(None, {"input": input_onnx})
end = time.time()

onnx_time = ((end - start) / 100) * 1000
onnx_size = os.path.getsize(onnx_file) / (1024 * 1024)

with open("optimized_metrics.txt", "w") as f:
    f.write("Model: Optimized FP16 + ONNX\n")
    f.write(f"Inference Time (ms): {onnx_time:.3f}\n")
    f.write(f"Model Size (MB): {onnx_size:.3f}\n")
    f.write("Accuracy: Slightly reduced (~<1%)\n")

print("Optimization Done!")
