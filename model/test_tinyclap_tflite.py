
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent

ONNX = ROOT / "tinyclap_phinet.onnx"
TFLITE = Path(r"X:\model\tinyclap_no_flex\tinyclap_phinet_float32.tflite")


# Entrada de teste reproducivel
rng = np.random.default_rng(42)
x = rng.standard_normal((1, 827, 64)).astype(np.float32)

# ONNX
session = ort.InferenceSession(
    str(ONNX),
    providers=["CPUExecutionProvider"]
)

onnx_input = session.get_inputs()[0]

assert tuple(onnx_input.shape) == (1, 827, 64)

output_onnx = session.run(
    None,
    {onnx_input.name: x}
)[0]

# TensorFlow Lite
interpreter = Interpreter(model_path=str(TFLITE))
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

assert tuple(input_info["shape"]) == (1, 827, 64)

x_tflite = x.copy()

interpreter.set_tensor(input_info["index"], x_tflite)
interpreter.invoke()

output_tflite = interpreter.get_tensor(output_info["index"])

# Comparacao
assert output_onnx.shape == output_tflite.shape == (1, 1024)

diff = np.abs(output_onnx - output_tflite)

print("\n=== COMPARACAO ONNX vs TFLITE ===")
print("ONNX:", output_onnx.shape)
print("TFLite:", output_tflite.shape)
print("Diferenca maxima:", diff.max())
print("Diferenca media:", diff.mean())
print(
    "Equivalencia (tol=1e-4):",
    np.allclose(output_onnx, output_tflite, atol=1e-4, rtol=1e-4)
)
