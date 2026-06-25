import glob
import os
import time

import numpy as np
import onnxruntime as ort

import onnxruntime_qnn  # noqa: F401 — register QNN EP plugin

print("ort", ort.__version__)
print("providers", ort.get_available_providers())
root = os.path.dirname(ort.__file__)
htp = glob.glob(os.path.join(root, "**", "libQnnHtp.so"), recursive=True)
gpu = glob.glob(os.path.join(root, "**", "libQnnGpu.so"), recursive=True)
print("htp", htp[0] if htp else None)
print("gpu", gpu[0] if gpu else None)

model = "/app/models/yolo26n.onnx"
x = np.random.rand(1, 3, 640, 640).astype(np.float32)

s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
inp = s.get_inputs()[0].name
for _ in range(3):
    s.run(None, {inp: x})
t0 = time.perf_counter()
for _ in range(20):
    s.run(None, {inp: x})
print("cpu_ms", round((time.perf_counter() - t0) / 20 * 1000, 1))

if "QNNExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("no QNN EP")

for label, bp in [("htp", htp[0] if htp else None), ("gpu", gpu[0] if gpu else None)]:
    if not bp:
        continue
    try:
        opts = {"backend_path": bp}
        if label == "htp":
            opts["htp_performance_mode"] = "burst"
        sess = ort.InferenceSession(
            model,
            providers=["QNNExecutionProvider", "CPUExecutionProvider"],
            provider_options=[opts, {}],
        )
        print(label, "session", sess.get_providers())
        inp = sess.get_inputs()[0].name
        for _ in range(5):
            sess.run(None, {inp: x})
        t0 = time.perf_counter()
        for _ in range(20):
            sess.run(None, {inp: x})
        print(label + "_ms", round((time.perf_counter() - t0) / 20 * 1000, 1))
    except Exception as exc:
        print(label + "_fail", type(exc).__name__, str(exc)[:300])
