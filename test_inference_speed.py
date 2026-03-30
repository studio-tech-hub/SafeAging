#!/usr/bin/env python3
import requests
import cv2
import base64
import time
import numpy as np

# Test inference speed
url = "http://127.0.0.1:18000/infer"

# Create a dummy frame (480x640x3 BGR)
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Encode to JPEG
ret, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
b64_image = base64.b64encode(jpeg_bytes).decode('utf-8')

# Test inference
payload = {
    "camera_id": "test_cam",
    "image": b64_image
}

print(f"Testing inference speed...")
print(f"Frame size: {frame.shape}")
print(f"Payload size: {len(b64_image) / 1024:.1f} KB")
print()

# Warm up
try:
    r = requests.post(url, json=payload, timeout=30)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# 5 runs
times = []
for i in range(5):
    start = time.time()
    try:
        r = requests.post(url, json=payload, timeout=30)
        elapsed = time.time() - start
        times.append(elapsed)
        num_detections = len(r.json()) if r.status_code == 200 else 0
        print(f"Run {i+1}: {elapsed*1000:.0f}ms - {num_detections} detections- Status {r.status_code}")
    except Exception as e:
        print(f"Run {i+1}: ERROR - {e}")

if times:
    print()
    print(f"Average: {np.mean(times)*1000:.0f}ms")
    print(f"Min: {np.min(times)*1000:.0f}ms")
    print(f"Max: {np.max(times)*1000:.0f}ms")
    
    if np.mean(times) < 0.3:
        print("✅ Inference speed GOOD (<300ms)")
    elif np.mean(times) < 1.0:
        print("⚠️ Inference speed OK (300-1000ms)")
    else:
        print("❌ Inference speed SLOW (>1000ms)")
