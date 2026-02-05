#!/usr/bin/env python3
"""
Test YOLO model directly on frame_samples to diagnose detection issues
Run: python test_model.py
"""
import os
import glob
import cv2
import torch

# ============================
# PATCH torch.load FIRST before importing YOLO
# ============================
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

# NOW import YOLO
from ultralytics import YOLO

print("=" * 60)
print("YOLOv8 Model Diagnostic Test")
print("=" * 60)

# Test 1: Check available frame samples
print("\n[1] Checking frame samples...")
samples = sorted(glob.glob("frame_samples/*.jpg"))
if not samples:
    print("❌ No frame samples found. Run service first to generate samples.")
    print("   After running service, frame samples will be in frame_samples/ folder")
    exit(1)

print(f"✓ Found {len(samples)} frame samples")
print(f"  Latest: {samples[-1]}")

# Test 2: Load model
print("\n[2] Loading YOLO model...")
try:
    model = YOLO("yolov8n.pt")
    model.to('cpu')
    print(f"✓ Model loaded: yolov8n.pt")
    print(f"  Type: {type(model)}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Test 3: Test on latest frame sample
print("\n[3] Testing model on latest frame sample...")
latest_sample = samples[-1]
frame = cv2.imread(latest_sample)
if frame is None:
    print(f"❌ Failed to read {latest_sample}")
    exit(1)

h, w = frame.shape[:2]
print(f"✓ Frame loaded: {w}x{h}")

# Test with different confidence thresholds
print("\n[4] Testing different confidence thresholds...")
print("   (Lower confidence = more detections, but more false positives)")
print()

thresholds = [0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]

for conf in thresholds:
    try:
        results = model.predict(
            frame,
            conf=conf,
            iou=0.45,
            classes=[0],  # person only
            verbose=False,
            device='cpu'
        )
        
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        
        if num_detections > 0:
            print(f"  conf={conf:.2f} → {num_detections} persons detected")
            for i, box in enumerate(results[0].boxes):
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w_box = x2 - x1
                h_box = y2 - y1
                print(f"    Box {i+1}: {int(w_box)}x{int(h_box)} px, score={score:.3f}")
        else:
            print(f"  conf={conf:.2f} → NO DETECTIONS ❌")
    except Exception as e:
        print(f"  conf={conf:.2f} → Error: {e}")

# Test 5: Check model info
print("\n[5] Model Information:")
print(f"  Model: yolov8n.pt")
print(f"  Input size: 640x640 (auto-scales)")
print(f"  Classes: Person (class 0)")

print("\n" + "=" * 60)
print("DIAGNOSIS RESULTS:")
print("=" * 60)

# Final test with recommended settings
print("\nTesting with RECOMMENDED settings (conf=0.25)...")
results = model.predict(
    frame,
    conf=0.25,
    iou=0.45,
    classes=[0],
    verbose=False,
    device='cpu'
)
num_final = len(results[0].boxes) if results[0].boxes is not None else 0
print(f"Result: {num_final} persons detected")

if num_final == 0:
    print("\n⚠️  ISSUE DETECTED: Model not detecting people at all!")
    print("    Possible causes:")
    print("    1. yolov8n (nano) model too weak for your camera")
    print("    2. Input format incorrect (BGR vs RGB)")
    print("    3. Camera resolution too high/low")
    print("\n    RECOMMENDATIONS:")
    print("    - Try yolov8s.pt (small) instead of yolov8n.pt")
    print("    - Run: python -m ultralytics.yolo detect predict model=yolov8s.pt source=frame_samples/frame_000001.jpg")
else:
    print(f"\n✓ Model IS detecting people correctly at conf=0.25")
    print(f"  Recommendation: Use CONFIDENCE_THRESHOLD=0.25 in production")

print("\n" + "=" * 60)
print("To improve detection:")
print("=" * 60)
print("1. Copy frames from frame_samples/ to debug visually")
print("2. If no detections: Try yolov8s.pt (larger model)")
print("3. If still bad: Check camera angle/resolution is correct")
print("=" * 60)
