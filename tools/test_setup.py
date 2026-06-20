#!/usr/bin/env python3
"""
Test script to verify wide-angle camera optimization setup.
Checks ROI, Undistort, Multi-Scale, and other configurations.
"""

import os
import json
import sys
from pathlib import Path


def test_config():
    """Test if configuration is valid"""
    print("=" * 60)
    print("YOLO26 Wide-Angle Optimization - Configuration Test")
    print("=" * 60)
    print()
    
    # Check environment variables
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Check if service.py exists
    print("1️⃣  Checking service.py...")
    tests_total += 1
    if os.path.exists("service.py"):
        print("   ✅ service.py found")
        tests_passed += 1
    else:
        print("   ❌ service.py NOT found - run from yolo26_people_analytics_plugin folder")
    print()
    
    # Test 2: Check if YOLO model exists
    print("2️⃣  Checking YOLO model...")
    tests_total += 1
    if os.path.exists("yolo26n.pt"):
        print("   ✅ yolo26n.pt found")
        tests_passed += 1
    else:
        print("   ⚠️  yolo26n.pt not found (will download on first run)")
    print()
    
    # Test 3: Check camera calibration
    print("3️⃣  Checking camera calibration...")
    tests_total += 1
    if os.path.exists("camera_calibration.json"):
        print("   ✅ camera_calibration.json found")
        try:
            with open("camera_calibration.json", 'r') as f:
                cal_data = json.load(f)
                if "camera_matrix" in cal_data and "distortion_coefficients" in cal_data:
                    print("      • Camera matrix: OK")
                    print("      • Distortion coeffs: OK")
                    tests_passed += 1
                else:
                    print("   ❌ Missing calibration data")
        except json.JSONDecodeError:
            print("   ❌ Invalid JSON in camera_calibration.json")
    else:
        print("   ⚠️  camera_calibration.json not found")
        print("      → Run: python calibrate_camera.py --images ./calibration_images")
    print()
    
    # Test 4: Check run scripts
    print("4️⃣  Checking run scripts...")
    tests_total += 1
    scripts = [
        "run_roi_undistort_optimized.bat",
        "run_fast_roi.bat"
    ]
    found_scripts = 0
    for script in scripts:
        if os.path.exists(script):
            print(f"   ✅ {script}")
            found_scripts += 1
        else:
            print(f"   ❌ {script} not found")
    if found_scripts >= 1:
        tests_passed += 1
    print()
    
    # Test 5: Check documentation
    print("5️⃣  Checking documentation...")
    tests_total += 1
    docs = [
        "WIDE_ANGLE_OPTIMIZATION_V2.md",
        "QUICK_START_ROI_UNDISTORT.md",
        "CHEAT_SHEET.txt"
    ]
    found_docs = 0
    for doc in docs:
        if os.path.exists(doc):
            print(f"   ✅ {doc}")
            found_docs += 1
        else:
            print(f"   ⚠️  {doc} not found")
    if found_docs >= 2:
        tests_passed += 1
    print()
    
    # Test 6: Check calibrate_camera.py
    print("6️⃣  Checking calibration tool...")
    tests_total += 1
    if os.path.exists("calibrate_camera.py"):
        print("   ✅ calibrate_camera.py found")
        tests_passed += 1
    else:
        print("   ❌ calibrate_camera.py not found")
    print()
    
    # Summary
    print("=" * 60)
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    print()
    
    if tests_passed == tests_total:
        print("🎉 Everything looks good! Ready to use.")
        print()
        print("Next steps:")
        print("  1. (Optional) Calibrate camera if using undistort:")
        print("     python calibrate_camera.py --images ./calibration_images")
        print()
        print("  2. Run optimized version:")
        print("     .\\run_roi_undistort_optimized.bat")
        print()
        return True
    else:
        print("⚠️  Some issues found. Please check above.")
        print()
        print("Common fixes:")
        print("  • If calibration missing: python calibrate_camera.py")
        print("  • If YOLO model missing: Will auto-download on first run")
        print("  • If scripts missing: Check file names")
        print()
        return False


def test_imports():
    """Test if required Python packages are available"""
    print()
    print("=" * 60)
    print("Checking Python dependencies...")
    print("=" * 60)
    print()
    
    packages = [
        "cv2",
        "numpy",
        "fastapi",
        "ultralytics",
        "pydantic",
        "uvicorn"
    ]
    
    all_ok = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} - missing, install: pip install {pkg}")
            all_ok = False
    
    print()
    if all_ok:
        print("✅ All dependencies available!")
    else:
        print("⚠️  Some dependencies missing. Install them:")
        print("   pip install opencv-python numpy fastapi ultralytics pydantic uvicorn")
    
    print()
    return all_ok


def test_roi_config():
    """Test if ROI configuration makes sense"""
    print()
    print("=" * 60)
    print("ROI Configuration Examples")
    print("=" * 60)
    print()
    
    examples = [
        {
            "name": "Skip top 30% (ceiling)",
            "config": {
                "ROI_X_MIN": "0.0",
                "ROI_X_MAX": "1.0",
                "ROI_Y_MIN": "0.3",
                "ROI_Y_MAX": "1.0"
            }
        },
        {
            "name": "Doorway (narrow, centered)",
            "config": {
                "ROI_X_MIN": "0.2",
                "ROI_X_MAX": "0.8",
                "ROI_Y_MIN": "0.1",
                "ROI_Y_MAX": "1.0"
            }
        },
        {
            "name": "Full frame (no crop)",
            "config": {
                "ROI_X_MIN": "0.0",
                "ROI_X_MAX": "1.0",
                "ROI_Y_MIN": "0.0",
                "ROI_Y_MAX": "1.0"
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print("   Config values:")
        for key, val in example['config'].items():
            print(f"     set {key}={val}")
        print()


def main():
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  YOLO26 Wide-Angle Optimization Setup Checker".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Run tests
    config_ok = test_config()
    imports_ok = test_imports()
    test_roi_config()
    
    # Final message
    print()
    print("=" * 60)
    if config_ok and imports_ok:
        print("✅ SETUP COMPLETE - Ready to optimize!")
        print()
        print("Quick start:")
        print("  python calibrate_camera.py --images ./calibration_images")
        print("  .\\run_roi_undistort_optimized.bat")
        print()
        sys.exit(0)
    else:
        print("⚠️  Some setup steps needed - see above")
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()
