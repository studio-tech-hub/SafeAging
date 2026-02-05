#!/usr/bin/env python3
"""
Test script for YOLOv8 Analytics Service

Usage:
    python test_service.py --help
    python test_service.py --health
    python test_service.py --status
    python test_service.py --infer <image_path>
"""

import argparse
import sys
import time
import json
import base64
import requests
from pathlib import Path

# Service config
SERVICE_URL = "http://127.0.0.1:18000"

def print_header(title):
    """Print a nice header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_health():
    """Test health endpoint"""
    print_header("Health Check")
    
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service is healthy")
            print(f"   Uptime: {data['service_uptime_seconds']:.1f} seconds")
            print(f"   Timestamp: {data['timestamp']}")
            return True
        else:
            print(f"❌ Service returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {SERVICE_URL}")
        print(f"   Make sure service is running: python service.py")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def test_status():
    """Test status endpoint"""
    print_header("Service Status")
    
    try:
        response = requests.get(f"{SERVICE_URL}/status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"Service: {data['service']}")
            print(f"Status: {data['status']}")
            print(f"Uptime: {data['uptime_seconds']:.1f}s")
            print(f"Total Requests: {data['total_requests']}")
            print(f"Total Errors: {data['total_errors']}")
            print(f"Error Rate: {data['error_rate']:.2f}%")
            print(f"Active Cameras: {data['active_cameras']}")
            print(f"Model: {data['model']}")
            print(f"\nCameras:")
            
            for cam_id, cam_data in data['cameras'].items():
                print(f"\n  {cam_id}:")
                print(f"    Tracks: {cam_data['tracks']}")
                print(f"    Unique Persons: {cam_data['unique_persons']}")
                print(f"    Avg Inference: {cam_data['avg_inference_ms']:.1f}ms")
                print(f"    Created: {cam_data['created_at']}")
            
            return True
        else:
            print(f"❌ Service returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {SERVICE_URL}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def test_infer(image_path):
    """Test inference endpoint"""
    print_header(f"Inference Test - {image_path}")
    
    # Read image file
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        print(f"Reading image: {image_path.name}")
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        # Encode to base64
        print(f"Image size: {len(img_bytes)} bytes")
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        print(f"Base64 encoded: {len(b64)} bytes")
        
        # Send request
        print(f"Sending request to {SERVICE_URL}/infer...")
        start_time = time.time()
        
        response = requests.post(
            f"{SERVICE_URL}/infer",
            json={
                "image": b64,
                "camera_id": "test_camera"
            },
            timeout=10
        )
        
        elapsed = time.time() - start_time
        
        print(f"Response received in {elapsed*1000:.1f}ms")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            detections = response.json()
            print(f"\n✅ Success! Detections: {len(detections)}")
            
            for i, det in enumerate(detections, 1):
                print(f"\n  Detection #{i}:")
                print(f"    Class: {det['cls']}")
                print(f"    Confidence: {det['score']:.2%}")
                print(f"    Position: ({det['x']:.1f}, {det['y']:.1f})")
                print(f"    Size: {det['w']:.1f}x{det['h']:.1f}")
                print(f"    Track ID: {det['track_id']}")
            
            return True
        else:
            print(f"❌ Service returned status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout (>10 seconds)")
        print(f"   Service may be processing a large image or model is slow")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {SERVICE_URL}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def test_all():
    """Run all tests"""
    print_header("Running All Tests")
    
    results = {
        "Health": test_health(),
        "Status": test_status(),
    }
    
    # Try to find a test image
    test_images = [
        "test_image.jpg",
        "test_image.png",
        Path.cwd().parent / "build" / "yolov8n.onnx",  # Not ideal but just checking
    ]
    
    found_image = None
    for img_path in test_images:
        if Path(img_path).exists():
            found_image = str(img_path)
            break
    
    if found_image:
        results["Inference"] = test_infer(found_image)
    else:
        print("\n⚠️ Skipping inference test (no test image found)")
        print(f"   Place a test image (test_image.jpg) in the current directory")
    
    # Summary
    print_header("Test Summary")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    return all_passed

def main():
    parser = argparse.ArgumentParser(
        description="Test YOLOv8 Analytics Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_service.py --health          # Test health endpoint
  python test_service.py --status          # Test status endpoint
  python test_service.py --infer image.jpg # Test inference
  python test_service.py --all             # Run all tests
        """
    )
    
    parser.add_argument(
        "--health",
        action="store_true",
        help="Test health endpoint"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Test status endpoint"
    )
    parser.add_argument(
        "--infer",
        type=str,
        help="Test inference endpoint with image file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    # If no arguments, run all tests
    if not any([args.health, args.status, args.infer, args.all]):
        args.all = True
    
    # Run tests
    if args.health:
        success = test_health()
    elif args.status:
        success = test_status()
    elif args.infer:
        success = test_infer(args.infer)
    elif args.all:
        success = test_all()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
