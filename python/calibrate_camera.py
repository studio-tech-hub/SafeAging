#!/usr/bin/env python3
"""
Camera Calibration Utility for Wide-Angle Lenses

This script performs camera calibration using a chessboard pattern.
It generates camera_calibration.json which is used for undistortion.

Usage:
    python calibrate_camera.py --images ./calibration_images --output camera_calibration.json

Steps:
    1. Print a chessboard pattern (8x6 or 9x7 inner corners)
    2. Take 15-30 photos of the chessboard at different angles and distances
    3. Save images to a folder (e.g., ./calibration_images/)
    4. Run this script with the image folder
    5. Check the generated camera_calibration.json

References:
    - OpenCV Calibration: https://docs.opencv.org/4.5.0/d9/df8/tutorial_root.html
    - Chessboard pattern: print from images/calibration_checkerboard_8x6.png
"""

import cv2
import numpy as np
import json
import os
import argparse
import glob
from pathlib import Path


def calibrate_camera(image_folder, chessboard_size=(9, 6), output_file="camera_calibration.json"):
    """
    Calibrate camera using chessboard pattern images.
    
    Args:
        image_folder: Path to folder containing chessboard images
        chessboard_size: Inner corners of chessboard (width, height)
        output_file: Path to save camera_calibration.json
    """
    print(f"📷 Camera Calibration Utility")
    print(f"================================")
    print(f"Chessboard size (inner corners): {chessboard_size}")
    print(f"Looking for images in: {image_folder}")
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ...
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    
    # Get all images
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(image_folder, "*.png")))
    image_paths.extend(glob.glob(os.path.join(image_folder, "*.bmp")))
    
    if not image_paths:
        print(f"❌ No images found in {image_folder}")
        return False
    
    print(f"✅ Found {len(image_paths)} images")
    print()
    
    successful_images = 0
    img_shape = None
    
    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️  [{idx+1}/{len(image_paths)}] Failed to read: {image_path}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            successful_images += 1
            print(f"✅ [{idx+1}/{len(image_paths)}] Success: {os.path.basename(image_path)}")
        else:
            print(f"❌ [{idx+1}/{len(image_paths)}] No chessboard found: {os.path.basename(image_path)}")
    
    print()
    if successful_images < 3:
        print(f"❌ Need at least 3 successful images, found {successful_images}")
        return False
    
    print(f"🎯 Using {successful_images} / {len(image_paths)} images for calibration")
    print()
    print(f"⏳ Calibrating camera... (this may take a minute)")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if not ret:
        print("❌ Calibration failed!")
        return False
    
    # Calculate reprojection error
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        total_points += 1
    
    reprojection_error = total_error / total_points
    
    print(f"✅ Calibration successful!")
    print(f"Reprojection error: {reprojection_error:.4f}")
    print()
    
    # Save to JSON
    calibration_data = {
        "camera_model": "Calibrated camera",
        "resolution": list(img_shape),
        "calibration_method": "opencv_chessboard",
        "chessboard_size": list(chessboard_size),
        "num_images_used": successful_images,
        "reprojection_error": float(reprojection_error),
        "calibration_date": str(Path(image_paths[0]).stat().st_mtime),
        
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
    }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"💾 Saved to: {output_file}")
    print()
    print(f"📊 Camera Matrix:")
    print(f"  {camera_matrix[0]}")
    print(f"  {camera_matrix[1]}")
    print(f"  {camera_matrix[2]}")
    print()
    print(f"🔧 Distortion Coefficients: {dist_coeffs.flatten().tolist()}")
    print()
    print(f"✅ You can now enable ENABLE_UNDISTORT=true in your run script!")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Camera calibration utility')
    parser.add_argument('--images', type=str, default='./calibration_images',
                        help='Folder containing calibration images')
    parser.add_argument('--output', type=str, default='camera_calibration.json',
                        help='Output file for calibration data')
    parser.add_argument('--chessboard', type=int, nargs=2, default=[9, 6],
                        help='Chessboard inner corners (width height)')
    
    args = parser.parse_args()
    
    # Create example chessboard if needed
    if not os.path.exists(args.images):
        print(f"📁 Creating calibration images folder: {args.images}")
        os.makedirs(args.images, exist_ok=True)
        print()
        print(f"📸 Steps to calibrate your camera:")
        print(f"  1. Print a chessboard pattern (example: 9x6 inner corners)")
        print(f"     - Try: https://markhedleyjones.com/projects/calibration")
        print(f"  2. Take 15-30 photos of the printed chessboard")
        print(f"     - Vary angles, distances, and positions")
        print(f"     - Ensure chessboard is well-lit and in focus")
        print(f"  3. Save images (JPG/PNG) to: {args.images}/")
        print(f"  4. Run this script again")
        return
    
    calibrate_camera(args.images, tuple(args.chessboard), args.output)


if __name__ == '__main__':
    main()
