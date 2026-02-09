#!/usr/bin/env python3
"""
FLOW 1: RTSP Camera Ingest Service
==================================

Purpose:
  Test and demonstrate FLOW 1 pipeline:
  RTSP Camera → Frame Ingest → AI Inference → Log Results

This service:
  1. Connects to an RTSP stream (with auto-reconnect)
  2. Processes frames at configurable FPS
  3. Calls existing AI service (/infer endpoint)
  4. Logs detections and fall events
  5. Optionally saves frames and shows preview

Usage:
  python rtsp_ingest_service.py [--rtsp URL] [--camera-id ID] [--fps FPS]

Author: SafeAging FLOW 1 Testing
Date: February 9, 2026
"""

import os
import sys
import cv2
import time
import logging
import argparse
import threading
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass

# ============================================================
# ENV LOADER (.env)
# ============================================================

def load_local_env() -> None:
    """Load KEY=VALUE pairs from .env (repo root or current directory)."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        script_dir.parent / ".env",
    ]

    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Keep startup resilient if .env has a malformed line.
        pass

load_local_env()

# ============================================================
# CONFIGURATION SECTION - EDIT HERE FOR YOUR SETUP
# ============================================================

# RTSP stream URL (supports credentials)
RTSP_URL = os.getenv("RTSP_URL", "")

# Camera identifier
CAMERA_ID = "flow1_public_cam"

# Target FPS for processing (lower = skip more frames)
TARGET_FPS = 5

# Maximum frames to queue while processing
MAX_QUEUE_SIZE = 2

# AI Service endpoint (existing /infer)
AI_SERVICE_URL = "http://127.0.0.1:18000/infer"

# Frame preprocessing
FRAME_DOWNSCALE_WIDTH = 640  # Resize to 640px width for faster inference
FRAME_ENCODE_QUALITY = 80    # JPEG quality (1-100)

# Output options
ENABLE_PREVIEW = False       # Show cv2.imshow live preview (set True to enable)
SAVE_FRAMES = False          # Save frames to disk (set True to enable)
SAVE_FRAMES_DIR = "rtsp_frames"  # Directory to save frames
SAVE_FRAME_INTERVAL = 30    # Save 1 frame every N inference calls

# Reconnect settings
RTSP_CONNECT_TIMEOUT = 5    # Timeout for RTSP connection attempt
RTSP_RECONNECT_DELAYS = [1, 2, 5, 10, 30]  # Backoff delays in seconds

# Logging
LOG_LEVEL = logging.INFO
LOG_FILE = "rtsp_ingest_service.log"
LOG_TO_FILE = False  # False = console only, no log file on disk

# ============================================================
# END CONFIGURATION
# ============================================================


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logger(name: str, log_file: str = LOG_FILE, level=LOG_LEVEL) -> logging.Logger:
    """Setup logger. By default, logs go to console only."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    if LOG_TO_FILE and log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger('FLOW1_RTSP_INGEST')


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Detection:
    """Represents a single detection from AI service."""
    cls: str
    score: float
    x: float
    y: float
    w: float
    h: float
    track_id: int
    fall_detected: bool


@dataclass
class FrameJob:
    """Frame job for processing."""
    frame: Any  # numpy array (BGR)
    timestamp: float
    frame_index: int
    camera_id: str


# ============================================================
# RTSP INGEST SERVICE
# ============================================================

class RTSPIngestService:
    """
    FLOW 1: RTSP camera ingest service.
    
    Ingests RTSP streams, processes frames at target FPS,
    calls AI inference, and logs results.
    """
    
    def __init__(
        self,
        rtsp_url: str,
        camera_id: str,
        target_fps: int = 5,
        ai_service_url: str = AI_SERVICE_URL,
        enable_preview: bool = False,
        save_frames: bool = False,
    ):
        """Initialize RTSP ingest service."""
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_interval_ms = 1000 / target_fps  # Milliseconds between frames
        self.ai_service_url = ai_service_url
        self.enable_preview = enable_preview
        self.save_frames = save_frames
        
        # State
        self.is_running = False
        self.cap = None
        self.frame_index = 0
        self.detection_count = 0
        self.fall_count = 0
        self.last_frame_time = time.time()
        
        # Thread control
        self.stop_event = threading.Event()
        self.ingest_thread = None
        self.process_thread = None
        self.frame_queue = []
        self.frame_queue_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'frames_ingested': 0,
            'frames_dropped': 0,
            'frames_processed': 0,
            'detections_total': 0,
            'falls_detected': 0,
            'errors': 0,
            'ai_service_errors': 0,
        }
        self.stats_lock = threading.Lock()
        
        # Create output directory if needed
        if self.save_frames:
            Path(SAVE_FRAMES_DIR).mkdir(exist_ok=True)
            logger.info(f"Frame save directory: {SAVE_FRAMES_DIR}")
    
    # ========================================================
    # RTSP Connection
    # ========================================================
    
    def connect_rtsp(self) -> bool:
        """
        Connect to RTSP stream with error handling.
        
        Returns:
            True if connected, False otherwise
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        try:
            logger.info(f"[RTSP] Attempting connection to: {self.rtsp_url}")
            
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
            
            # Try to read one frame to verify connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to read initial frame from RTSP")
                self.cap.release()
                self.cap = None
                return False
            
            logger.info(f"[RTSP] Successfully connected. Resolution: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            logger.error(f"[RTSP] Connection error: {type(e).__name__}: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def reconnect_with_backoff(self):
        """Reconnect to RTSP with exponential backoff."""
        attempt = 0
        while not self.stop_event.is_set():
            delay = RTSP_RECONNECT_DELAYS[min(attempt, len(RTSP_RECONNECT_DELAYS) - 1)]
            logger.warning(f"[RTSP] Reconnecting in {delay}s (attempt {attempt + 1})...")
            
            # Wait with stop event check
            if self.stop_event.wait(delay):
                return  # Stop requested
            
            if self.connect_rtsp():
                return  # Success
            
            attempt += 1
    
    # ========================================================
    # FRAME INGESTION
    # ========================================================
    
    def ingest_frames(self):
        """
        Main RTSP frame ingestion loop.
        Runs in background thread.
        """
        logger.info("[INGEST] Frame ingestion thread started")
        
        while not self.stop_event.is_set():
            # Check connection
            if self.cap is None or not self.cap.isOpened():
                logger.warning("[INGEST] RTSP connection lost, attempting reconnect...")
                self.reconnect_with_backoff()
                continue
            
            try:
                # Read frame from RTSP
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.error("[INGEST] Failed to read frame from RTSP")
                    self.reconnect_with_backoff()
                    continue
                
                # FPS throttling - check if enough time has passed
                now = time.time()
                elapsed_ms = (now - self.last_frame_time) * 1000
                
                if elapsed_ms < self.frame_interval_ms:
                    # Skip frame, not enough time elapsed
                    with self.stats_lock:
                        self.stats['frames_dropped'] += 1
                    continue
                
                self.last_frame_time = now
                
                # Create frame job
                job = FrameJob(
                    frame=frame.copy(),
                    timestamp=now,
                    frame_index=self.frame_index,
                    camera_id=self.camera_id,
                )
                
                # Enqueue frame (bounded queue)
                with self.frame_queue_lock:
                    if len(self.frame_queue) >= MAX_QUEUE_SIZE:
                        # Queue full, drop oldest
                        old_job = self.frame_queue.pop(0)
                        with self.stats_lock:
                            self.stats['frames_dropped'] += 1
                        logger.debug(f"[INGEST] Frame queue full, dropped frame #{old_job.frame_index}")
                    
                    self.frame_queue.append(job)
                
                self.frame_index += 1
                
                with self.stats_lock:
                    self.stats['frames_ingested'] += 1
                
                if self.frame_index % 50 == 0:
                    logger.debug(f"[INGEST] Ingested {self.frame_index} frames")
                
            except Exception as e:
                logger.error(f"[INGEST] Error in frame ingestion: {type(e).__name__}: {e}")
                with self.stats_lock:
                    self.stats['errors'] += 1
                self.reconnect_with_backoff()
        
        logger.info("[INGEST] Frame ingestion thread stopped")
    
    # ========================================================
    # AI INFERENCE
    # ========================================================
    
    def call_ai_service(self, frame: Any, camera_id: str) -> Tuple[List[Detection], Optional[Exception], Optional[Any]]:
        """
        Call existing AI service /infer endpoint.
        
        Args:
            frame: numpy BGR frame
            camera_id: camera identifier
        
        Returns:
            Tuple of (detections list, exception or None, frame_sent_to_ai)
        """
        try:
            # Resize frame for faster inference.
            # Keep the exact frame sent to AI so drawn boxes align with output coords.
            if FRAME_DOWNSCALE_WIDTH > 0 and frame.shape[1] > FRAME_DOWNSCALE_WIDTH:
                scale = FRAME_DOWNSCALE_WIDTH / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                frame_resized = cv2.resize(frame, (FRAME_DOWNSCALE_WIDTH, new_h))
            else:
                frame_resized = frame.copy()
            
            # Encode frame to JPEG
            ret, jpeg_bytes = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, FRAME_ENCODE_QUALITY])
            if not ret:
                return [], Exception("Failed to encode frame to JPEG"), None
            
            # Base64 encode JPEG
            b64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
            
            # Create request
            payload = {
                'camera_id': camera_id,
                'image': b64_image,
            }
            
            # POST to /infer
            response = requests.post(
                self.ai_service_url,
                json=payload,
                timeout=5.0,  # 5 second timeout
            )
            
            if response.status_code != 200:
                error_msg = f"AI service returned status {response.status_code}: {response.text[:100]}"
                return [], Exception(error_msg), None
            
            # Parse response
            detections_raw = response.json()
            
            if not isinstance(detections_raw, list):
                return [], Exception(f"Expected JSON array, got {type(detections_raw)}"), None
            
            # Convert to Detection objects
            detections = []
            for det_dict in detections_raw:
                try:
                    detection = Detection(
                        cls=det_dict.get('cls', 'unknown'),
                        score=float(det_dict.get('score', 0.0)),
                        x=float(det_dict.get('x', 0.0)),
                        y=float(det_dict.get('y', 0.0)),
                        w=float(det_dict.get('w', 0.0)),
                        h=float(det_dict.get('h', 0.0)),
                        track_id=int(det_dict.get('track_id', 0)),
                        fall_detected=bool(det_dict.get('fall_detected', False)),
                    )
                    detections.append(detection)
                except Exception as e:
                    logger.warning(f"Failed to parse detection: {e}")
                    continue
            
            return detections, None, frame_resized
            
        except requests.exceptions.Timeout:
            return [], Exception("AI service timeout (5s)"), None
        except requests.exceptions.ConnectionError:
            return [], Exception("Cannot connect to AI service"), None
        except Exception as e:
            return [], e, None

    def draw_detections(self, frame: Any, detections: List[Detection]) -> Any:
        """Draw bbox + labels onto frame for local debug/preview."""
        if frame is None:
            return frame

        drawn = frame.copy()
        h, w = drawn.shape[:2]

        for det in detections:
            if det.cls != "person":
                continue

            x1 = max(0, min(int(det.x), w - 1))
            y1 = max(0, min(int(det.y), h - 1))
            x2 = max(0, min(int(det.x + det.w), w - 1))
            y2 = max(0, min(int(det.y + det.h), h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            is_fall = bool(det.fall_detected)
            color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)

            label = f"ID={det.track_id} {det.score:.2f}"
            if is_fall:
                label += " FALL"

            text_y = y1 - 8 if y1 > 20 else y1 + 18
            cv2.putText(
                drawn,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return drawn
    
    # ========================================================
    # FRAME PROCESSING
    # ========================================================
    
    def process_frames(self):
        """
        Main frame processing loop.
        Dequeues frames and calls AI inference.
        Runs in background thread.
        """
        logger.info("[PROCESS] Frame processing thread started")
        
        save_frame_count = 0
        
        while not self.stop_event.is_set():
            # Dequeue frame
            job = None
            with self.frame_queue_lock:
                if len(self.frame_queue) > 0:
                    job = self.frame_queue.pop(0)
            
            if job is None:
                # No frames in queue, wait a bit
                time.sleep(0.01)
                continue
            
            # Process frame
            try:
                # Call AI service
                detections, ai_error, frame_sent_to_ai = self.call_ai_service(job.frame, job.camera_id)
                
                if ai_error:
                    logger.error(f"[PROCESS] AI service error: {ai_error}")
                    with self.stats_lock:
                        self.stats['ai_service_errors'] += 1
                    continue
                
                # Log results
                persons = [d for d in detections if d.cls == 'person']
                falls = [d for d in detections if d.fall_detected]
                
                with self.stats_lock:
                    self.stats['frames_processed'] += 1
                    self.stats['detections_total'] += len(detections)
                    self.stats['falls_detected'] += len(falls)
                
                # Log to console
                log_msg = f"[PROCESS] Frame #{job.frame_index} | Persons: {len(persons)}"
                if len(persons) > 0:
                    person_ids = ', '.join([str(p.track_id) for p in persons])
                    log_msg += f" (IDs: {person_ids})"
                
                if len(falls) > 0:
                    log_msg += f" | ⚠️  FALLS: {len(falls)}"
                    for fall_det in falls:
                        log_msg += f" [ID={fall_det.track_id}, score={fall_det.score:.2f}]"
                
                logger.info(log_msg)

                # Draw bboxes on the same frame size that was sent to AI service.
                frame_for_output = frame_sent_to_ai if frame_sent_to_ai is not None else job.frame
                annotated_frame = self.draw_detections(frame_for_output, detections)
                
                # Save frame if enabled (sample)
                if self.save_frames:
                    save_frame_count += 1
                    if save_frame_count % SAVE_FRAME_INTERVAL == 0:
                        self.save_frame(annotated_frame, len(persons), len(falls))
                
                # Show preview if enabled
                if self.enable_preview:
                    self.show_preview(annotated_frame, persons, falls)
                
            except Exception as e:
                logger.error(f"[PROCESS] Error processing frame: {type(e).__name__}: {e}")
                with self.stats_lock:
                    self.stats['errors'] += 1
        
        logger.info("[PROCESS] Frame processing thread stopped")
    
    # ========================================================
    # OUTPUT / PREVIEW
    # ========================================================
    
    def save_frame(self, frame: Any, num_persons: int, num_falls: int):
        """Save frame to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{SAVE_FRAMES_DIR}/frame_{timestamp}_p{num_persons}_f{num_falls}.jpg"
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.debug(f"[OUTPUT] Saved frame: {filename}")
        except Exception as e:
            logger.error(f"[OUTPUT] Failed to save frame: {e}")
    
    def show_preview(self, frame: Any, persons: List[Detection], falls: List[Detection]):
        """Show live preview with detections (no drawing, just title)."""
        try:
            # Create title
            title = f"FLOW1 - Persons: {len(persons)} | Falls: {len(falls)}"
            
            # Resize for display
            display_frame = frame.copy()
            if display_frame.shape[1] > 1280:
                scale = 1280 / display_frame.shape[1]
                new_h = int(display_frame.shape[0] * scale)
                display_frame = cv2.resize(display_frame, (1280, new_h))
            
            # Show window
            cv2.imshow(title, display_frame)
            
            # Wait for keypress (1ms)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("[PREVIEW] User quit (pressed 'q')")
                self.stop_event.set()
            
        except Exception as e:
            logger.error(f"[PREVIEW] Error showing preview: {e}")
    
    # ========================================================
    # SERVICE CONTROL
    # ========================================================
    
    def start(self):
        """Start the ingest service."""
        if self.is_running:
            logger.warning("Service already running")
            return
        
        logger.info("=" * 70)
        logger.info("FLOW 1: RTSP Ingest Service Starting")
        logger.info("=" * 70)
        logger.info(f"RTSP URL: {self.rtsp_url}")
        logger.info(f"Camera ID: {self.camera_id}")
        logger.info(f"Target FPS: {self.target_fps}")
        logger.info(f"AI Service: {self.ai_service_url}")
        logger.info(f"Preview: {self.enable_preview}")
        logger.info(f"Save Frames: {self.save_frames}")
        logger.info("=" * 70)
        
        self.is_running = True
        self.stop_event.clear()
        
        # Connect to RTSP
        if not self.connect_rtsp():
            self.reconnect_with_backoff()
        
        # Start threads
        self.ingest_thread = threading.Thread(target=self.ingest_frames, daemon=False)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=False)
        
        self.ingest_thread.start()
        self.process_thread.start()
        
        logger.info("Service started successfully")
    
    def stop(self):
        """Stop the ingest service."""
        logger.info("Stopping service...")
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.ingest_thread and self.ingest_thread.is_alive():
            self.ingest_thread.join(timeout=5)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5)
        
        # Close RTSP connection
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Close preview window
        cv2.destroyAllWindows()
        
        self.is_running = False
        logger.info("Service stopped")
    
    def print_stats(self):
        """Print service statistics."""
        with self.stats_lock:
            logger.info("=" * 70)
            logger.info("SERVICE STATISTICS")
            logger.info("=" * 70)
            logger.info(f"Frames Ingested:      {self.stats['frames_ingested']}")
            logger.info(f"Frames Dropped:       {self.stats['frames_dropped']}")
            logger.info(f"Frames Processed:     {self.stats['frames_processed']}")
            logger.info(f"Detections Total:     {self.stats['detections_total']}")
            logger.info(f"Falls Detected:       {self.stats['falls_detected']}")
            logger.info(f"Errors:               {self.stats['errors']}")
            logger.info(f"AI Service Errors:    {self.stats['ai_service_errors']}")
            
            if self.stats['frames_ingested'] > 0:
                drop_rate = self.stats['frames_dropped'] / self.stats['frames_ingested'] * 100
                logger.info(f"Drop Rate:            {drop_rate:.1f}%")
            
            logger.info("=" * 70)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='FLOW 1: RTSP Camera Ingest Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage with defaults
  python rtsp_ingest_service.py

  # Custom RTSP URL and FPS
  python rtsp_ingest_service.py --rtsp rtsp://192.168.1.100:554/stream --fps 10

  # Enable preview
  python rtsp_ingest_service.py --preview

  # Save frames
  python rtsp_ingest_service.py --save-frames

  # All options
  python rtsp_ingest_service.py --rtsp rtsp://admin:pass@192.168.1.100:554/stream \\
                                  --camera-id my_camera \\
                                  --fps 5 \\
                                  --ai-service http://localhost:18000/infer \\
                                  --preview \\
                                  --save-frames
        '''
    )
    
    parser.add_argument('--rtsp', default=RTSP_URL, help=f'RTSP URL (default: {RTSP_URL})')
    parser.add_argument('--camera-id', default=CAMERA_ID, help=f'Camera ID (default: {CAMERA_ID})')
    parser.add_argument('--fps', type=int, default=TARGET_FPS, help=f'Target FPS (default: {TARGET_FPS})')
    parser.add_argument('--ai-service', default=AI_SERVICE_URL, help=f'AI service URL (default: {AI_SERVICE_URL})')
    parser.add_argument('--preview', action='store_true', help='Enable live preview')
    parser.add_argument('--save-frames', action='store_true', help='Save frames to disk')
    
    args = parser.parse_args()
    if not args.rtsp:
        parser.error("RTSP URL is required. Set RTSP_URL in .env or pass --rtsp.")
    
    # Create and start service
    service = RTSPIngestService(
        rtsp_url=args.rtsp,
        camera_id=args.camera_id,
        target_fps=args.fps,
        ai_service_url=args.ai_service,
        enable_preview=args.preview,
        save_frames=args.save_frames,
    )
    
    service.start()
    
    # Main loop
    try:
        while service.is_running:
            time.sleep(1)
            
            # Print stats periodically
            if service.frame_index % 100 == 0 and service.frame_index > 0:
                logger.info(
                    f"Progress: {service.stats['frames_ingested']} ingested, "
                    f"{service.stats['frames_processed']} processed, "
                    f"{service.stats['falls_detected']} falls"
                )
    
    except KeyboardInterrupt:
        logger.info("\nCtrl+C detected, shutting down...")
    finally:
        service.stop()
        time.sleep(0.5)
        service.print_stats()


if __name__ == '__main__':
    main()
