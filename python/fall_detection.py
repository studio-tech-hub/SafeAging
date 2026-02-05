"""
Fall Detection Module
Detects falls using velocity, angle change, aspect ratio, and position tracking.
Integrates with person tracking system.
"""

import math
import logging
from typing import Dict, Tuple, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class FallDetectionTracker:
    """
    Tracks person motion and detects fall events.
    Uses smoothed position/angle history to avoid false positives.
    """
    
    def __init__(
        self,
        person_id: int,
        velocity_threshold: float = 20.0,
        angle_change_threshold: float = 45.0,
        aspect_ratio_threshold: float = 1.5,
        position_history_size: int = 5,
        confidence_threshold: float = 0.8
    ):
        """
        Args:
            person_id: Unique person ID from tracker
            velocity_threshold: Vertical position change threshold (pixels)
            angle_change_threshold: Angle change threshold (degrees)
            aspect_ratio_threshold: Width/Height ratio threshold
            position_history_size: Number of frames to keep in history
            confidence_threshold: Minimum confidence for detection (0.8 or higher)
        """
        self.person_id = person_id
        self.velocity_threshold = velocity_threshold
        self.angle_change_threshold = angle_change_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.position_history_size = position_history_size
        self.confidence_threshold = confidence_threshold
        
        # Position tracking (center Y coordinate)
        self.position_history = deque(maxlen=position_history_size)
        # Angle tracking (body orientation)
        self.angle_history = deque(maxlen=position_history_size)
        # Aspect ratio tracking
        self.aspect_ratio_history = deque(maxlen=position_history_size)
        # Fall event tracking
        self.fall_detected = False
        self.fall_frame_count = 0
        self.last_detection_frame = 0
        
    def update(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        frame_idx: int
    ) -> bool:
        """
        Update tracker with new detection.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            confidence: Detection confidence (0-1)
            frame_idx: Current frame index
            
        Returns:
            True if fall detected in this frame
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        # Ignore very small detections
        if height < 10 or width < 10:
            return False
        
        # Calculate metrics
        center_y = (y1 + y2) // 2
        aspect_ratio = width / height if height > 0 else 0
        angle = math.degrees(math.atan2(height, width))
        
        # Store in history
        self.position_history.append(center_y)
        self.angle_history.append(angle)
        self.aspect_ratio_history.append(aspect_ratio)
        
        self.last_detection_frame = frame_idx
        
        # Need at least 2 frames to detect change
        if len(self.position_history) < 2:
            return False
        
        # Calculate metrics only if we have enough history
        velocity = abs(self.position_history[-1] - self.position_history[-2])
        angle_change = abs(self.angle_history[-1] - self.angle_history[-2]) if len(self.angle_history) >= 2 else 0
        
        # Smoothed aspect ratio (use average of last few frames)
        avg_aspect_ratio = sum(self.aspect_ratio_history) / len(self.aspect_ratio_history)
        
        # Fall detection heuristics:
        # 1. High velocity downward (person dropping)
        # 2. Large angle change (rotating/falling)
        # 3. Aspect ratio becomes more horizontal (lying down)
        # 4. Combination of above factors
        
        is_falling = False
        fall_reasons = []
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False
        
        # Check velocity (quick vertical movement)
        if velocity > self.velocity_threshold:
            is_falling = True
            fall_reasons.append(f"high_velocity({velocity:.1f})")
        
        # Check angle change (body orientation change)
        if angle_change > self.angle_change_threshold:
            is_falling = True
            fall_reasons.append(f"angle_change({angle_change:.1f}°)")
        
        # Check aspect ratio (becomes horizontal/wider)
        if avg_aspect_ratio > self.aspect_ratio_threshold:
            is_falling = True
            fall_reasons.append(f"aspect_ratio({avg_aspect_ratio:.2f})")
        
        # Additional check: height becomes significantly narrower (lying down)
        if len(self.aspect_ratio_history) >= 2:
            ratio_change = avg_aspect_ratio - self.aspect_ratio_history[0]
            if ratio_change > 0.5 and aspect_ratio > 1.2:
                is_falling = True
                fall_reasons.append(f"ratio_increase({ratio_change:.2f})")
        
        if is_falling:
            self.fall_detected = True
            self.fall_frame_count += 1
            logger.info(f"Fall detected for person {self.person_id}: {', '.join(fall_reasons)}")
        
        return is_falling
    
    def is_stale(self, current_frame: int, ttl: int = 30) -> bool:
        """Check if tracker hasn't been updated in a while (person lost)."""
        return (current_frame - self.last_detection_frame) > ttl
    
    def reset_fall(self):
        """Reset fall detection state."""
        self.fall_detected = False
        self.fall_frame_count = 0


class FallDetectionManager:
    """
    Manages fall detection for multiple people.
    Coordinates between person tracking and fall detection.
    """
    
    def __init__(
        self,
        velocity_threshold: float = 20.0,
        angle_change_threshold: float = 45.0,
        aspect_ratio_threshold: float = 1.5,
        confidence_threshold: float = 0.8,
        tracker_ttl: int = 30
    ):
        """
        Args:
            velocity_threshold: Vertical motion threshold (pixels)
            angle_change_threshold: Body angle change threshold (degrees)
            aspect_ratio_threshold: Aspect ratio threshold
            confidence_threshold: Min confidence for fall detection
            tracker_ttl: Frames to keep inactive trackers
        """
        self.velocity_threshold = velocity_threshold
        self.angle_change_threshold = angle_change_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.confidence_threshold = confidence_threshold
        self.tracker_ttl = tracker_ttl
        
        # Track persons by ID
        self.trackers: Dict[int, FallDetectionTracker] = {}
        self.current_frame = 0
        
    def update(
        self,
        detections: List[Dict],
        frame_idx: int
    ) -> Dict[int, bool]:
        """
        Update all trackers with current detections.
        
        Args:
            detections: List of dicts with 'track_id', 'bbox', 'confidence'
            frame_idx: Current frame index
            
        Returns:
            Dict mapping person_id -> fall_detected
        """
        self.current_frame = frame_idx
        fall_results = {}
        
        # Update existing trackers
        active_ids = set()
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id < 0:
                continue
                
            active_ids.add(track_id)
            
            # Create tracker if needed
            if track_id not in self.trackers:
                self.trackers[track_id] = FallDetectionTracker(
                    person_id=track_id,
                    velocity_threshold=self.velocity_threshold,
                    angle_change_threshold=self.angle_change_threshold,
                    aspect_ratio_threshold=self.aspect_ratio_threshold,
                    confidence_threshold=self.confidence_threshold
                )
            
            # Update tracker
            bbox = det.get('bbox', (0, 0, 0, 0))
            conf = det.get('confidence', 0.0)
            is_falling = self.trackers[track_id].update(bbox, conf, frame_idx)
            fall_results[track_id] = is_falling
        
        # Remove stale trackers
        stale_ids = [
            pid for pid, tracker in self.trackers.items()
            if tracker.is_stale(frame_idx, self.tracker_ttl)
        ]
        for pid in stale_ids:
            del self.trackers[pid]
        
        return fall_results
    
    def get_fall_detections(self) -> List[Tuple[int, Dict]]:
        """
        Get all persons currently detected as falling.
        
        Returns:
            List of (person_id, tracker_info) tuples
        """
        results = []
        for person_id, tracker in self.trackers.items():
            if tracker.fall_detected:
                results.append((person_id, {
                    'person_id': person_id,
                    'frame_count': tracker.fall_frame_count,
                    'position_history': list(tracker.position_history),
                    'angle_history': list(tracker.angle_history),
                }))
        return results
    
    def reset_fall(self, person_id: Optional[int] = None):
        """Reset fall state for person(s)."""
        if person_id is None:
            # Reset all
            for tracker in self.trackers.values():
                tracker.reset_fall()
        elif person_id in self.trackers:
            self.trackers[person_id].reset_fall()
    
    def get_stats(self) -> Dict:
        """Get statistics about current tracking."""
        return {
            'total_tracked': len(self.trackers),
            'total_fallen': sum(1 for t in self.trackers.values() if t.fall_detected),
            'current_frame': self.current_frame,
        }
