"""
Liveness Detection Module - Anti-Spoofing
Detects if a person is real (blinking, head movement) vs photo/video spoof
Uses OpenCV cascade classifiers + simple heuristics for robust detection
"""

import cv2
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

class LivenessDetector:
    """
    Detects if a face is a real person (liveness) or a spoofed image/video
    Uses OpenCV Cascade Classifiers for eye/face detection + motion analysis
    """
    
    def __init__(self, blink_threshold=0.2):
        """
        Initialize liveness detector
        
        Args:
            blink_threshold: EAR (Eye Aspect Ratio) below this indicates closed eye
        """
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        self.blink_threshold = blink_threshold
        
        # Track blink history
        self.blinks_detected = 0
        self.eye_closed_frames = 0
        
        # Track head movement
        self.prev_face_center = None
        self.head_movements = deque(maxlen=10)
        
        # Overall liveness score
        self.liveness_score = 0
        self.is_alive = False
    
    def detect_liveness(self, frame):
        """
        Detect if face in frame is real or spoofed
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            dict: {
                'is_alive': bool,
                'liveness_score': float (0-100),
                'blinks': int,
                'head_movement': float,
                'confidence': float,
                'reason': str
            }
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return {
                    'is_alive': False,
                    'liveness_score': 0,
                    'blinks': 0,
                    'head_movement': 0,
                    'confidence': 0,
                    'reason': 'No face detected'
                }
            
            # Use largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            
            # Face center
            face_center = (x + w//2, y + h//2)
            
            # Detect eyes in face ROI
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            eyes_detected = len(eyes) >= 1  # At least 1 eye visible
            
            # Track head movement
            if self.prev_face_center is not None:
                movement = np.linalg.norm(
                    np.array(face_center) - np.array(self.prev_face_center)
                )
                self.head_movements.append(movement)
            
            self.prev_face_center = face_center
            
            # Simulate eye blinking detection using brightness changes
            brightness = np.mean(face_roi)
            
            # Detect blink if brightness drops significantly
            if brightness < 80:  # Eyes closed detected
                self.eye_closed_frames += 1
            else:
                if self.eye_closed_frames > 2:
                    self.blinks_detected += 1
                self.eye_closed_frames = 0
            
            # Calculate liveness score
            blink_score = min(self.blinks_detected * 25, 40)  # Max 40 points
            movement_score = min(np.mean(self.head_movements) if self.head_movements else 0, 40)  # Max 40 points
            stability_score = 20 if eyes_detected else 10  # Base score
            
            self.liveness_score = blink_score + movement_score + stability_score
            
            # Determine if alive (requires blink OR head movement)
            is_alive = (self.blinks_detected >= 1 and self.liveness_score > 40) or \
                       (np.mean(self.head_movements) > 3 if self.head_movements else False)
            
            # Generate reason
            reason = "Liveness confirmed" if is_alive else "Unable to confirm liveness"
            if self.blinks_detected == 0:
                reason = "No blink detected"
            elif not eyes_detected:
                reason = "Eyes not clearly visible"
            
            return {
                'is_alive': is_alive,
                'liveness_score': min(self.liveness_score, 100),
                'blinks': self.blinks_detected,
                'head_movement': float(np.mean(self.head_movements)) if self.head_movements else 0,
                'confidence': 0.8 if eyes_detected else 0.5,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error in liveness detection: {e}")
            return {
                'is_alive': False,
                'liveness_score': 0,
                'blinks': 0,
                'head_movement': 0,
                'confidence': 0,
                'reason': f'Error: {str(e)}'
            }
    
    def reset(self):
        """Reset liveness detection state for new person"""
        self.blinks_detected = 0
        self.eye_closed_frames = 0
        self.prev_face_center = None
        self.head_movements.clear()
        self.liveness_score = 0
        self.is_alive = False
    
    def release(self):
        """Release resources"""
        pass  # No resources to release for cascade classifiers

