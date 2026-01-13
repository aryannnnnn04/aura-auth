from recognition_utils import FaceRecognitionUtils
import cv2
import numpy as np
import face_recognition
import logging
from datetime import datetime
import time
import threading
import collections

class EnhancedFaceRecognitionSystem:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_employee_ids = []
        self.recent_detections = collections.deque(maxlen=10)
        self.camera = None
        self.lock = threading.Lock()
        self.face_utils = FaceRecognitionUtils()
        self.last_detection_time = {}  # To prevent duplicate detections
        self.detection_cooldown = 30  # Seconds between repeated detections
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces with error handling"""
        try:
            with self.lock:
                encodings, names, emp_ids = self.db_manager.get_employee_encodings()
                self.known_face_encodings = encodings
                self.known_face_names = names
                self.known_employee_ids = emp_ids
                self.logger.info(f"Loaded {len(self.known_face_encodings)} face encodings")
        except Exception as e:
            self.logger.error(f"Error loading known faces: {str(e)}")
    
    def process_frame(self, frame):
        """Process a single frame with improved recognition"""
        try:
            # Preprocess frame
            frame = self.face_utils.preprocess_frame(frame)
            
            # Detect and encode faces
            face_locations, face_encodings = self.face_utils.detect_and_encode_faces(frame)
            
            current_time = time.time()
            detections = []
            
            with self.lock:
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    match_found, best_match_idx, confidence = self.face_utils.compare_faces(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    name = "Unknown"
                    employee_id = None
                    
                    if match_found and best_match_idx is not None:
                        name = self.known_face_names[best_match_idx]
                        employee_id = self.known_employee_ids[best_match_idx]
                        
                        # Check cooldown period
                        if employee_id:
                            last_time = self.last_detection_time.get(employee_id, 0)
                            if current_time - last_time >= self.detection_cooldown:
                                # Mark attendance
                                self.db_manager.mark_attendance(employee_id)
                                self.last_detection_time[employee_id] = current_time
                                
                                # Add to recent detections
                                detection_event = {
                                    "name": name,
                                    "employee_id": employee_id,
                                    "time": datetime.now().strftime('%I:%M:%S %p'),
                                    "confidence": float(1 - confidence) if confidence else None
                                }
                                self.recent_detections.appendleft(detection_event)
                    
                    # Draw face box with confidence
                    confidence_score = None if confidence is None else (1 - confidence)
                    self.face_utils.draw_face_box(frame, face_location, name, confidence_score)
                    
                    detections.append({
                        "name": name,
                        "employee_id": employee_id,
                        "confidence": confidence_score
                    })
            
            return frame, detections
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame, []
    
    def release_camera(self):
        """Safely release camera resources"""
        try:
            with self.lock:
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
                    self.logger.info("Camera released successfully")
        except Exception as e:
            self.logger.error(f"Error releasing camera: {str(e)}")