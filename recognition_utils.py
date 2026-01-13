import cv2
import numpy as np
import face_recognition
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FaceRecognitionUtils:
    def __init__(self):
        # Initialize cascade classifier for faster initial detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.detection_model = "hog"  # Default to HOG for better performance
        self.frame_skip = 0  # Process every frame initially
        self.last_frame = None
        self.last_encodings = None
        self.last_locations = None
        
        # Camera preprocessing settings (Recommended Standards)
        self.color_space = 'RGB'  # Standard RGB for skin-tone segmentation
        self.gamma = 2.2  # Standard gamma to prevent dark mid-tones
        self.brightness = 0.5  # Balanced (50%) to avoid clipping
        
        # CLAHE for preprocessing (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def apply_gamma_correction(self, frame, gamma=2.2):
        """
        Apply gamma correction to prevent dark mid-tones
        Gamma = 2.2 (standard) prevents image from being too dark
        """
        try:
            # Build a lookup table mapping pixel values [0, 255] to adjusted gamma values
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            
            # Apply gamma correction using the lookup table
            return cv2.LUT(frame, table)
        except Exception as e:
            logger.error(f"Error applying gamma correction: {str(e)}")
            return frame
    
    def apply_clahe(self, frame):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Prevents over-amplification of noise while improving contrast
        """
        try:
            # Convert to LAB color space for CLAHE application
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel only (brightness)
            l_clahe = self.clahe.apply(l)
            
            # Merge and convert back to RGB
            lab_clahe = cv2.merge([l_clahe, a, b])
            frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            
            return frame_clahe
        except Exception as e:
            logger.error(f"Error applying CLAHE: {str(e)}")
            return frame
    
    def adjust_brightness(self, frame, brightness=0.5):
        """
        Adjust brightness to balanced level (50%)
        Avoids clipping (losing detail in highlights/shadows)
        """
        try:
            # Balanced brightness adjustment
            if brightness > 0.5:
                # Increase brightness slightly
                alpha = brightness  # Contrast control (1.0 = no change)
                beta = (brightness - 0.5) * 50  # Brightness control
                return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            else:
                # Decrease brightness slightly
                alpha = brightness * 2.0 if brightness > 0 else 0.1
                return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        except Exception as e:
            logger.error(f"Error adjusting brightness: {str(e)}")
            return frame
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for optimal performance with recommended settings:
        - Color Space: RGB (Standard) for skin-tone segmentation
        - Gamma: 2.2 (Standard) to prevent dark mid-tones
        - Brightness: Balanced (50%) to avoid clipping
        - CLAHE: Contrast Limited Adaptive Histogram Equalization
        """
        try:
            # Step 1: Convert BGR to RGB (camera captures in BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Step 2: Apply gamma correction (2.2 standard)
            frame_gamma = self.apply_gamma_correction(frame_rgb, gamma=self.gamma)
            
            # Step 3: Adjust brightness to balanced level (50%)
            frame_brightness = self.adjust_brightness(frame_gamma, brightness=self.brightness)
            
            # Step 4: Apply CLAHE for enhanced contrast
            frame_clahe = self.apply_clahe(frame_brightness)
            
            # Step 5: Resize for faster processing if needed
            height, width = frame_clahe.shape[:2]
            if width > 640:
                scale = 640 / width
                frame_clahe = cv2.resize(frame_clahe, None, fx=scale, fy=scale, 
                                        interpolation=cv2.INTER_LINEAR)
            
            return frame_clahe
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {str(e)}")
            # Return original frame converted to RGB if preprocessing fails
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect_and_encode_faces(self, frame):
        """
        Detect faces using face_recognition library directly for better accuracy
        """
        try:
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use face_recognition library directly for better accuracy
            # This is more reliable than cascade classifier
            face_locations = face_recognition.face_locations(
                rgb_frame, model='hog', number_of_times_to_upsample=1
            )
            
            if len(face_locations) == 0:
                return [], []
            
            # Get face encodings for detected faces
            face_encodings = []
            valid_locations = []
            
            for face_location in face_locations:
                try:
                    # Get face encoding with minimal jitter for speed
                    face_encoding = face_recognition.face_encodings(
                        rgb_frame, [face_location], num_jitters=1, model="small"
                    )
                    if face_encoding:
                        face_encodings.append(face_encoding[0])
                        valid_locations.append(face_location)
                except Exception as e:
                    logger.error(f"Error encoding face: {str(e)}")
                    continue
            
            return valid_locations, face_encodings
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return [], []

    @staticmethod
    def compare_faces(known_encodings, face_encoding, tolerance=0.5):
        """
        Compare faces with adaptive tolerance
        """
        try:
            # Get face distances
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if len(face_distances) == 0:
                return False, None, None
            
            # Find best match
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            # Adaptive tolerance based on distance
            if min_distance < tolerance:
                matches = face_recognition.compare_faces([known_encodings[best_match_index]], face_encoding, tolerance=tolerance)
                if matches[0]:
                    return True, best_match_index, min_distance
            
            return False, None, None
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return False, None, None

    @staticmethod
    def draw_face_box(frame, face_location, name, confidence=None):
        """
        Draw improved face box and labels
        """
        try:
            top, right, bottom, left = face_location
            
            # Draw face box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw background for text
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            
            # Add name and confidence if available
            label = name
            if confidence is not None:
                label = f"{name} ({confidence:.2%})"
            
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        except Exception as e:
            logger.error(f"Error drawing face box: {str(e)}")