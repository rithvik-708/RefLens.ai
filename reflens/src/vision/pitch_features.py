import cv2
import numpy as np
from typing import List, Tuple
from loguru import logger

class PitchFeatureExtractor:
    def __init__(self):
        # HSV range for soccer pitch grass (tuneable based on lighting/stadium)
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([85, 255, 255])
        
    def extract_lines(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Extracts structural pitch lines using color segmentation and Hough transforms.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Color Segmentation
        mask_green = cv2.inRange(hsv_frame, self.lower_green, self.upper_green)
        
        # 2. Morphological Operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        
        # Mask out everything except the pitch, then find white lines within it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pitch_only = cv2.bitwise_and(gray, gray, mask=mask_cleaned)
        
        # Extract white pixels (lines) on the pitch
        _, white_mask = cv2.threshold(pitch_only, 200, 255, cv2.THRESH_BINARY)
        
        # 3. Edge Detection & Hough Lines
        edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50, 
            minLineLength=100, 
            maxLineGap=20
        )
        
        return lines if lines is not None else []
