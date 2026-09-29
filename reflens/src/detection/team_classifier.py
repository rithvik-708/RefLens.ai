import cv2
import numpy as np
from typing import Optional, Tuple
from sklearn.cluster import KMeans
from src.detection.types import TeamClassification

class TeamClassifier:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.is_fitted = False
        self.feature_buffer = []

    def extract_jersey_features(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Crops the upper/middle bounding box to isolate the jersey."""
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        
        crop_y1 = y1 + int(0.15 * h)
        crop_y2 = y1 + int(0.60 * h)
        
        # Boundary safety
        crop_y1 = max(0, min(crop_y1, frame.shape[0]))
        crop_y2 = max(0, min(crop_y2, frame.shape[0]))
        x1 = max(0, min(x1, frame.shape[1]))
        x2 = max(0, min(x2, frame.shape[1]))

        if crop_y2 <= crop_y1 or x2 <= x1:
            return None

        jersey_crop = frame[crop_y1:crop_y2, x1:x2]
        hsv_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate a 2D histogram for Hue and Saturation
        hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def classify(self, frame: np.ndarray, bbox: np.ndarray, track_id: int) -> TeamClassification:
        features = self.extract_jersey_features(frame, bbox)
        if features is None:
            return TeamClassification(team_id=None, confidence=0.0, jersey_features=np.zeros(256))

        # Dynamic clustering logic for MVP
        if not self.is_fitted:
            self.feature_buffer.append(features)
            # Wait for enough diverse players before locking clusters
            if len(self.feature_buffer) > 20: 
                self.kmeans.fit(self.feature_buffer)
                self.is_fitted = True
                self.feature_buffer.clear()
            return TeamClassification(team_id=None, confidence=0.0, jersey_features=features)
        
        team_id = int(self.kmeans.predict([features])[0])
        distances = self.kmeans.transform([features])[0]
        
        # Simple confidence based on distance ratio to the two centroids
        dist_ratio = min(distances) / (max(distances) + 1e-6)
        confidence = 1.0 - dist_ratio 

        return TeamClassification(team_id=team_id, confidence=confidence, jersey_features=features)
