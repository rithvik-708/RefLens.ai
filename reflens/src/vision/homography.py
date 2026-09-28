import cv2
import numpy as np
import json
from typing import Tuple, Optional, Dict

class PitchHomography:
    def __init__(self):
        self.H_matrix: Optional[np.ndarray] = None
        
    def calibrate(self, image_points: np.ndarray, pitch_metric_points: np.ndarray) -> bool:
        """
        Computes the perspective transform matrix mapping (u, v) to metric (X, Y).
        image_points: Nx2 array of pixel coordinates (e.g., penalty box corners)
        pitch_metric_points: Nx2 array of real-world coordinates in meters
        """
        if len(image_points) < 4:
            raise ValueError("At least 4 points are required to compute Homography.")
            
        H, status = cv2.findHomography(image_points, pitch_metric_points, cv2.RANSAC, 5.0)
        
        if H is not None:
            self.H_matrix = H
            return True
        return False
        
    def load_from_json(self, json_path: str) -> Dict:
        """
        Loads calibration from JSON, computes H matrix, and returns reprojection errors.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        img_pts = []
        pitch_pts = []
        for pt in data.get("points", []):
            img_pts.append(pt["image"])
            pitch_pts.append(pt["pitch"])
            
        img_pts = np.array(img_pts, dtype=np.float32)
        pitch_pts = np.array(pitch_pts, dtype=np.float32)
        
        if not self.calibrate(img_pts, pitch_pts):
            raise ValueError("Homography computation failed.")
            
        # Compute reprojection errors
        projected = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), self.H_matrix).reshape(-1, 2)
        errors = np.linalg.norm(projected - pitch_pts, axis=1)
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        
        return {
            "mean_reprojection_error_m": mean_err,
            "max_reprojection_error_m": max_err,
            "pitch_dimensions": {
                "length": data.get("pitch_length_m", 105.0),
                "width": data.get("pitch_width_m", 68.0)
            },
            "num_points": len(img_pts),
            "frame": data.get("calibration_frame", 0),
            "image_points": img_pts,
            "pitch_points": pitch_pts,
            "projected_points": projected
        }

    def pixels_to_meters(self, u: float, v: float) -> Tuple[float, float]:
        """Projects a 2D pixel coordinate to 3D metric ground plane."""
        if self.H_matrix is None:
            raise ValueError("Homography matrix not calibrated.")
            
        point = np.array([[[u, v]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.H_matrix)
        return float(transformed[0][0][0]), float(transformed[0][0][1])
