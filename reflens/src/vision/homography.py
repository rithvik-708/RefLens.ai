import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

class HomographyEstimator:
    def __init__(self):
        self.H_matrix: np.ndarray | None = None
        self.image_points: List[Tuple[float, float]] = []
        self.pitch_points: List[Tuple[float, float]] = []
        self.validation_metrics: Dict[str, float] = {}

    def fit(self, image_points: List[Tuple[float, float]], pitch_points: List[Tuple[float, float]]) -> bool:
        if len(image_points) < 4 or len(pitch_points) < 4:
            raise ValueError("At least 4 non-collinear correspondences are required.")
            
        img_pts = np.array(image_points, dtype=np.float32)
        ptc_pts = np.array(pitch_points, dtype=np.float32)
        
        # Use basic findHomography since user will click explicit landmarks
        H, status = cv2.findHomography(img_pts, ptc_pts)
        
        if H is not None and H.shape == (3, 3):
            # Check for degenerate configuration (e.g. all points collinear or determinant near zero)
            det = np.linalg.det(H)
            if abs(det) < 1e-7:
                return False

            self.H_matrix = H
            self.image_points = image_points
            self.pitch_points = pitch_points
            self.validate()
            
            # Reject if error is obviously excessive (e.g. > 10m on average)
            if self.validation_metrics["mean_reprojection_error_m"] > 10.0:
                self.H_matrix = None
                return False
                
            return True
        return False

    def image_to_pitch(self, image_point: Any) -> Tuple[float, float]:
        if self.H_matrix is None:
            raise ValueError("Homography matrix not calibrated.")
            
        if hasattr(image_point, 'x') and hasattr(image_point, 'y'):
            u, v = float(image_point.x), float(image_point.y)
        else:
            u, v = float(image_point[0]), float(image_point[1])
        pt = np.array([[[u, v]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H_matrix)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    def pitch_to_image(self, pitch_point: Any) -> Tuple[float, float]:
        if self.H_matrix is None:
            raise ValueError("Homography matrix not calibrated.")
            
        if hasattr(pitch_point, 'x') and hasattr(pitch_point, 'y'):
            x, y = float(pitch_point.x), float(pitch_point.y)
        else:
            x, y = float(pitch_point[0]), float(pitch_point[1])
        H_inv = np.linalg.inv(self.H_matrix)
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H_inv)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    def reprojection_error(self, image_point: Any, pitch_point: Any) -> float:
        pred_x, pred_y = self.image_to_pitch(image_point)
        if hasattr(pitch_point, 'x') and hasattr(pitch_point, 'y'):
            true_x, true_y = float(pitch_point.x), float(pitch_point.y)
        else:
            true_x, true_y = float(pitch_point[0]), float(pitch_point[1])
        return float(np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2))

    def validate(self) -> Dict[str, float]:
        if self.H_matrix is None or not self.image_points or len(self.image_points) < 4:
            self.validation_metrics = {
                "mean_reprojection_error_m": 0.0,
                "median_reprojection_error_m": 0.0,
                "max_reprojection_error_m": 0.0,
                "calibration_valid": False
            }
            self.H_matrix = None
            return self.validation_metrics

        errors = []
        for img_pt, ptc_pt in zip(self.image_points, self.pitch_points):
            error = self.reprojection_error(img_pt, ptc_pt)
            errors.append(error)

        mean_err = float(np.mean(errors))
        if mean_err == 0.0:
            # Fake/synthetic calibration detected
            self.H_matrix = None
            self.validation_metrics = {
                "mean_reprojection_error_m": 0.0,
                "median_reprojection_error_m": 0.0,
                "max_reprojection_error_m": 0.0,
                "calibration_valid": False
            }
            return self.validation_metrics

        self.validation_metrics = {
            "mean_reprojection_error_m": mean_err,
            "median_reprojection_error_m": float(np.median(errors)),
            "max_reprojection_error_m": float(np.max(errors)),
            "calibration_valid": True
        }
        return self.validation_metrics
