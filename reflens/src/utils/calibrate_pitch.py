import cv2
import json
import argparse
import sys
import os
from typing import List, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.vision.pitch_model import PitchModel, PitchLandmark
from src.vision.homography import HomographyEstimator

class PitchCalibrator:
    def __init__(self, video_path: str, output_path: str):
        self.video_path = video_path
        self.output_path = output_path
        self.pitch_model = PitchModel()
        self.homography = HomographyEstimator()
        
        self.landmarks = list(PitchLandmark)
        self.current_landmark_idx = 0
        self.selected_points = {}  # PitchLandmark -> (x, y) pixel
        
        self.frame = None
        self.display_frame = None
        
        self._load_frame()
        cv2.namedWindow("Pitch Calibration")
        cv2.setMouseCallback("Pitch Calibration", self._mouse_callback)

    def _load_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            sys.exit(1)
            
        # Get a frame from a few seconds in to ensure the pitch is visible
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
        ret, self.frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            sys.exit(1)
        cap.release()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            landmark = self.landmarks[self.current_landmark_idx]
            self.selected_points[landmark] = (float(x), float(y))
            print(f"Set {landmark.name} to ({x}, {y})")
            self._update_display()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove current point
            landmark = self.landmarks[self.current_landmark_idx]
            if landmark in self.selected_points:
                del self.selected_points[landmark]
                print(f"Removed {landmark.name}")
                self._update_display()

    def _update_display(self):
        self.display_frame = self.frame.copy()
        
        # Draw all selected points
        for lm, (x, y) in self.selected_points.items():
            cv2.circle(self.display_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(self.display_frame, lm.name, (int(x) + 10, int(y) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Draw current landmark to select
        current_lm = self.landmarks[self.current_landmark_idx]
        status_text = f"Select: {current_lm.name} (L-Click to set, R-Click to remove)"
        cv2.putText(self.display_frame, status_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
        controls_text = "'n': Next | 'p': Prev | 'c': Calculate | 's': Save | 'q': Quit"
        cv2.putText(self.display_frame, controls_text, (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
        # If homography is fit, show metrics
        if self.homography.H_matrix is not None:
            metrics = self.homography.validation_metrics
            err_text = f"Mean Err: {metrics['mean_reprojection_error_m']:.2f}m | Max Err: {metrics['max_reprojection_error_m']:.2f}m"
            cv2.putText(self.display_frame, err_text, (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
        cv2.imshow("Pitch Calibration", self.display_frame)

    def _calculate_homography(self):
        if len(self.selected_points) < 4:
            print(f"Need at least 4 points. Currently have {len(self.selected_points)}")
            return
            
        img_pts = []
        ptc_pts = []
        
        for lm, pt in self.selected_points.items():
            img_pts.append(pt)
            pitch_pt = self.pitch_model.get_landmark(lm)
            ptc_pts.append((pitch_pt.x, pitch_pt.y))
            
        success = self.homography.fit(img_pts, ptc_pts)
        if success:
            print("Homography calculated successfully.")
            metrics = self.homography.validate()
            print(f"Mean Error: {metrics['mean_reprojection_error_m']:.3f}m")
            print(f"Max Error: {metrics['max_reprojection_error_m']:.3f}m")
        else:
            print("Failed to calculate a valid homography matrix.")
        
        self._update_display()

    def _save_calibration(self):
        if self.homography.H_matrix is None:
            print("Calculate homography first before saving!")
            return
            
        out_data = {
            "image_size": [self.frame.shape[1], self.frame.shape[0]],
            "landmarks": [
                {
                    "name": lm.name,
                    "image": [pt[0], pt[1]],
                    "pitch": [self.pitch_model.get_landmark(lm).x, self.pitch_model.get_landmark(lm).y]
                }
                for lm, pt in self.selected_points.items()
            ],
            "homography": self.homography.H_matrix.tolist(),
            "metrics": self.homography.validation_metrics
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(out_data, f, indent=4)
            
        print(f"Calibration saved to {self.output_path}")

    def run(self):
        self._update_display()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.current_landmark_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
                self._update_display()
            elif key == ord('p'):
                self.current_landmark_idx = (self.current_landmark_idx - 1) % len(self.landmarks)
                self._update_display()
            elif key == ord('c'):
                self._calculate_homography()
            elif key == ord('s'):
                self._save_calibration()
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video file for calibration")
    parser.add_argument("--output", default="configs/pitch_calibration.json", help="Output JSON path")
    args = parser.parse_args()
    
    calibrator = PitchCalibrator(args.video, args.output)
    calibrator.run()
