import cv2
import numpy as np
import os
import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.homography import PitchHomography

def draw_pitch_lines(frame: np.ndarray, H_inv: np.ndarray, pitch_l: float, pitch_w: float):
    """Draws major pitch lines by projecting metric coordinates to image coordinates."""
    
    def proj(x, y):
        pt = np.array([[[x, y]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, H_inv)
        return int(res[0][0][0]), int(res[0][0][1])

    # 1. Pitch Boundary
    pts = [proj(0, 0), proj(pitch_l, 0), proj(pitch_l, pitch_w), proj(0, pitch_w)]
    cv2.polylines(frame, [np.array(pts)], True, (255, 255, 255), 2)
    
    # 2. Center Line
    pt1 = proj(pitch_l/2, 0)
    pt2 = proj(pitch_l/2, pitch_w)
    cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
    
    # Center Circle (approximation via polygon)
    center_x, center_y = pitch_l/2, pitch_w/2
    radius = 9.15
    circle_pts = []
    for angle in np.linspace(0, 2*np.pi, 36):
        cx = center_x + radius * np.cos(angle)
        cy = center_y + radius * np.sin(angle)
        circle_pts.append(proj(cx, cy))
    cv2.polylines(frame, [np.array(circle_pts)], True, (255, 255, 255), 2)
    
    # 3. Penalty Boxes
    # Left box
    lb_pts = [proj(0, pitch_w/2 - 20.15), proj(16.5, pitch_w/2 - 20.15), 
              proj(16.5, pitch_w/2 + 20.15), proj(0, pitch_w/2 + 20.15)]
    cv2.polylines(frame, [np.array(lb_pts)], False, (255, 255, 255), 2)
    
    # Right box
    rb_pts = [proj(pitch_l, pitch_w/2 - 20.15), proj(pitch_l - 16.5, pitch_w/2 - 20.15), 
              proj(pitch_l - 16.5, pitch_w/2 + 20.15), proj(pitch_l, pitch_w/2 + 20.15)]
    cv2.polylines(frame, [np.array(rb_pts)], False, (255, 255, 255), 2)

def validate_calibration(json_path: str):
    print("="*50)
    print("RefLens Pitch Calibration Validation")
    print("="*50)

    if not os.path.exists(json_path):
        print(f"Error: Calibration file not found at {json_path}")
        return

    homography = PitchHomography()
    
    try:
        results = homography.load_from_json(json_path)
    except Exception as e:
        print(f"\nStatus: FAIL")
        print(f"Error: {e}")
        return

    num_pts = results["num_points"]
    mean_err = results["mean_reprojection_error_m"]
    max_err = results["max_reprojection_error_m"]
    pitch_l = results["pitch_dimensions"]["length"]
    pitch_w = results["pitch_dimensions"]["width"]
    
    print(f"\nCalibration points: {num_pts}")
    print(f"Homography: VALID")
    print(f"Mean reprojection error: {mean_err:.2f} m")
    print(f"Max reprojection error: {max_err:.2f} m")

    status = "FAIL"
    if mean_err < 1.0 and max_err < 3.0:
        status = "PASS"
    elif mean_err < 3.0:
        status = "PARTIAL"
        
    print(f"\nStatus: {status}\n")

    # Visual Validation
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    video_path = data.get("video_path")
    frame_num = data.get("calibration_frame", 0)
    
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Draw calibration points (ground truth user clicks)
            for pt in data.get("points", []):
                u, v = int(pt["image"][0]), int(pt["image"][1])
                cv2.circle(frame, (u, v), 6, (0, 0, 255), -1) # Red for clicked points
                
            # Draw reprojections from pitch to image
            H_inv = np.linalg.inv(homography.H_matrix)
            
            for pt in data.get("points", []):
                px, py = pt["pitch"][0], pt["pitch"][1]
                ppt = np.array([[[px, py]]], dtype=np.float32)
                res = cv2.perspectiveTransform(ppt, H_inv)
                pu, pv = int(res[0][0][0]), int(res[0][0][1])
                cv2.circle(frame, (pu, pv), 4, (0, 255, 0), -1) # Green for reprojected
                
                # Draw line connecting them showing error magnitude visually
                u, v = int(pt["image"][0]), int(pt["image"][1])
                cv2.line(frame, (u, v), (pu, pv), (0, 255, 255), 1)

            # Draw pitch grid
            draw_pitch_lines(frame, H_inv, pitch_l, pitch_w)

            out_dir = "outputs/phase3"
            os.makedirs(out_dir, exist_ok=True)
            out_img = os.path.join(out_dir, "calibration_debug.png")
            cv2.imwrite(out_img, frame)
            print(f"Visual validation saved to {out_img}")
        else:
            print("Could not read frame from video for visual validation.")
    else:
        print("Video path from calibration file not found. Skipping visual validation.")

    # Write report
    report_path = os.path.join(out_dir, "PHASE3A_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# RefLens Phase 3A Interactive Pitch Calibration Report\n\n")
        f.write(f"- **Calibration frame**: {frame_num}\n")
        f.write(f"- **Number of points**: {num_pts}\n")
        f.write(f"- **Pitch dimensions**: {pitch_l}m x {pitch_w}m\n")
        f.write(f"- **Mean reprojection error**: {mean_err:.2f} m\n")
        f.write(f"- **Max reprojection error**: {max_err:.2f} m\n\n")
        f.write("## Limitations\n")
        f.write("- The homography matrix strictly maps points on the ground plane (Z=0). Objects above ground (e.g. players' heads or a flying ball) will reproject incorrectly. Always project the bottom-center of bounding boxes.\n")
        f.write("- Calibration is currently fixed per frame. If the camera pans, tilts, or zooms, this matrix will instantly become invalid. Dynamic camera calibration is deferred to later phases if needed.\n\n")
        f.write(f"## Final Status\n")
        f.write(f"**{status}**\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RefLens Pitch Calibration Validation")
    parser.add_argument("--calibration", type=str, required=True, help="Path to the calibration JSON file")
    args = parser.parse_args()
    
    validate_calibration(args.calibration)
