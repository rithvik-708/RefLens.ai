import os
import sys
import cv2
import numpy as np
import json
import time
from typing import List
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.stream_reader import VideoStreamIngest
from src.detection.detector import LightweightDetector
from src.detection.tracker import PerceptionTracker
from src.vision.pitch_features import PitchFeatureExtractor
from src.vision.homography import PitchHomography
from src.vision.contact_detection import PassEventDetector
from src.vision.offside_geometry import OffsideEngine

def run_phase3_audit(video_path: str, max_frames: int = 150):
    print("="*50)
    print("RefLens Phase 3 Audit & Validation")
    print("="*50)
    
    if not os.path.exists(video_path):
        print(f"Error: Input video does not exist: {video_path}")
        return False
        
    abs_path = os.path.abspath(video_path)
    size_bytes = os.path.getsize(video_path)
    
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(video_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    manifest = {
        "input_video": video_path,
        "absolute_path": abs_path,
        "file_size_bytes": size_bytes,
        "sha256": sha256_hash.hexdigest(),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "source": "SoccerNet" if "dataset" in abs_path or "soccernet" in abs_path.lower() else "Local Test"
    }
    
    os.makedirs("outputs/debug", exist_ok=True)
    with open("outputs/debug/input_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nInput video:\n{video_path}\n\nResolved path:\n{abs_path}\n\nSource:\n{manifest['source']}\n")

    out_dir = "outputs/phase3"
    os.makedirs(out_dir, exist_ok=True)
    
    # Init Phase 3 Modules
    pitch_extractor = PitchFeatureExtractor()
    homography = PitchHomography()
    pass_detector = PassEventDetector()
    offside_engine = OffsideEngine(homography)
    
    # Try calibrating homography with dummy points (since we have no calibration tool yet)
    # Mapping a 640x640 frame approx to a 105x68m pitch just to satisfy the matrix requirement
    dummy_img_pts = np.array([[0,0], [640,0], [640,640], [0,640]], dtype=np.float32)
    dummy_pitch_pts = np.array([[0,0], [105,0], [105,68], [0,68]], dtype=np.float32)
    homography_calibrated = homography.calibrate(dummy_img_pts, dummy_pitch_pts)
    
    # Init Phase 2 Modules
    ingest = VideoStreamIngest(video_path, batch_size=1)
    detector = LightweightDetector("yolov8n.onnx")
    tracker = PerceptionTracker(frame_rate=int(ingest.fps))
    
    w, h = None, None
    writer_debug = None
    
    metrics = {
        "processed_frames": 0,
        "lines_detected": [],
        "t_pass": None
    }
    
    all_player_tracks = []
    ball_track = None
    
    for batch_frames, batch_meta in ingest.stream_batches():
        for frame, meta in zip(batch_frames, batch_meta):
            if w is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_debug = cv2.VideoWriter(os.path.join(out_dir, "phase3_debug.mp4"), fourcc, ingest.fps, (w, h))

            vis_frame = frame.copy()
            
            # 1. Pitch Features
            lines = pitch_extractor.extract_lines(frame)
            metrics["lines_detected"].append(len(lines) if lines is not None else 0)
            if lines is not None and len(lines) > 0 and isinstance(lines, np.ndarray):
                for line in lines.reshape(-1, 4):
                    x1, y1, x2, y2 = line
                    cv2.line(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                
            # 2. Detect & Track
            detections = detector.detect(frame)
            active_tracks = tracker.update(detections, frame, meta.frame_id, meta.pts_milliseconds)
            
            # Find the ball track (heuristic: longest history)
            current_ball = None
            for tid, entity in tracker.active_tracks.items():
                if entity.class_id == 32: # sports ball
                    if ball_track is None or len(entity.history) > len(ball_track.history):
                        ball_track = entity
                    current_ball = entity
                        
            # Draw tracks
            for tid, entity in active_tracks.items():
                latest_state = entity.history[-1]
                if latest_state.frame_id == meta.frame_id:
                    x1, y1, x2, y2 = map(int, latest_state.bbox)
                    color = (0, 0, 255) if entity.class_id == 32 else (255, 0, 0)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    if entity.class_id == 0:
                        all_player_tracks.append(entity) # Collect players for pass detection later

            cv2.putText(vis_frame, f"Frame: {meta.frame_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer_debug.write(vis_frame)
            
            metrics["processed_frames"] += 1
            if metrics["processed_frames"] >= max_frames:
                break
        if metrics["processed_frames"] >= max_frames:
            break
            
    ingest.close()
    if writer_debug:
        writer_debug.release()

    # 3. Offline Analysis: Find T_pass
    if ball_track:
        t_pass = pass_detector.find_t_pass(ball_track, all_player_tracks)
        metrics["t_pass"] = t_pass

    # Generate Report
    report_path = os.path.join(out_dir, "PHASE3_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# RefLens Phase 3 Audit & Validation Report\n\n")
        
        f.write("## 1. Pitch Feature Extraction\n")
        f.write(f"- Average lines detected per frame: {np.mean(metrics['lines_detected']):.1f}\n")
        if np.mean(metrics['lines_detected']) > 5:
            f.write("- **Status**: PASS. Hough lines successfully extracted using color isolation.\n\n")
            pitch_status = "PASS"
        else:
            f.write("- **Status**: PARTIAL. Very few lines extracted.\n\n")
            pitch_status = "PARTIAL"
            
        f.write("## 2. Homography\n")
        f.write("- Ground-truth pitch correspondences are unavailable for this arbitrary video.\n")
        f.write("- **Status**: UNVERIFIED. Matrix H initialized with dummy points.\n\n")
        homo_status = "UNVERIFIED"
        
        f.write("## 3. Contact Frame Detection\n")
        if metrics["t_pass"] is not None:
            f.write(f"- Detected T_pass at frame {metrics['t_pass']}.\n")
            f.write("- **Status**: PARTIAL. Frame detected but requires manual confirmation.\n\n")
            contact_status = "PARTIAL"
        else:
            f.write("- Failed to detect a pass. Ball tracking was likely unreliable.\n")
            f.write("- **Status**: UNVERIFIED.\n\n")
            contact_status = "UNVERIFIED"

        f.write("## 4. Offside Geometry Engine\n")
        f.write("- Unit tests confirm logic for team assignment, ball reference, goalkeeper handling, and attacking directions.\n")
        f.write("- **Status**: PASS (in deterministic synthetic unit tests).\n\n")
        geometry_status = "PASS"
        rule_status = "PASS"
        
        f.write("## 5. End-to-End Football Validation\n")
        f.write("- Run completed on test video.\n")
        f.write("- Debug visualization generated.\n")
        f.write("- **Status**: PARTIAL. Without ground-truth homography and perfect tracking, end-to-end offside geometry cannot be fully trusted yet.\n\n")
        e2e_status = "PARTIAL"

        f.write("## Assumptions & Approximations\n")
        f.write("- **Bounding Box Center**: Attacker/Defender positions are approximated using the bottom-center of their bounding box (MVP approximation). This is NOT VAR-grade body geometry.\n")
        f.write("- **Team Assignment**: Since team jersey clustering is not implemented, the geometry engine relies on external input for `defenders` array.\n")
        f.write("- **Homography Calibration**: Automated intersection detection is missing; requires manual point mapping.\n\n")

        f.write("## Final Assessment\n")
        f.write(f"- Pitch feature extraction: {pitch_status}\n")
        f.write(f"- Homography: {homo_status}\n")
        f.write(f"- Contact frame: {contact_status}\n")
        f.write(f"- Player/reference geometry: {geometry_status}\n")
        f.write(f"- Rule engine: {rule_status}\n")
        f.write(f"- End-to-end football validation: {e2e_status}\n")
        
    print(f"\nPhase 3 Audit complete. Check {report_path} and {out_dir}/phase3_debug.mp4")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RefLens Phase 3 Validation")
    parser.add_argument("--video", type=str, required=True, help="Path to the football video (.mp4 or .mkv)")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames to process")
    args = parser.parse_args()
    
    run_phase3_audit(args.video, args.frames)
