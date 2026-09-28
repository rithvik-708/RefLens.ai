import sys
import os
import cv2
import numpy as np
import time
import argparse
import json
from loguru import logger
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.stream_reader import VideoStreamIngest
from src.detection.detector import LightweightDetector
from src.detection.tracker import PerceptionTracker

def verify_phase2(video_path: str, max_frames: int):
    print("="*50)
    print("RefLens Phase 2 Football Validation")
    print("="*50)

    # 1. OpenCV DNN Backend Verification
    cv_ver = cv2.__version__
    build_info = cv2.getBuildInformation()
    cuda_compiled = "CUDA" in build_info
    cudnn_compiled = "cuDNN" in build_info
    
    print("\n--- OpenCV Backend Check ---")
    print(f"OpenCV version: {cv_ver}")
    print(f"CUDA compiled into OpenCV: {'YES' if cuda_compiled else 'NO'}")
    print(f"cuDNN compiled into OpenCV: {'YES' if cudnn_compiled else 'NO'}")

    # 2. Export ONNX Detector
    model_path = "yolov8n.onnx"
    if not os.path.exists(model_path):
        print("\n--- Exporting YOLOv8n to ONNX ---")
        try:
            model = YOLO("yolov8n.pt")
            model.export(format="onnx", imgsz=640, dynamic=True)
            print("Successfully exported YOLOv8n to ONNX.")
        except Exception as e:
            print(f"Failed to export ONNX model: {e}")
            return False

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

    # Validation criteria and metrics
    metrics = {
        "processed_frames": 0,
        "inference_times": [],
        "total_times": [],
        "person_detections": 0,
        "max_persons_per_frame": 0,
        "ball_detections": 0,
        "ball_frames": 0,
        "sum_ball_conf": 0.0,
        "sum_person_conf": 0.0,
        "highest_raw_ball_conf": 0.0,
        "active_tracks_counts": [],
        "tracks_gt_10": 0
    }

    # 3. Setup Components
    ingest = VideoStreamIngest(video_path, batch_size=1)
    
    detector = LightweightDetector(model_path)
    detector.debug = True
    detector.thresholds = {0: 0.20, 32: 0.05}  # Diagnostic thresholds
    
    tracker = PerceptionTracker(frame_rate=int(ingest.fps))

    out_dir = "outputs/phase2"
    os.makedirs(out_dir, exist_ok=True)
    
    w, h = None, None
    writers = {}
    
    start_pipeline = time.time()
    
    for batch_frames, batch_meta in ingest.stream_batches():
        for frame, meta in zip(batch_frames, batch_meta):
            if w is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writers["detector"] = cv2.VideoWriter(os.path.join(out_dir, "detector_only.mp4"), fourcc, ingest.fps, (w, h))
                writers["tracker"] = cv2.VideoWriter(os.path.join(out_dir, "tracker_only.mp4"), fourcc, ingest.fps, (w, h))
                writers["final"] = cv2.VideoWriter(os.path.join(out_dir, "final_pipeline.mp4"), fourcc, ingest.fps, (w, h))

            t0 = time.time()
            
            # 1. Detection
            t_det0 = time.time()
            detections = detector.detect(frame)
            t_det1 = time.time()
            
            metrics["inference_times"].append(t_det1 - t_det0)
            
            # Keep track of highest raw ball conf seen
            if detector.last_max_ball_conf > metrics["highest_raw_ball_conf"]:
                metrics["highest_raw_ball_conf"] = detector.last_max_ball_conf
                
            # Log debug info before tracker
            logger.debug(f"[DEBUG-PRE-TRACKER] Frame {meta.frame_id}: {len(detections)} detections")
            for d in detections:
                logger.debug(f"  -> Class:{d.class_id} ({d.class_name}), Conf:{d.confidence:.4f}, BBox:{d.bbox}")
                
            balls = [d for d in detections if d.class_id == 32]
            persons = [d for d in detections if d.class_id == 0]
            
            metrics["person_detections"] += len(persons)
            metrics["max_persons_per_frame"] = max(metrics["max_persons_per_frame"], len(persons))
            metrics["ball_detections"] += len(balls)
            
            if balls:
                metrics["ball_frames"] += 1
                metrics["sum_ball_conf"] += sum(b.confidence for b in balls)
            if persons:
                metrics["sum_person_conf"] += sum(p.confidence for p in persons)
                
            # detector_only visualization
            vis_det = frame.copy()
            for d in detections:
                x1, y1, x2, y2 = map(int, d.bbox)
                color = (0, 0, 255) if d.class_id == 32 else (0, 255, 0)
                cv2.rectangle(vis_det, (x1, y1), (x2, y2), color, 2)
            writers["detector"].write(vis_det)

            # 2. Tracking
            active_tracks = tracker.update(detections, frame, meta.frame_id, meta.pts_milliseconds)
            metrics["active_tracks_counts"].append(len(active_tracks))
            
            # tracker_only visualization
            vis_trk = frame.copy()
            cv2.putText(vis_trk, f"Active Tracks: {len(active_tracks)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            for track_id, entity in active_tracks.items():
                latest_state = entity.history[-1]
                if latest_state.frame_id == meta.frame_id:
                    x1, y1, x2, y2 = map(int, latest_state.bbox)
                    color = (0, 0, 255) if entity.class_id == 32 else (255, 0, 0)
                    cv2.rectangle(vis_trk, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_trk, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            writers["tracker"].write(vis_trk)

            # final pipeline visualization
            vis_final = frame.copy()
            for track_id, entity in active_tracks.items():
                latest_state = entity.history[-1]
                if latest_state.frame_id == meta.frame_id:
                    x1, y1, x2, y2 = map(int, latest_state.bbox)
                    color = (0, 0, 255) if entity.class_id == 32 else (255, 0, 0)
                    cv2.rectangle(vis_final, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} | Vx:{latest_state.v_x:.1f} Vy:{latest_state.v_y:.1f}"
                    cv2.putText(vis_final, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            writers["final"].write(vis_final)

            t1 = time.time()
            metrics["total_times"].append(t1 - t0)
            metrics["processed_frames"] += 1
            
            if metrics["processed_frames"] >= max_frames:
                logger.info(f"Reached {max_frames} frames limit for testing. Stopping early.")
                break
        
        if metrics["processed_frames"] >= max_frames:
            break
            
    ingest.close()
    for w_name, writer in writers.items():
        writer.release()
        
    end_pipeline = time.time()
    total_duration = end_pipeline - start_pipeline
    
    # Calculate track length metric
    for track_id, entity in tracker.active_tracks.items():
        if len(entity.history) > 10:
            metrics["tracks_gt_10"] += 1

    video_pass = "PASS" if metrics["processed_frames"] > 0 else "FAIL"
    person_pass = "PASS" if metrics["person_detections"] > 0 else "FAIL"
    
    if metrics["ball_detections"] > 10:
        ball_pass = "PASS"
    elif metrics["ball_detections"] > 0:
        ball_pass = "PARTIAL"
    else:
        ball_pass = "FAIL"
        
    if len(tracker.active_tracks) > 5 and metrics["tracks_gt_10"] > 5:
        tracking_pass = "PASS"
    elif len(tracker.active_tracks) > 0 and metrics["tracks_gt_10"] > 0:
        tracking_pass = "PARTIAL"
    else:
        tracking_pass = "FAIL"
    
    avg_tot = np.mean(metrics["total_times"]) * 1000 if metrics["total_times"] else 0
    processing_fps = 1.0 / (avg_tot / 1000) if avg_tot > 0 else 0
    rt_ratio = processing_fps / ingest.fps if ingest.fps > 0 else 0
    
    if rt_ratio > 0.8:
        perf_pass = "PASS"
    elif rt_ratio > 0.3:
        perf_pass = "PARTIAL"
    else:
        perf_pass = "FAIL"

    # Report writing
    report_path = os.path.join(out_dir, "PHASE2_FOOTBALL_VALIDATION.md")
    with open(report_path, "w") as f:
        f.write("# RefLens Phase 2 Football Validation Report\n\n")
        f.write("## Validation Results\n\n")
        f.write(f"- **Video decoding**: {video_pass}\n")
        f.write(f"- **Person detection**: {person_pass}\n")
        f.write(f"- **Ball detection**: {ball_pass}\n")
        f.write(f"- **Multi-object tracking**: {tracking_pass}\n")
        f.write(f"- **Performance**: {perf_pass}\n\n")
        
        f.write("## Video Metrics\n")
        f.write(f"- Resolution: {w}x{h}\n")
        f.write(f"- FPS: {ingest.fps}\n")
        f.write(f"- Total processed frames: {metrics['processed_frames']}\n")
        f.write(f"- Duration: {total_duration:.2f} s\n\n")
        
        avg_persons = metrics["person_detections"] / metrics["processed_frames"] if metrics["processed_frames"] > 0 else 0
        avg_ball_conf = metrics["sum_ball_conf"] / metrics["ball_detections"] if metrics["ball_detections"] > 0 else 0
        ball_det_rate = (metrics["ball_frames"] / metrics["processed_frames"]) * 100 if metrics["processed_frames"] > 0 else 0
        
        f.write("## Detection Metrics\n")
        f.write(f"- Person detections: {metrics['person_detections']}\n")
        f.write(f"- Average persons/frame: {avg_persons:.2f}\n")
        f.write(f"- Maximum persons/frame: {metrics['max_persons_per_frame']}\n")
        f.write(f"- Ball detections: {metrics['ball_detections']}\n")
        f.write(f"- Ball detection rate: {ball_det_rate:.1f}%\n")
        f.write(f"- Maximum ball confidence (raw, pre-threshold): {metrics['highest_raw_ball_conf']:.4f}\n")
        f.write(f"- Average ball confidence (post-threshold): {avg_ball_conf:.4f}\n\n")
        
        avg_active = np.mean(metrics["active_tracks_counts"]) if metrics["active_tracks_counts"] else 0
        max_active = np.max(metrics["active_tracks_counts"]) if metrics["active_tracks_counts"] else 0
        total_unique = tracker.next_id - 1
        
        f.write("## Tracking Metrics\n")
        f.write(f"- Unique track IDs: {total_unique}\n")
        f.write(f"- Average active tracks/frame: {avg_active:.1f}\n")
        f.write(f"- Maximum active tracks/frame: {max_active}\n")
        f.write(f"- Tracks lasting > 10 frames: {metrics['tracks_gt_10']}\n\n")
        
        avg_inf = np.mean(metrics["inference_times"]) * 1000 if metrics["inference_times"] else 0
        f.write("## Performance Metrics\n")
        f.write(f"- Average inference latency: {avg_inf:.1f} ms\n")
        f.write(f"- Average frame latency: {avg_tot:.1f} ms\n")
        f.write(f"- Processing FPS: {processing_fps:.1f}\n")
        f.write(f"- Realtime ratio: {rt_ratio:.2f}\n\n")

        f.write("## Root Cause Analysis\n")
        if video_pass != "PASS":
            f.write("- **Video decoding FAIL**: Could not read frames or video file not found/supported.\n")
        if person_pass != "PASS":
            f.write("- **Person detection FAIL**: YOLOv8n failed to detect any persons, even with threshold 0.20.\n")
        if ball_pass != "PASS":
            f.write(f"- **Ball detection ISSUE**: The highest raw ball confidence observed was {metrics['highest_raw_ball_conf']:.4f}. Standard COCO YOLOv8n is often insufficient for small footballs at 640x640.\n")
        if tracking_pass != "PASS":
            f.write("- **Tracking ISSUE**: Insufficient tracks persisted for more than 10 frames. ByteTrack may be discarding detections or matching failed.\n")
        if perf_pass != "PASS":
            f.write("- **Performance ISSUE**: Realtime ratio was below optimal, possibly due to CPU inference.\n")

    print(f"\nPhase 2 Football Validation Done! Check {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RefLens Phase 2 Football Validation")
    parser.add_argument("--video", type=str, required=True, help="Path to the football video (.mp4 or .mkv)")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to process")
    args = parser.parse_args()
    
    verify_phase2(args.video, args.frames)
