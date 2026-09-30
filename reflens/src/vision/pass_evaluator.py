import cv2
import json
import numpy as np
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.pipeline import PerceptionEngine
from src.vision.pass_detector import PassEventDetector
from src.vision.events import PassCandidate
from src.vision.visualizer import VARVisualizer

class PassEvaluator:
    def __init__(self, video_path: str, model_path: str, output_dir: str = "outputs/phase3"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        debug_video_path = str(self.output_dir / "ball_tracking_debug.mp4")
        self.perception = PerceptionEngine(
            model_path=model_path, 
            source_video=video_path, 
            output_video=debug_video_path
        )
        self.event_detector = PassEventDetector()
        self.visualizer = VARVisualizer()
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    def export_json(self, candidates: List[PassCandidate], filepath: str):
        """Exports the candidate events to a structured JSON file."""
        data = []
        for c in candidates:
            data.append({
                "frame_id": c.frame_id,
                "timestamp": round(c.frame_id / self.fps, 2),
                "player_id": c.player_id,
                "ball_distance_px": round(c.ball_distance_px, 2),
                "ball_speed_before": round(c.ball_speed_before, 2),
                "ball_speed_after": round(c.ball_speed_after, 2),
                "acceleration": round(c.acceleration, 2),
                "confidence": round(c.confidence, 2)
            })
            
        with open(filepath, 'w') as f:
            json.dump({
                "total_candidates": len(data),
                "high_confidence_count": sum(1 for x in data if x["confidence"] > 0.8),
                "events": data
            }, f, indent=4)
        logger.info(f"Exported {len(candidates)} candidates to {filepath}")

    def export_diagnostics(self, diagnostics: Dict[str, Any], filepath: str):
        """Exports ball tracking diagnostics to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(diagnostics, f, indent=4)
        logger.info(f"Exported ball tracking diagnostics to {filepath}")

    def generate_candidate_video(self, candidates: List[PassCandidate], window_frames: int = 25):
        """Render a smooth continuous video showing pass events with dynamic top-left HUD without freezing."""
        out_path = str(self.output_dir / "pass_candidates.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.width, self.height))
        
        logger.info(f"Rendering smooth candidate events video to {out_path}...")
        
        # Map frame_id -> (event_index, candidate)
        frame_to_candidate = {}
        intervals = []
        for idx, candidate in enumerate(candidates):
            sf = max(0, candidate.frame_id - window_frames)
            ef = candidate.frame_id + window_frames
            intervals.append((sf, ef, idx + 1, candidate))
            for f in range(sf, ef + 1):
                if f not in frame_to_candidate:
                    frame_to_candidate[f] = (idx + 1, candidate)
                else:
                    curr_dist = abs(f - candidate.frame_id)
                    prev_dist = abs(f - frame_to_candidate[f][1].frame_id)
                    if curr_dist < prev_dist:
                        frame_to_candidate[f] = (idx + 1, candidate)
                        
        # Merge overlapping intervals for smooth continuous sequential playback
        merged_intervals = []
        for sf, ef, _, _ in sorted(intervals, key=lambda x: x[0]):
            if not merged_intervals:
                merged_intervals.append([sf, ef])
            else:
                last_sf, last_ef = merged_intervals[-1]
                if sf <= last_ef + 15:  # merge if overlapping or adjacent
                    merged_intervals[-1][1] = max(last_ef, ef)
                else:
                    merged_intervals.append([sf, ef])
                    
        for start_frame, end_frame in merged_intervals:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            
            while current_frame <= end_frame:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                if current_frame in frame_to_candidate:
                    event_idx, candidate = frame_to_candidate[current_frame]
                    is_exact = (current_frame == candidate.frame_id)
                    frame = self.visualizer.draw_candidate_overlay(
                        frame, 
                        candidate, 
                        event_idx=event_idx, 
                        is_exact_frame=is_exact
                    )
                    
                writer.write(frame)
                current_frame += 1

        writer.release()
        logger.info("Smooth video generation complete.")

    def run(self, max_frames: int = None):
        ball_track, player_tracks = self._generate_tracks(max_frames=max_frames)
        
        # Diagnostics
        diagnostics = ball_track.get_diagnostics() if hasattr(ball_track, "get_diagnostics") else {}
        diag_path = str(self.output_dir / "ball_diagnostics.json")
        self.export_diagnostics(diagnostics, diag_path)

        logger.info("=" * 40)
        logger.info("BALL TRACKING DIAGNOSTICS:")
        for k, v in diagnostics.items():
            logger.info(f"  - {k}: {v}")
        logger.info("=" * 40)
        
        candidates = self.event_detector.detect_candidates(ball_track, player_tracks)
        candidates.sort(key=lambda x: x.frame_id)
        
        json_path = str(self.output_dir / "candidates.json")
        self.export_json(candidates, json_path)
        
        if candidates:
            self.generate_candidate_video(candidates)
        else:
            logger.warning("No pass candidates detected in the provided video.")
            
        self.cap.release()
        return diagnostics, candidates

    def _generate_tracks(self, max_frames: int = None):
        """Pass 1: Run the perception engine to build complete tracks."""
        logger.info("Pass 1: Extracting tracks from video...")
        ball_track, player_tracks = self.perception.generate_full_tracks(max_frames=max_frames)
        return ball_track, player_tracks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Pass Events on Real Video")
    parser.add_argument("--video", required=True, help="Path to 720p broadcast video")
    parser.add_argument("--model", required=True, help="Path to YOLO ONNX model")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
    args = parser.parse_args()
    
    evaluator = PassEvaluator(video_path=args.video, model_path=args.model)
    evaluator.run(max_frames=args.max_frames)
