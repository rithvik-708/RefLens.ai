import cv2
import json
import numpy as np
import sys
import os
import time
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.vision.pipeline import PerceptionEngine
from src.vision.pass_detector import PassEventDetector
from src.vision.homography import HomographyEstimator
from src.vision.player_geometry import PlayerGeometry
from src.vision.attacking_direction import AttackingDirection, Direction
from src.vision.offside_reference import ReferenceSelector
from src.vision.offside_engine import OffsideEngine, OffsideDecision
from src.vision.offside_visualizer import OffsideVisualizer
from src.vision.pitch_model import PitchModel

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class OffsideEvaluator:
    def __init__(self, video_path: str, model_path: str, calibration_path: str, attacking_dir: str, output_dir: str = "outputs/phase4", debug_geometry: bool = False):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_geometry = debug_geometry
        
        self.perception = PerceptionEngine(model_path=model_path, source_video=video_path, output_video=None)
        self.event_detector = PassEventDetector()
        
        self.homography = HomographyEstimator()
        self._load_calibration(calibration_path)
        
        self.pitch_model = PitchModel()
        self.player_geom = PlayerGeometry(self.homography)
        self.offside_engine = OffsideEngine(uncertainty_margin_m=0.15)
        self.visualizer = OffsideVisualizer(self.homography, self.pitch_model)
        self.visualizer.debug_geometry = self.debug_geometry
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # We assume Team 1 is attacking the requested direction
        d_enum = Direction.ATTACK_RIGHT if attacking_dir == "ATTACK_RIGHT" else Direction.ATTACK_LEFT
        self.attacking_direction = AttackingDirection.create(team_id=1, direction=d_enum)

    def _load_calibration(self, filepath: str):
        if not os.path.exists(filepath):
            logger.warning(f"Calibration file {filepath} not found. Homography will be invalid.")
            return
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        H = np.array(data["homography"])
        if H.shape == (3, 3):
            self.homography.H_matrix = H
            # Load points if we need validation
            img_pts = []
            ptc_pts = []
            for lm in data.get("landmarks", []):
                img_pts.append((lm["image"][0], lm["image"][1]))
                ptc_pts.append((lm["pitch"][0], lm["pitch"][1]))
            
            self.homography.image_points = img_pts
            self.homography.pitch_points = ptc_pts
            self.homography.validate()

    def _generate_calibration_debug(self):
        debug_path = str(self.output_dir / "calibration_debug.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(debug_path, fourcc, self.fps, (self.width, self.height))
        
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        while frame_idx < 100:
            ret, frame = cap.read()
            if not ret: break
            
            # Draw original clicked landmarks (Red)
            for pt in self.homography.image_points:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                
            # Draw projected canonical landmarks (Green)
            for landmark_name, pitch_coord in self.pitch_model.landmarks.items():
                try:
                    ix, iy = self.homography.pitch_to_image(pitch_coord)
                    if 0 <= ix < self.width and 0 <= iy < self.height:
                        cv2.circle(frame, (int(ix), int(iy)), 4, (0, 255, 0), -1)
                        lm_str = landmark_name.name if hasattr(landmark_name, 'name') else str(landmark_name)
                        cv2.putText(frame, lm_str, (int(ix)+5, int(iy)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                except ValueError:
                    pass
                    
            # Draw calibration metrics text
            metrics = self.homography.validation_metrics
            cv2.putText(frame, f"CALIBRATION", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Landmarks: {len(self.homography.image_points)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Mean error: {metrics.get('mean_reprojection_error_m', 0.0):.2f}m", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            writer.write(frame)
            frame_idx += 1
            
        cap.release()
        writer.release()
        logger.info(f"Calibration debug video saved to {debug_path}")

    def run(self, max_frames: int = None):
        if not self.homography.validation_metrics.get("calibration_valid", False):
            logger.error("INVALID PITCH CALIBRATION. Cannot proceed with offside evaluation.")
            sys.exit(1)
            
        self._generate_calibration_debug()
        
        start_time = time.time()
        logger.info("Pass 1: Extracting tracks from video...")
        ball_track, player_tracks = self.perception.generate_full_tracks(max_frames=max_frames)
        
        frames_processed = min(max_frames, self.total_frames) if max_frames else self.total_frames
        
        logger.info("Pass 2: Detecting contact candidates...")
        candidates = self.event_detector.detect_candidates(ball_track, player_tracks)
        candidates.sort(key=lambda x: x.frame_id)
        
        decisions = []
        
        for cand in candidates:
            frame_id = cand.frame_id
            
            # Get tracks at this frame
            frame_players = []
            for p_id, p_track in player_tracks.items():
                p_state = next((s for s in p_track.history if s.frame_id == frame_id), None)
                if p_state:
                    p_state.track_id = p_id
                    if hasattr(p_track, 'team_id'):
                        p_state.team_id = p_track.team_id
                    frame_players.append(p_state)
            ball_state = next((r for r in ball_track.records if r.frame_id == frame_id), None)
            
            # Identify attacking team (for MVP, we assume it's always Team 1)
            # In a real engine, we'd infer the team of the contact player
            attacking_team_id = 1
            contact_player = next((p for p in frame_players if p.track_id == cand.player_id), None)
            if contact_player and hasattr(contact_player, 'team_id'):
                attacking_team_id = contact_player.team_id
                
            self.attacking_direction = AttackingDirection.create(
                team_id=attacking_team_id, 
                direction=self.attacking_direction.direction
            )

            # Project all players
            projected_players = []
            for p in frame_players:
                # Assign team if not assigned (mock for now, assume 1 is attacking, 0 is defending, unless classified)
                tid = p.team_id if hasattr(p, 'team_id') and p.team_id is not None else 0
                proj = self.player_geom.project_player(frame_id, cand.frame_id / self.fps, p, tid)
                projected_players.append(proj)
                
            # Filter attackers and defenders
            attackers = [p for p in projected_players if p.team_id == attacking_team_id]
            defenders = [p for p in projected_players if p.team_id != attacking_team_id]
            
            # Find SLO and Reference
            slo = ReferenceSelector.find_second_last_opponent(defenders, self.attacking_direction)
            
            ball_pitch = None
            ball_image = None
            if ball_state:
                bx, by = self.player_geom.calculate_foot_point(ball_state.bbox)
                ball_image = (bx, by)
                try:
                    px, py = self.homography.image_to_pitch(ball_image)
                    ball_pitch = (px, py)
                except Exception:
                    pass
                    
            reference = ReferenceSelector.get_offside_reference(slo, ball_pitch, self.attacking_direction)
            
            # Evaluate
            decision = self.offside_engine.generate_decision(
                frame_id=frame_id,
                timestamp=cand.frame_id / self.fps,
                attacking_direction=self.attacking_direction,
                contact_player_id=cand.player_id,
                ball_pitch=ball_pitch,
                ball_image=ball_image,
                second_last_opponent=slo,
                reference=reference,
                attackers=attackers
            )
            decisions.append(decision)

        # Generate outputs
        self._export_json(decisions)
        self._generate_visualizations(decisions)
        self._generate_validation_report(frames_processed, time.time() - start_time, ball_track, decisions)
        self.cap.release()

    def _export_json(self, decisions: List[OffsideDecision]):
        out_path = self.output_dir / "offside_events.json"
        data = {
            "total_events": len(decisions),
            "events": [d.to_dict() for d in decisions]
        }
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Exported JSON to {out_path}")

    def _generate_visualizations(self, decisions: List[OffsideDecision], window_frames: int = 25):
        if not decisions:
            return
            
        review_path = str(self.output_dir / "offside_review.mp4")
        debug_path = str(self.output_dir / "offside_topdown_debug.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        review_writer = cv2.VideoWriter(review_path, fourcc, self.fps, (self.width, self.height))
        debug_writer = cv2.VideoWriter(debug_path, fourcc, self.fps, (self.visualizer.radar_width, self.visualizer.radar_height))
        
        # Render intervals
        intervals = []
        frame_to_dec = {}
        for idx, dec in enumerate(decisions):
            sf = max(0, dec.frame_id - window_frames)
            ef = dec.frame_id + window_frames
            intervals.append([sf, ef])
            for f in range(sf, ef + 1):
                frame_to_dec[f] = dec
                
        merged = []
        for sf, ef in sorted(intervals, key=lambda x: x[0]):
            if not merged:
                merged.append([sf, ef])
            else:
                last_sf, last_ef = merged[-1]
                if sf <= last_ef + 15:
                    merged[-1][1] = max(last_ef, ef)
                else:
                    merged.append([sf, ef])
                    
        for sf, ef in merged:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
            curr = sf
            while curr <= ef:
                ret, frame = self.cap.read()
                if not ret: break
                
                if curr in frame_to_dec:
                    dec = frame_to_dec[curr]
                    rev_frame = self.visualizer.draw_broadcast_review(frame, dec)
                    review_writer.write(rev_frame)
                    
                    # Dump radar map for the specific exact candidate frame to the debug video, 
                    # but actually we can draw radar for the entire window holding the state of the candidate frame
                    # since we only have player projections at the contact frame exactly.
                    # Wait, we need to show radar for the whole clip. For MVP, just show the frozen radar of the contact.
                    # Build a dictionary of players from the decision:
                    radar_players = []
                    # We stored attackers in decision, but not defenders. 
                    # To do this right for debug video, we should just re-project all tracks for the frame.
                    # For simplicity, we just use the frozen `attackers` and `slo` in the decision object
                    # for the radar, since radar is static for the 2-second window.
                    radar_players.extend([
                        {'team_id': dec.attacking_team, 'player_id': a.player_id, 'pitch_x': a.pitch_x, 'pitch_y': a.pitch_y}
                        for a in dec.attackers
                    ])
                    if dec.second_last_opponent:
                        slo = dec.second_last_opponent
                        # Opponent team ID is usually not attacking team. Let's say 0.
                        radar_players.append({
                            'team_id': 0, 'player_id': slo['player_id'], 'pitch_x': slo['pitch_x'], 'pitch_y': slo['pitch_y']
                        })
                        
                    radar_frame = self.visualizer.draw_topdown_radar(dec, radar_players)
                    debug_writer.write(radar_frame)
                else:
                    review_writer.write(frame)
                    # For frames outside exact candidate but inside window, we can just write blank radar or last radar
                    # For simplicity we write blank radar
                    blank = np.zeros((self.visualizer.radar_height, self.visualizer.radar_width, 3), dtype=np.uint8)
                    debug_writer.write(blank)
                    
                curr += 1
                
        review_writer.release()
        debug_writer.release()
        logger.info(f"Visualizations saved to {review_path} and {debug_path}")

    def _generate_validation_report(self, frames_processed: int, duration: float, ball_track: Any, decisions: List[OffsideDecision]):
        analyzed = len(decisions)
        onside = sum(1 for d in decisions if d.decision == "ONSIDE_POSITION")
        offside = sum(1 for d in decisions if d.decision == "OFFSIDE_POSITION")
        uncertain = sum(1 for d in decisions if d.decision == "UNCERTAIN")
        
        metrics = self.homography.validation_metrics
        
        ball_cov = 0.0
        longest_gap = 0
        if hasattr(ball_track, "get_diagnostics"):
            diag = ball_track.get_diagnostics()
            ball_cov = diag.get("continuous_coverage_pct", 0.0)
            longest_gap = diag.get("longest_gap", 0)
            
        report = {
            "video": self.video_path,
            "fps": self.fps,
            "frames_processed": frames_processed,
            "duration": round(duration, 2),
            "calibration": {
                "landmark_count": len(self.homography.image_points),
                "mean_reprojection_error": metrics.get("mean_reprojection_error_m", 0.0),
                "median_reprojection_error": metrics.get("median_reprojection_error_m", 0.0),
                "max_reprojection_error": metrics.get("max_reprojection_error_m", 0.0)
            },
            "contacts": {
                "total_candidates": analyzed
            },
            "offside_analysis": {
                "analyzed": analyzed,
                "onside": onside,
                "offside": offside,
                "uncertain": uncertain
            },
            "tracking": {
                "ball_coverage": ball_cov,
                "longest_ball_gap": longest_gap
            }
        }
        
        out_path = self.output_dir / "offside_validation.json"
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Validation report saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--calibration", required=True, help="Path to calibration json")
    parser.add_argument("--attack-dir", required=True, choices=["ATTACK_LEFT", "ATTACK_RIGHT"], help="Attacking direction for Team 1")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--debug-geometry", action="store_true", help="Enable geometry debug mode")
    
    args = parser.parse_args()
    
    evaluator = OffsideEvaluator(
        video_path=args.video,
        model_path=args.model,
        calibration_path=args.calibration,
        attacking_dir=args.attack_dir,
        debug_geometry=args.debug_geometry
    )
    evaluator.run(max_frames=args.max_frames)
