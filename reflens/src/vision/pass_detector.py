import numpy as np
from typing import List, Dict, Optional
from src.detection.types import TrackedEntity
from src.vision.events import PassCandidate

class PassEventDetector:
    def __init__(self, proximity_threshold_px: float = 40.0, min_acceleration: float = 5.0):
        self.prox_thresh = proximity_threshold_px
        self.min_accel = min_acceleration

    def _get_foot_point(self, bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]])

    def _get_center_point(self, bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])

    def detect_candidates(
        self, 
        ball_track: Optional[TrackedEntity], 
        player_tracks: Dict[int, TrackedEntity]
    ) -> List[PassCandidate]:
        if not ball_track or len(ball_track.history) < 5:
            return []

        candidates: List[PassCandidate] = []
        history = sorted(ball_track.history, key=lambda s: s.frame_id)
        
        # Precompute kinematics
        speeds = []
        for i in range(1, len(history)):
            curr, prev = history[i], history[i-1]
            dt = max(1, curr.frame_id - prev.frame_id)
            
            p_curr = self._get_center_point(curr.bbox)
            p_prev = self._get_center_point(prev.bbox)
            
            # Pixels per frame
            dist = float(np.linalg.norm(p_curr - p_prev))
            speeds.append(dist / dt)
            
        # Pad initial speed to match history length
        speeds = [0.0] + speeds 
        
        for i in range(2, len(history) - 2):
            curr_state = history[i]
            speed_before = float(np.mean(speeds[max(0, i-2):i]))
            speed_after = float(np.mean(speeds[i+1:i+3]))
            accel = speed_after - speed_before
            
            if accel < self.min_accel:
                continue
                
            # Find nearest player at this frame
            ball_pos = self._get_center_point(curr_state.bbox)
            min_dist = float('inf')
            closest_player_id = -1
            
            for p_id, p_track in player_tracks.items():
                p_state = next((s for s in p_track.history if s.frame_id == curr_state.frame_id), None)
                if p_state is not None:
                    foot_pos = self._get_foot_point(p_state.bbox)
                    dist = float(np.linalg.norm(ball_pos - foot_pos))
                    if dist < min_dist:
                        min_dist = dist
                        closest_player_id = p_id
                        
            if min_dist <= self.prox_thresh:
                # Base confidence on proximity (closer is better) and acceleration magnitude
                prox_score = max(0.0, 1.0 - (min_dist / self.prox_thresh))
                accel_score = min(1.0, accel / (self.min_accel * 3))
                confidence = float((prox_score * 0.6) + (accel_score * 0.4))
                
                candidates.append(PassCandidate(
                    frame_id=curr_state.frame_id,
                    player_id=closest_player_id,
                    ball_distance_px=float(min_dist),
                    ball_speed_before=float(speed_before),
                    ball_speed_after=float(speed_after),
                    acceleration=float(accel),
                    confidence=float(confidence)
                ))
                
        # Non-maximum suppression to avoid overlapping events
        return self._nms_candidates(candidates)

    def _nms_candidates(self, candidates: List[PassCandidate], window: int = 5) -> List[PassCandidate]:
        if not candidates:
            return []
        
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        kept: List[PassCandidate] = []
        for c in candidates:
            if not any(abs(c.frame_id - k.frame_id) <= window for k in kept):
                kept.append(c)
        return sorted(kept, key=lambda c: c.frame_id)
