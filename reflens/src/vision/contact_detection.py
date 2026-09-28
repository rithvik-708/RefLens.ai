import numpy as np
from typing import Dict, List, Optional
from src.detection.types import TrackedEntity

class PassEventDetector:
    def __init__(self, distance_threshold_px: float = 30.0, velocity_inflection_thresh: float = 1.5):
        self.dist_thresh = distance_threshold_px
        self.vel_thresh = velocity_inflection_thresh
        
    def find_t_pass(self, ball_track: TrackedEntity, player_tracks: List[TrackedEntity]) -> Optional[int]:
        """
        Detects the pass frame based on minimal Euclidean distance and velocity inflection.
        """
        if not ball_track.history:
            return None
            
        pass_candidates = []
        
        for state in ball_track.history:
            # 1. Calculate velocity magnitude
            v_mag = np.sqrt(state.v_x**2 + state.v_y**2)
            
            # Find the closest player at this frame
            min_dist = float('inf')
            closest_player_id = None
            
            for player in player_tracks:
                p_state = next((s for s in player.history if s.frame_id == state.frame_id), None)
                if p_state:
                    dist = np.sqrt((state.x - p_state.x)**2 + (state.y - p_state.y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_player_id = player.track_id
                        
            # 2. Check conditions: proximity to foot and potential velocity spike
            if min_dist < self.dist_thresh:
                pass_candidates.append({
                    "frame_id": state.frame_id,
                    "velocity": v_mag,
                    "distance": min_dist
                })
                
        if not pass_candidates:
            return None
            
        # 3. Locate the inflection point (highest velocity delta right after proximity)
        candidates_sorted = sorted(pass_candidates, key=lambda x: x["frame_id"])
        best_frame = None
        max_v_delta = 0.0
        
        for i in range(1, len(candidates_sorted)):
            v_delta = candidates_sorted[i]["velocity"] - candidates_sorted[i-1]["velocity"]
            if v_delta > max_v_delta and v_delta > self.vel_thresh:
                max_v_delta = v_delta
                best_frame = candidates_sorted[i]["frame_id"]
                
        return best_frame
