from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional
from src.vision.homography import HomographyEstimator
from src.detection.types import TrackedState

@dataclass
class PlayerPitchPosition:
    frame_id: int
    timestamp: float
    player_id: int
    team_id: int
    image_x: float
    image_y: float
    pitch_x: float
    pitch_y: float
    projection_valid: bool

class PlayerGeometry:
    def __init__(self, homography: HomographyEstimator):
        self.homography = homography

    def calculate_foot_point(self, bbox: np.ndarray) -> Tuple[float, float]:
        """
        V1 estimates the player's ground position using the bottom-center of the detected bounding box.
        This is an MVP approximation, not an exact body-part localization.
        """
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_bottom = float(bbox[3])
        return x_center, y_bottom

    def project_player(self, frame_id: int, timestamp: float, player: TrackedState, team_id: int) -> PlayerPitchPosition:
        image_x, image_y = self.calculate_foot_point(player.bbox)
        
        try:
            pitch_x, pitch_y = self.homography.image_to_pitch((image_x, image_y))
            projection_valid = True
        except Exception:
            pitch_x, pitch_y = 0.0, 0.0
            projection_valid = False
            
        return PlayerPitchPosition(
            frame_id=frame_id,
            timestamp=timestamp,
            player_id=player.track_id,
            team_id=team_id,
            image_x=image_x,
            image_y=image_y,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            projection_valid=projection_valid
        )
