from dataclasses import dataclass
from typing import Optional

@dataclass
class PassCandidate:
    frame_id: int
    player_id: int
    ball_distance_px: float
    ball_speed_before: float
    ball_speed_after: float
    acceleration: float
    confidence: float
