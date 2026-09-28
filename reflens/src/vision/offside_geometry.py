import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from src.detection.types import TrackedState
from src.vision.homography import PitchHomography

@dataclass
class OffsideVerdict:
    is_offside: bool
    delta_x_meters: float
    attacker_metric_x: float
    reference_metric_x: float
    second_last_opponent_x: float
    ball_x: float
    attacking_direction: int
    confidence: float

class OffsideEngine:
    def __init__(self, homography: PitchHomography):
        self.homography = homography

    def _get_foot_contact_point(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Extracts the bottom-center of a bounding box. (MVP Approximation)"""
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_bottom = float(bbox[3])
        return x_center, y_bottom

    def evaluate_frame(
        self,
        attacking_direction_x: int,
        attacker: TrackedState,
        defenders: List[TrackedState],
        ball: Optional[TrackedState]
    ) -> OffsideVerdict:
        """
        Determines if the attacker is offside.
        defenders list must include GK and exclude referee before passing.
        """
        if len(defenders) < 2:
            raise ValueError("Not enough defenders to find second-last opponent.")
            
        # Transform attacker
        att_u, att_v = self._get_foot_contact_point(attacker.bbox)
        att_x, att_y = self.homography.pixels_to_meters(att_u, att_v)
        
        # Transform defenders
        def_xs = []
        for d in defenders:
            du, dv = self._get_foot_contact_point(d.bbox)
            dx, dy = self.homography.pixels_to_meters(du, dv)
            def_xs.append(dx)
            
        # Sort defenders by closeness to attacking goal
        # If attacking positive X, higher X is closer to goal
        def_xs_sorted = sorted(def_xs, key=lambda x: x * attacking_direction_x, reverse=True)
        second_last_opponent_x = def_xs_sorted[1]
        
        # Transform ball
        if ball is not None:
            ball_u, ball_v = self._get_foot_contact_point(ball.bbox)
            ball_x, ball_y = self.homography.pixels_to_meters(ball_u, ball_v)
        else:
            ball_x = float('-inf') if attacking_direction_x == 1 else float('inf')
            
        # Reference line is closer of ball or second-last opponent to the goal
        ref_x = second_last_opponent_x
        if ball is not None:
            if (ball_x * attacking_direction_x) > (second_last_opponent_x * attacking_direction_x):
                ref_x = ball_x

        # Calculate spatial delta (positive means attacker is closer to goal than reference)
        delta_x = (att_x - ref_x) * attacking_direction_x
        
        is_offside = delta_x > 0.0
        
        margin_of_error_meters = 0.15 
        confidence = min(1.0, abs(delta_x) / margin_of_error_meters) if abs(delta_x) > 0 else 0.0
        
        return OffsideVerdict(
            is_offside=is_offside,
            delta_x_meters=delta_x,
            attacker_metric_x=att_x,
            reference_metric_x=ref_x,
            second_last_opponent_x=second_last_opponent_x,
            ball_x=ball_x if ball is not None else 0.0,
            attacking_direction=attacking_direction_x,
            confidence=float(confidence)
        )
