from dataclasses import dataclass
from typing import List, Optional, Tuple
from src.vision.player_geometry import PlayerPitchPosition
from src.vision.attacking_direction import AttackingDirection, Direction

@dataclass
class OffsideReference:
    type: str  # "BALL" or "SECOND_LAST_OPPONENT"
    pitch_x: float
    pitch_y: float
    player_id: Optional[int]

class ReferenceSelector:
    @staticmethod
    def find_second_last_opponent(
        defenders: List[PlayerPitchPosition], 
        attacking_direction: AttackingDirection
    ) -> Optional[PlayerPitchPosition]:
        if len(defenders) < 2:
            return None
            
        valid_defenders = [d for d in defenders if d.projection_valid]
        if len(valid_defenders) < 2:
            return None
            
        # Sort defenders based on distance to the defending goal (which is the attacking team's target goal)
        # If attacking LEFT (goal at 0), smaller X is closer to goal
        # If attacking RIGHT (goal at 105), larger X is closer to goal
        if attacking_direction.direction == Direction.ATTACK_LEFT:
            valid_defenders.sort(key=lambda d: d.pitch_x)
        else:
            valid_defenders.sort(key=lambda d: d.pitch_x, reverse=True)
            
        # The first is the last opponent (often the GK), the second is the second-last opponent
        return valid_defenders[1]

    @staticmethod
    def get_offside_reference(
        second_last_opponent: Optional[PlayerPitchPosition],
        ball_position: Optional[Tuple[float, float]],
        attacking_direction: AttackingDirection
    ) -> Optional[OffsideReference]:
        
        if not second_last_opponent:
            return None
            
        slo_x = second_last_opponent.pitch_x
        slo_y = second_last_opponent.pitch_y
        
        if not ball_position:
            # If no ball, the reference is strictly the second last opponent
            return OffsideReference(
                type="SECOND_LAST_OPPONENT",
                pitch_x=slo_x,
                pitch_y=slo_y,
                player_id=second_last_opponent.player_id
            )
            
        ball_x, ball_y = ball_position
        
        # Determine which is closer to the attacking goal
        if attacking_direction.direction == Direction.ATTACK_LEFT:
            # Closer to goal means smaller X
            if ball_x < slo_x:
                return OffsideReference(
                    type="BALL",
                    pitch_x=ball_x,
                    pitch_y=ball_y,
                    player_id=None
                )
        else:
            # Closer to goal means larger X
            if ball_x > slo_x:
                return OffsideReference(
                    type="BALL",
                    pitch_x=ball_x,
                    pitch_y=ball_y,
                    player_id=None
                )
                
        return OffsideReference(
            type="SECOND_LAST_OPPONENT",
            pitch_x=slo_x,
            pitch_y=slo_y,
            player_id=second_last_opponent.player_id
        )
