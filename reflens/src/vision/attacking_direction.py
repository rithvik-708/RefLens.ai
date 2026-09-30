from dataclasses import dataclass
from enum import Enum

class Direction(Enum):
    ATTACK_LEFT = -1
    ATTACK_RIGHT = 1

@dataclass
class AttackingDirection:
    team_id: int
    direction: Direction
    attacking_goal_x: float
    defending_goal_x: float

    @classmethod
    def create(cls, team_id: int, direction: Direction, pitch_length: float = 105.0) -> 'AttackingDirection':
        if direction == Direction.ATTACK_LEFT:
            attacking_goal_x = 0.0
            defending_goal_x = pitch_length
        else:
            attacking_goal_x = pitch_length
            defending_goal_x = 0.0
            
        return cls(
            team_id=team_id,
            direction=direction,
            attacking_goal_x=attacking_goal_x,
            defending_goal_x=defending_goal_x
        )
