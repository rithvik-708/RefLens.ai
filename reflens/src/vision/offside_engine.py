from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from src.vision.player_geometry import PlayerPitchPosition
from src.vision.attacking_direction import AttackingDirection, Direction
from src.vision.offside_reference import OffsideReference

@dataclass
class AttackerEvaluation:
    player_id: int
    team_id: int
    pitch_x: float
    pitch_y: float
    status: str  # ONSIDE, OFFSIDE, UNCERTAIN
    distance_to_reference: float

@dataclass
class OffsideDecision:
    frame_id: int
    timestamp: float
    attacking_team: int
    attacking_direction: str
    contact_player_id: int
    ball_position: Optional[Dict[str, float]]
    second_last_opponent: Optional[Dict[str, float]]
    reference: Optional[Dict[str, float]]
    attackers: List[AttackerEvaluation]
    decision: str
    uncertainty_margin_m: float
    explanation: str
    
    def to_dict(self):
        return asdict(self)

class OffsideEngine:
    def __init__(self, uncertainty_margin_m: float = 0.15):
        self.uncertainty_margin_m = uncertainty_margin_m

    def evaluate_attacker(
        self,
        attacker: PlayerPitchPosition,
        reference: OffsideReference,
        attacking_direction: AttackingDirection,
        pitch_length: float = 105.0
    ) -> AttackerEvaluation:
        
        if not attacker.projection_valid:
            return AttackerEvaluation(
                player_id=attacker.player_id,
                team_id=attacker.team_id,
                pitch_x=attacker.pitch_x,
                pitch_y=attacker.pitch_y,
                status="UNCERTAIN",
                distance_to_reference=0.0
            )

        att_x = attacker.pitch_x
        ref_x = reference.pitch_x
        
        # 1. Determine whether the player is in the opponents' half
        # Halfway line is at pitch_length / 2 (52.5)
        # If attacking LEFT, opponents' half is X < 52.5
        # If attacking RIGHT, opponents' half is X > 52.5
        halfway_line = pitch_length / 2.0
        in_opponents_half = False
        
        if attacking_direction.direction == Direction.ATTACK_LEFT:
            in_opponents_half = att_x < halfway_line
        else:
            in_opponents_half = att_x > halfway_line
            
        # Calculate geometric distance relative to reference
        # Positive distance means closer to goal than reference
        if attacking_direction.direction == Direction.ATTACK_LEFT:
            distance_beyond_ref = ref_x - att_x
        else:
            distance_beyond_ref = att_x - ref_x
            
        status = "UNCERTAIN"
        
        if not in_opponents_half:
            # Player in own half (or exactly on halfway line) is not in offside position
            status = "ONSIDE"
        else:
            # Player is in opponents' half
            if abs(distance_beyond_ref) <= self.uncertainty_margin_m:
                # Player is "level" with reference within uncertainty margin
                # A player level with second-last opponent is NOT offside.
                # However, since it's within the margin of error, we can either call it ONSIDE or UNCERTAIN.
                # Law says level is onside, so if we can't be sure they are beyond, they are onside (or we say uncertain).
                # The prompt says: "Use UNCERTAIN whenever calibration/tracking geometry is insufficient."
                # But it also says: "If the separation is comfortably beyond the uncertainty margin: OFFSIDE. If comfortably behind: ONSIDE. If smaller than margin: UNCERTAIN."
                status = "UNCERTAIN"
            elif distance_beyond_ref > self.uncertainty_margin_m:
                status = "OFFSIDE"
            elif distance_beyond_ref < -self.uncertainty_margin_m:
                status = "ONSIDE"
                
        return AttackerEvaluation(
            player_id=attacker.player_id,
            team_id=attacker.team_id,
            pitch_x=att_x,
            pitch_y=attacker.pitch_y,
            status=status,
            distance_to_reference=distance_beyond_ref
        )

    def generate_decision(
        self,
        frame_id: int,
        timestamp: float,
        attacking_direction: AttackingDirection,
        contact_player_id: int,
        ball_pitch: Optional[Tuple[float, float]],
        ball_image: Optional[Tuple[float, float]],
        second_last_opponent: Optional[PlayerPitchPosition],
        reference: Optional[OffsideReference],
        attackers: List[PlayerPitchPosition]
    ) -> OffsideDecision:
        
        ball_dict = None
        if ball_pitch and ball_image:
            ball_dict = {
                "image_x": ball_image[0],
                "image_y": ball_image[1],
                "pitch_x": ball_pitch[0],
                "pitch_y": ball_pitch[1]
            }
            
        slo_dict = None
        if second_last_opponent:
            slo_dict = {
                "player_id": second_last_opponent.player_id,
                "image_x": second_last_opponent.image_x,
                "image_y": second_last_opponent.image_y,
                "pitch_x": second_last_opponent.pitch_x,
                "pitch_y": second_last_opponent.pitch_y
            }
            
        ref_dict = None
        if reference:
            ref_dict = {
                "type": reference.type,
                "pitch_x": reference.pitch_x,
                "pitch_y": reference.pitch_y
            }

        evaluated_attackers = []
        offside_count = 0
        uncertain_count = 0
        
        for att in attackers:
            # Skip the contact player themselves from being offside (they are playing the ball)
            # Actually, the contact player can't be offside from their own pass, but we'll still evaluate them to display.
            if not reference:
                eval_status = "UNCERTAIN"
                dist = 0.0
            else:
                evaluation = self.evaluate_attacker(att, reference, attacking_direction)
                eval_status = evaluation.status
                dist = evaluation.distance_to_reference
                
            att_eval = AttackerEvaluation(
                player_id=att.player_id,
                team_id=att.team_id,
                pitch_x=att.pitch_x,
                pitch_y=att.pitch_y,
                status=eval_status,
                distance_to_reference=dist
            )
            evaluated_attackers.append(att_eval)
            
            if att.player_id != contact_player_id:
                if eval_status == "OFFSIDE":
                    offside_count += 1
                elif eval_status == "UNCERTAIN":
                    uncertain_count += 1
                    
        if not reference:
            decision = "UNCERTAIN"
            explanation = "Could not determine offside reference."
        elif offside_count > 0:
            decision = "OFFSIDE_POSITION"
            offending = [a for a in evaluated_attackers if a.status == "OFFSIDE" and a.player_id != contact_player_id]
            off_ids = [str(a.player_id) for a in offending]
            explanation = f"Player(s) {', '.join(off_ids)} were beyond the offside reference at the detected contact frame."
        elif uncertain_count > 0:
            decision = "UNCERTAIN"
            explanation = "One or more attackers were within the uncertainty margin or had invalid projections."
        else:
            decision = "ONSIDE_POSITION"
            explanation = "All attackers were comfortably behind the offside reference."

        return OffsideDecision(
            frame_id=frame_id,
            timestamp=timestamp,
            attacking_team=attacking_direction.team_id,
            attacking_direction=attacking_direction.direction.name,
            contact_player_id=contact_player_id,
            ball_position=ball_dict,
            second_last_opponent=slo_dict,
            reference=ref_dict,
            attackers=evaluated_attackers,
            decision=decision,
            uncertainty_margin_m=self.uncertainty_margin_m,
            explanation=explanation
        )
