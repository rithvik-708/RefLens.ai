import unittest
from src.vision.offside_engine import OffsideEngine
from src.vision.player_geometry import PlayerPitchPosition
from src.vision.attacking_direction import AttackingDirection, Direction
from src.vision.offside_reference import ReferenceSelector

class TestOffsideGeometry(unittest.TestCase):
    def setUp(self):
        self.engine = OffsideEngine(uncertainty_margin_m=0.15)
        self.dir_left = AttackingDirection.create(1, Direction.ATTACK_LEFT)
        self.dir_right = AttackingDirection.create(1, Direction.ATTACK_RIGHT)
        
    def create_player(self, pid, tid, px):
        return PlayerPitchPosition(
            frame_id=1, timestamp=1.0, player_id=pid, team_id=tid,
            image_x=0.0, image_y=0.0, pitch_x=px, pitch_y=34.0, projection_valid=True
        )

    def test_attacker_clearly_onside(self):
        # Attacking RIGHT (Goal at 105). Opponents half > 52.5
        attacker = self.create_player(10, 1, 60.0)
        defenders = [self.create_player(20, 2, 80.0), self.create_player(21, 2, 70.0)]
        slo = ReferenceSelector.find_second_last_opponent(defenders, self.dir_right) # Should be 70.0
        ref = ReferenceSelector.get_offside_reference(slo, None, self.dir_right)
        
        evaluation = self.engine.evaluate_attacker(attacker, ref, self.dir_right)
        self.assertEqual(evaluation.status, "ONSIDE")
        self.assertEqual(evaluation.distance_to_reference, -10.0)

    def test_attacker_clearly_offside(self):
        # Attacking RIGHT
        attacker = self.create_player(10, 1, 75.0)
        defenders = [self.create_player(20, 2, 80.0), self.create_player(21, 2, 70.0)]
        slo = ReferenceSelector.find_second_last_opponent(defenders, self.dir_right) # 70.0
        ref = ReferenceSelector.get_offside_reference(slo, None, self.dir_right)
        
        evaluation = self.engine.evaluate_attacker(attacker, ref, self.dir_right)
        self.assertEqual(evaluation.status, "OFFSIDE")
        self.assertEqual(evaluation.distance_to_reference, 5.0)

    def test_level_with_second_last_opponent(self):
        attacker = self.create_player(10, 1, 70.05)
        defenders = [self.create_player(20, 2, 80.0), self.create_player(21, 2, 70.0)]
        slo = ReferenceSelector.find_second_last_opponent(defenders, self.dir_right)
        ref = ReferenceSelector.get_offside_reference(slo, None, self.dir_right)
        
        evaluation = self.engine.evaluate_attacker(attacker, ref, self.dir_right)
        self.assertEqual(evaluation.status, "UNCERTAIN") # Within 0.15 margin

    def test_ball_is_reference(self):
        # Ball is closer to goal than SLO
        defenders = [self.create_player(20, 2, 80.0), self.create_player(21, 2, 70.0)]
        slo = ReferenceSelector.find_second_last_opponent(defenders, self.dir_right) # 70.0
        ball_pos = (75.0, 34.0)
        ref = ReferenceSelector.get_offside_reference(slo, ball_pos, self.dir_right)
        
        self.assertEqual(ref.type, "BALL")
        self.assertEqual(ref.pitch_x, 75.0)

    def test_slo_is_reference(self):
        # Ball is further from goal than SLO
        defenders = [self.create_player(20, 2, 80.0), self.create_player(21, 2, 70.0)]
        slo = ReferenceSelector.find_second_last_opponent(defenders, self.dir_right) # 70.0
        ball_pos = (60.0, 34.0)
        ref = ReferenceSelector.get_offside_reference(slo, ball_pos, self.dir_right)
        
        self.assertEqual(ref.type, "SECOND_LAST_OPPONENT")
        self.assertEqual(ref.pitch_x, 70.0)
        
    def test_player_in_own_half(self):
        # Player at 40.0, attacking right (own half)
        attacker = self.create_player(10, 1, 40.0)
        defenders = [self.create_player(20, 2, 30.0), self.create_player(21, 2, 20.0)]
        slo = ReferenceSelector.find_second_last_opponent(defenders, self.dir_right) # 30.0
        ref = ReferenceSelector.get_offside_reference(slo, None, self.dir_right)
        
        evaluation = self.engine.evaluate_attacker(attacker, ref, self.dir_right)
        self.assertEqual(evaluation.status, "ONSIDE")

    def test_attack_left(self):
        # Goal at 0
        attacker = self.create_player(10, 1, 20.0)
        defenders = [self.create_player(20, 2, 10.0), self.create_player(21, 2, 30.0)]
        slo = ReferenceSelector.find_second_last_opponent(defenders, self.dir_left) # 30.0 is SLO
        ref = ReferenceSelector.get_offside_reference(slo, None, self.dir_left)
        
        evaluation = self.engine.evaluate_attacker(attacker, ref, self.dir_left)
        # Attacker is at 20, SLO is at 30. Goal at 0. Attacker is closer to goal.
        self.assertEqual(evaluation.status, "OFFSIDE")
        
    def test_multiple_defenders_at_same_x(self):
        # Two defenders at 70.0. Attacking right.
        defenders = [self.create_player(20, 2, 80.0), self.create_player(21, 2, 70.0), self.create_player(22, 2, 70.0)]
        slo = ReferenceSelector.find_second_last_opponent(defenders, self.dir_right)
        # Since sorted, [80, 70, 70], index 1 is 70
        self.assertEqual(slo.pitch_x, 70.0)

if __name__ == '__main__':
    unittest.main()
