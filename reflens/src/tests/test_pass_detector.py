import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.detection.types import TrackedEntity, TrackedState
from src.vision.events import PassCandidate
from src.vision.pass_detector import PassEventDetector
from src.vision.visualizer import VARVisualizer

def create_state(frame_id: int, x: float, y: float, w: float = 20.0, h: float = 40.0) -> TrackedState:
    # bbox format: [x1, y1, x2, y2]
    # center is (x, y), foot is (x, y + h/2)
    bbox = np.array([x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0])
    return TrackedState(
        frame_id=frame_id,
        timestamp_ms=float(frame_id * 40.0), # 25 fps = 40ms/frame
        bbox=bbox,
        x=x,
        y=y
    )

class TestPassDetector(unittest.TestCase):
    def setUp(self):
        self.detector = PassEventDetector(proximity_threshold_px=40.0, min_acceleration=5.0)
        self.visualizer = VARVisualizer()

    def test_pass_detected_on_kick(self):
        # Ball starts stationary near player, then accelerates fast
        # Frame 0-2: stationary at (100, 200)
        # Frame 3+: moves rapidly towards (250, 200)
        ball_track = TrackedEntity(track_id=1, class_id=0)
        ball_positions = [
            (0, 100.0, 200.0),
            (1, 100.0, 200.0),
            (2, 100.0, 200.0),
            (3, 120.0, 200.0),
            (4, 150.0, 200.0),
            (5, 180.0, 200.0),
        ]
        for f, x, y in ball_positions:
            ball_track.history.append(create_state(f, x, y, w=10.0, h=10.0))

        # Player at frame 2 whose foot is near (100, 200)
        # Foot point is (x, y + h/2) -> if foot is (100, 200) and h=40, center is (100, 180)
        player_track = TrackedEntity(track_id=10, class_id=1)
        for f in range(6):
            player_track.history.append(create_state(f, 100.0, 180.0, w=20.0, h=40.0))

        candidates = self.detector.detect_candidates(ball_track, {10: player_track})
        self.assertGreater(len(candidates), 0)
        top = candidates[0]
        self.assertEqual(top.player_id, 10)
        self.assertGreater(top.confidence, 0.0)
        self.assertGreaterEqual(top.acceleration, 5.0)

    def test_low_acceleration_ignored(self):
        # Ball rolling at constant speed (no acceleration spike)
        ball_track = TrackedEntity(track_id=1, class_id=0)
        for f in range(10):
            ball_track.history.append(create_state(f, 100.0 + f * 5.0, 200.0, w=10.0, h=10.0))

        player_track = TrackedEntity(track_id=10, class_id=1)
        for f in range(10):
            player_track.history.append(create_state(f, 100.0 + f * 5.0, 180.0, w=20.0, h=40.0))

        candidates = self.detector.detect_candidates(ball_track, {10: player_track})
        self.assertEqual(len(candidates), 0)

    def test_distant_player_ignored(self):
        # Acceleration spike occurs, but player is far away (e.g. 200px away)
        ball_track = TrackedEntity(track_id=1, class_id=0)
        ball_positions = [
            (0, 100.0, 200.0),
            (1, 100.0, 200.0),
            (2, 100.0, 200.0),
            (3, 130.0, 200.0),
            (4, 170.0, 200.0),
            (5, 210.0, 200.0),
        ]
        for f, x, y in ball_positions:
            ball_track.history.append(create_state(f, x, y, w=10.0, h=10.0))

        player_track = TrackedEntity(track_id=10, class_id=1)
        for f in range(6):
            player_track.history.append(create_state(f, 500.0, 500.0, w=20.0, h=40.0))

        candidates = self.detector.detect_candidates(ball_track, {10: player_track})
        self.assertEqual(len(candidates), 0)

    def test_short_ball_history(self):
        ball_track = TrackedEntity(track_id=1, class_id=0)
        ball_track.history = [create_state(0, 100, 100), create_state(1, 105, 100)]
        candidates = self.detector.detect_candidates(ball_track, {})
        self.assertEqual(len(candidates), 0)

    def test_nms_candidates(self):
        candidates = [
            PassCandidate(frame_id=10, player_id=1, ball_distance_px=10, ball_speed_before=1, ball_speed_after=10, acceleration=9, confidence=0.8),
            PassCandidate(frame_id=11, player_id=1, ball_distance_px=8, ball_speed_before=1, ball_speed_after=12, acceleration=11, confidence=0.9),
            PassCandidate(frame_id=12, player_id=1, ball_distance_px=15, ball_speed_before=2, ball_speed_after=11, acceleration=9, confidence=0.7),
            PassCandidate(frame_id=30, player_id=2, ball_distance_px=5, ball_speed_before=0, ball_speed_after=15, acceleration=15, confidence=0.95),
        ]
        filtered = self.detector._nms_candidates(candidates, window=5)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].frame_id, 11)
        self.assertEqual(filtered[1].frame_id, 30)

    def test_visualizer_overlay(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        candidate = PassCandidate(
            frame_id=100,
            player_id=7,
            ball_distance_px=12.5,
            ball_speed_before=1.2,
            ball_speed_after=18.4,
            acceleration=17.2,
            confidence=0.85
        )
        overlayed = self.visualizer.draw_candidate_overlay(frame, candidate)
        self.assertEqual(overlayed.shape, frame.shape)
        self.assertFalse(np.array_equal(overlayed, frame))

if __name__ == "__main__":
    unittest.main()
