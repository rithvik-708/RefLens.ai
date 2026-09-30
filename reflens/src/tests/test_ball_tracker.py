import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.detection.types import Detection
from src.detection.ball_tracker import BallTracker, BallTrackStatus


def make_ball_det(x: float, y: float, w: float = 14.0, h: float = 14.0, conf: float = 0.8) -> Detection:
    bbox = np.array([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], dtype=np.float32)
    return Detection(
        bbox=bbox,
        confidence=conf,
        class_id=32,
        class_name="sports_ball"
    )


class TestBallTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = BallTracker(max_missed_frames=5)

    def test_normal_movement(self):
        # Ball moves at 5 px/frame across 10 consecutive frames
        pos_x = 100.0
        pos_y = 200.0
        records = []
        for f in range(10):
            det = make_ball_det(pos_x + f * 5.0, pos_y, conf=0.85)
            rec = self.tracker.update([det], frame_id=f, timestamp_ms=f * 40.0)
            records.append(rec)

        ball_track = self.tracker.finalize_track()
        self.assertEqual(len(ball_track.history), 10)
        self.assertEqual(ball_track.diagnostics["total_raw_ball_detections"], 10)
        self.assertEqual(ball_track.diagnostics["frames_with_ball_detection"], 10)
        self.assertEqual(ball_track.diagnostics["final_continuous_track_coverage"], 100.0)
        self.assertEqual(records[-1].status, BallTrackStatus.TRACKED)
        self.assertAlmostEqual(records[-1].center[0], 145.0, delta=1.0)

    def test_fast_movement_with_zero_iou(self):
        # Ball diameter is 10px, moves 60px/frame (Zero IoU between consecutive bboxes)
        # IoU-based tracker would fail; velocity prediction + adaptive gating should succeed.
        det0 = make_ball_det(100.0, 200.0, w=10.0, h=10.0)
        rec0 = self.tracker.update([det0], frame_id=0, timestamp_ms=0.0)
        self.assertEqual(rec0.status, BallTrackStatus.TRACKED)

        # Move by 50px
        det1 = make_ball_det(150.0, 200.0, w=10.0, h=10.0)
        rec1 = self.tracker.update([det1], frame_id=1, timestamp_ms=40.0)
        self.assertEqual(rec1.status, BallTrackStatus.TRACKED)

        # Move by another 60px -> center 210.0 (Zero IoU with [145, 195, 155, 205])
        det2 = make_ball_det(210.0, 200.0, w=10.0, h=10.0)
        rec2 = self.tracker.update([det2], frame_id=2, timestamp_ms=80.0)
        self.assertEqual(rec2.status, BallTrackStatus.TRACKED)
        self.assertAlmostEqual(rec2.center[0], 210.0, delta=1.0)
        self.assertGreater(rec2.speed, 25.0)

    def test_detection_gaps_and_prediction(self):
        # 1-3 frame detection gaps: frames 0-2 detected, frames 3, 4, 5 missed, frame 6 detected
        self.tracker.update([make_ball_det(100.0, 200.0)], frame_id=0, timestamp_ms=0.0)
        self.tracker.update([make_ball_det(110.0, 200.0)], frame_id=1, timestamp_ms=40.0)
        self.tracker.update([make_ball_det(120.0, 200.0)], frame_id=2, timestamp_ms=80.0)

        # Frame 3: gap 1
        rec3 = self.tracker.update([], frame_id=3, timestamp_ms=120.0)
        self.assertEqual(rec3.status, BallTrackStatus.PREDICTED)
        self.assertTrue(rec3.is_interpolated)

        # Frame 4: gap 2
        rec4 = self.tracker.update([], frame_id=4, timestamp_ms=160.0)
        self.assertEqual(rec4.status, BallTrackStatus.PREDICTED)

        # Frame 5: gap 3
        rec5 = self.tracker.update([], frame_id=5, timestamp_ms=200.0)
        self.assertEqual(rec5.status, BallTrackStatus.PREDICTED)

        # Total history length up to frame 5 is 6 (continuous coverage)
        self.assertEqual(len(self.tracker.ball_track.history), 6)

    def test_reconnecting_after_gap(self):
        # Frames 0, 1 detected. Frames 2, 3 missed (2-frame gap). Frame 4 detected at predicted location.
        self.tracker.update([make_ball_det(100.0, 200.0)], frame_id=0, timestamp_ms=0.0)
        self.tracker.update([make_ball_det(120.0, 200.0)], frame_id=1, timestamp_ms=40.0)

        # Missed at frame 2, 3
        self.tracker.update([], frame_id=2, timestamp_ms=80.0)
        self.tracker.update([], frame_id=3, timestamp_ms=120.0)

        # Reappears at frame 4 at (180.0, 200.0)
        rec4 = self.tracker.update([make_ball_det(180.0, 200.0)], frame_id=4, timestamp_ms=160.0)
        self.assertEqual(rec4.status, BallTrackStatus.RECONNECTED)

        track = self.tracker.finalize_track()
        self.assertEqual(track.diagnostics["longest_gap"], 2)
        # All frame IDs 0, 1, 2, 3, 4 must exist in history
        frame_ids = [s.frame_id for s in track.history]
        self.assertEqual(frame_ids, [0, 1, 2, 3, 4])

    def test_rejecting_implausibly_distant_detection(self):
        # Ball at (100, 100) with velocity (0, 0).
        # A false positive detection appears at (800, 800) -> distance ~990px (exceeds max_phys_speed/gate)
        self.tracker.update([make_ball_det(100.0, 100.0)], frame_id=0, timestamp_ms=0.0)
        
        # Far false positive
        rec1 = self.tracker.update([make_ball_det(800.0, 800.0)], frame_id=1, timestamp_ms=40.0)
        
        # Should NOT associate with (800, 800); should treat as missed/predicted or lost
        self.assertEqual(rec1.status, BallTrackStatus.PREDICTED)
        self.assertLess(rec1.center[0], 200.0)
        self.assertLess(rec1.center[1], 200.0)


if __name__ == "__main__":
    unittest.main()
