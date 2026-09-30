import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger

from src.detection.types import Detection, TrackedState, TrackedEntity


class BallTrackStatus(Enum):
    INACTIVE = "INACTIVE"
    TRACKED = "TRACKED"
    PREDICTED = "PREDICTED"
    RECONNECTED = "RECONNECTED"
    LOST = "LOST"


@dataclass
class BallStateRecord:
    frame_id: int
    timestamp_ms: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    center: np.ndarray  # [x, y]
    predicted_center: Optional[np.ndarray]  # [x, y]
    velocity: np.ndarray  # [v_x, v_y] in px/frame
    speed: float  # px/frame
    status: BallTrackStatus
    confidence: float
    is_interpolated: bool = False
    gate_radius: float = 0.0


@dataclass
class BallTrack(TrackedEntity):
    """
    Continuous Ball Track representation that adheres to TrackedEntity interface
    while maintaining continuous frame IDs, gap bridging, and rich diagnostics.
    """
    track_id: int = 0
    class_id: int = 32
    records: List[BallStateRecord] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def get_diagnostics(self) -> Dict[str, Any]:
        return self.diagnostics


class BallTracker:
    """
    Specialized ball tracker using center coordinates, velocity kinematics,
    adaptive distance gating, and gap reconnection.
    """
    def __init__(
        self,
        max_missed_frames: int = 5,
        min_detection_conf: float = 0.15,
        r_min: float = 35.0,
        r_base: float = 25.0,
        alpha_vel: float = 1.5,
        beta_gap: float = 12.0,
        max_phys_speed: float = 220.0,  # Max plausible px/frame jump
        vel_smoothing: float = 0.35,
        drag_damping: float = 0.98,
        default_ball_size: Tuple[float, float] = (14.0, 14.0)
    ):
        self.max_missed_frames = max_missed_frames
        self.min_detection_conf = min_detection_conf
        self.r_min = r_min
        self.r_base = r_base
        self.alpha_vel = alpha_vel
        self.beta_gap = beta_gap
        self.max_phys_speed = max_phys_speed
        self.vel_smoothing = vel_smoothing
        self.drag_damping = drag_damping
        self.default_ball_size = default_ball_size

        # Internal state
        self.ball_track = BallTrack()
        self.status = BallTrackStatus.INACTIVE
        self.pos: Optional[np.ndarray] = None  # [x, y]
        self.vel: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)  # px/frame
        self.last_bbox: Optional[np.ndarray] = None
        self.last_frame_id: Optional[int] = None
        self.last_timestamp_ms: Optional[float] = None
        self.missed_frames: int = 0
        self.last_conf: float = 0.0

        # Fragment and connection tracking
        self.last_detected_pos: Optional[np.ndarray] = None
        self.last_detected_frame: Optional[int] = None
        self.last_detected_timestamp: Optional[float] = None
        self.last_detected_bbox: Optional[np.ndarray] = None

        # Diagnostics counters
        self.total_raw_ball_detections: int = 0
        self.frames_with_ball_detection: int = 0
        self.number_of_ball_track_fragments: int = 0
        self.longest_gap: int = 0
        self.processed_frames: int = 0
        self.recorded_speeds: List[float] = []

    def reset(self):
        """Resets tracker state."""
        self.ball_track = BallTrack()
        self.status = BallTrackStatus.INACTIVE
        self.pos = None
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.last_bbox = None
        self.last_frame_id = None
        self.last_timestamp_ms = None
        self.missed_frames = 0
        self.last_conf = 0.0
        self.last_detected_pos = None
        self.last_detected_frame = None
        self.last_detected_timestamp = None
        self.last_detected_bbox = None
        self.total_raw_ball_detections = 0
        self.frames_with_ball_detection = 0
        self.number_of_ball_track_fragments = 0
        self.longest_gap = 0
        self.processed_frames = 0
        self.recorded_speeds = []

    def _get_center(self, bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)

    def _make_bbox(self, center: np.ndarray, w: float, h: float) -> np.ndarray:
        return np.array([
            center[0] - w / 2.0,
            center[1] - h / 2.0,
            center[0] + w / 2.0,
            center[1] + h / 2.0
        ], dtype=np.float32)

    def _compute_gate_radius(self, speed: float, gap_frames: int) -> float:
        # If ball is nearly stationary, allow sudden kick radius (up to 80px/frame)
        base = self.r_base if speed > 10.0 else max(self.r_base, 65.0)
        radius = max(self.r_min, base + self.alpha_vel * speed + self.beta_gap * gap_frames)
        return min(radius, self.max_phys_speed)

    def update(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp_ms: float
    ) -> BallStateRecord:
        """
        Updates the ball tracker with new detections for frame_id.
        """
        self.processed_frames += 1

        # 1. Filter ball detections (COCO class 32)
        ball_dets = [
            d for d in detections 
            if d.class_id == 32 and d.confidence >= self.min_detection_conf
        ]
        self.total_raw_ball_detections += len(ball_dets)
        if len(ball_dets) > 0:
            self.frames_with_ball_detection += 1

        # 2. Predict next position if active
        speed = float(np.linalg.norm(self.vel))
        predicted_pos: Optional[np.ndarray] = None
        gate_radius = self._compute_gate_radius(speed, self.missed_frames)

        if self.pos is not None and self.last_frame_id is not None:
            dt_frames = max(1, frame_id - self.last_frame_id)
            damping = self.drag_damping ** dt_frames
            predicted_pos = self.pos + (self.vel * dt_frames * damping)
        else:
            predicted_pos = None

        # 3. Associate detections via Adaptive Distance Gating
        best_det: Optional[Detection] = None
        best_dist = float('inf')

        if predicted_pos is not None:
            for det in ball_dets:
                c = self._get_center(det.bbox)
                dist = float(np.linalg.norm(c - predicted_pos))
                if dist <= gate_radius and dist < best_dist:
                    best_dist = dist
                    best_det = det
        elif len(ball_dets) > 0:
            # First initialization or picking highest confidence candidate
            best_det = max(ball_dets, key=lambda d: d.confidence)

        # 4. State Update & Reconnection Handling
        if best_det is not None:
            det_pos = self._get_center(best_det.bbox)
            det_w = float(best_det.bbox[2] - best_det.bbox[0])
            det_h = float(best_det.bbox[3] - best_det.bbox[1])

            if self.pos is None or self.status == BallTrackStatus.INACTIVE:
                # Fresh initialization
                self.number_of_ball_track_fragments += 1
                self.vel = np.array([0.0, 0.0], dtype=np.float32)
                cur_status = BallTrackStatus.TRACKED
            elif self.missed_frames > 0:
                # Reconnection after a gap <= max_missed_frames
                gap_len = self.missed_frames
                if gap_len > self.longest_gap:
                    self.longest_gap = gap_len

                # Remove provisional predicted records during the gap to replace with smooth interpolated states
                if gap_len > 0 and len(self.ball_track.records) >= gap_len:
                    self.ball_track.records = self.ball_track.records[:-gap_len]
                    self.ball_track.history = self.ball_track.history[:-gap_len]

                # Linearly interpolate missing intermediate frames for a seamless continuous track
                if self.last_detected_frame is not None and self.last_detected_pos is not None:
                    prev_f = self.last_detected_frame
                    prev_t = self.last_detected_timestamp or timestamp_ms
                    prev_p = self.last_detected_pos
                    prev_bb = self.last_detected_bbox if self.last_detected_bbox is not None else best_det.bbox
                    prev_w = float(prev_bb[2] - prev_bb[0])
                    prev_h = float(prev_bb[3] - prev_bb[1])

                    total_f_delta = frame_id - prev_f
                    for gap_f in range(prev_f + 1, frame_id):
                        alpha = (gap_f - prev_f) / float(total_f_delta)
                        interp_pos = (1.0 - alpha) * prev_p + alpha * det_pos
                        interp_w = (1.0 - alpha) * prev_w + alpha * det_w
                        interp_h = (1.0 - alpha) * prev_h + alpha * det_h
                        interp_bbox = self._make_bbox(interp_pos, interp_w, interp_h)
                        interp_t = prev_t + alpha * (timestamp_ms - prev_t)
                        interp_vel = (det_pos - prev_p) / float(total_f_delta)
                        interp_speed = float(np.linalg.norm(interp_vel))
                        self.recorded_speeds.append(interp_speed)

                        interp_rec = BallStateRecord(
                            frame_id=gap_f,
                            timestamp_ms=interp_t,
                            bbox=interp_bbox,
                            center=interp_pos,
                            predicted_center=interp_pos,
                            velocity=interp_vel,
                            speed=interp_speed,
                            status=BallTrackStatus.PREDICTED,
                            confidence=float(best_det.confidence * 0.7),
                            is_interpolated=True,
                            gate_radius=gate_radius
                        )
                        self.ball_track.records.append(interp_rec)
                        self._append_tracked_state(interp_rec)

                # Instantaneous velocity across the gap
                dt_frames = max(1, frame_id - (self.last_detected_frame or (frame_id - 1)))
                inst_vel = (det_pos - (self.last_detected_pos if self.last_detected_pos is not None else det_pos)) / float(dt_frames)
                self.vel = (1.0 - self.vel_smoothing) * self.vel + self.vel_smoothing * inst_vel
                cur_status = BallTrackStatus.RECONNECTED
            else:
                # Continuous consecutive track
                dt_frames = max(1, frame_id - self.last_frame_id) if self.last_frame_id else 1
                inst_vel = (det_pos - self.pos) / float(dt_frames)
                self.vel = (1.0 - self.vel_smoothing) * self.vel + self.vel_smoothing * inst_vel
                cur_status = BallTrackStatus.TRACKED

            self.pos = det_pos
            self.last_bbox = best_det.bbox
            self.last_frame_id = frame_id
            self.last_timestamp_ms = timestamp_ms
            self.missed_frames = 0
            self.last_conf = float(best_det.confidence)
            self.status = cur_status
            self.last_detected_pos = det_pos.copy()
            self.last_detected_frame = frame_id
            self.last_detected_timestamp = timestamp_ms
            self.last_detected_bbox = best_det.bbox.copy()

            cur_speed = float(np.linalg.norm(self.vel))
            self.recorded_speeds.append(cur_speed)

            record = BallStateRecord(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                bbox=best_det.bbox,
                center=det_pos,
                predicted_center=predicted_pos if predicted_pos is not None else det_pos,
                velocity=self.vel.copy(),
                speed=cur_speed,
                status=cur_status,
                confidence=self.last_conf,
                is_interpolated=False,
                gate_radius=gate_radius
            )
            self.ball_track.records.append(record)
            self._append_tracked_state(record)
            return record

        else:
            # 5. Missed Detection
            self.missed_frames += 1

            if self.missed_frames <= self.max_missed_frames and predicted_pos is not None:
                # Maintain prediction within allowed gap
                self.pos = predicted_pos
                bw, bh = self.default_ball_size
                if self.last_bbox is not None:
                    bw = float(self.last_bbox[2] - self.last_bbox[0])
                    bh = float(self.last_bbox[3] - self.last_bbox[1])
                pred_bbox = self._make_bbox(self.pos, bw, bh)

                self.last_frame_id = frame_id
                self.last_timestamp_ms = timestamp_ms
                self.status = BallTrackStatus.PREDICTED
                cur_speed = float(np.linalg.norm(self.vel))
                self.recorded_speeds.append(cur_speed)

                record = BallStateRecord(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    bbox=pred_bbox,
                    center=self.pos,
                    predicted_center=predicted_pos,
                    velocity=self.vel.copy(),
                    speed=cur_speed,
                    status=BallTrackStatus.PREDICTED,
                    confidence=max(0.05, self.last_conf * (0.85 ** self.missed_frames)),
                    is_interpolated=True,
                    gate_radius=gate_radius
                )
                self.ball_track.records.append(record)
                self._append_tracked_state(record)
                return record
            else:
                # Exceeded max_missed_frames -> Track lost/inactive
                self.status = BallTrackStatus.LOST
                dummy_pos = self.pos if self.pos is not None else np.array([0.0, 0.0], dtype=np.float32)
                bw, bh = self.default_ball_size
                dummy_bbox = self.last_bbox if self.last_bbox is not None else np.array([0.0, 0.0, bw, bh], dtype=np.float32)

                record = BallStateRecord(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    bbox=dummy_bbox,
                    center=dummy_pos,
                    predicted_center=predicted_pos,
                    velocity=np.array([0.0, 0.0], dtype=np.float32),
                    speed=0.0,
                    status=BallTrackStatus.LOST,
                    confidence=0.0,
                    is_interpolated=False,
                    gate_radius=gate_radius
                )
                self.pos = None
                return record

    def _append_tracked_state(self, record: BallStateRecord):
        """Appends a TrackedState to ball_track.history for pipeline compatibility."""
        vx_sec = float(record.velocity[0] * 25.0)  # approx px/sec
        vy_sec = float(record.velocity[1] * 25.0)
        
        st = TrackedState(
            frame_id=record.frame_id,
            timestamp_ms=record.timestamp_ms,
            bbox=record.bbox,
            x=float(record.center[0]),
            y=float(record.center[1]),
            v_x=vx_sec,
            v_y=vy_sec
        )
        self.ball_track.history.append(st)

    def finalize_track(self) -> BallTrack:
        """Computes diagnostics and returns final continuous BallTrack."""
        valid_records = [r for r in self.ball_track.records if r.status != BallTrackStatus.LOST and r.status != BallTrackStatus.INACTIVE]
        coverage_pct = (len(valid_records) / max(1, self.processed_frames)) * 100.0
        
        speeds = [r.speed for r in valid_records if r.speed > 0]
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0

        self.ball_track.diagnostics = {
            "total_raw_ball_detections": self.total_raw_ball_detections,
            "frames_with_ball_detection": self.frames_with_ball_detection,
            "number_of_ball_track_fragments": max(1, self.number_of_ball_track_fragments),
            "final_continuous_track_coverage": round(coverage_pct, 2),
            "total_processed_frames": self.processed_frames,
            "valid_track_frames": len(valid_records),
            "longest_gap": self.longest_gap,
            "average_ball_speed": round(avg_speed, 2),
            "max_ball_speed": round(max_speed, 2)
        }
        return self.ball_track


def render_ball_debug_overlay(
    frame: np.ndarray,
    record: BallStateRecord,
    raw_ball_detections: Optional[List[Detection]] = None,
    diagnostics: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Renders detailed ball tracking debug visualization with HUD and telemetry.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw all raw candidate detections in yellow
    if raw_ball_detections:
        for d in raw_ball_detections:
            bx1, by1, bx2, by2 = map(int, d.bbox)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 1)
            cv2.putText(vis, f"{d.confidence:.2f}", (bx1, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Colors for track state
    status_colors = {
        BallTrackStatus.TRACKED: (0, 255, 0),       # Vibrant Green
        BallTrackStatus.RECONNECTED: (255, 255, 0), # Cyan
        BallTrackStatus.PREDICTED: (0, 165, 255),   # Orange
        BallTrackStatus.LOST: (0, 0, 255),          # Red
        BallTrackStatus.INACTIVE: (128, 128, 128)   # Gray
    }
    color = status_colors.get(record.status, (255, 255, 255))

    if record.status in [BallTrackStatus.TRACKED, BallTrackStatus.RECONNECTED, BallTrackStatus.PREDICTED]:
        cx, cy = int(record.center[0]), int(record.center[1])
        x1, y1, x2, y2 = map(int, record.bbox)

        # Draw ball bounding box & circle center
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (cx, cy), 4, color, -1)

        # Draw predicted position (if available)
        if record.predicted_center is not None:
            px, py = int(record.predicted_center[0]), int(record.predicted_center[1])
            cv2.drawMarker(vis, (px, py), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)

        # Draw adaptive gate circle
        if record.gate_radius > 0:
            cv2.circle(vis, (cx, cy), int(record.gate_radius), (180, 180, 180), 1, cv2.LINE_AA)

        # Draw velocity vector
        vx, vy = record.velocity[0], record.velocity[1]
        end_pt = (int(cx + vx * 4.0), int(cy + vy * 4.0))
        cv2.arrowedLine(vis, (cx, cy), end_pt, (0, 255, 255), 2, tipLength=0.3)

    # Telemetry HUD in top-right
    hud_w, hud_h = 340, 190
    hud_x = w - hud_w - 20
    hud_y = 20

    overlay = vis.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
    cv2.rectangle(vis, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (100, 100, 100), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "BALL TRACKER TELEMETRY", (hud_x + 10, hud_y + 24), font, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, f"Frame: {record.frame_id}", (hud_x + 10, hud_y + 50), font, 0.5, (200, 200, 200), 1)
    
    status_str = record.status.value
    cv2.putText(vis, f"State: {status_str}", (hud_x + 10, hud_y + 75), font, 0.55, color, 2)
    cv2.putText(vis, f"Speed: {record.speed:.1f} px/f ({record.speed * 25.0:.0f} px/s)", (hud_x + 10, hud_y + 100), font, 0.5, (220, 220, 220), 1)
    cv2.putText(vis, f"Confidence: {record.confidence:.2f}", (hud_x + 10, hud_y + 125), font, 0.5, (220, 220, 220), 1)
    cv2.putText(vis, f"Gate Radius: {record.gate_radius:.1f} px", (hud_x + 10, hud_y + 150), font, 0.5, (200, 200, 200), 1)
    
    if diagnostics:
        cov = diagnostics.get("final_continuous_track_coverage", 0)
        cv2.putText(vis, f"Coverage: {cov:.1f}%", (hud_x + 10, hud_y + 175), font, 0.5, (100, 255, 100), 1)

    return vis
