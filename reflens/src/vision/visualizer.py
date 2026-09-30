import cv2
import numpy as np
from typing import List, Optional
from src.vision.events import PassCandidate

class VARVisualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_candidate_overlay(
        self, 
        frame: np.ndarray, 
        candidate: PassCandidate, 
        event_idx: Optional[int] = None,
        is_exact_frame: bool = False
    ) -> np.ndarray:
        """
        Renders a sleek, non-intrusive top-left corner HUD overlay displaying
        the pass candidate metrics for smooth real-time video playback.
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # HUD Panel Dimensions in top-left
        hud_x, hud_y = 25, 25
        hud_w, hud_h = 360, 185
        
        # Semi-transparent dark background
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.82, vis_frame, 0.18, 0, vis_frame)
        
        # Border (Cyan on exact contact frame, sleek gray during review window)
        border_color = (0, 255, 255) if is_exact_frame else (80, 80, 90)
        cv2.rectangle(vis_frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), border_color, 2)
        
        # Header Badge
        header_text = f"PASS EVENT #{event_idx}" if event_idx is not None else "PASS CANDIDATE"
        cv2.putText(vis_frame, header_text, (hud_x + 14, hud_y + 28), self.font, 0.65, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"F:{candidate.frame_id}", (hud_x + hud_w - 90, hud_y + 28), self.font, 0.55, (200, 200, 200), 1)
        
        # Divider Line
        cv2.line(vis_frame, (hud_x + 10, hud_y + 38), (hud_x + hud_w - 10, hud_y + 38), (70, 70, 80), 1)
        
        # Nearest Player
        cv2.putText(vis_frame, "NEAREST PLAYER:", (hud_x + 14, hud_y + 64), self.font, 0.45, (170, 170, 170), 1)
        cv2.putText(vis_frame, f"ID {candidate.player_id} ({candidate.ball_distance_px:.1f}px)", (hud_x + 155, hud_y + 64), self.font, 0.5, (255, 255, 255), 1)
        
        # Ball Velocity & Acceleration
        cv2.putText(vis_frame, "BALL SPEED:", (hud_x + 14, hud_y + 92), self.font, 0.45, (170, 170, 170), 1)
        speed_str = f"{candidate.ball_speed_before:.1f} -> {candidate.ball_speed_after:.1f} px/f"
        cv2.putText(vis_frame, speed_str, (hud_x + 155, hud_y + 92), self.font, 0.5, (255, 255, 255), 1)
        
        cv2.putText(vis_frame, "ACCELERATION:", (hud_x + 14, hud_y + 120), self.font, 0.45, (170, 170, 170), 1)
        accel_str = f"+{candidate.acceleration:.1f} px/f"
        cv2.putText(vis_frame, accel_str, (hud_x + 155, hud_y + 120), self.font, 0.5, (0, 230, 255), 1)
        
        # Confidence Progress Bar
        conf_pct = int(np.clip(candidate.confidence, 0.0, 1.0) * 100)
        cv2.putText(vis_frame, f"CONFIDENCE: {conf_pct}%", (hud_x + 14, hud_y + 150), self.font, 0.45, (170, 170, 170), 1)
        
        bar_x = hud_x + 155
        bar_y = hud_y + 140
        bar_w = hud_w - 175
        bar_h = 12
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 50), -1)
        fill_w = int((conf_pct / 100.0) * bar_w)
        bar_color = (0, 255, 120) if conf_pct >= 75 else ((0, 200, 255) if conf_pct >= 50 else (0, 120, 255))
        if fill_w > 0:
            cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 110), 1)
        
        return vis_frame
