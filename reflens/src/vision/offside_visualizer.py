import cv2
import numpy as np
from typing import List
from src.vision.offside_engine import OffsideDecision
from src.vision.homography import HomographyEstimator
from src.vision.pitch_model import PitchModel

class OffsideVisualizer:
    def __init__(self, homography: HomographyEstimator, pitch_model: PitchModel):
        self.homography = homography
        self.pitch_model = pitch_model
        # Precompute pitch corners for top-down radar
        self.radar_width = 1050
        self.radar_height = 680
        self.radar_scale = 10.0  # 1 meter = 10 pixels

    def _meters_to_radar(self, x: float, y: float) -> tuple:
        rx = int(x * self.radar_scale)
        ry = int(y * self.radar_scale)
        return rx, ry

    def draw_topdown_radar(self, decision: OffsideDecision, all_players_pitch: List[dict]) -> np.ndarray:
        """
        Creates a top-down radar view of the offside geometry.
        """
        img = np.zeros((self.radar_height, self.radar_width, 3), dtype=np.uint8)
        
        # Draw pitch background
        cv2.rectangle(img, (0, 0), (self.radar_width, self.radar_height), (0, 100, 0), -1)
        
        # Draw pitch lines (midline)
        mid_x, mid_y = self._meters_to_radar(52.5, 0)
        cv2.line(img, (mid_x, 0), (mid_x, self.radar_height), (255, 255, 255), 2)
        
        # Draw all players
        for p in all_players_pitch:
            color = (0, 0, 255) if p['team_id'] == decision.attacking_team else (255, 0, 0)
            rx, ry = self._meters_to_radar(p['pitch_x'], p['pitch_y'])
            cv2.circle(img, (rx, ry), 8, color, -1)
            cv2.putText(img, str(p['player_id']), (rx - 10, ry - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw offside reference line
        if decision.reference:
            ref_x = decision.reference['pitch_x']
            rx, _ = self._meters_to_radar(ref_x, 0)
            cv2.line(img, (rx, 0), (rx, self.radar_height), (0, 255, 255), 2)
            cv2.putText(img, f"REF: {ref_x:.2f}m", (rx + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw ball
        if decision.ball_position:
            bx = decision.ball_position['pitch_x']
            by = decision.ball_position['pitch_y']
            brx, bry = self._meters_to_radar(bx, by)
            cv2.circle(img, (brx, bry), 6, (0, 255, 255), -1)
            
        return img

    def draw_broadcast_review(self, frame: np.ndarray, decision: OffsideDecision) -> np.ndarray:
        """
        Draws the HUD and reference lines on the broadcast video frame.
        """
        out_frame = frame.copy()
        
        # Helper to draw a point
        def draw_pitch_point(px, py, label, color):
            try:
                ix, iy = self.homography.pitch_to_image((px, py))
                ix, iy = int(ix), int(iy)
                h, w = out_frame.shape[:2]
                if -1000 < ix < w + 1000 and -1000 < iy < h + 1000:
                    cv2.circle(out_frame, (ix, iy), 6, color, -1)
                    cv2.putText(out_frame, label, (ix+10, iy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(out_frame, f"X={px:.1f}m Y={py:.1f}m", (ix+10, iy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            except ValueError:
                pass

        # Draw contact player (Yellow)
        # We need the contact player's pitch coordinate
        contact_player = next((a for a in decision.attackers if a.player_id == decision.contact_player_id), None)
        if contact_player:
            draw_pitch_point(contact_player.pitch_x, contact_player.pitch_y, f"CONTACT ID: {contact_player.player_id}", (0, 255, 255))
            
        # Draw second-last opponent (Blue/Red depending on team)
        slo = decision.second_last_opponent
        if slo:
            draw_pitch_point(slo['pitch_x'], slo['pitch_y'], f"SECOND-LAST ID: {slo['player_id']}", (255, 0, 0))

        # Draw reference line in broadcast view
        if decision.reference and self.homography.H_matrix is not None:
            ref_x = decision.reference['pitch_x']
            y_pts = np.linspace(0, 68, 20)
            img_pts = []
            h, w = out_frame.shape[:2]
            for y in y_pts:
                try:
                    ix, iy = self.homography.pitch_to_image((ref_x, float(y)))
                    # Only add if reasonably near screen
                    if -10000 < ix < w + 10000 and -10000 < iy < h + 10000:
                        img_pts.append((int(ix), int(iy)))
                except ValueError:
                    pass
            
            for i in range(len(img_pts) - 1):
                cv2.line(out_frame, img_pts[i], img_pts[i+1], (0, 255, 255), 3)

        # Draw HUD
        hud_bg = out_frame[10:450, 10:350]
        rect = np.zeros(hud_bg.shape, dtype=np.uint8)
        rect[:] = (30, 30, 30)
        cv2.addWeighted(rect, 0.8, hud_bg, 0.2, 0, hud_bg)
        out_frame[10:450, 10:350] = hud_bg

        y_offset = 35
        def write_hud(text, color=(255, 255, 255), size=0.5):
            nonlocal y_offset
            cv2.putText(out_frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
            y_offset += 25

        if hasattr(self, 'debug_geometry') and self.debug_geometry:
            write_hud("OFFSIDE GEOMETRY DEBUG", (0, 255, 255), 0.6)
            write_hud("-" * 35)
            write_hud(f"FRAME: {decision.frame_id}")
            write_hud(f"TIME: {decision.timestamp:.2f}s")
            write_hud("")
            write_hud(f"CONTACT:")
            write_hud(f"Player {decision.contact_player_id}")
            write_hud("")
            slo_id = slo['player_id'] if slo else "N/A"
            write_hud(f"SECOND-LAST:")
            write_hud(f"Player {slo_id}")
            write_hud("")
            
            ref_type = decision.reference['type'] if decision.reference else "N/A"
            ref_x_str = f"{decision.reference['pitch_x']:.2f}m" if decision.reference else "N/A"
            write_hud(f"REFERENCE:")
            write_hud(f"{ref_type}")
            write_hud(f"REFERENCE X: {ref_x_str}")
            
            bx_str = f"{decision.ball_position['pitch_x']:.2f}m" if decision.ball_position else "N/A"
            write_hud(f"BALL X: {bx_str}")
            
            slo_x_str = f"{slo['pitch_x']:.2f}m" if slo else "N/A"
            write_hud(f"SECOND-LAST X: {slo_x_str}")
            
            write_hud(f"ATTACKING DIRECTION:")
            dir_name = getattr(getattr(decision.attacking_direction, 'direction', decision.attacking_direction), 'name', str(decision.attacking_direction))
            write_hud(f"{dir_name}")
            
            write_hud(f"CALIBRATION:")
            write_hud(f"{len(self.homography.image_points)} landmarks")
            write_hud(f"REPROJECTION:")
            err = self.homography.validation_metrics.get("mean_reprojection_error_m", 0.0)
            write_hud(f"{err:.2f} m")
        else:
            write_hud("OFFSIDE POSITION REVIEW", (0, 255, 255), 0.6)
            write_hud("-" * 35)
            write_hud(f"TIME: {decision.timestamp:.2f}s | FRAME: {decision.frame_id}")
            write_hud(f"CONTACT: Player {decision.contact_player_id}")
            
            slo_id = slo['player_id'] if slo else "N/A"
            write_hud(f"SECOND-LAST: Player {slo_id}")
            
            ref_type = decision.reference['type'] if decision.reference else "N/A"
            write_hud(f"REFERENCE: {ref_type}")
            write_hud("-" * 35)

            for attacker in decision.attackers:
                if attacker.status != "ONSIDE":  # Highlight offside/uncertain attackers
                    color = (0, 0, 255) if attacker.status == "OFFSIDE" else (0, 165, 255)
                    write_hud(f"ATTACKER {attacker.player_id}: {attacker.status}", color)
                    write_hud(f"  Pos: {attacker.pitch_x:.2f}m", color)
                    write_hud(f"  Margin: {attacker.distance_to_reference:.2f}m", color)

            write_hud("-" * 35)
            decision_color = (0, 0, 255) if decision.decision == "OFFSIDE_POSITION" else (0, 255, 0)
            write_hud(f"STATUS: {decision.decision}", decision_color, 0.6)

        return out_frame
