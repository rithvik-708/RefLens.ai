import numpy as np
from typing import Dict, List
from scipy.optimize import linear_sum_assignment
from src.detection.types import Detection, TrackedEntity
from loguru import logger

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

class PerceptionTracker:
    def __init__(self, frame_rate: int = 30):
        self.active_tracks: Dict[int, TrackedEntity] = {}
        self.lost_tracks: Dict[int, tuple] = {}  # track_id -> (TrackedEntity, missed_frames)
        self.next_id = 1
        self.track_high_thresh = 0.4
        self.track_low_thresh = 0.1
        self.match_thresh = 0.8
        self.max_lost = 30
        
    def update(self, detections: List[Detection], frame: np.ndarray, frame_id: int, timestamp_ms: float) -> Dict[int, TrackedEntity]:
        dets_high = [d for d in detections if d.confidence >= self.track_high_thresh]
        dets_low = [d for d in detections if self.track_low_thresh <= d.confidence < self.track_high_thresh]
        
        tracks = list(self.active_tracks.values()) + [t[0] for t in self.lost_tracks.values()]
        
        # 1. Match high conf detections
        unmatched_dets_high, unmatched_tracks = self._match(dets_high, tracks, self.match_thresh, frame_id, timestamp_ms)
        
        # 2. Match low conf detections with unmatched tracks
        remaining_tracks = [t for i, t in enumerate(tracks) if i in unmatched_tracks]
        unmatched_dets_low, unmatched_tracks_final = self._match(dets_low, remaining_tracks, 0.5, frame_id, timestamp_ms)
        
        new_active = {}
        
        # 3. Create new tracks from unmatched high detections
        for i in unmatched_dets_high:
            d = dets_high[i]
            ent = TrackedEntity(self.next_id, d.class_id)
            ent.update_state(frame_id, timestamp_ms, d.bbox)
            new_active[self.next_id] = ent
            self.next_id += 1
            
        # Gather all updated tracks
        for t in self.active_tracks.values():
            if t.history[-1].frame_id == frame_id:
                new_active[t.track_id] = t
        for t, _ in self.lost_tracks.values():
            if t.history[-1].frame_id == frame_id:
                new_active[t.track_id] = t
                
        # Update lost tracks
        new_lost = {}
        for t_idx in unmatched_tracks_final:
            t = remaining_tracks[t_idx]
            missed = self.lost_tracks[t.track_id][1] if t.track_id in self.lost_tracks else 0
            if missed < self.max_lost:
                new_lost[t.track_id] = (t, missed + 1)
                
        for t in self.active_tracks.values():
            if t.track_id not in new_active:
                new_lost[t.track_id] = (t, 1)
                
        self.active_tracks = new_active
        self.lost_tracks = new_lost
        
        return self.active_tracks
        
    def _match(self, dets, tracks, thresh, frame_id, timestamp_ms):
        if not dets or not tracks:
            return list(range(len(dets))), list(range(len(tracks)))
            
        cost_matrix = np.zeros((len(dets), len(tracks)), dtype=np.float32)
        for i, d in enumerate(dets):
            for j, t in enumerate(tracks):
                iou = box_iou(d.bbox, t.history[-1].bbox)
                cost_matrix[i, j] = 1.0 - iou
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        unmatched_dets = set(range(len(dets)))
        unmatched_tracks = set(range(len(tracks)))
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                d = dets[r]
                t = tracks[c]
                t.update_state(frame_id, timestamp_ms, d.bbox)
                unmatched_dets.discard(r)
                unmatched_tracks.discard(c)
                
        return list(unmatched_dets), list(unmatched_tracks)
