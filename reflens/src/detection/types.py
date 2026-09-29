from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class TeamClassification:
    team_id: Optional[int]
    confidence: float
    jersey_features: np.ndarray

@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

@dataclass
class TrackedState:
    frame_id: int
    timestamp_ms: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    x: float          # Center X
    y: float          # Center Y
    v_x: float = 0.0  # Velocity X (pixels/sec)
    v_y: float = 0.0  # Velocity Y (pixels/sec)
    team_classification: Optional[TeamClassification] = None

@dataclass
class TrackedEntity:
    track_id: int
    class_id: int
    history: List[TrackedState] = field(default_factory=list)
    team_history: List[int] = field(default_factory=list)
    
    @property
    def aggregated_team_id(self) -> Optional[int]:
        """Returns the most frequent team_id across the track's history."""
        if not self.team_history:
            return None
        counts = {0: 0, 1: 0}
        for t_id in self.team_history:
            if t_id in counts:
                counts[t_id] += 1
        return max(counts, key=counts.get) if sum(counts.values()) > 0 else None

    @property
    def aggregated_team_confidence(self) -> float:
        """Calculates confidence based on temporal consistency."""
        if not self.team_history:
            return 0.0
        team_id = self.aggregated_team_id
        if team_id is None:
            return 0.0
        matches = sum(1 for t in self.team_history if t == team_id)
        return matches / len(self.team_history)
        
    def update_state(self, frame_id: int, timestamp_ms: float, bbox: np.ndarray):
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        
        v_x, v_y = 0.0, 0.0
        if self.history:
            last_state = self.history[-1]
            dt_sec = (timestamp_ms - last_state.timestamp_ms) / 1000.0
            if dt_sec > 0:
                v_x = (x_center - last_state.x) / dt_sec
                v_y = (y_center - last_state.y) / dt_sec
                
        new_state = TrackedState(frame_id, timestamp_ms, bbox, x_center, y_center, v_x, v_y)
        self.history.append(new_state)
