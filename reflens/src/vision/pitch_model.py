from dataclasses import dataclass
from enum import Enum
from typing import Dict

@dataclass
class Point2D:
    x: float
    y: float

class PitchLandmark(Enum):
    # Field corners
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
    
    # Halfway line
    TOP_MID = "TOP_MID"
    BOTTOM_MID = "BOTTOM_MID"
    CENTER_MARK = "CENTER_MARK"
    
    # Left Penalty Box (assuming left side is X=0)
    LEFT_PENALTY_TOP_LEFT = "LEFT_PENALTY_TOP_LEFT"
    LEFT_PENALTY_BOTTOM_LEFT = "LEFT_PENALTY_BOTTOM_LEFT"
    LEFT_PENALTY_TOP_RIGHT = "LEFT_PENALTY_TOP_RIGHT"
    LEFT_PENALTY_BOTTOM_RIGHT = "LEFT_PENALTY_BOTTOM_RIGHT"
    LEFT_PENALTY_SPOT = "LEFT_PENALTY_SPOT"
    
    # Right Penalty Box (assuming right side is X=105)
    RIGHT_PENALTY_TOP_LEFT = "RIGHT_PENALTY_TOP_LEFT"
    RIGHT_PENALTY_BOTTOM_LEFT = "RIGHT_PENALTY_BOTTOM_LEFT"
    RIGHT_PENALTY_TOP_RIGHT = "RIGHT_PENALTY_TOP_RIGHT"
    RIGHT_PENALTY_BOTTOM_RIGHT = "RIGHT_PENALTY_BOTTOM_RIGHT"
    RIGHT_PENALTY_SPOT = "RIGHT_PENALTY_SPOT"
    
    # Left Goal Area
    LEFT_GOAL_TOP_LEFT = "LEFT_GOAL_TOP_LEFT"
    LEFT_GOAL_BOTTOM_LEFT = "LEFT_GOAL_BOTTOM_LEFT"
    LEFT_GOAL_TOP_RIGHT = "LEFT_GOAL_TOP_RIGHT"
    LEFT_GOAL_BOTTOM_RIGHT = "LEFT_GOAL_BOTTOM_RIGHT"
    
    # Right Goal Area
    RIGHT_GOAL_TOP_LEFT = "RIGHT_GOAL_TOP_LEFT"
    RIGHT_GOAL_BOTTOM_LEFT = "RIGHT_GOAL_BOTTOM_LEFT"
    RIGHT_GOAL_TOP_RIGHT = "RIGHT_GOAL_TOP_RIGHT"
    RIGHT_GOAL_BOTTOM_RIGHT = "RIGHT_GOAL_BOTTOM_RIGHT"

class PitchModel:
    """
    Canonical 2D football pitch coordinate system.
    Length: 105 meters (X-axis, 0 to 105).
    Width: 68 meters (Y-axis, 0 to 68).
    Origin (0,0): Top-Left corner of the pitch.
    """
    LENGTH = 105.0
    WIDTH = 68.0
    
    # Penalty area dimensions
    PENALTY_AREA_LENGTH = 16.5
    PENALTY_AREA_WIDTH = 40.32
    
    # Goal area dimensions
    GOAL_AREA_LENGTH = 5.5
    GOAL_AREA_WIDTH = 18.32
    
    # Penalty spot distance
    PENALTY_SPOT_DIST = 11.0
    
    # Goal width
    GOAL_WIDTH = 7.32

    def __init__(self):
        self.center_x = self.LENGTH / 2
        self.center_y = self.WIDTH / 2
        
        self.landmarks: Dict[PitchLandmark, Point2D] = self._initialize_landmarks()

    def _initialize_landmarks(self) -> Dict[PitchLandmark, Point2D]:
        lm = {}
        
        # Corners
        lm[PitchLandmark.TOP_LEFT] = Point2D(0.0, 0.0)
        lm[PitchLandmark.TOP_RIGHT] = Point2D(self.LENGTH, 0.0)
        lm[PitchLandmark.BOTTOM_LEFT] = Point2D(0.0, self.WIDTH)
        lm[PitchLandmark.BOTTOM_RIGHT] = Point2D(self.LENGTH, self.WIDTH)
        
        # Midline
        lm[PitchLandmark.TOP_MID] = Point2D(self.center_x, 0.0)
        lm[PitchLandmark.BOTTOM_MID] = Point2D(self.center_x, self.WIDTH)
        lm[PitchLandmark.CENTER_MARK] = Point2D(self.center_x, self.center_y)
        
        # Y-coordinates for boxes (centered on Y-axis)
        penalty_y_top = self.center_y - (self.PENALTY_AREA_WIDTH / 2)
        penalty_y_bottom = self.center_y + (self.PENALTY_AREA_WIDTH / 2)
        
        goal_area_y_top = self.center_y - (self.GOAL_AREA_WIDTH / 2)
        goal_area_y_bottom = self.center_y + (self.GOAL_AREA_WIDTH / 2)
        
        # Left Penalty Box
        lm[PitchLandmark.LEFT_PENALTY_TOP_LEFT] = Point2D(0.0, penalty_y_top)
        lm[PitchLandmark.LEFT_PENALTY_BOTTOM_LEFT] = Point2D(0.0, penalty_y_bottom)
        lm[PitchLandmark.LEFT_PENALTY_TOP_RIGHT] = Point2D(self.PENALTY_AREA_LENGTH, penalty_y_top)
        lm[PitchLandmark.LEFT_PENALTY_BOTTOM_RIGHT] = Point2D(self.PENALTY_AREA_LENGTH, penalty_y_bottom)
        lm[PitchLandmark.LEFT_PENALTY_SPOT] = Point2D(self.PENALTY_SPOT_DIST, self.center_y)
        
        # Right Penalty Box
        lm[PitchLandmark.RIGHT_PENALTY_TOP_RIGHT] = Point2D(self.LENGTH, penalty_y_top)
        lm[PitchLandmark.RIGHT_PENALTY_BOTTOM_RIGHT] = Point2D(self.LENGTH, penalty_y_bottom)
        lm[PitchLandmark.RIGHT_PENALTY_TOP_LEFT] = Point2D(self.LENGTH - self.PENALTY_AREA_LENGTH, penalty_y_top)
        lm[PitchLandmark.RIGHT_PENALTY_BOTTOM_LEFT] = Point2D(self.LENGTH - self.PENALTY_AREA_LENGTH, penalty_y_bottom)
        lm[PitchLandmark.RIGHT_PENALTY_SPOT] = Point2D(self.LENGTH - self.PENALTY_SPOT_DIST, self.center_y)
        
        # Left Goal Area
        lm[PitchLandmark.LEFT_GOAL_TOP_LEFT] = Point2D(0.0, goal_area_y_top)
        lm[PitchLandmark.LEFT_GOAL_BOTTOM_LEFT] = Point2D(0.0, goal_area_y_bottom)
        lm[PitchLandmark.LEFT_GOAL_TOP_RIGHT] = Point2D(self.GOAL_AREA_LENGTH, goal_area_y_top)
        lm[PitchLandmark.LEFT_GOAL_BOTTOM_RIGHT] = Point2D(self.GOAL_AREA_LENGTH, goal_area_y_bottom)
        
        # Right Goal Area
        lm[PitchLandmark.RIGHT_GOAL_TOP_RIGHT] = Point2D(self.LENGTH, goal_area_y_top)
        lm[PitchLandmark.RIGHT_GOAL_BOTTOM_RIGHT] = Point2D(self.LENGTH, goal_area_y_bottom)
        lm[PitchLandmark.RIGHT_GOAL_TOP_LEFT] = Point2D(self.LENGTH - self.GOAL_AREA_LENGTH, goal_area_y_top)
        lm[PitchLandmark.RIGHT_GOAL_BOTTOM_LEFT] = Point2D(self.LENGTH - self.GOAL_AREA_LENGTH, goal_area_y_bottom)
        
        return lm

    def get_landmark(self, landmark: PitchLandmark) -> Point2D:
        return self.landmarks[landmark]
