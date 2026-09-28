from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class Uncertainty(BaseModel):
    contact_frame_uncertain: bool = False
    ball_occluded: bool = False
    attacker_occluded: bool = False
    defender_occluded: bool = False
    defender_foot_occluded: bool = False
    poor_tracking: bool = False
    poor_homography: bool = False
    insufficient_camera_angle: bool = False
    conflicting_evidence: bool = False

class Observation(BaseModel):
    frame_id: int
    camera_id: str
    attacker_position: Optional[tuple] = None
    defender_position: Optional[tuple] = None
    ball_position: Optional[tuple] = None
    ball_visibility: float = 0.0
    attacker_visibility: float = 0.0
    defender_visibility: float = 0.0
    defender_foot_visibility: float = 0.0
    tracking_confidence: float = 0.0
    homography_confidence: float = 0.0
    geometric_confidence: float = 0.0

class EvidenceState(BaseModel):
    incident_id: str
    timestamp: float
    camera_id: str
    contact_frame: Optional[int] = None
    contact_frame_confidence: float = 0.0
    attacker_id: Optional[int] = None
    defender_id: Optional[int] = None
    decision: Optional[str] = None
    decision_confidence: float = 0.0
    uncertainties: Uncertainty = Field(default_factory=Uncertainty)
    observations: List[Observation] = Field(default_factory=list)
    evidence_sources: List[str] = Field(default_factory=list)
    tool_call_history: List[Any] = Field(default_factory=list)
    tool_result_history: List[Dict[str, Any]] = Field(default_factory=list)

class AgentAction(BaseModel):
    action: str
    reason: str
    target_uncertainty: str
    expected_information_gain: float
    previous_confidence: float

class AgentDecision(BaseModel):
    decision: str
    confidence: float
    reason: str

class EvidencePacket(BaseModel):
    incident_id: str
    recommendation: str
    confidence: float
    contact_frame: Optional[int]
    uncertainties: List[str]
    agent_actions: List[str]
    evidence_sources: List[str]
    tool_results: List[Dict[str, Any]]
