from typing import Dict, Any
import numpy as np

def compare_evidence(evidence_a: Dict[str, Any], evidence_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compares two evidence observations.
    """
    contact_frame_delta = abs(evidence_a.get("contact_frame", 0) - evidence_b.get("contact_frame", 0))
    
    pos_a = evidence_a.get("attacker_position")
    pos_b = evidence_b.get("attacker_position")
    
    if pos_a and pos_b:
        attacker_position_delta_m = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
    else:
        attacker_position_delta_m = 0.0

    dpos_a = evidence_a.get("defender_position")
    dpos_b = evidence_b.get("defender_position")
    if dpos_a and dpos_b:
        defender_position_delta_m = np.linalg.norm(np.array(dpos_a) - np.array(dpos_b))
    else:
        defender_position_delta_m = 0.0
        
    dec_a = evidence_a.get("decision")
    dec_b = evidence_b.get("decision")

    return {
        "decision_a": dec_a,
        "decision_b": dec_b,
        "decision_agreement": dec_a == dec_b,
        "contact_frame_delta": contact_frame_delta,
        "attacker_position_delta_m": attacker_position_delta_m,
        "defender_position_delta_m": defender_position_delta_m,
        "offside_line_delta_m": defender_position_delta_m, # simplified
        "decision_stability": 1.0 if dec_a == dec_b else 0.0
    }
