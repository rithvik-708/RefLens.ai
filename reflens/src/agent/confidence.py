from typing import Dict, Any
from src.agent.evidence import EvidenceState, Uncertainty

def compute_evidence_confidence(state: EvidenceState) -> Dict[str, Any]:
    """
    Computes overall confidence based on multi-dimensional signals from observations.
    """
    if not state.observations:
        return {
            "overall_confidence": 0.0,
            "components": {},
            "uncertainties": state.uncertainties.dict(),
            "explanation": "No observations available."
        }

    latest_obs = state.observations[-1]
    
    components = {
        "contact_confidence": state.contact_frame_confidence,
        "tracking_confidence": latest_obs.tracking_confidence,
        "homography_confidence": latest_obs.homography_confidence,
        "ball_visibility": latest_obs.ball_visibility,
        "attacker_visibility": latest_obs.attacker_visibility,
        "defender_visibility": latest_obs.defender_visibility,
        "defender_foot_visibility": latest_obs.defender_foot_visibility,
        "geometric_confidence": latest_obs.geometric_confidence
    }

    # Weighting the components
    weights = {
        "contact_confidence": 0.2,
        "tracking_confidence": 0.1,
        "homography_confidence": 0.2,
        "ball_visibility": 0.1,
        "defender_foot_visibility": 0.2,
        "geometric_confidence": 0.2
    }

    overall_confidence = sum(components[k] * w for k, w in weights.items())

    # Update uncertainties based on thresholds
    unc = Uncertainty()
    unc.contact_frame_uncertain = components["contact_confidence"] < 0.7
    unc.ball_occluded = components["ball_visibility"] < 0.5
    unc.attacker_occluded = components["attacker_visibility"] < 0.5
    unc.defender_occluded = components["defender_visibility"] < 0.5
    unc.defender_foot_occluded = components["defender_foot_visibility"] < 0.6
    unc.poor_tracking = components["tracking_confidence"] < 0.6
    unc.poor_homography = components["homography_confidence"] < 0.6
    unc.insufficient_camera_angle = components["defender_foot_visibility"] < 0.3
    
    state.uncertainties = unc

    return {
        "overall_confidence": overall_confidence,
        "components": components,
        "uncertainties": unc.model_dump(),
        "explanation": f"Computed overall confidence {overall_confidence:.2f} from {len(components)} components."
    }
