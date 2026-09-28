import os
import json
from src.agent.evidence import EvidencePacket, EvidenceState

def flag_for_human_review(state: EvidenceState, reason: str, output_dir: str = "outputs/agent/incidents"):
    """
    Halts agent loop and packages telemetry for the Phase 6 dashboard.
    """
    incident_dir = os.path.join(output_dir, state.incident_id)
    os.makedirs(incident_dir, exist_ok=True)
    
    uncertainties = [k for k, v in state.uncertainties.model_dump().items() if v]
    
    packet = EvidencePacket(
        incident_id=state.incident_id,
        recommendation="HUMAN_REVIEW",
        confidence=state.decision_confidence,
        contact_frame=state.contact_frame,
        uncertainties=uncertainties,
        agent_actions=[a.action for a in getattr(state, "tool_call_history", [])],
        evidence_sources=state.evidence_sources,
        tool_results=getattr(state, "tool_result_history", [])
    )
    
    packet_path = os.path.join(incident_dir, "evidence_packet.json")
    with open(packet_path, "w") as f:
        json.dump(packet.model_dump(), f, indent=4)
    
    return packet.model_dump()
