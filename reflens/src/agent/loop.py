import time
from typing import Dict, Any, List
from loguru import logger
from src.agent.tools import AgentTools
from src.agent.evidence import EvidenceState, AgentAction, Observation
from src.agent.confidence import compute_evidence_confidence
from src.agent.evidence_fusion import compare_evidence
from src.agent.human_review import flag_for_human_review

class AutonomousVARAgent:
    def __init__(self, tools: AgentTools, target_confidence: float = 0.80, max_iterations: int = 5):
        self.tools = tools
        self.target_confidence = target_confidence
        self.max_iterations = max_iterations
        self.agent_trace: List[Dict[str, Any]] = []

    def _log_thought(self, event: str, step: int, **kwargs):
        logger.info(f"[AGENT] {event}")
        trace_entry = {"step": step, "event": event}
        trace_entry.update(kwargs)
        self.agent_trace.append(trace_entry)

    def resolve_incident(self, initial_state: EvidenceState) -> Dict[str, Any]:
        """
        Executes the agentic reasoning loop for a specific pass event.
        """
        state = initial_state
        state.tool_call_history = getattr(state, "tool_call_history", [])
        state.tool_result_history = getattr(state, "tool_result_history", [])
        
        step = 0
        self._log_thought("INCIDENT_DETECTED", step, incident_id=state.incident_id)
        
        while step < self.max_iterations:
            step += 1
            
            # 1. ASSESS
            conf_data = compute_evidence_confidence(state)
            current_confidence = conf_data["overall_confidence"]
            state.decision_confidence = current_confidence
            
            self._log_thought(
                "EVIDENCE_ASSESSMENT", step, 
                confidence=current_confidence, 
                uncertainties=conf_data["uncertainties"]
            )
            
            # Base Case: High Confidence
            if current_confidence >= self.target_confidence:
                self._log_thought("FINAL_DECISION", step, decision=state.decision, confidence=current_confidence)
                return {"status": "resolved", "decision": state.decision, "confidence": current_confidence, "trace": self.agent_trace}

            # 2. IDENTIFY MISSING EVIDENCE & SELECT TOOL
            action = None
            unc = state.uncertainties
            
            if unc.contact_frame_uncertain and "inspect_frame_window" not in [a.action for a in state.tool_call_history]:
                action = AgentAction(
                    action="inspect_frame_window",
                    reason="Contact frame uncertainty is high",
                    target_uncertainty="contact_frame_uncertain",
                    expected_information_gain=0.3,
                    previous_confidence=current_confidence
                )
            elif (unc.defender_foot_occluded or unc.defender_occluded) and "request_alternate_camera" not in [a.action for a in state.tool_call_history]:
                action = AgentAction(
                    action="request_alternate_camera",
                    reason="Defender foot is occluded",
                    target_uncertainty="defender_foot_occluded",
                    expected_information_gain=0.4,
                    previous_confidence=current_confidence
                )
            elif unc.poor_homography and "recalibrate_homography" not in [a.action for a in state.tool_call_history]:
                action = AgentAction(
                    action="recalibrate_homography",
                    reason="Homography quality is poor",
                    target_uncertainty="poor_homography",
                    expected_information_gain=0.2,
                    previous_confidence=current_confidence
                )
            
            if not action:
                self._log_thought("NO_AVAILABLE_ACTIONS", step, message="All autonomous recovery paths exhausted.")
                break
                
            # 3. EXECUTE TOOL
            self._log_thought("TOOL_SELECTED", step, tool=action.action, reason=action.reason)
            state.tool_call_history.append(action)
            
            t0 = time.time()
            if action.action == "inspect_frame_window":
                t_pass = state.contact_frame if state.contact_frame else 0
                result = self.tools.inspect_frame_window(t_pass - 15, t_pass + 15, 1)
            elif action.action == "request_alternate_camera":
                result = self.tools.request_alternate_camera(state.incident_id, "CAM_OPPOSITE")
            elif action.action == "recalibrate_homography":
                t_pass = state.contact_frame if state.contact_frame else 0
                result = self.tools.recalibrate_homography(t_pass)
            latency = time.time() - t0
            
            result["latency_ms"] = int(latency * 1000)
            state.tool_result_history.append(result)
            
            # 4. UPDATE BELIEF (FUSION)
            self._log_thought("EVIDENCE_UPDATED", step, tool_result=result)
            
            if state.observations:
                old_obs = state.observations[-1]
                new_obs = Observation(
                    frame_id=state.contact_frame if state.contact_frame else 0,
                    camera_id=state.camera_id,
                    attacker_position=old_obs.attacker_position,
                    defender_position=old_obs.defender_position,
                    ball_position=old_obs.ball_position,
                    ball_visibility=old_obs.ball_visibility,
                    attacker_visibility=old_obs.attacker_visibility,
                    defender_visibility=old_obs.defender_visibility,
                    defender_foot_visibility=old_obs.defender_foot_visibility,
                    tracking_confidence=old_obs.tracking_confidence,
                    homography_confidence=old_obs.homography_confidence,
                    geometric_confidence=old_obs.geometric_confidence
                )
            else:
                new_obs = Observation(
                    frame_id=state.contact_frame if state.contact_frame else 0,
                    camera_id=state.camera_id
                )
            
            if action.action == "inspect_frame_window":
                state.contact_frame = result.get("best_contact_frame", state.contact_frame)
                state.contact_frame_confidence = 0.9 # assumed improved
                
                new_obs.ball_visibility = result.get("ball_visibility", 0)
                new_obs.attacker_visibility = result.get("attacker_visibility", 0)
                new_obs.defender_visibility = result.get("defender_visibility", 0)
                new_obs.defender_foot_visibility = result.get("defender_foot_visibility", 0)
                new_obs.tracking_confidence = result.get("tracking_continuity", 0)
                
                # Compare evidence
                if state.observations:
                    comparison = compare_evidence(state.observations[-1].model_dump(), new_obs.model_dump())
                    self._log_thought("EVIDENCE_COMPARISON", step, comparison=comparison)
                    
                state.observations.append(new_obs)
                
            elif action.action == "recalibrate_homography":
                if result.get("homography_updated"):
                    new_obs.homography_confidence = result.get("calibration_confidence", 0.9)
                    state.observations.append(new_obs)

        # 5. ESCALATE
        escalation_result = flag_for_human_review(state, "Unable to reach target confidence autonomously.")
        self._log_thought("ESCALATED_TO_HUMAN", step, packet=escalation_result)
        
        return {
            "status": "escalated_to_human",
            "decision": state.decision,
            "confidence": current_confidence,
            "escalation_data": escalation_result,
            "trace": self.agent_trace
        }
