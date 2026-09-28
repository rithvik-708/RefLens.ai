import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agent.evidence import EvidenceState, Observation, Uncertainty
from src.agent.tools import AgentTools
from src.agent.loop import AutonomousVARAgent

class MockAgentTools(AgentTools):
    def inspect_frame_window(self, start_frame: int, end_frame: int, step: int = 1):
        # Mocking improved visibility after inspection
        return {
            "tool": "inspect_frame_window",
            "best_contact_frame": 102,
            "ball_visibility": 0.9,
            "attacker_visibility": 0.9,
            "defender_visibility": 0.9,
            "defender_foot_visibility": 0.9,
            "tracking_continuity": 0.9,
            "occlusion_detected": False
        }

class TestTemporalAmbiguity(unittest.TestCase):
    def test_temporal_ambiguity(self):
        # Initial: contact-frame confidence low
        obs = Observation(
            frame_id=100,
            camera_id="CAM_MAIN",
            ball_visibility=0.4,
            attacker_visibility=0.9,
            defender_visibility=0.9,
            defender_foot_visibility=0.9,
            tracking_confidence=0.4,
            homography_confidence=0.9,
            geometric_confidence=0.9
        )
        
        state = EvidenceState(
            incident_id="TEST_002",
            timestamp=1.0,
            camera_id="CAM_MAIN",
            contact_frame_confidence=0.4,
            observations=[obs],
            decision="OFFSIDE"
        )
        
        tools = MockAgentTools()
        agent = AutonomousVARAgent(tools=tools, target_confidence=0.85)
        
        result = agent.resolve_incident(state)
        
        # Expected: agent selects inspect_frame_window, confidence improves, decision recomputed
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["decision"], "OFFSIDE")
        
        tool_calls = [t for t in result["trace"] if t["event"] == "TOOL_SELECTED"]
        self.assertGreater(len(tool_calls), 0)
        self.assertEqual(tool_calls[0]["tool"], "inspect_frame_window")

if __name__ == '__main__':
    unittest.main()
