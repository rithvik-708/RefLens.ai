import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agent.evidence import EvidenceState, Observation, Uncertainty
from src.agent.tools import AgentTools
from src.agent.loop import AutonomousVARAgent

class TestClearIncident(unittest.TestCase):
    def test_clear_incident(self):
        # Initial evidence: confidence > 0.85
        obs = Observation(
            frame_id=100,
            camera_id="CAM_MAIN",
            ball_visibility=0.9,
            attacker_visibility=0.9,
            defender_visibility=0.9,
            defender_foot_visibility=0.9,
            tracking_confidence=0.9,
            homography_confidence=0.9,
            geometric_confidence=0.9
        )
        
        state = EvidenceState(
            incident_id="TEST_001",
            timestamp=1.0,
            camera_id="CAM_MAIN",
            contact_frame_confidence=0.9,
            observations=[obs],
            decision="OFFSIDE"
        )
        
        tools = AgentTools()
        agent = AutonomousVARAgent(tools=tools, target_confidence=0.85)
        
        result = agent.resolve_incident(state)
        
        # Expected: agent does not unnecessarily request additional evidence.
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["decision"], "OFFSIDE")
        
        # Verify: tool calls == 0 or minimal.
        tool_calls = [t for t in result["trace"] if t["event"] == "TOOL_SELECTED"]
        self.assertEqual(len(tool_calls), 0)

if __name__ == '__main__':
    unittest.main()
