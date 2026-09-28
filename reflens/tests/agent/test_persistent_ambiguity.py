import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agent.evidence import EvidenceState, Observation
from src.agent.tools import AgentTools
from src.agent.loop import AutonomousVARAgent

class MockAgentTools(AgentTools):
    def inspect_frame_window(self, start_frame: int, end_frame: int, step: int = 1):
        # Mocking persistent occlusion
        return {
            "tool": "inspect_frame_window",
            "best_contact_frame": 100,
            "ball_visibility": 0.4,
            "attacker_visibility": 0.9,
            "defender_visibility": 0.4,
            "defender_foot_visibility": 0.2,
            "tracking_continuity": 0.4,
            "occlusion_detected": True
        }
        
    def request_alternate_camera(self, incident_id: str, camera_id: str):
        return {
            "available": False,
            "reason": "No alternate camera source available for this incident"
        }

class TestPersistentAmbiguity(unittest.TestCase):
    def test_persistent_ambiguity(self):
        # Initial: confidence low, persistent occlusion
        obs = Observation(
            frame_id=100,
            camera_id="CAM_MAIN",
            ball_visibility=0.4,
            attacker_visibility=0.9,
            defender_visibility=0.4,
            defender_foot_visibility=0.2,
            tracking_confidence=0.4,
            homography_confidence=0.9,
            geometric_confidence=0.9
        )
        
        state = EvidenceState(
            incident_id="TEST_003",
            timestamp=1.0,
            camera_id="CAM_MAIN",
            contact_frame_confidence=0.4,
            observations=[obs],
            decision="OFFSIDE"
        )
        
        tools = MockAgentTools()
        agent = AutonomousVARAgent(tools=tools, target_confidence=0.85)
        
        result = agent.resolve_incident(state)
        
        # Expected: agent eventually calls flag_for_human_review, does not loop indefinitely
        self.assertEqual(result["status"], "escalated_to_human")
        self.assertEqual(result["decision"], "OFFSIDE")
        
        tool_calls = [t for t in result["trace"] if t["event"] == "TOOL_SELECTED"]
        self.assertGreater(len(tool_calls), 0)
        self.assertLessEqual(len(tool_calls), 5) # Default max iterations

if __name__ == '__main__':
    unittest.main()
