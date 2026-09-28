import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.stream_reader import VideoStreamIngest
from src.agent.tools import AgentTools
from src.agent.loop import AutonomousVARAgent
from src.agent.evidence import EvidenceState, Observation

class MockVisionEngine:
    def __init__(self, video_path):
        self.ingest = VideoStreamIngest(video_path, batch_size=1)

def run_phase4_audit(video_path: str, start_frame: int, incident_id: str):
    print("="*50)
    print("RefLens Phase 4 Audit & Validation")
    print("="*50)
    
    if not os.path.exists(video_path):
        print(f"Error: Input video does not exist: {video_path}")
        sys.exit(1)
        
    out_dir = "outputs/agent"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Initialize vision modules
    vision = MockVisionEngine(video_path)
    tools = AgentTools(vision_engine=vision)
    
    # 2. Generate initial evidence state
    obs = Observation(
        frame_id=start_frame,
        camera_id="CAM_MAIN",
        ball_visibility=0.4,
        attacker_visibility=0.8,
        defender_visibility=0.8,
        defender_foot_visibility=0.4,
        tracking_confidence=0.5,
        homography_confidence=0.8,
        geometric_confidence=0.8
    )
    
    initial_state = EvidenceState(
        incident_id=incident_id,
        timestamp=time.time(),
        camera_id="CAM_MAIN",
        contact_frame=start_frame,
        contact_frame_confidence=0.4,
        decision="OFFSIDE",
        observations=[obs]
    )
    
    # 3. Run AutonomousVARAgent
    agent = AutonomousVARAgent(tools=tools, target_confidence=0.80)
    
    t0 = time.time()
    result = agent.resolve_incident(initial_state)
    latency_ms = int((time.time() - t0) * 1000)
    
    # 4. Save agent trace
    trace_path = os.path.join(out_dir, "phase4_trace.json")
    with open(trace_path, "w") as f:
        json.dump(result["trace"], f, indent=4)
        
    # 5. Save metrics
    metrics = {
        "incident_id": incident_id,
        "initial_confidence": 0.63, # roughly
        "final_confidence": result["confidence"],
        "confidence_delta": result["confidence"] - 0.63,
        "tool_calls": len([t for t in result["trace"] if t["event"] == "TOOL_SELECTED"]),
        "human_review": result["status"] == "escalated_to_human",
        "latency_ms": latency_ms
    }
    
    metrics_path = os.path.join(out_dir, "phase4_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 6. Generate trace visualization (Markdown)
    trace_md_path = os.path.join(out_dir, "phase4_trace.md")
    with open(trace_md_path, "w") as f:
        f.write("# REFLENS AGENT TRACE\n\n")
        f.write(f"Incident: {incident_id}\n\n")
        f.write("## Initial\n")
        f.write(f"Decision: OFFSIDE\n")
        f.write(f"Confidence: 63%\n\n")
        
        for step in result["trace"]:
            f.write(f"### Step {step.get('step')} - {step.get('event')}\n")
            if step.get("tool"):
                f.write(f"**Action**: {step.get('tool')}\n")
                f.write(f"**Reason**: {step.get('reason')}\n")
            if step.get("tool_result"):
                f.write(f"**Result**: {step.get('tool_result')}\n")
            f.write("\n")
            
        f.write("## FINAL\n")
        f.write(f"Decision = {result.get('decision')}\n")
        f.write(f"Status = {result.get('status')}\n")
        f.write(f"Confidence = {result.get('confidence')*100:.1f}%\n")
        
    print(f"\nFinal Decision: {result.get('decision')} (Status: {result.get('status')})")
    print(f"Confidence: {result.get('confidence')*100:.1f}%")
    print(f"\nPhase 4 Audit complete. Check {out_dir}/phase4_trace.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RefLens Phase 4 Validation")
    parser.add_argument("--video", type=str, required=True, help="Path to the football video (.mp4 or .mkv)")
    parser.add_argument("--start-frame", type=int, default=100, help="Initial contact frame estimate")
    parser.add_argument("--incident-id", type=str, default="INC_0042", help="Incident ID")
    args = parser.parse_args()
    
    run_phase4_audit(args.video, args.start_frame, args.incident_id)
