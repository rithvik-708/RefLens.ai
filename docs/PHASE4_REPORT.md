# Phase 4 — Agentic Reasoning Loop & Active Perception

## 1. Objective
To complete Phase 4 of RefLens by replacing the deterministic fallback chain with an active perception agent that identifies missing evidence, formulates a strategy, executes tools to acquire that evidence, and fuses the results to reach a confident decision or escalate to human review.

## 2. Existing Implementation Audit
- Evaluated `src/agent/tools.py` and `src/agent/loop.py`. They originally contained generic fallback if/else logic without maintaining state or understanding specific uncertainties.
- The previous implementation was a placeholder, not a true agentic loop. We replaced it entirely while preserving the interfaces for Phase 2/3 dependencies.

## 3. Agent Architecture
The agent operates via a loop: OBSERVE → ASSESS → IDENTIFY MISSING EVIDENCE → SELECT TOOL → EXECUTE TOOL → UPDATE BELIEF → REASSESS.
We implemented `AutonomousVARAgent` in `src/agent/loop.py` to handle this logic dynamically up to a maximum number of iterations.

## 4. Evidence Model
Created `src/agent/evidence.py` introducing `EvidenceState`, `Observation`, and `Uncertainty`. This strongly-typed model maintains the current belief state of the incident, decoupling raw confidence from the underlying reasons for uncertainty (e.g., `contact_frame_uncertain`, `defender_foot_occluded`).

## 5. Tool Protocol
Rewrote `AgentTools` in `src/agent/tools.py`.
- `inspect_frame_window`: Scans the video frames utilizing actual computer vision logic from Phase 2/3.
- `recalibrate_homography`: Attempts to extract pitch lines using `PitchFeatureExtractor`.
- `request_alternate_camera`: Refers to `CameraRegistry` to determine actual availability.

## 6. Evidence-Seeking Policy
The agent selects tools based on the specific uncertainty that caused the confidence drop, instead of following a fixed chain. For instance, if the homography is poor, it invokes `recalibrate_homography`. If the contact frame is uncertain, it invokes `inspect_frame_window`.

## 7. Evidence Fusion
Implemented in `src/agent/evidence_fusion.py`. `compare_evidence` analyzes the delta between two observations, ensuring the agent can determine if uncertainty decreased after a tool call.

## 8. Confidence / Uncertainty Model
Implemented in `src/agent/confidence.py`. It computes a multi-dimensional confidence score by assigning weights to variables like contact confidence, tracking confidence, and geometric confidence, generating discrete uncertainty flags based on thresholds.

## 9. Human Review
Implemented in `src/agent/human_review.py`. If autonomous paths are exhausted (e.g., persistent occlusion and no alternate cameras), the agent generates an `EvidencePacket` containing its trace, uncertainties, and observations, escalating to a human official.

## 10. Agent Trace
Maintained in `outputs/agent/incidents/<incident_id>`. The trace captures every step of the reasoning loop, the tools selected, the reasons for selection, and the results, generating a human-readable markdown visualization (`phase4_trace.md`).

## 11. Synthetic Test Results
- **Clear Incident**: Evaluated successfully. Agent made 0 tool calls.
- **Temporal Ambiguity**: Evaluated successfully. Agent requested `inspect_frame_window` and successfully improved confidence.
- **Persistent Occlusion**: Evaluated successfully. Agent attempted recovery but correctly escalated to human review, avoiding infinite loops.

## 12. Real SoccerNet Test Results
Executed the `verify_phase4.py` script on the provided `1_224p.mkv` match video. The agent effectively processed the real video frames, evaluated the available evidence, triggered tools, and reached a documented state.

## 13. Metrics
Captured in `outputs/agent/phase4_metrics.json`. Tracks `initial_confidence`, `final_confidence`, `tool_calls`, `human_review` status, and `latency_ms`.

## 14. Failure Cases
- **No Alternate Camera**: When requested, the tool gracefully reports `available: false` because SoccerNet standard feeds do not provide alternate broadcast angles in this dataset.
- **Insufficient Pitch Lines**: Homography tool fails safely if the camera is zoomed in too closely to identify 4 pitch lines.

## 15. Known Limitations
- The current SoccerNet dataset primarily features a single main broadcast camera, limiting the effectiveness of the alternate camera tool in this specific dataset.
- The `inspect_frame_window` currently relies on general YOLO detections; for true sub-frame accuracy, an interpolated skeletal model would be required.

## 16. How This Demonstrates Active Perception
The system does not process the entire video and output a single result. It processes an initial snapshot, recognizes what it does not know (e.g., "I cannot see the defender's trailing foot"), actively requests a specific temporal or spatial view to gain that knowledge, and updates its belief.

## 17. Files Added / Modified
- **Added**: `src/agent/evidence.py`, `src/agent/confidence.py`, `src/agent/camera_registry.py`, `src/agent/evidence_fusion.py`, `src/agent/human_review.py`
- **Modified**: `src/agent/tools.py`, `src/agent/loop.py`
- **Tests Added**: `tests/agent/test_clear_incident.py`, `tests/agent/test_temporal_ambiguity.py`, `tests/agent/test_persistent_ambiguity.py`
- **Utils Added**: `src/utils/verify_phase4.py`

## 18. Reproduction Commands
```bash
# Run unit tests
python -m pytest tests/agent -v

# Run real SoccerNet integration test
python -m src.utils.verify_phase4 --video "dataset/raw/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"
```
