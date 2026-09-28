# RefLens

RefLens is an agentic football VAR/referee assistant designed to process raw match footage, track players and the ball, extract pitch geometry, and make autonomous offside decisions using an active perception agentic reasoning loop.

## Phases
RefLens is developed in iterative phases:
- **Phase 1**: Base repository and dataset acquisition tools.
- **Phase 2**: Core CV models, object detection (YOLOv8), tracking, and ingestion.
- **Phase 3**: Pitch homography, contact frame detection, and deterministic offside geometry evaluation.
- **Phase 4**: Agentic Reasoning Loop (`AutonomousVARAgent`), active perception for evidence retrieval, multi-dimensional confidence formulation, and human review escalation.

## Getting Started

### 1. Requirements
- Python 3.12+
- Packages listed in `requirements.txt`

### 2. Setup
Run the setup script or manually create a virtual environment and install dependencies:
```bash
python -m venv venv312
venv312\Scripts\activate
pip install -r requirements.txt
```

### 3. Fetching Dataset
Run the download script to fetch the SoccerNet sample footage for testing:
```bash
python reflens/test_soccernet_legacy.py
```

### 4. Running Validation Tests
You can validate the implementation phases using the provided audit scripts.

**Phase 3 Audit** (Tests homography and deterministic logic):
```bash
cd reflens
python -m src.utils.verify_phase3 --video "dataset/raw/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv" --frames 150
```

**Phase 4 Audit** (Tests Agentic loop and tool selection):
```bash
cd reflens
python -m src.utils.verify_phase4 --video "dataset/raw/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv"
```

## Structure
- `reflens/src/agent`: Core AI agent logic, multi-dimensional confidence, evidence fusion, and tool execution.
- `reflens/src/vision`: Core CV logic, tracking, homography mapping, and camera ingestion.
- `reflens/tests`: Unit tests covering deterministic outcomes and ambiguous scenarios.

## Limitations
RefLens requires high-resolution inputs to reliably track small fast-moving objects like footballs. Running on standard non-CUDA CPU backend is currently supported but slow. Camera stream capabilities depend heavily on the multi-view support provided by the dataset feed.
