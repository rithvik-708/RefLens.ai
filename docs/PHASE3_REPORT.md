# RefLens Phase 3 Audit & Validation Report

## 1. Pitch Feature Extraction
- Average lines detected per frame: 7.3
- **Status**: PASS. Hough lines successfully extracted using color isolation.

## 2. Homography
- Ground-truth pitch correspondences are unavailable for this arbitrary video.
- **Status**: UNVERIFIED. Matrix H initialized with dummy points.

## 3. Contact Frame Detection
- Failed to detect a pass. Ball tracking was likely unreliable.
- **Status**: UNVERIFIED.

## 4. Offside Geometry Engine
- Unit tests confirm logic for team assignment, ball reference, goalkeeper handling, and attacking directions.
- **Status**: PASS (in deterministic synthetic unit tests).

## 5. End-to-End Football Validation
- Run completed on test video.
- Debug visualization generated.
- **Status**: PARTIAL. Without ground-truth homography and perfect tracking, end-to-end offside geometry cannot be fully trusted yet.

## Assumptions & Approximations
- **Bounding Box Center**: Attacker/Defender positions are approximated using the bottom-center of their bounding box (MVP approximation). This is NOT VAR-grade body geometry.
- **Team Assignment**: Since team jersey clustering is not implemented, the geometry engine relies on external input for `defenders` array.
- **Homography Calibration**: Automated intersection detection is missing; requires manual point mapping.

## Final Assessment
- Pitch feature extraction: PASS
- Homography: UNVERIFIED
- Contact frame: UNVERIFIED
- Player/reference geometry: PASS
- Rule engine: PASS
- End-to-end football validation: PARTIAL
