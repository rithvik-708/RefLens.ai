import cv2
import time
from loguru import logger
from src.vision.stream_reader import VideoStreamIngest
from src.detection.detector import LightweightDetector
from src.detection.tracker import PerceptionTracker

class PerceptionEngine:
    def __init__(self, model_path: str, source_video: str, output_video: str):
        self.ingest = VideoStreamIngest(source_video, batch_size=1)
        self.detector = LightweightDetector(model_path)
        self.tracker = PerceptionTracker(frame_rate=int(self.ingest.fps))
        self.output_video = output_video
        
        self.writer = None
        self.metrics = {
            "processed_frames": 0,
            "inference_times": [],
            "total_times": [],
            "person_detections": 0,
            "ball_detections": 0,
            "ball_frames": 0,
            "sum_ball_conf": 0.0,
            "sum_person_conf": 0.0
        }

    def _init_writer(self, frame_shape: tuple):
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        import os
        os.makedirs(os.path.dirname(self.output_video), exist_ok=True)
        self.writer = cv2.VideoWriter(self.output_video, fourcc, self.ingest.fps, (w, h))

    def run(self):
        logger.info("Starting Perception Engine Pipeline...")
        
        start_pipeline = time.time()
        
        for batch_frames, batch_meta in self.ingest.stream_batches():
            for frame, meta in zip(batch_frames, batch_meta):
                t0 = time.time()
                if self.writer is None:
                    self._init_writer(frame.shape)
                
                # 1. Detection
                t_det0 = time.time()
                detections = self.detector.detect(frame)
                t_det1 = time.time()
                
                self.metrics["inference_times"].append(t_det1 - t_det0)
                
                # Track detection stats
                balls = [d for d in detections if d.class_id == 32]
                persons = [d for d in detections if d.class_id == 0]
                
                self.metrics["person_detections"] += len(persons)
                self.metrics["ball_detections"] += len(balls)
                
                if balls:
                    self.metrics["ball_frames"] += 1
                    self.metrics["sum_ball_conf"] += sum(b.confidence for b in balls)
                if persons:
                    self.metrics["sum_person_conf"] += sum(p.confidence for p in persons)
                
                # 2. Tracking & State Estimation
                active_tracks = self.tracker.update(
                    detections, frame, meta.frame_id, meta.pts_milliseconds
                )
                
                # 3. Visualization and Video Writing
                vis_frame = frame.copy()
                for track_id, entity in active_tracks.items():
                    latest_state = entity.history[-1]
                    if latest_state.frame_id == meta.frame_id:
                        x1, y1, x2, y2 = map(int, latest_state.bbox)
                        
                        color = (0, 0, 255) if entity.class_id == 32 else (255, 0, 0)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"ID:{track_id} | Vx:{latest_state.v_x:.1f} Vy:{latest_state.v_y:.1f}"
                        cv2.putText(vis_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                self.writer.write(vis_frame)
                
                t1 = time.time()
                self.metrics["total_times"].append(t1 - t0)
                self.metrics["processed_frames"] += 1
                
                if self.metrics["processed_frames"] >= 150:
                    logger.info("Reached 150 frames limit for testing. Stopping early.")
                    break
            
            if self.metrics["processed_frames"] >= 150:
                break
                
        self.ingest.close()
        if self.writer:
            self.writer.release()
            
        end_pipeline = time.time()
        self.metrics["total_duration"] = end_pipeline - start_pipeline
        
        logger.info(f"Pipeline complete. Output saved to {self.output_video}")
        return self.metrics, self.tracker.active_tracks
