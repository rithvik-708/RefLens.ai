from dataclasses import dataclass
from typing import Generator, Tuple
import cv2
import numpy as np
from loguru import logger

@dataclass
class FramePacket:
    frame_id: int
    pts_milliseconds: float
    frame: np.ndarray

class VideoStreamIngest:
    def __init__(self, video_path: str, batch_size: int = 16):
        self.video_path = video_path
        self.batch_size = batch_size
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to access source stream: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def stream_batches(self) -> Generator[Tuple[np.ndarray, list[FramePacket]], None, None]:
        """
        Yields contiguous arrays of shape (B, H, W, C) alongside precise PTS metadata.
        """
        batch_frames = []
        batch_meta = []
        
        while self.cap.isOpened():
            current_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            pts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            
            ret, frame = self.cap.read()
            if not ret:
                break
                
            packet = FramePacket(frame_id=current_idx, pts_milliseconds=pts_ms, frame=frame)
            batch_frames.append(frame)
            batch_meta.append(packet)
            
            if len(batch_frames) == self.batch_size:
                yield np.stack(batch_frames, axis=0), batch_meta
                batch_frames.clear()
                batch_meta.clear()
                
        if batch_frames:
            yield np.stack(batch_frames, axis=0), batch_meta

    def close(self):
        if self.cap.isOpened():
            self.cap.release()
