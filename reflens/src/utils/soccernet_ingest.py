import os
import json
from pathlib import Path
from typing import List, Optional
from loguru import logger
from SoccerNet.Downloader import SoccerNetDownloader
import cv2

class SoccerNetIngestion:
    def __init__(self, data_root: str = "dataset/raw", password: Optional[str] = None):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        self.password = password or os.getenv("SOCCERNET_PASSWORD")
        if not self.password:
            raise ValueError("SoccerNet video password must be set via SOCCERNET_PASSWORD environment variable or passed directly.")
            
        self.downloader = SoccerNetDownloader(LocalDirectory=str(self.data_root))
        self.downloader.password = self.password

    def download_action_labels(self, split: str = "train"):
        """Downloads label annotations containing event action timestamps."""
        logger.info(f"Fetching action annotations for split: {split}")
        self.downloader.downloadGames(
            files=["Labels-v2.json"],
            split=[split],
            task="action-spotting"
        )

    def download_game_video(self, game_path: str, video_res: str = "720p.mkv"):
        """
        Downloads specific video halves for a single match.
        Example game_path: 'england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley'
        """
        logger.info(f"Downloading high-res video for: {game_path}")
        self.downloader.downloadGames(
            files=[f"1_{video_res}", f"2_{video_res}"],
            split=[],
            task="frames"
        )

    def extract_incident_clip(
        self,
        video_path: str,
        output_path: str,
        timestamp_seconds: float,
        pre_buffer_sec: float = 4.0,
        post_buffer_sec: float = 4.0
    ) -> bool:
        """
        Cuts a high-precision spatial temporal window around a flagged event.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = max(0, int((timestamp_seconds - pre_buffer_sec) * fps))
        end_frame = int((timestamp_seconds + post_buffer_sec) * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()
        logger.info(f"Generated incident window [{start_frame} -> {end_frame}] at {output_path}")
        return True
