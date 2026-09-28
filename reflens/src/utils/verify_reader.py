import os
import cv2
import numpy as np
from loguru import logger
import sys

# Add src to path so we can import vision.stream_reader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.vision.stream_reader import VideoStreamIngest

def create_synthetic_video(output_path, width=640, height=480, fps=25.0, num_frames=50):
    """Generates a tiny synthetic video for smoke testing the reader."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create a moving rectangle to simulate video
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = int((i / num_frames) * width)
        cv2.rectangle(frame, (x, 100), (x+50, 150), (0, 255, 0), -1)
        out.write(frame)
        
    out.release()
    logger.info(f"Created synthetic test video at {output_path}")

def verify_stream_reader():
    video_path = "dataset/raw/synthetic_test.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # 1. Create a tiny synthetic MP4 video
    create_synthetic_video(video_path, fps=25.0, num_frames=50)
    
    # 2. Use stream_reader to parse it
    try:
        reader = VideoStreamIngest(video_path, batch_size=10)
        
        # Read the first batch to verify decoding
        generator = reader.stream_batches()
        first_batch_frames, first_batch_meta = next(generator)
        
        duration = reader.total_frames / reader.fps if reader.fps > 0 else 0
        
        logger.info(f"Resolution: {first_batch_frames.shape[2]}x{first_batch_frames.shape[1]}")
        logger.info(f"FPS: {reader.fps}")
        logger.info(f"Frames: {reader.total_frames}")
        logger.info(f"Duration: {duration:.2f} sec")
        
        if len(first_batch_frames) > 0:
            logger.info("First frame: PASS")
        
        reader.close()
        
        return True, {
            "path": video_path,
            "width": first_batch_frames.shape[2],
            "height": first_batch_frames.shape[1],
            "fps": reader.fps,
            "frames": reader.total_frames,
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"Stream reader verification failed: {e}")
        return False, None
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    success, data = verify_stream_reader()
    print("Stream Reader Verification:", "PASS" if success else "FAIL")
