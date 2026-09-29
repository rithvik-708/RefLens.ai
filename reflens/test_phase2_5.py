import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.vision.pipeline import PerceptionEngine

def main():
    model_path = "yolov8n.onnx"
    source_video = r"c:\Coding Stuff\RefLens\reflens\dataset\raw\SoccerNet\england_epl\2014-2015\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley\1_720p.mkv"
    output_video = "outputs/phase2_5/team_classification_test.mp4"
    
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
        
    if not os.path.exists(source_video):
        print(f"Source video {source_video} not found!")
        return

    print("Initializing PerceptionEngine...")
    engine = PerceptionEngine(
        model_path=model_path,
        source_video=source_video,
        output_video=output_video
    )
    
    print("Running Pipeline...")
    metrics, tracks = engine.run()
    
    print(f"\nFinished processing! Processed {metrics.get('processed_frames', 0)} frames.")
    print(f"Output saved to {output_video}")

if __name__ == "__main__":
    main()
