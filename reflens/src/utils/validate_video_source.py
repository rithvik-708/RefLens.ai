import cv2
import hashlib
import os
import argparse
import json

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def validate_video_source(video_path):
    print("="*50)
    print("RefLens Video Source Validator")
    print("="*50)
    
    abs_path = os.path.abspath(video_path)
    exists = os.path.exists(video_path)
    
    print(f"Requested video: {video_path}")
    print(f"Resolved absolute path: {abs_path}")
    print(f"File exists: {'YES' if exists else 'NO'}")
    
    if not exists:
        print("\nStatus: INVALID_VIDEO")
        return False
        
    size_bytes = os.path.getsize(video_path)
    size_mb = size_bytes / (1024 * 1024)
    extension = os.path.splitext(video_path)[1]
    filename = os.path.basename(video_path)
    
    print(f"File size: {size_mb:.2f} MB")
    print(f"Extension: {extension}")
    print(f"Video: {filename}")
    print(f"OpenCV: {cv2.__version__}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: OpenCV could not open the video.")
        print("\nStatus: INVALID_VIDEO")
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    file_hash = get_sha256(video_path)
    print(f"\nSHA-256: {file_hash}")
    
    ret, frame = cap.read()
    cap.release()
    
    os.makedirs("outputs/debug", exist_ok=True)
    preview_path = f"outputs/debug/{filename}_preview.jpg"
    
    if ret:
        cv2.imwrite(preview_path, frame)
        print(f"First-frame preview path: {preview_path}")
    
    print("\nStatus: VALID_VIDEO")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Video Source")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()
    
    validate_video_source(args.video)
