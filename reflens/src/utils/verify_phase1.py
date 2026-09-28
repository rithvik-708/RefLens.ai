import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import cv2
import numpy as np
from loguru import logger

from src.utils.verify_onnx import verify_opencv_onnx
from src.utils.verify_soccernet import verify_soccernet_labels
from src.utils.verify_reader import verify_stream_reader

def print_banner(text):
    print("\n" + "="*40)
    print(text)
    print("="*40 + "\n")

def check_python():
    print(f"[1/8] Python ................. PASS ({sys.version.split()[0]})")
    return True

def check_opencv():
    cv_ver = cv2.__version__
    try:
        assert cv_ver.startswith("5.")
        print(f"[2/8] OpenCV 5 ............... PASS ({cv_ver})")
        return True
    except AssertionError:
        print(f"[2/8] OpenCV 5 ............... FAIL (Version is {cv_ver}, expected 5.x)")
        return False

def check_pytorch():
    try:
        ver = torch.__version__
        print(f"[3/8] PyTorch ................ PASS ({ver})")
        return True
    except Exception as e:
        print(f"[3/8] PyTorch ................ FAIL ({e})")
        return False

def check_cuda():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        try:
            x = torch.randn(2048, 2048, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            print(f"[4/8] CUDA / RTX 4050 ........ PASS ({gpu_name})")
            return True
        except Exception as e:
            print(f"[4/8] CUDA / RTX 4050 ........ FAIL (CUDA computation error: {e})")
            return False
    else:
        print("[4/8] CUDA / RTX 4050 ........ FAIL (CUDA not available in current torch build)")
        return False

def main():
    print_banner("RefLens Phase 1 Verification")
    
    results = []
    
    # 1. Python
    results.append(check_python())
    
    # 2. OpenCV 5
    results.append(check_opencv())
    
    # 3. PyTorch
    results.append(check_pytorch())
    
    # 4. CUDA / RTX 4050
    results.append(check_cuda())
    
    # 5. OpenCV ONNX
    try:
        success, msg = verify_opencv_onnx()
        if success:
            print("[5/8] OpenCV ONNX ............ PASS")
        else:
            print(f"[5/8] OpenCV ONNX ............ FAIL ({msg})")
        results.append(success)
    except Exception as e:
        print(f"[5/8] OpenCV ONNX ............ FAIL ({e})")
        results.append(False)
        
    # 6 & 7. SoccerNet & OFFSIDE
    try:
        success, ev, game = verify_soccernet_labels()
        if success:
            print("[6/8] SoccerNet .............. PASS")
            print("[7/8] OFFSIDE parsing ........ PASS")
        else:
            print("[6/8] SoccerNet .............. FAIL")
            print("[7/8] OFFSIDE parsing ........ FAIL")
        results.append(success)
        results.append(success)  # One for SoccerNet, one for parsing
    except Exception as e:
        print(f"[6/8] SoccerNet .............. FAIL ({e})")
        print(f"[7/8] OFFSIDE parsing ........ FAIL ({e})")
        results.append(False)
        results.append(False)
        
    # 8. Stream Reader
    try:
        success, data = verify_stream_reader()
        if success:
            print("[8/8] MP4 reader ............. PASS")
        else:
            print("[8/8] MP4 reader ............. FAIL")
        results.append(success)
    except Exception as e:
        print(f"[8/8] MP4 reader ............. FAIL ({e})")
        results.append(False)
        
    if all(results):
        print_banner("PHASE 1: COMPLETE")
    else:
        print_banner("PHASE 1: NOT COMPLETE")
        print("Please review the failed checks above.")

if __name__ == "__main__":
    main()
