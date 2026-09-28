import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from loguru import logger

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

def verify_opencv_onnx():
    onnx_path = "dummy_model.onnx"
    
    # 1. Create and export a simple PyTorch model to ONNX
    try:
        model = DummyModel()
        model.eval()
        dummy_input = torch.randn(1, 10)
        torch.onnx.export(
            model, dummy_input, onnx_path, 
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Exported dummy PyTorch model to {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        return False, f"Export error: {e}"

    # 2. Load the ONNX model using OpenCV 5
    try:
        net = cv2.dnn.readNetFromONNX(onnx_path)
        logger.info("Loaded ONNX model via cv2.dnn.readNetFromONNX")
        
        # 3. Create input tensor and run forward
        np_input = np.random.randn(1, 10).astype(np.float32)
        net.setInput(np_input)
        output = net.forward()
        
        # 4. Verify output
        if output.shape == (1, 2):
            logger.info(f"OpenCV ONNX Inference PASS. Output shape: {output.shape}")
            return True, "PASS"
        else:
            logger.error(f"Unexpected output shape: {output.shape}")
            return False, f"Unexpected output shape: {output.shape}"
            
    except Exception as e:
        logger.error(f"OpenCV ONNX Inference failed: {e}")
        return False, f"Inference error: {e}"
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

if __name__ == "__main__":
    success, msg = verify_opencv_onnx()
    print("ONNX Verification:", "PASS" if success else f"FAIL ({msg})")
