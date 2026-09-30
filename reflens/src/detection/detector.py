import cv2
import numpy as np
from typing import List
from loguru import logger
from src.detection.types import Detection

class LightweightDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.35, nms_threshold: float = 0.4, thresholds: dict = None):
        """
        Initializes an OpenCV 5 DNN instance running an ONNX YOLOv8 model.
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = {0: "person", 32: "sports_ball"}
        self.debug = False
        self.thresholds = thresholds if thresholds is not None else {0: 0.35, 32: 0.15}
        self.last_max_ball_conf = 0.0
        
        logger.info(f"Loading ONNX model into OpenCV DNN: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        
        # Check CUDA availability in OpenCV build
        build_info = cv2.getBuildInformation()
        cuda_support = "CUDA" in build_info and "DNN_BACKEND_CUDA" in build_info
        
        if cuda_support:
            logger.info("OpenCV built with CUDA support. Using CUDA FP16 backend.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            logger.warning("OpenCV DNN CUDA unavailable in current wheel. Falling back to CPU backend.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        # YOLOv8 default input size
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        return self._postprocess(outputs[0], frame.shape)

    def _postprocess(self, output: np.ndarray, frame_shape: tuple) -> List[Detection]:
        h_orig, w_orig = frame_shape[:2]
        x_factor = w_orig / 640.0
        y_factor = h_orig / 640.0
        
        if self.debug:
            logger.debug(f"[DEBUG] Raw ONNX output shape: {output.shape}")
            
        # output shape: (84, 8400) for YOLOv8
        output = output.T
        
        if self.debug:
            logger.debug(f"[DEBUG] Transposed output shape: {output.shape}")
        
        classes_scores = output[:, 4:]
        class_ids = np.argmax(classes_scores, axis=1)
        confidences = np.max(classes_scores, axis=1)
        
        # Track max raw ball confidence for debugging
        ball_mask = (class_ids == 32)
        if np.any(ball_mask):
            self.last_max_ball_conf = float(np.max(confidences[ball_mask]))
        else:
            self.last_max_ball_conf = 0.0

        if self.debug:
            logger.debug(f"[DEBUG] Max raw BALL confidence in frame: {self.last_max_ball_conf:.4f}")

        # Basic vectorized thresholding mask
        min_thresh = self.conf_threshold
        if self.thresholds:
            min_thresh = min(self.thresholds.values())
            
        target_ids = list(self.target_classes.keys())
        mask = (confidences >= min_thresh) & np.isin(class_ids, target_ids)
        
        # Filter arrays
        valid_output = output[mask]
        valid_class_ids = class_ids[mask]
        valid_confidences = confidences[mask]
        
        boxes = []
        final_class_ids = []
        final_confidences = []
        
        for i in range(len(valid_output)):
            c_id = int(valid_class_ids[i])
            conf = float(valid_confidences[i])
            
            # Apply per-class thresholding if specified
            thresh = self.thresholds[c_id] if self.thresholds and c_id in self.thresholds else self.conf_threshold
            if conf >= thresh:
                x, y, w, h = valid_output[i, 0], valid_output[i, 1], valid_output[i, 2], valid_output[i, 3]
                
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([left, top, width, height])
                final_class_ids.append(c_id)
                final_confidences.append(conf)
                
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, final_confidences, min_thresh, self.nms_threshold)
        detections = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                left, top, width, height = box[0], box[1], box[2], box[3]
                detections.append(Detection(
                    bbox=np.array([left, top, left + width, top + height]),
                    confidence=final_confidences[i],
                    class_id=final_class_ids[i],
                    class_name=self.target_classes[final_class_ids[i]]
                ))
                
        if self.debug:
            logger.debug(f"[DEBUG] Post-processing complete. Detections passed NMS: {len(detections)}")
            for d in detections:
                logger.debug(f"[DEBUG] Detection -> class:{d.class_name} conf:{d.confidence:.4f} bbox:{d.bbox}")
                
        return detections
