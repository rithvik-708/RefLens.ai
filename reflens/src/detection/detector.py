import cv2
import numpy as np
from typing import List
from loguru import logger
from src.detection.types import Detection

class LightweightDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.45, nms_threshold: float = 0.4):
        """
        Initializes an OpenCV 5 DNN instance running an ONNX YOLOv8 model.
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = {0: "person", 32: "sports_ball"}
        self.debug = False
        self.thresholds = None
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
        
        class_ids = []
        confidences = []
        boxes = []
        
        max_raw_ball_conf = 0.0
        
        for row in output:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]
            
            if class_id == 32 and confidence > max_raw_ball_conf:
                max_raw_ball_conf = confidence
                
            threshold = self.conf_threshold
            if self.thresholds and class_id in self.thresholds:
                threshold = self.thresholds[class_id]
            
            if class_id in self.target_classes and confidence >= threshold:
                x, y, w, h = row[0], row[1], row[2], row[3]
                
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                class_ids.append(int(class_id))
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
        # NMS
        # For multiple classes, ideally NMS should be per-class. We will pass a single min threshold to NMSBoxes.
        min_thresh = self.conf_threshold
        if self.thresholds:
            min_thresh = min(self.thresholds.values())
            
        indices = cv2.dnn.NMSBoxes(boxes, confidences, min_thresh, self.nms_threshold)
        detections = []
        self.last_max_ball_conf = max_raw_ball_conf
        
        if self.debug:
            logger.debug(f"[DEBUG] Max raw BALL confidence in frame: {max_raw_ball_conf:.4f}")
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                left, top, width, height = box[0], box[1], box[2], box[3]
                detections.append(Detection(
                    bbox=np.array([left, top, left + width, top + height]),
                    confidence=confidences[i],
                    class_id=class_ids[i],
                    class_name=self.target_classes[class_ids[i]]
                ))
                
        if self.debug:
            logger.debug(f"[DEBUG] Post-processing complete. Detections passed NMS: {len(detections)}")
            for d in detections:
                logger.debug(f"[DEBUG] Detection -> class:{d.class_name} conf:{d.confidence:.4f} bbox:{d.bbox}")
                
        return detections
