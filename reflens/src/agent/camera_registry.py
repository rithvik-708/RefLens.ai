from pydantic import BaseModel
from typing import Dict, Optional

class CameraInfo(BaseModel):
    camera_id: str
    video_path: str
    fps: float
    frame_count: int
    type: str

class CameraRegistry:
    def __init__(self):
        self.cameras: Dict[str, CameraInfo] = {}

    def register_camera(self, camera: CameraInfo):
        self.cameras[camera.camera_id] = camera

    def get_camera(self, camera_id: str) -> Optional[CameraInfo]:
        return self.cameras.get(camera_id)

    def is_available(self, camera_id: str) -> bool:
        return camera_id in self.cameras
