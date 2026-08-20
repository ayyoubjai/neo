import importlib.util
import os
from typing import Any, Dict, Optional


class CameraAdapterError(RuntimeError):
    pass


class OpenCvCameraAdapter:
    def __init__(self, camera_index: int = 0, cv2_module=None):
        self._camera_index = int(camera_index)
        self._cv2_module = cv2_module

    @staticmethod
    def is_available() -> bool:
        try:
            return importlib.util.find_spec("cv2") is not None
        except Exception:
            return False

    def _cv2(self):
        if self._cv2_module is not None:
            return self._cv2_module
        try:
            import cv2
        except ImportError as e:
            raise CameraAdapterError(f"opencv-python not available: {e}") from e
        self._cv2_module = cv2
        return cv2

    def capture_frame(self, abs_path: str, *, width: Optional[int] = None, height: Optional[int] = None) -> Dict[str, Any]:
        cv2 = self._cv2()
        cap = cv2.VideoCapture(self._camera_index)
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        try:
            if not cap.isOpened():
                raise CameraAdapterError(f"Unable to open camera index {self._camera_index}")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise CameraAdapterError(f"Unable to read frame from camera index {self._camera_index}")
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            if not cv2.imwrite(abs_path, frame):
                raise CameraAdapterError(f"Unable to write camera frame to {abs_path}")
            frame_height, frame_width = frame.shape[:2]
            return {
                "width": int(frame_width),
                "height": int(frame_height),
                "camera_index": self._camera_index,
            }
        finally:
            cap.release()
