import os
import cv2
import numpy as np


class LaneSegmenter:
    """Builds segmentation masks for lane lines and obstacles.

    - Yellow + white lanes: HSV threshold (fast, reliable for this map).
    - Obstacles: YOLOv8-seg if weights are provided; fallback to color threshold.
    """

    def __init__(
        self,
        yolo_weights: str | None = None,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        imgsz: int = 320,
        device: str | None = None,
        yolo_every: int = 1,
    ) -> None:
        self.yolo = None
        self.yolo_conf = float(yolo_conf)
        self.yolo_iou = float(yolo_iou)
        self.imgsz = int(imgsz)
        self.device = device
        self.yolo_every = max(1, int(yolo_every))
        self._yolo_count = 0
        self._last_road_mask = None

        self.road_class_name = "road"
        self.road_class_id = None

        if yolo_weights:
            try:
                from ultralytics import YOLO

                self.yolo = YOLO(yolo_weights)
                try:
                    names = self.yolo.model.names
                    if isinstance(names, dict):
                        for k, v in names.items():
                            if v == self.road_class_name:
                                self.road_class_id = int(k)
                                break
                    elif isinstance(names, list):
                        if self.road_class_name in names:
                            self.road_class_id = names.index(self.road_class_name)
                except Exception:
                    self.road_class_id = None
            except Exception as exc:
                print(f"[segmentation] YOLO not available: {exc}")
                self.yolo = None

        # HSV thresholds (tuned for this map)
        self.yellow_low = np.array([20, 100, 100], dtype=np.uint8)
        self.yellow_high = np.array([35, 255, 255], dtype=np.uint8)

        # White lines: raise V threshold to avoid gray floor
        self.white_low = np.array([0, 0, 220], dtype=np.uint8)
        self.white_high = np.array([180, 40, 255], dtype=np.uint8)

        # Fallback obstacle color (orange/red)
        self.obs_low1 = np.array([0, 100, 80], dtype=np.uint8)
        self.obs_high1 = np.array([15, 255, 255], dtype=np.uint8)
        self.obs_low2 = np.array([160, 100, 80], dtype=np.uint8)
        self.obs_high2 = np.array([180, 255, 255], dtype=np.uint8)

    def _lane_masks(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, self.yellow_low, self.yellow_high)
        white = cv2.inRange(hsv, self.white_low, self.white_high)
        return yellow, white

    def _road_mask(self, bgr: np.ndarray) -> np.ndarray:
        if self.yolo is not None and self.road_class_id is not None:
            if self.yolo_every > 1 and (self._yolo_count % self.yolo_every) != 0:
                self._yolo_count += 1
                if self._last_road_mask is not None:
                    return self._last_road_mask
                return np.zeros(bgr.shape[:2], dtype=np.uint8)
            try:
                results = self.yolo.predict(
                    bgr,
                    imgsz=self.imgsz,
                    conf=self.yolo_conf,
                    iou=self.yolo_iou,
                    verbose=False,
                    device=self.device,
                )
                if results and results[0].masks is not None and results[0].boxes is not None:
                    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
                    cls_ids = results[0].boxes.cls.detach().cpu().numpy().astype(int)
                    for idx, m in enumerate(results[0].masks.data):
                        if idx >= len(cls_ids):
                            break
                        if cls_ids[idx] != self.road_class_id:
                            continue
                        m_np = (m.detach().cpu().numpy() * 255).astype(np.uint8)
                        m_np = cv2.resize(m_np, (bgr.shape[1], bgr.shape[0]))
                        mask = cv2.bitwise_or(mask, m_np)
                    self._last_road_mask = mask
                    self._yolo_count += 1
                    return mask
            except Exception as exc:
                print(f"[segmentation] YOLO inference failed: {exc}")
        self._yolo_count += 1
        return np.zeros(bgr.shape[:2], dtype=np.uint8)

    def build_obs(self, bgr: np.ndarray, size: int = 84) -> np.ndarray:
        bgr_small = cv2.resize(bgr, (size, size))
        road = self._road_mask(bgr_small)
        yellow, white = self._lane_masks(bgr_small)
        obs = np.stack([road, yellow, white], axis=-1).astype(np.uint8)
        return obs
