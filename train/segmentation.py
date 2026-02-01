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
        self._printed_class_info = False

        self.road_class_name = os.getenv("SEG_ROAD_CLASS", "road")
        self.obstacle_class_name = os.getenv("SEG_OBS_CLASS", "object")
        self.road_class_id = None
        self.obstacle_class_id = None

        if yolo_weights:
            try:
                from ultralytics import YOLO

                self.yolo = YOLO(yolo_weights)
                try:
                    names = self.yolo.model.names
                    self._resolve_class_ids(names)
                    self._print_class_info(names)
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

        # Road mask cleanup (reduce noise)
        self.road_open = int(os.getenv("SEG_ROAD_OPEN", "0"))
        self.road_close = int(os.getenv("SEG_ROAD_CLOSE", "15"))
        self.road_min_area = float(os.getenv("SEG_ROAD_MIN_AREA", "0.05"))  # ratio of image
        self.road_keep_largest = os.getenv("SEG_ROAD_KEEP_LARGEST", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.road_y_start = float(os.getenv("SEG_ROAD_Y_START", "0.0"))
        self.road_median = int(os.getenv("SEG_ROAD_MEDIAN", "5"))
        self.road_fill_holes = os.getenv("SEG_ROAD_FILL_HOLES", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.road_hole_max_area = float(os.getenv("SEG_ROAD_HOLE_MAX_AREA", "1.0"))  # ratio
        self.road_dilate = int(os.getenv("SEG_ROAD_DILATE", "3"))
        self.road_keep_bottom = os.getenv("SEG_ROAD_KEEP_BOTTOM", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.road_bottom_margin = int(os.getenv("SEG_ROAD_BOTTOM_MARGIN", "8"))
        self.road_use_convex = os.getenv("SEG_ROAD_CONVEX", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.road_temporal = os.getenv("SEG_ROAD_TEMPORAL", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.road_temporal_alpha = float(os.getenv("SEG_ROAD_ALPHA", "0.8"))
        self.subtract_obstacles = os.getenv("SEG_SUBTRACT_OBS", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # Obstacle mask cleanup before subtracting from road
        self.obs_min_area = float(os.getenv("SEG_OBS_MIN_AREA", "0.01"))  # ratio
        self.obs_erode = int(os.getenv("SEG_OBS_ERODE", "0"))

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return mask
        out = mask.copy()
        if self.road_median >= 3 and (self.road_median % 2) == 1:
            out = cv2.medianBlur(out, self.road_median)
        if self.road_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.road_close, self.road_close))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
        if self.road_open > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.road_open, self.road_open))
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
        if self.road_dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.road_dilate, self.road_dilate))
            out = cv2.dilate(out, k)
        if self.road_y_start > 0.0:
            h = out.shape[0]
            y0 = int(h * self.road_y_start)
            if y0 > 0:
                out[:y0, :] = 0
        if self.road_min_area > 0:
            min_area = int(self.road_min_area * out.size)
            bin_mask = (out > 0).astype(np.uint8)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            if num > 1:
                if self.road_keep_largest:
                    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    keep = np.zeros_like(out)
                    if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                        keep[labels == idx] = 255
                    out = keep
                else:
                    keep = np.zeros_like(out)
                    for i in range(1, num):
                        if stats[i, cv2.CC_STAT_AREA] >= min_area:
                            keep[labels == i] = 255
                    out = keep
        if self.road_keep_bottom:
            bin_mask = (out > 0).astype(np.uint8)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            if num > 1:
                h = labels.shape[0]
                bm = max(1, min(self.road_bottom_margin, h))
                bottom_labels = set(labels[h - bm : h, :].ravel())
                bottom_labels.discard(0)
                if bottom_labels:
                    keep_label = max(bottom_labels, key=lambda l: stats[l, cv2.CC_STAT_AREA])
                else:
                    keep_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                keep = np.zeros_like(out)
                keep[labels == keep_label] = 255
                out = keep
        if self.road_fill_holes:
            hole_max_area = int(self.road_hole_max_area * out.size)
            if hole_max_area > 0:
                inv = (out == 0).astype(np.uint8)
                h, w = inv.shape
                flood = inv.copy()
                mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(flood, mask, (0, 0), 0)
                holes = (inv == 1) & (flood == 1)
                if np.any(holes):
                    holes_u8 = holes.astype(np.uint8)
                    num_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(holes_u8, connectivity=8)
                    for i in range(1, num_h):
                        if stats_h[i, cv2.CC_STAT_AREA] <= hole_max_area:
                            out[labels_h == i] = 255
        if self.road_use_convex:
            contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(cnt)
                filled = np.zeros_like(out)
                cv2.fillConvexPoly(filled, hull, 255)
                out = filled
        if self.road_temporal and self._last_road_mask is not None:
            alpha = max(0.0, min(1.0, self.road_temporal_alpha))
            blend = cv2.addWeighted(out, alpha, self._last_road_mask, 1.0 - alpha, 0)
            out = (blend >= 128).astype(np.uint8) * 255
        return out

    def _resolve_class_ids(self, names) -> None:
        if isinstance(names, dict):
            for k, v in names.items():
                if v == self.road_class_name:
                    self.road_class_id = int(k)
                if v == self.obstacle_class_name:
                    self.obstacle_class_id = int(k)
        elif isinstance(names, list):
            if self.road_class_name in names:
                self.road_class_id = names.index(self.road_class_name)
            if self.obstacle_class_name in names:
                self.obstacle_class_id = names.index(self.obstacle_class_name)

    def _print_class_info(self, names) -> None:
        if self._printed_class_info:
            return
        print(
            f"[segmentation] names={names} road_id={self.road_class_id} "
            f"obstacles_id={self.obstacle_class_id}",
            flush=True,
        )
        self._printed_class_info = True

    def _lane_masks(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, self.yellow_low, self.yellow_high)
        white = cv2.inRange(hsv, self.white_low, self.white_high)
        return yellow, white

    def _road_mask(self, bgr: np.ndarray) -> np.ndarray:
        if self.yolo is not None:
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
                if results and results[0].masks is not None:
                    if self.road_class_id is None:
                        try:
                            self._resolve_class_ids(results[0].names)
                            self._print_class_info(results[0].names)
                        except Exception:
                            pass
                    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
                    obs_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
                    if results[0].boxes is None:
                        for m in results[0].masks.data:
                            m_np = (m.detach().cpu().numpy() * 255).astype(np.uint8)
                            m_np = cv2.resize(m_np, (bgr.shape[1], bgr.shape[0]))
                            mask = cv2.bitwise_or(mask, m_np)
                    else:
                        cls_ids = results[0].boxes.cls.detach().cpu().numpy().astype(int)
                        for idx, m in enumerate(results[0].masks.data):
                            if idx >= len(cls_ids):
                                break
                            m_np = (m.detach().cpu().numpy() * 255).astype(np.uint8)
                            m_np = cv2.resize(m_np, (bgr.shape[1], bgr.shape[0]))
                            cid = cls_ids[idx]
                            if self.road_class_id is not None and cid == self.road_class_id:
                                mask = cv2.bitwise_or(mask, m_np)
                            elif self.obstacle_class_id is not None and cid == self.obstacle_class_id:
                                obs_mask = cv2.bitwise_or(obs_mask, m_np)
                            elif self.road_class_id is None:
                                # fallback: treat non-obstacle as road
                                if self.obstacle_class_id is None or cid != self.obstacle_class_id:
                                    mask = cv2.bitwise_or(mask, m_np)
                    if self.subtract_obstacles and self.obstacle_class_id is not None and obs_mask.any():
                        if self.obs_erode > 0:
                            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.obs_erode, self.obs_erode))
                            obs_mask = cv2.erode(obs_mask, k)
                        obs_area = float(np.count_nonzero(obs_mask))
                        min_area = self.obs_min_area * obs_mask.size
                        if obs_area >= min_area:
                            mask = cv2.bitwise_and(mask, cv2.bitwise_not(obs_mask))
                    mask = self._clean_mask(mask)
                    self._last_road_mask = mask
                    self._yolo_count += 1
                    return mask
            except Exception as exc:
                print(f"[segmentation] YOLO inference failed: {exc}")
        self._yolo_count += 1
        return np.zeros(bgr.shape[:2], dtype=np.uint8)

    def build_obs(self, bgr: np.ndarray, size: int = 84) -> np.ndarray:
        # Run YOLO on full-res frame for better mask quality, then resize
        road_full = self._road_mask(bgr)
        bgr_small = cv2.resize(bgr, (size, size))
        road = cv2.resize(road_full, (size, size), interpolation=cv2.INTER_NEAREST)
        yellow, white = self._lane_masks(bgr_small)
        obs = np.stack([road, yellow, white], axis=-1).astype(np.uint8)
        return obs
