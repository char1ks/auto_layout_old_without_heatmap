from dataclasses import dataclass

import cv2
import numpy as np

from flashbone.core.detection.base import DetectionResult


@dataclass
class ResizeContext:
    scale: float
    orig_shape: tuple[int, int]


class ImageResizer:
    def __init__(self, max_side: int):
        self.max_side = int(max_side)

    def _resize_cv(self, img: np.ndarray, new_h: int, new_w: int, s: float) -> np.ndarray:
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    def resize(self, img: np.ndarray) -> tuple[np.ndarray, ResizeContext]:
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= self.max_side:
            return img, {"scale": 1.0, "orig_shape": (h, w)}
        s = self.max_side / m
        new_h = int(np.floor(h * s + 0.5))
        new_w = int(np.floor(w * s + 0.5))
        out = self._resize_cv(img, new_h, new_w, s)
        return out, ResizeContext(scale=s, orig_shape=(h, w))

    def _scaled_copy(self, det: DetectionResult, inv: float) -> DetectionResult:
        if det.polygons:
            polys = [np.rint(np.asarray(p, dtype=np.float32) * inv).astype(np.int32).tolist() for p in det.polygons]
        else:
            polys = []
        if det.bbox and len(det.bbox) == 4:
            b = np.asarray(det.bbox, dtype=np.float32)
            b[:2] *= inv
            b[2:] *= inv
            bbox = np.rint(b).astype(np.int32).tolist()
        else:
            bbox = []
        area = float(det.area * (inv * inv))
        return DetectionResult(polys, bbox, area, det.score, det.class_id)

    def restore(self, img: np.ndarray, dets, ctx: ResizeContext) -> tuple[np.ndarray, list[DetectionResult]]:
        s = float(ctx.scale)
        oh, ow = ctx.orig_shape
        if s == 1.0:
            out = img
            inv = 1.0
        else:
            out = self._resize(img, oh, ow, 1.0 / s)
            inv = 1.0 / s
        if isinstance(dets, list):
            new_dets = [self._scaled_copy(d, inv) for d in dets]
        else:
            new_dets = [self._scaled_copy(dets, inv)]
        return out, new_dets

    def restore_dets(self, dets, ctx: ResizeContext) -> list[DetectionResult]:
        s = float(ctx.scale)
        inv = 1.0 if s == 1.0 else 1.0 / s
        if isinstance(dets, list):
            return [self._scaled_copy(d, inv) for d in dets]
        return [self._scaled_copy(dets, inv)]
