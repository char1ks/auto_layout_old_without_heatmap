from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2


@dataclass(slots=True, frozen=True)
class ExtractConfig:
    # Peak detection (no hard thresholding)
    gaussian_sigma: float = 1.0         # 0 disables. Smooths noise, stabilizes peaks.
    min_peak_distance: int = 3          # pixel radius for seed spacing (NMS)
    max_seeds: Optional[int] = None     # limit number of seeds (None = no limit)

    # Peak "prominence" (keeps only meaningful maxima)
    # Prominence is (peak_value - local_min) in a window.
    prominence_window: int = 9          # odd size; effective radius ~ window//2
    min_prominence_rel: float = 0.02    # keep peaks with prominence >= rel * (max - min)
    min_prominence_abs: float = 0.0     # absolute fallback in heat units (added to rel)

    # Post-segmentation cleanup / sampling
    min_area: int = 12                  # drop tiny basins
    top_k_per_cluster: int = 3
    nms_radius: int = 3                 # min pixel spacing among returned top points

@dataclass(slots=True)
class Cluster:
    id: int
    points: np.ndarray                   # (M, 2) [[x,y],...] in image coordinates
    scores: np.ndarray                   # (M,) values at those points, desc
    area: int                            # number of pixels
    bbox: Tuple[int, int, int, int]      # (x, y, w, h)
    weighted_center: Tuple[float, float] # (cx, cy) intensity-weighted centroid

@dataclass(slots=True)
class ExtractResult:
    clusters: List[Cluster]
    labels: np.ndarray                   # int32; -1 boundary, 0..K-1 clusters


class BrightClusterExtractor:
    """
    Weighted, threshold-free hotspot extraction:
      1) Smooth (optional), then find local maxima.
      2) Filter by prominence (value - local_min) using heat values (weights).
      3) Distance-NMS on seeds (min spacing) and optional top-N.
      4) Seeded watershed on the *inverted* heatmap -> labels.
      5) For each label: weighted center, top-K high-value points (with NMS).
    """

    def __init__(self, config: ExtractConfig):
        self.cfg = config

    def __call__(self, heatmap: np.ndarray) -> ExtractResult:
        return self.run(heatmap)

    # ---- public API ---------------------------------------------------------
    def run(self, heatmap: np.ndarray) -> ExtractResult:
        assert heatmap.ndim == 2, "heatmap must be 2D"
        hm = heatmap.astype(np.float32, copy=False)
        H, W = hm.shape

        # 1) optional smoothing (stabilizes peaks & watershed)
        if self.cfg.gaussian_sigma and self.cfg.gaussian_sigma > 0:
            hm_s = cv2.GaussianBlur(hm, ksize=(0, 0), sigmaX=self.cfg.gaussian_sigma)
        else:
            hm_s = hm

        # 2) local maxima (fast, vectorized)
        seeds_xy = self._find_seeds(hm_s)

        # If no seeds -> no clusters; labels all -1
        labels = np.full((H, W), -1, dtype=np.int32)
        if len(seeds_xy) == 0:
            return ExtractResult(clusters=[], labels=labels)

        # 3) Watershed on inverted height map (bright regions become basins)
        labels_ws = self._watershed_labels(hm_s, seeds_xy)

        # 4) Build clusters from labels (compact to 0..K-1)
        clusters, labels_compact = self._clusters_from_labels(hm_s, labels_ws)

        return ExtractResult(clusters=clusters, labels=labels_compact)

    # ---- internals ----------------------------------------------------------
    def _find_seeds(self, hm_s: np.ndarray) -> np.ndarray:
        """Find strong peaks without thresholds: local maxima + prominence + NMS."""
        H, W = hm_s.shape
        # local maxima via grayscale dilation
        r = max(1, self.cfg.min_peak_distance)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r + 1, 2*r + 1))
        local_max = cv2.dilate(hm_s, k)
        peaks_mask = (hm_s >= local_max - 1e-12)  # allow plateau tops

        # prominence: peak_value - local_min in a larger window
        pw = max(3, self.cfg.prominence_window | 1)  # odd
        kprom = cv2.getStructuringElement(cv2.MORPH_RECT, (pw, pw))
        local_min = cv2.erode(hm_s, kprom)
        prominence = hm_s - local_min

        # relative prominence scale
        dyn = float(hm_s.max() - hm_s.min()) if np.isfinite(hm_s).all() else float(hm_s.max())
        rel_thr = self.cfg.min_prominence_rel * max(dyn, 1e-6) + self.cfg.min_prominence_abs

        cand_mask = peaks_mask & (prominence >= rel_thr)
        ys, xs = np.where(cand_mask)
        if xs.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        vals = hm_s[ys, xs]
        # Sort by value descending and perform distance-NMS
        order = np.argsort(vals)[::-1]
        xs, ys, vals = xs[order], ys[order], vals[order]

        seeds = []
        r2 = float(self.cfg.min_peak_distance * self.cfg.min_peak_distance)
        for x, y in zip(xs, ys):
            keep = True
            for xx, yy in seeds:
                dx, dy = x - xx, y - yy
                if (dx*dx + dy*dy) <= r2:
                    keep = False
                    break
            if keep:
                seeds.append((int(x), int(y)))
                if self.cfg.max_seeds is not None and len(seeds) >= self.cfg.max_seeds:
                    break

        return np.asarray(seeds, dtype=np.int32)

    def _watershed_labels(self, hm_s: np.ndarray, seeds_xy: np.ndarray) -> np.ndarray:
        """Run OpenCV watershed with given seed markers on inverted intensities."""
        H, W = hm_s.shape

        # build markers: >0 for seeds, 0 elsewhere (int32)
        markers = np.zeros((H, W), dtype=np.int32)
        for i, (x, y) in enumerate(seeds_xy, start=1):
            markers[y, x] = i

        # Invert & scale to 8-bit for watershed (bright = deep basins)
        elev = hm_s.max() - hm_s
        elev_u8 = cv2.normalize(elev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img3 = cv2.merge([elev_u8, elev_u8, elev_u8])

        cv2.watershed(img3, markers)  # in-place update
        # Markers: -1 = boundary; >=1 = seed id. There is no background seed.
        return markers

    def _clusters_from_labels(self, hm_s: np.ndarray, markers: np.ndarray) -> Tuple[List[Cluster], np.ndarray]:
        """Pack clusters; relabel to compact 0..K-1; compute weighted centers & top-K."""
        H, W = hm_s.shape
        # keep only positive labels
        pos = markers > 0
        if not np.any(pos):
            return [], np.full((H, W), -1, dtype=np.int32)

        unique_labels = np.unique(markers[pos])
        clusters: List[Cluster] = []
        labels_compact = np.full((H, W), -1, dtype=np.int32)

        new_id = 0
        for lab in unique_labels:
            ys, xs = np.where(markers == lab)
            if ys.size < self.cfg.min_area:
                continue

            vals = hm_s[ys, xs]
            area = int(ys.size)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

            # weighted centroid (heat as weight)
            wsum = float(vals.sum())
            cx = float((xs.astype(np.float32) * vals).sum() / max(wsum, 1e-6))
            cy = float((ys.astype(np.float32) * vals).sum() / max(wsum, 1e-6))

            # Top-K high-value points within this label (with spacing NMS)
            pts_xy, scores = self._take_topk_with_nms_xy(xs, ys, vals,
                                                         self.cfg.top_k_per_cluster,
                                                         self.cfg.nms_radius)

            clusters.append(Cluster(
                id=new_id,
                points=pts_xy,
                scores=scores,
                area=area,
                bbox=bbox,
                weighted_center=(cx, cy),
            ))
            labels_compact[markers == lab] = new_id
            new_id += 1

        return clusters, labels_compact

    @staticmethod
    def _take_topk_with_nms_xy(
        xs: np.ndarray, ys: np.ndarray, vals: np.ndarray, k: int, radius: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy top-K with pixel-distance NMS inside a cluster."""
        if k <= 0 or len(vals) == 0:
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.float32))

        cand = min(len(vals), max(32, 5 * k))
        if cand < len(vals):
            idx = np.argpartition(vals, -cand)[-cand:]
        else:
            idx = np.arange(len(vals))
        idx = idx[np.argsort(vals[idx])[::-1]]  # sort desc

        chosen_xy: List[Tuple[int, int]] = []
        chosen_scores: List[float] = []
        r2 = float(radius * radius)
        for i in idx:
            x, y, v = int(xs[i]), int(ys[i]), float(vals[i])
            ok = True
            for xx, yy in chosen_xy:
                dx, dy = x - xx, y - yy
                if (dx*dx + dy*dy) <= r2:
                    ok = False
                    break
            if ok:
                chosen_xy.append((x, y))
                chosen_scores.append(v)
                if len(chosen_xy) >= k:
                    break

        if not chosen_xy:
            m = int(np.argmax(vals))
            chosen_xy = [(int(xs[m]), int(ys[m]))]
            chosen_scores = [float(vals[m])]

        return (np.asarray(chosen_xy, dtype=np.float32),
                np.asarray(chosen_scores, dtype=np.float32))



def sample_points_with_value(
    arr: np.ndarray,
    value: float,
    n: int,
    *,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    match_nan: bool = False,
    replace: bool = False,
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Return N random (x, y) image coordinates from a 2D array where arr[y, x] == value.

    - Exact match by default; use atol/rtol for float tolerance via np.isclose.
    - Set match_nan=True and value=np.nan to sample NaN positions.
    - If replace=False, raises if fewer than N matches exist.

    Returns: list of (x, y) integer tuples.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("arr must be 2D")

    if match_nan and np.isnan(value):
        mask = np.isnan(a)
    elif atol is not None or rtol is not None:
        mask = np.isclose(a, value, atol=atol or 0.0, rtol=rtol or 0.0, equal_nan=False)
    else:
        mask = (a == value)

    flat_idx = np.flatnonzero(mask)
    if flat_idx.size == 0:
        return []

    if not replace and n > flat_idx.size:
        raise ValueError(f"Requested n={n} but only {flat_idx.size} matching points exist.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(flat_idx, size=n, replace=replace)
    rows, cols = np.unravel_index(chosen, a.shape)  # rows=y, cols=x

    # Convert to list of (x, y) ints
    return [(int(x), int(y)) for y, x in zip(rows, cols)]
