import cv2
import numpy as np


def to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def color_distance(frame: np.ndarray, bg_lab: np.ndarray) -> np.ndarray:
    """Per-pixel CIE76 colour difference (ΔE) to the background."""
    d = np.linalg.norm(to_lab(frame) - bg_lab, axis=2)
    return cv2.GaussianBlur(d, (0, 0), 1.0)


def _disk(r: float) -> np.ndarray:
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    flood = np.pad(mask, 1)
    cv2.floodFill(flood, None, (0, 0), 1)
    return mask | (flood[1:-1, 1:-1] == 0).astype(np.uint8)


def object_mask(dist: np.ndarray, hi: float, min_area_frac: float = 2e-4) -> np.ndarray | None:
    """Binary mask of the main moving object: the largest blob plus fragments near it, holes filled."""
    h, w = dist.shape
    s = h / 1080
    core = (dist > hi).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, _disk(1.5 * s))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _disk(6 * s))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    if areas[main - 1] < min_area_frac * h * w:
        return None
    # Thin dark parts (legs on a dark floor) can split the object; keep pieces near the main blob.
    x, y, bw, bh = stats[main, :4]
    pad = 0.3 * max(bw, bh)
    x0, y0, x1, y1 = x - pad, y - pad, x + bw + pad, y + bh + pad
    keep = np.zeros(n, bool)
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        keep[i] = area >= 0.02 * areas[main - 1] and cx < x1 and cx + cw > x0 and cy < y1 and cy + ch > y0
    return _fill_holes(keep[labels].astype(np.uint8))


def soft_alpha(dist: np.ndarray, mask: np.ndarray, lo: float, hi: float, reach: float = 40) -> np.ndarray:
    """Opaque inside the object mask; elsewhere a ΔE ramp that fades out with distance from the
    object, so soft contact shadows come along without hard cut-out edges."""
    s = dist.shape[0] / 1080
    t = np.clip((dist - lo) / (hi - lo), 0, 1)
    ramp = t * t * (3 - 2 * t)
    to_obj = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    falloff = np.clip(1 - to_obj / (reach * s), 0, 1)
    alpha = np.maximum(mask.astype(np.float32), ramp * falloff)
    return cv2.GaussianBlur(alpha, (0, 0), 0.8 * s + 0.3)


def centroid(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def select_indices(centroids: list, n: int, spacing: str) -> list[int]:
    """Pick n frame indices, evenly spaced in time or along the object's path."""
    valid = [i for i, c in enumerate(centroids) if c is not None]
    if len(valid) < n:
        raise ValueError(f"object detected in only {len(valid)} frames, fewer than the {n} requested")
    if spacing == "time":
        targets = np.linspace(valid[0], valid[-1], n)
        return [min(valid, key=lambda i: abs(i - t)) for t in targets]
    pts = np.array([centroids[i] for i in valid])
    k = min(9, len(pts) // 2 * 2 - 1)
    if k >= 3:  # smooth so detection jitter while standing still doesn't count as travel
        padded = np.pad(pts, ((k // 2, k // 2), (0, 0)), mode="edge")
        pts = np.stack([np.convolve(padded[:, j], np.ones(k) / k, mode="valid") for j in range(2)], 1)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    targets = np.linspace(0, cum[-1], n)
    return [valid[int(np.argmin(np.abs(cum - t)))] for t in targets]


def composite(bg: np.ndarray, frames: list, alphas: list, opacities: list, later_on_top: bool = True) -> np.ndarray:
    out = bg.astype(np.float32)
    order = range(len(frames)) if later_on_top else reversed(range(len(frames)))
    for i in order:
        a = (alphas[i] * opacities[i])[..., None]
        out = out * (1 - a) + frames[i].astype(np.float32) * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def crop_box(masks: list, shape: tuple, margin: float, aspect: float | None = None) -> tuple[int, int, int, int]:
    """Union bbox of all masks, padded by margin × object height, optionally grown to an aspect ratio (w/h)."""
    h, w = shape[:2]
    union = np.zeros((h, w), np.uint8)
    for m in masks:
        union |= m
    ys, xs = np.nonzero(union)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    obj_h = max(cv2.boundingRect(m)[3] for m in masks)
    pad = margin * obj_h
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    if aspect:
        cw, ch = x1 - x0, y1 - y0
        if cw / ch < aspect:
            grow = (ch * aspect - cw) / 2
            x0, x1 = x0 - grow, x1 + grow
        else:
            grow = (cw / aspect - ch) / 2
            y0, y1 = y0 - grow, y1 + grow
    # Shift (rather than shrink) boxes that run off the image edge.
    cw, ch = min(x1 - x0, w), min(y1 - y0, h)
    x0 = min(max(x0, 0), w - cw)
    y0 = min(max(y0, 0), h - ch)
    return int(x0), int(y0), int(x0 + cw), int(y0 + ch)
