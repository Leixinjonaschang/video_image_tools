"""Tile N keyframes of a video clip side by side into one figure.

Examples:
    uv run keyframes video.mov --start 4 --end 7 -n 4
    uv run keyframes video.mov --start 4 --end 7 -n 4 --crop 380 0 1000 720
    uv run keyframes video.mov --start 4 --end 7 -n 4 --follow
    uv run keyframes video.mov --start 4 --end 7 -n 4 --follow --box 670 250 820 540
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from .cli import draw_grid, parse_time
from .video import probe, read_clip


def small_gray(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def camera_motion(gray: list, k: int) -> float:
    """Global image shift per frame around frame k, a proxy for camera-shake blur.

    Sharpness scores like Laplacian variance get fooled by sensor noise in dim footage; how fast
    the whole picture is moving is a more reliable sign of blur."""
    a, b = max(k - 1, 0), min(k + 1, len(gray) - 1)
    if a == b:
        return 0.0
    (dx, dy), _ = cv2.phaseCorrelate(gray[a], gray[b], cv2.createHanningWindow(gray[a].shape[::-1], cv2.CV_32F))
    return float(np.hypot(dx, dy)) / (b - a)


def cut(img: np.ndarray, box) -> np.ndarray:
    if box is None:
        return img
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return img[y0:y1, x0:x1]


def follow_boxes(boxes: list, frame_shape: tuple, margin: float, aspect: float | None) -> tuple[list, list]:
    """Equal-size crops centred on each robot box, leaving margin × (largest robot height) on every side.

    Returns the crops and, per crop, the smallest gap actually left between robot and crop edge
    (smaller than requested when the robot is near the edge of the video frame)."""
    fh, fw = frame_shape[:2]
    bh = max(b[3] for b in boxes)
    pad = margin * bh
    cw = max(b[2] for b in boxes) + 2 * pad
    ch = bh + 2 * pad
    if aspect:
        cw, ch = max(cw, ch * aspect), max(ch, cw / aspect)
    cw, ch = int(min(cw, fw)), int(min(ch, fh))
    crops, gaps = [], []
    for x, y, w, h in boxes:
        x0 = min(max(round(x + w / 2 - cw / 2), 0), fw - cw)
        y0 = min(max(round(y + h / 2 - ch / 2), 0), fh - ch)
        crops.append((x0, y0, x0 + cw, y0 + ch))
        gaps.append(min(x - x0, y - y0, x0 + cw - (x + w), y0 + ch - (y + h)))
    return crops, gaps


def tile(panels: list, cols: int, gap: int) -> np.ndarray:
    """Lay panels out in a grid with white gaps; panels are scaled to a common height."""
    h = max(p.shape[0] for p in panels)
    panels = [p if p.shape[0] == h else cv2.resize(p, (round(p.shape[1] * h / p.shape[0]), h), interpolation=cv2.INTER_AREA)
              for p in panels]
    white = 255
    rows = []
    for r in range(math.ceil(len(panels) / cols)):
        row = panels[r * cols:(r + 1) * cols]
        parts = []
        for k, p in enumerate(row):
            if k:
                parts.append(np.full((h, gap, 3), white, np.uint8))
            parts.append(p)
        rows.append(np.hstack(parts))
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((h, width - r.shape[1], 3), white, np.uint8)]) if r.shape[1] < width else r for r in rows]
    out = []
    for k, r in enumerate(rows):
        if k:
            out.append(np.full((gap, width, 3), white, np.uint8))
        out.append(r)
    return np.vstack(out)


def pick_static(info, targets: list, snap: float, crop, clip: tuple[float, float]) -> list:
    """For each target time, the steadiest frame within ±snap seconds, kept inside the clip."""
    step = 1 / info.fps
    picked = []
    for t in targets:
        lo, hi = max(clip[0], t - snap), min(clip[1] - 0.5 * step, t + snap + 0.5 * step)
        # one extra frame on each side so the candidates' motion can be measured
        frames = [(ts, f.copy()) for ts, f in read_clip(info, max(0.0, lo - step), min(info.duration, hi + step))]
        cands = [k for k, (ts, _) in enumerate(frames) if lo - 1e-3 <= ts <= hi]
        if not cands:
            raise SystemExit(f"could not decode a frame at t={t:.2f}s")
        gray = [small_gray(f) for _, f in frames]
        ts, f = frames[min(cands, key=lambda k: camera_motion(gray, k))]
        picked.append((ts, cut(f, crop).copy()))
    return picked


def pick_following(args, info, targets: list) -> list:
    """Track the robot with SAM 2 and crop every keyframe around it at a constant size."""
    from .moving import robot_box, segment_robot

    fps = args.fps or info.fps
    frames = [f.copy() for _, f in read_clip(info, args.start, args.end, fps=args.fps)]
    box = robot_box(frames, args.box, args.detect, fps)
    print(f"tracking the robot in {len(frames)} frames with {args.sam_model}")
    robot = [(p > 0.5).astype(np.uint8) for p in segment_robot(frames, box, args.sam_model)]
    gray = [small_gray(f) for f in frames]
    k = round(args.snap * fps)
    chosen = []
    for t in targets:
        c = min(round((t - args.start) * fps), len(frames) - 1)
        cands = [i for i in range(max(0, c - k), min(len(frames), c + k + 1)) if robot[i].any()]
        if not cands:
            raise SystemExit(f"robot lost near t={t:.2f}s; try a tighter --box or a different --detect text")
        chosen.append(min(cands, key=lambda i: camera_motion(gray, i)))
    boxes = [cv2.boundingRect(robot[i]) for i in chosen]
    crops, gaps = follow_boxes(boxes, frames[0].shape, args.margin, args.aspect)
    wanted = args.margin * max(b[3] for b in boxes)
    for i, g in zip(chosen, gaps):
        if g < 0.5 * wanted:
            print(f"  note: at t={args.start + i / fps:.2f}s the robot is {max(g, 0):.0f}px from the panel edge "
                  f"(wanted {wanted:.0f}px) because it is near the edge of the video frame")
    return [(args.start + i / fps, cut(frames[i], b).copy()) for i, b in zip(chosen, crops)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--start", type=parse_time, required=True, help="clip start, seconds or m:ss")
    p.add_argument("--end", type=parse_time, required=True, help="clip end, seconds or m:ss")
    p.add_argument("-n", "--num", type=int, default=4, help="number of keyframes, evenly spaced in time (default 4)")
    p.add_argument("--times", type=parse_time, nargs="+", help="explicit timestamps instead of --num")
    p.add_argument("--snap", type=float, default=0.1,
                   help="pick the least motion-blurred frame within ±SNAP seconds of each target (default 0.1, 0 = exact)")
    p.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="crop every frame to this box, in video pixels (see --preview)")
    p.add_argument("--cols", type=int, help="panels per row (default: all in one row)")
    p.add_argument("--gap", type=int, default=6, help="white gap between panels in pixels (default 6)")
    p.add_argument("-o", "--out", type=Path, help="output image (default: outputs/<video>_<start>-<end>_keyframes.png)")
    p.add_argument("--preview", action="store_true",
                   help="save the clip's first frame with a pixel grid (to read off --crop / --box) and exit")
    f = p.add_argument_group("follow the robot (needs `uv sync --extra sam`)")
    f.add_argument("--follow", action="store_true", help="track the robot with SAM 2 and centre each panel on it")
    f.add_argument("--detect", default="robot", metavar="TEXT",
                   help='what to look for in the first frame when --box is not given (default "robot")')
    f.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="robot bounding box in the first frame of the clip, in video pixels; overrides --detect")
    f.add_argument("--margin", type=float, default=0.25,
                   help="space left around the robot on every side, as a fraction of its height (default 0.25)")
    f.add_argument("--aspect", type=float, help="panel width/height ratio (default: fit the robot)")
    f.add_argument("--fps", type=float, help="track at this frame rate (default: source rate; lower is faster)")
    f.add_argument("--sam-model", default="facebook/sam2.1-hiera-small", help="Hugging Face SAM 2 video checkpoint")
    args = p.parse_args(argv)

    info = probe(args.video)
    start, end = args.start, args.end
    if not 0 <= start < end <= info.duration:
        p.error(f"need 0 <= start < end <= {info.duration:.2f}s")
    if args.times and not all(start <= t < end for t in args.times):
        p.error("all --times must lie within [--start, --end)")
    if args.follow and args.crop:
        p.error("use either --crop or --follow, not both")
    out = args.out or Path("outputs") / f"{args.video.stem}_{start:g}-{end:g}_keyframes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.video.name}: {info.width}x{info.height} @ {info.fps:g} fps{', HDR -> SDR' if info.is_hdr else ''}")

    if args.preview:
        _, first = next(read_clip(info, start, end))
        path = out.with_name(out.stem + "_preview.png")
        cv2.imwrite(str(path), draw_grid(first))
        print(f"saved {path}")
        return

    last = end - 1 / info.fps
    targets = sorted(args.times) if args.times else list(np.linspace(start, last, args.num)) if args.num > 1 else [start]
    if args.follow:
        picked = pick_following(args, info, targets)
    else:
        picked = pick_static(info, targets, args.snap, args.crop, (start, end))
    print("using frames at t =", ", ".join(f"{t:.2f}s" for t, _ in picked))

    result = tile([f for _, f in picked], args.cols or len(picked), args.gap)
    cv2.imwrite(str(out), result)
    print(f"saved {out} ({result.shape[1]}x{result.shape[0]})")
