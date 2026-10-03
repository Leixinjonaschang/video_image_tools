"""Turn a video clip of a moving robot into a single multi-exposure ("motion trail") image.

Example:
    uv run motion-trail video.mov --start 1:39 --end 1:45
    uv run motion-trail video.mov --start 1:39 --end 1:45 -n 8 --opacity 0.35 1 -o outputs/fade.png
    uv run motion-trail video.mov --start 4 --end 7 --moving-camera
    uv run motion-trail video.mov --start 4 --end 7 --moving-camera --box 670 250 820 540
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .composite import centroid, color_distance, composite, crop_box, object_mask, select_indices, soft_alpha, to_lab
from .video import probe, read_clip

ANALYSIS_SCALE = 0.25
BG_SAMPLES = 60


def parse_time(s: str) -> float:
    """Accept seconds (99.5) or m:ss(.f) (1:39.5)."""
    parts = s.split(":")
    return sum(float(p) * 60**i for i, p in enumerate(reversed(parts)))


def median_background(info, start: float, end: float, scale: float = 1.0) -> np.ndarray:
    fps = BG_SAMPLES / (end - start)
    frames = np.stack([f for _, f in read_clip(info, start, end, scale=scale, fps=fps)])
    return np.median(frames, axis=0).astype(np.uint8)


def track(info, start: float, end: float, bg_range: tuple[float, float], hi: float) -> list:
    """Object centroid per frame of the clip, from a fast low-resolution pass."""
    bg_lab = to_lab(median_background(info, *bg_range, scale=ANALYSIS_SCALE))
    return [centroid(object_mask(color_distance(f, bg_lab), hi)) for _, f in read_clip(info, start, end, scale=ANALYSIS_SCALE)]


def overlay_debug(frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    tint = np.zeros_like(frame)
    tint[..., 2] = 255
    a = (alpha * 0.6)[..., None]
    return (frame * (1 - a) + tint * a).astype(np.uint8)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--start", type=parse_time, required=True, help="clip start, seconds or m:ss")
    p.add_argument("--end", type=parse_time, required=True, help="clip end, seconds or m:ss")
    p.add_argument("-n", "--num", type=int, default=6, help="number of robot instances (default 6)")
    p.add_argument("--spacing", choices=["distance", "time"], default="distance",
                   help="space instances evenly along the path travelled (default) or evenly in time")
    p.add_argument("--times", type=parse_time, nargs="+", help="explicit timestamps to use instead of --num/--spacing")
    p.add_argument("--order", choices=["later", "earlier"], default="later", help="which instance is drawn on top where they overlap")
    p.add_argument("--opacity", type=float, nargs=2, default=(1.0, 1.0), metavar=("FIRST", "LAST"),
                   help="opacity of the first and last instance, interpolated in between (e.g. 0.4 1.0 to fade in)")
    p.add_argument("--thresh", type=float, nargs=2, default=(5.0, 14.0), metavar=("LO", "HI"),
                   help="colour-difference (ΔE) thresholds: below LO is background, above HI is solid object")
    p.add_argument("--bg-range", type=parse_time, nargs=2, metavar=("START", "END"),
                   help="time range to estimate the empty background from (default: the clip itself); "
                        "pick a range where the robot never sits still")
    p.add_argument("--margin", type=float, default=0.25, help="crop padding as a fraction of robot height")
    p.add_argument("--aspect", type=float, help="force the crop to this width/height ratio")
    p.add_argument("--no-crop", action="store_true", help="keep the full frame")
    p.add_argument("-o", "--out", type=Path, help="output image (default: outputs/<video>_<start>-<end>.png)")
    p.add_argument("--debug", action="store_true", help="also save per-instance mask overlays")
    m = p.add_argument_group("moving camera (needs `uv sync --extra sam`)")
    m.add_argument("--moving-camera", action="store_true",
                   help="for hand-held / tracking shots: segment with SAM 2, register frames, stitch a panorama")
    m.add_argument("--detect", default="robot", metavar="TEXT",
                   help='what to look for in the first frame when --box is not given (default "robot")')
    m.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="robot bounding box in the first frame of the clip, in video pixels; overrides --detect")
    m.add_argument("--keep-people", action="store_true", help="don't remove people (e.g. operators) from the background")
    m.add_argument("--fps", type=float, help="analyse the clip at this frame rate (default: source rate; lower is faster)")
    m.add_argument("--sam-model", default="facebook/sam2.1-hiera-small", help="Hugging Face SAM 2 video checkpoint")
    m.add_argument("--no-cache", action="store_true",
                   help="recompute robot detection and tracking instead of reusing the cached result for this clip")
    p.add_argument("--preview", action="store_true",
                   help="save the clip's first frame with a pixel grid (to read off a manual --box) and exit")
    args = p.parse_args(argv)

    info = probe(args.video)
    start, end = args.start, args.end
    if not 0 <= start < end <= info.duration:
        p.error(f"need 0 <= start < end <= {info.duration:.2f}s")
    if args.times and not all(start <= t < end for t in args.times):
        p.error("all --times must lie within [--start, --end)")
    out = args.out or Path("outputs") / f"{args.video.stem}_{start:g}-{end:g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.video.name}: {info.width}x{info.height} @ {info.fps:g} fps{', HDR -> SDR' if info.is_hdr else ''}")

    if args.preview:
        _, first = next(read_clip(info, start, end))
        path = out.with_name(out.stem + "_preview.png")
        cv2.imwrite(str(path), draw_grid(first))
        print(f"saved {path}")
    elif args.moving_camera:
        if tuple(args.opacity) != (1.0, 1.0):
            p.error("--opacity is not supported with --moving-camera: the background behind each robot is unknown")
        run_moving(args, info, out)
    else:
        run_static(args, info, out)


def draw_grid(img: np.ndarray, step: int = 100) -> np.ndarray:
    img = img.copy()
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(img, str(x), (x + 3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    for y in range(step, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return img


def save(result: np.ndarray, masks: list, args, out: Path, coverage: np.ndarray | None = None) -> None:
    if not args.no_crop:
        box = crop_box(masks, result.shape, args.margin, args.aspect)
        if coverage is not None:
            from .moving import fit_inside

            box = fit_inside(box, coverage)
        cv2.imwrite(str(out.with_name(out.stem + "_full" + out.suffix)), result)
        x0, y0, x1, y1 = box
        result = result[y0:y1, x0:x1]
    cv2.imwrite(str(out), result)
    print(f"saved {out} ({result.shape[1]}x{result.shape[0]})")


def opacity_ramp(args, n: int) -> np.ndarray:
    return np.linspace(*args.opacity, n) if n > 1 else np.array([args.opacity[1]])


def run_moving(args, info, out: Path) -> None:
    from . import moving

    fps = args.fps or info.fps
    frames = [f.copy() for _, f in read_clip(info, args.start, args.end, fps=args.fps)]
    probs, box = moving.track_robot(frames, args.video, args.start, args.end, fps, args.box, args.detect,
                                    args.sam_model, cache=not args.no_cache)
    robot = [(p > 0.5).astype(np.uint8) for p in probs]
    print("registering frames")
    ground = moving.ground_motion(frames, robot)

    if args.times:
        indices = sorted({min(round((t - args.start) * fps), len(frames) - 1) for t in args.times})
    else:
        feet = []
        for a, r in zip(ground, robot):
            if not r.any():
                feet.append(None)
                continue
            x, y, w, h = cv2.boundingRect(r)
            feet.append((a @ [x + w / 2, y + h, 1])[:2])
        indices = select_indices(feet, args.num, args.spacing)
    print("using frames at t =", ", ".join(f"{args.start + i / fps:.2f}s" for i in indices))

    clean = {i: frames[i] for i in indices}
    if not args.keep_people:
        print("removing people")
        people = moving.detect_people(frames, indices, robot)
        feats = moving.far_features(frames, robot)
        for i in indices:
            if people[i].any():
                clean[i] = moving.remove_people(frames, i, people[i], feats, robot)

    result, coverage, masks = moving.stitch(clean, ground, probs, indices, later_on_top=args.order == "later")
    save(result, masks, args, out, coverage)

    if args.debug:
        dbg_dir = out.with_name(out.stem + "_debug")
        dbg_dir.mkdir(exist_ok=True)
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.imwrite(str(dbg_dir / "box.jpg"), cv2.rectangle(frames[0].copy(), (x0, y0), (x1, y1), (0, 0, 255), 3))
        for k, i in enumerate(indices):
            cv2.imwrite(str(dbg_dir / f"mask_{k:02d}.jpg"), overlay_debug(clean[i], probs[i]))
        print(f"debug images in {dbg_dir}")


def run_static(args, info, out: Path) -> None:
    start, end = args.start, args.end
    bg_range = tuple(args.bg_range) if args.bg_range else (start, end)
    lo, hi = args.thresh
    if args.times:
        indices = sorted({round((t - start) * info.fps) for t in args.times})
    else:
        centroids = track(info, start, end, bg_range, hi)
        indices = select_indices(centroids, args.num, args.spacing)
    print("using frames at t =", ", ".join(f"{start + i / info.fps:.2f}s" for i in indices))

    bg = median_background(info, *bg_range)
    bg_lab = to_lab(bg)
    wanted = set(indices)
    frames, masks, alphas = [], [], []
    for k, (_, f) in enumerate(read_clip(info, start, end)):
        if k not in wanted:
            continue
        dist = color_distance(f, bg_lab)
        mask = object_mask(dist, hi)
        if mask is None:
            print(f"  warning: no object found at t={start + k / info.fps:.2f}s, skipping")
            continue
        frames.append(f.copy())
        masks.append(mask)
        alphas.append(soft_alpha(dist, mask, lo, hi))
    if not frames:
        raise SystemExit("no object found in any selected frame; try lowering --thresh")

    result = composite(bg, frames, alphas, opacity_ramp(args, len(frames)), later_on_top=args.order == "later")
    save(result, masks, args, out)

    if args.debug:
        dbg_dir = out.with_name(out.stem + "_debug")
        dbg_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(dbg_dir / "background.png"), bg)
        for i, (f, a) in enumerate(zip(frames, alphas)):
            cv2.imwrite(str(dbg_dir / f"mask_{i:02d}.jpg"), overlay_debug(f, a))
        print(f"debug images in {dbg_dir}")
