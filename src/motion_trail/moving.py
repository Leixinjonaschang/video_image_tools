"""Motion trails from a moving (hand-held / tracking) camera.

Pipeline: SAM 2 tracks the robot -> frames are registered with a similarity transform fitted to the
ground around the robot's feet (keeps the robot upright and undistorted despite parallax) -> people
are removed from the chosen frames using a far-background-aligned median of the other frames ->
the chosen frames are stitched as feathered strips, one per robot instance, onto a wide canvas.
"""

import warnings

import cv2
import numpy as np

from .composite import to_lab


def torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DETECTOR = "IDEA-Research/grounding-dino-tiny"


def detect(frame: np.ndarray, text: str, threshold: float = 0.2) -> list[tuple[list[float], float]]:
    """Boxes (x0, y0, x1, y1) matching `text` with their scores, best first, from Grounding DINO."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    dev = torch_device()
    proc = AutoProcessor.from_pretrained(DETECTOR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).to(dev).eval()
    inputs = proc(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), text=[[text]], return_tensors="pt").to(dev)
    with torch.inference_mode():
        outputs = model(**inputs)
    found = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold, text_threshold=threshold, target_sizes=[frame.shape[:2]]
    )[0]
    h, w = frame.shape[:2]
    limits = (w, h, w, h)
    boxes = [([round(min(max(float(v), 0.0), lim), 1) for v, lim in zip(b, limits)], float(s))
             for b, s in zip(found["boxes"], found["scores"])]
    return sorted(boxes, key=lambda c: -c[1])


def motion_in_boxes(frames: list, boxes: list, fps: float) -> np.ndarray:
    """Mean colour change (ΔE) inside each box between the first frame and frames up to 2 s later,
    after aligning those frames to the first one: how much each box moves relative to the scene."""
    h, w = frames[0].shape[:2]
    sift = cv2.SIFT_create(3000)
    p0, d0 = _sift(sift, frames[0])
    lab0 = to_lab(frames[0])
    ones = np.ones((h, w), np.uint8)
    motion = np.zeros(len(boxes))
    for k in sorted({min(len(frames) - 1, round(s * fps)) for s in (0.3, 0.6, 1.0, 1.5, 2.0)} - {0}):
        pk, dk = _sift(sift, frames[k])
        hm = None
        if len(p0) >= 8 and len(pk) >= 8:
            good = [a for a, b in cv2.BFMatcher().knnMatch(dk, d0, k=2) if a.distance < 0.75 * b.distance]
            if len(good) >= 8:
                hm, _ = cv2.findHomography(pk[[a.queryIdx for a in good]], p0[[a.trainIdx for a in good]], cv2.RANSAC, 3.0)
        hm = np.eye(3) if hm is None else hm
        valid = cv2.warpPerspective(ones, hm, (w, h))
        diff = np.linalg.norm(to_lab(cv2.warpPerspective(frames[k], hm, (w, h))) - lab0, axis=2) * valid
        for c, (x0, y0, x1, y1) in enumerate(boxes):
            sl = (slice(int(y0), int(y1)), slice(int(x0), int(x1)))
            if (n := valid[sl].sum()) > 0:
                motion[c] = max(motion[c], diff[sl].sum() / n)
    return motion


def _sift(sift, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    kps, desc = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    return np.float32([k.pt for k in kps]).reshape(-1, 2), desc


def robot_box(frames: list, box, text: str, fps: float) -> list[float]:
    """The user's --box if given, otherwise the `text` detection in the first frame that best combines
    detector confidence with actually moving (so robots parked in the background are skipped)."""
    if box:
        return list(box)
    print(f'detecting "{text}" in the first frame with {DETECTOR}')
    cands = detect(frames[0], text)
    if not cands:
        raise SystemExit(f'no "{text}" found in the first frame of the clip; '
                         f'describe it differently with --detect, or give --box (see --preview)')
    pick = 0
    if len(cands) > 1:
        motion = motion_in_boxes(frames, [b for b, _ in cands], fps)
        if motion.max() >= 4:  # static boxes measure ~2 (noise); below 4 trust the detector alone
            pick = int(np.argmax([s * m for (_, s), m in zip(cands, motion)]))
    box, score = cands[pick]
    note = f", the moving one of {len(cands)} candidates" if pick else ""
    print(f"  found at {' '.join(f'{v:.0f}' for v in box)} (score {score:.2f}{note}); pass --box to override")
    return box


def segment_robot(frames: list, box: tuple, model_id: str) -> np.ndarray:
    """Per-frame robot probability maps (N, H, W) from SAM 2, prompted with a box on the first frame."""
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    dev = torch_device()
    # bfloat16 is ~2.5x faster than float32 on GPUs (measured on Apple M5) with near-identical masks.
    fast = dev == "mps" or (dev == "cuda" and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if fast else torch.float32
    model = Sam2VideoModel.from_pretrained(model_id).to(dev, dtype=dtype)
    proc = Sam2VideoProcessor.from_pretrained(model_id)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    sess = proc.init_video_session(video=rgb, inference_device=dev, video_storage_device="cpu", dtype=dtype)
    proc.add_inputs_to_inference_session(
        inference_session=sess, frame_idx=0, obj_ids=1, input_boxes=[[[float(v) for v in box]]]
    )
    probs = np.zeros((len(frames), *frames[0].shape[:2]), np.float32)
    size = [[sess.video_height, sess.video_width]]
    with torch.inference_mode():
        model(inference_session=sess, frame_idx=0)
        for out in model.propagate_in_video_iterator(sess, show_progress_bar=True):
            m = proc.post_process_masks([out.pred_masks.float()], original_sizes=size, binarize=False)[0]
            probs[out.frame_idx] = torch.sigmoid(m[0, 0]).cpu().numpy()
    return probs


def detect_people(frames: list, indices: list[int], robot: list) -> dict[int, np.ndarray]:
    """Dilated person masks (excluding the robot) for the given frames, from COCO Mask R-CNN."""
    import torch
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

    dev = torch_device()
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().to(dev)
    h, w = frames[0].shape[:2]
    grow = np.ones((25, 25), np.uint8)
    people = {}
    with torch.inference_mode():
        for i in indices:
            x = torch.from_numpy(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
            out = model([x.to(dev)])[0]
            keep = (out["labels"] == 1) & (out["scores"] > 0.5)
            m = np.zeros((h, w), np.uint8)
            if keep.any():
                m = (out["masks"][keep, 0] > 0.5).any(0).cpu().numpy().astype(np.uint8)
            people[i] = cv2.dilate(m & (1 - robot[i]), grow)
    return people


def _chain(steps: list) -> np.ndarray:
    """Compose per-step transforms (frame i -> frame i-1) into transforms frame i -> frame 0."""
    out = [np.eye(3)]
    for m in steps:
        out.append(out[-1] @ m)
    return np.stack(out)


def far_features(frames: list, robot: list) -> list:
    """SIFT keypoints/descriptors outside the robot; at night these are mostly distant lights."""
    sift = cv2.SIFT_create(4000)
    grow = np.ones((21, 21), np.uint8)
    feats = []
    for f, r in zip(frames, robot):
        kps, desc = sift.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (1 - cv2.dilate(r, grow)) * 255)
        feats.append((np.float32([k.pt for k in kps]), desc))
    return feats


def far_homography(src: tuple, dst: tuple, min_inliers: int = 20) -> np.ndarray | None:
    """Homography src -> dst on the dominant feature plane (the distant background), or None if unreliable."""
    (p1, d1), (p0, d0) = src, dst
    if d1 is None or d0 is None or len(p1) < min_inliers or len(p0) < min_inliers:
        return None
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d0, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < min_inliers:
        return None
    hm, inl = cv2.findHomography(p1[[a.queryIdx for a in good]], p0[[a.trainIdx for a in good]], cv2.RANSAC, 3.0)
    if hm is None or inl.sum() < max(min_inliers, 0.3 * len(good)):
        return None
    return hm


def ground_motion(frames: list, robot: list) -> np.ndarray:
    """A[i] maps frame i to frame 0 with a similarity transform fitted to the ground near the robot's feet."""
    clahe = cv2.createCLAHE(3.0, (8, 8))
    gray = [clahe.apply(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    h, w = gray[0].shape
    ys, xs = np.mgrid[4:h:8, 4:w:8]
    grid = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    grow = np.ones((31, 31), np.uint8)
    steps, last, band = [], np.eye(3), (0.5 * h, h)
    for i in range(1, len(frames)):
        if robot[i].any():
            _, y, _, bh = cv2.boundingRect(robot[i])
            band = (y + 0.5 * bh, y + 1.5 * bh)
        flow = dis.calc(gray[i], gray[i - 1], None)
        free = cv2.dilate(robot[i], grow)[grid[:, 1].astype(int), grid[:, 0].astype(int)] == 0
        keep = free & (grid[:, 1] > band[0]) & (grid[:, 1] < band[1])
        src = grid[keep]
        dst = src + flow[src[:, 1].astype(int), src[:, 0].astype(int)]
        if len(src) >= 20:  # otherwise assume the camera kept its previous motion
            m, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5)
            if m is not None and inl.sum() >= 20:
                last = np.vstack([m, [0, 0, 1]])
        steps.append(last)
    return _chain(steps)


def remove_people(frames: list, i: int, hole: np.ndarray, feats: list, robot: list, sources: int = 40) -> np.ndarray:
    """Fill `hole` in frame i with the median of other frames aligned on the distant background.

    Standing people sit at a different depth than the far background, so in far-aligned frames they
    drift and the median drops them, while the textured far background stays sharp."""
    h, w = hole.shape
    x, y, bw, bh = cv2.boundingRect(cv2.dilate(hole, np.ones((61, 61), np.uint8)))
    grow = np.ones((15, 15), np.uint8)
    ones = np.ones((h, w), np.uint8)
    stride = max(1, len(frames) // sources)
    stack = []
    for j in range(0, len(frames), stride):
        if abs(j - i) < stride or (hj := far_homography(feats[j], feats[i])) is None:
            continue
        m = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]]) @ hj
        warped = cv2.warpPerspective(frames[j], m, (bw, bh)).astype(np.float32)
        ok = cv2.warpPerspective(ones - cv2.dilate(robot[j], grow), m, (bw, bh), flags=cv2.INTER_NEAREST)
        ok &= cv2.warpPerspective(ones, m, (bw, bh), flags=cv2.INTER_NEAREST)
        warped[ok == 0] = np.nan
        stack.append(warped)
    out = frames[i].astype(np.float32)
    if not stack:
        return frames[i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels are handled below
        med = np.nanmedian(np.stack(stack), axis=0)
    patch = out[y:y + bh, x:x + bw]
    hp = hole[y:y + bh, x:x + bw].astype(bool)
    missing = np.isnan(med).any(2)
    ring = cv2.dilate(hp.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & ~hp & ~missing
    if ring.any():
        med = med * ((patch[ring].mean(0) + 1) / (med[ring].mean(0) + 1))
    med = np.where(missing[..., None], patch, med)
    a = np.maximum(cv2.GaussianBlur(hp.astype(np.float32), (0, 0), 5), hp)[..., None]
    out[y:y + bh, x:x + bw] = patch * (1 - a) + med * a
    leftover = (hp & missing).astype(np.uint8)
    result = np.clip(out, 0, 255).astype(np.uint8)
    if leftover.any():
        full = np.zeros((h, w), np.uint8)
        full[y:y + bh, x:x + bw] = leftover
        result = cv2.inpaint(result, full, 5, cv2.INPAINT_TELEA)
    return result


def stitch(frames: dict, A: np.ndarray, probs: np.ndarray, indices: list[int],
           later_on_top: bool = True, feather: float = 40.0):
    """Blend the chosen frames as feathered strips around each robot, then paste the robots on top.

    Returns (canvas, coverage mask, robot masks in canvas coordinates)."""
    h, w = probs.shape[1:]
    ref = indices[len(indices) // 2]
    A = np.linalg.inv(A[ref]) @ A
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    pts = np.hstack([(A[i] @ corners)[:2] for i in indices])
    x0, y0 = np.floor(pts.min(1))
    x1, y1 = np.ceil(pts.max(1))
    cw, ch = int(x1 - x0), int(y1 - y0)
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    warps = [(shift @ A[i])[:2] for i in indices]

    robot = [(probs[i] > 0.5).astype(np.uint8) for i in indices]
    feet = []
    for m, r in zip(warps, robot):
        x, y, bw, bh = cv2.boundingRect(r)
        feet.append(m @ [x + bw / 2, y + bh, 1])
    feet = np.array(feet)
    u = feet[-1] - feet[0]
    u = u / (np.linalg.norm(u) or 1)
    gy, gx = np.mgrid[0:ch, 0:cw]
    s = (gx * u[0] + gy * u[1]).astype(np.float32)
    dist = np.abs(s[None] - (feet @ u)[:, None, None].astype(np.float32))
    pref = np.exp(-(dist - dist.min(0, keepdims=True)) / feather)
    del dist

    acc = np.zeros((ch, cw, 3), np.float32)
    wsum = np.zeros((ch, cw), np.float32)
    coverage = np.zeros((ch, cw), np.uint8)
    layers = []
    for k, (i, m) in enumerate(zip(indices, warps)):
        img = cv2.warpAffine(frames[i], m, (cw, ch), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        valid = cv2.erode(cv2.warpAffine(np.ones((h, w), np.uint8), m, (cw, ch)), np.ones((5, 5), np.uint8))
        alpha = cv2.GaussianBlur(cv2.warpAffine(probs[i], m, (cw, ch)), (0, 0), 0.7)
        wk = pref[k] * valid
        acc += img * wk[..., None]
        wsum += wk
        coverage |= valid
        layers.append((img, alpha[..., None]))
    bg = acc / np.maximum(wsum, 1e-12)[..., None]
    # Each robot already sits in its own strip; re-pasting fixes the spots where a neighbour's strip covers it.
    order = range(len(layers)) if later_on_top else reversed(range(len(layers)))
    for k in order:
        img, a = layers[k]
        bg = bg * (1 - a) + img * a
    masks = [(a[..., 0] > 0.5).astype(np.uint8) for _, a in layers]
    return np.clip(bg + 0.5, 0, 255).astype(np.uint8), coverage, masks


def fit_inside(box: tuple, coverage: np.ndarray) -> tuple:
    """Shrink a crop box edge by edge until its border has no uncovered (black) canvas pixels.

    The covered area is a union of overlapping warped frames, so a clean border means a clean box."""
    x0, y0, x1, y1 = box
    bad = coverage == 0
    while x1 - x0 > 1 and y1 - y0 > 1:
        edges = [bad[y0, x0:x1].mean(), bad[y1 - 1, x0:x1].mean(), bad[y0:y1, x0].mean(), bad[y0:y1, x1 - 1].mean()]
        if max(edges) == 0:
            break
        e = int(np.argmax(edges))
        if e == 0:
            y0 += 1
        elif e == 1:
            y1 -= 1
        elif e == 2:
            x0 += 1
        else:
            x1 -= 1
    return x0, y0, x1, y1
