import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# 203 nits = BT.2408 HDR reference white, so diffuse white lands near SDR white.
HDR_TO_SDR = (
    "zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=mobius:param=0.5:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv"
)


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    is_hdr: bool


def probe(path: str | Path) -> VideoInfo:
    gen = imageio_ffmpeg.read_frames(str(path))
    meta = next(gen)
    gen.close()
    pix_fmt = meta.get("pix_fmt", "")
    w, h = meta["size"]
    return VideoInfo(
        path=Path(path),
        width=w,
        height=h,
        fps=meta["fps"],
        duration=meta["duration"],
        is_hdr="arib-std-b67" in pix_fmt or "smpte2084" in pix_fmt,
    )


def read_clip(info: VideoInfo, start: float, end: float, scale: float = 1.0, fps: float | None = None):
    """Yield (timestamp, BGR uint8 frame) for [start, end), tone-mapped to SDR if the source is HDR."""
    out_w = int(round(info.width * scale / 2)) * 2
    out_h = int(round(info.height * scale / 2)) * 2
    filters = []
    if fps:
        filters.append(f"fps={fps}")
    if info.is_hdr:
        filters.append(HDR_TO_SDR)
    filters.append(f"scale={out_w}:{out_h}:flags=area")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.4f}", "-t", f"{end - start:.4f}", "-i", str(info.path),
        "-an", "-vf", ",".join(filters),
        # iPhone clips are often variable-frame-rate with a 120 tbr time base; without passthrough
        # ffmpeg pads them to 120 fps with duplicate frames (4x the work, wrong timestamps).
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    step = 1.0 / (fps or info.fps)
    nbytes = out_w * out_h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finished = False
    try:
        k = 0
        while len(buf := proc.stdout.read(nbytes)) == nbytes:
            yield start + k * step, np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
            k += 1
        finished = True
    finally:
        if not finished:
            proc.kill()
        proc.stdout.close()
        err = proc.stderr.read().decode()
        proc.stderr.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed: {err}")
