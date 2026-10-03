# video_image_tools

Tools for turning robot videos into paper figures.

## motion-trail

Composes several poses of a moving robot from a video clip into one still image (a "multi-exposure" / motion-trail figure).

![motion-trail example](assets/motion_trail_example.png)

<sub>A wheel-legged robot crossing a step, 6 instances from a 6-second iPhone HDR clip (`--start 1:39 --end 1:45`).</sub>

How it works:

1. Decodes the clip with ffmpeg; iPhone HDR (HLG/PQ) footage is tone-mapped to SDR automatically.
2. Estimates the empty background as the per-pixel median of the clip.
3. Segments the robot in each frame by colour difference to the background, keeping thin parts and soft contact shadows.
4. Picks frames evenly spaced along the robot's path, pastes them onto the background, and crops to the motion.

Assumes a **static camera** and **one moving object** in the clip.

### Install

Requires [uv](https://docs.astral.sh/uv/). ffmpeg is bundled via `imageio-ffmpeg`, so no system install is needed.

```bash
git clone https://github.com/Leixinjonaschang/video_image_tools.git
cd video_image_tools
uv sync
```

### Usage

```bash
# 6 instances from 1:39 to 1:45
uv run motion-trail path/to/video.mov --start 1:39 --end 1:45

# 8 instances, earlier poses fading in, custom output path
uv run motion-trail path/to/video.mov --start 99 --end 105 -n 8 --opacity 0.35 1 -o fig.png

# hand-pick the exact moments
uv run motion-trail path/to/video.mov --start 99 --end 105 --times 99.0 100.8 102.1 103.4 104.9
```

Outputs `outputs/<video>_<start>-<end>.png` (cropped) and `..._full.png` (full frame).

### Options

| Option | Default | Description |
| --- | --- | --- |
| `video` | – | Input video file |
| `--start`, `--end` | required | Clip range, in seconds (`99.5`) or `m:ss` (`1:39.5`) |
| `-n`, `--num` | `6` | Number of robot instances |
| `--spacing {distance,time}` | `distance` | Space instances evenly along the path travelled, or evenly in time |
| `--times T [T ...]` | – | Explicit timestamps; overrides `--num` / `--spacing` |
| `--order {later,earlier}` | `later` | Which instance is drawn on top where they overlap |
| `--opacity FIRST LAST` | `1 1` | Opacity of the first and last instance, interpolated in between |
| `--thresh LO HI` | `5 14` | Colour-difference (ΔE) thresholds: below `LO` is background, above `HI` is solid robot |
| `--bg-range START END` | the clip | Time range used to estimate the empty background |
| `--margin` | `0.25` | Crop padding as a fraction of robot height |
| `--aspect` | – | Force the crop to this width/height ratio (e.g. `3`) |
| `--no-crop` | off | Keep the full frame |
| `-o`, `--out` | `outputs/...png` | Output image path |
| `--debug` | off | Also save the background and per-instance mask overlays |

### Tips

- **Ghost of the robot in the background**: the robot stood still for most of the clip. Pass a `--bg-range` where it keeps moving or is out of view.
- **Parts of the robot missing**: lower `--thresh` (e.g. `4 10`). **Background speckles pasted in**: raise it. Check with `--debug`.
- **Instances overlap too much**: reduce `-n` or use a longer clip.
