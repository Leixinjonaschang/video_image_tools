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

This default mode assumes a **static camera** and **one moving object** in the clip. For hand-held or tracking shots, see [Moving camera](#moving-camera).

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
| `--preview` | off | Save the clip's first frame with a pixel grid and exit (to read off `--box`) |

### Moving camera

For footage where the camera follows the robot (hand-held, walking alongside), add `--moving-camera`:

1. [SAM 2](https://huggingface.co/facebook/sam2.1-hiera-small) tracks the robot from a box you draw on the first frame.
2. Frames are registered with a similarity transform fitted to the ground around the robot's feet, so robots stay upright and undistorted despite parallax.
3. People (e.g. operators) are detected with Mask R-CNN and removed by filling in the background from other frames.
4. The chosen frames are stitched into a wide panorama, one feathered strip per robot instance.

Needs the optional ML dependencies (PyTorch, transformers). Model weights (~0.4 GB) download on first use; runs on CUDA, Apple MPS or CPU.

```bash
uv sync --extra sam

# 1. find the robot's box (x0 y0 x1 y1) in the first frame of the clip
uv run motion-trail path/to/video.mov --start 4 --end 7 --preview

# 2. make the figure
uv run motion-trail path/to/video.mov --start 4 --end 7 --moving-camera --box 670 250 820 540
```

| Option | Default | Description |
| --- | --- | --- |
| `--moving-camera` | off | Enable the moving-camera pipeline |
| `--box X0 Y0 X1 Y1` | required | Robot bounding box in the clip's first frame, in video pixels |
| `--keep-people` | off | Don't remove people from the background |
| `--fps` | source rate | Analyse the clip at a lower frame rate; `15` is about 2× faster |
| `--sam-model` | `facebook/sam2.1-hiera-small` | SAM 2 video checkpoint on Hugging Face |

`--num`, `--spacing`, `--times`, `--order`, `--margin`, `--aspect`, `--no-crop`, `-o` and `--debug` work as above; `--opacity`, `--thresh` and `--bg-range` don't apply.

### Tips

- **Ghost of the robot in the background**: the robot stood still for most of the clip. Pass a `--bg-range` where it keeps moving or is out of view.
- **Parts of the robot missing**: lower `--thresh` (e.g. `4 10`). **Background speckles pasted in**: raise it. Check with `--debug`.
- **Instances overlap too much**: reduce `-n` or use a longer clip.
- **Moving camera, robot mask wrong**: check `--debug` overlays; tighten `--box`, or try `--sam-model facebook/sam2.1-hiera-large`.
- **Moving camera, ghosts of people left over**: use the full frame rate (drop `--fps`) so more frames are available to fill from.
