# Roto Local

Roto Local is a local-first macOS video rotoscoping application. The current build
uses SAM 2.1 Hiera Base+ for point-prompted masks and offline video propagation.
Video stays on the selected source path. By default, models, projects, frame
windows, logs, cache, temporary files, and PNG exports stay inside this
application folder; video exports can be saved to any writable location.

## Requirements

- macOS on Apple Silicon; tested here on an M4 with 16 GiB unified memory.
- Python 3.12.
- FFmpeg and ffprobe available on `PATH`.
- About 2 GiB free on the volume containing this folder for the current venv and
  Base+ checkpoint; projects and exports require additional space.
- `models/sam2.1_hiera_base_plus.pt` (323,606,802 bytes), downloaded from the
  [official Meta checkpoint URL](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt).
  Its SHA-256 is
  `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`.

The SAM 2 repository and the Hugging Face model card identify Apache-2.0
licensing. Preserve the relevant license and notices if redistributing code or
weights. See the [SAM 2 repository](https://github.com/facebookresearch/sam2)
and [official Base+ model card](https://huggingface.co/facebook/sam2.1-hiera-base-plus).

## Start

Create the virtual environment on any writable volume with enough free space.

```sh
python3.12 -m venv .venv
mkdir -p runtime/tmp runtime/cache/pip
TMPDIR="$PWD/runtime/tmp" \
PIP_CACHE_DIR="$PWD/runtime/cache/pip" \
.venv/bin/python -m pip install -e .
```

Set the model file at `models/sam2.1_hiera_base_plus.pt`, then launch:

```sh
.venv/bin/python run.py
```

`run.py` redirects Python temp files, pip/PyTorch/Hugging Face caches, logs and
bytecode caches into `runtime/` before importing Qt or PyTorch. PyTorch's
`PYTORCH_ENABLE_MPS_FALLBACK=1` is enabled because the tested PyTorch MPS backend
does not implement SAM 2's bicubic upsample operator; that operation falls back
to CPU and can reduce throughput. This does not send video or masks to a cloud
service.

## Workflow

1. Open or drag a video into the viewer. While dragging the timeline, Roto Local
   displays downscaled previews from a bounded in-memory frame cache; on release
   it loads the selected frame at full resolution for annotation.
2. Select a subject in the subject list. Add more with **+ Soggetto** or
   remove the selected subject with **−**. Each subject has its own point
   prompts, brush corrections, mask, and tracking pass. Positive points and
   mask overlays use a distinct color per subject; negative points remain red.
3. Choose a positive point to include the selected subject or a negative point
   to exclude an area. Each point refines only that subject's current-frame mask.
   Use the add/subtract brush to paint corrections on the selected subject. AI
   updates do not erase these manual strokes; the brush preview uses that
   subject's color. Undo and redo are available for point annotations and brush
   strokes with the buttons or standard macOS shortcuts (Cmd+Z / Cmd+Shift+Z).
4. Set the tracking range and select any frame inside it. Add at least one
   positive point for every subject on that frame, choose forward, backward,
   or both directions, choose mask resolution, then start tracking. Subjects
   are propagated independently and their masks are combined for export.
   Original resolution keeps full mask detail; 1080p reduces mask memory and
   storage for larger videos. SAM 2 runs in a worker thread; progress,
   processed-frame count, approximate remaining time, and cancellation are
   shown in the window. The viewer follows the most recently processed frame
   and displays each saved mask in its subject color.
5. Save the project or export PNG sequences, a black-and-white mask MP4, or an
   isolated-subject MOV with ProRes 4444 transparency.

Projects are autosaved as annotations or masks change under `projects/`.
Exports are written beneath `exports/`. Source videos are referenced in place
and are not copied into the application directory.

## Memory and storage

The upstream SAM 2 video predictor loads every frame passed to `init_state` as
float32 at 1024×1024, even when video frames are offloaded to CPU. Roto Local
therefore sends bounded windows and carries a boundary mask into the next
window. The default is 12 frames to limit memory with the larger Base+ model;
12/24/48-frame windows remain selectable. The previous 1.5 GiB peak-RSS
measurement was made with the Small model and must not be treated as a Base+
benchmark. Actual memory depends on model state, frame size, and other
applications. Tracking states and output masks are released between windows.
The decoded-frame cache is capped at 48 MiB for full-resolution frames and
24 MiB for previews; previews decode at most six frames with a 960-pixel
maximum dimension.

The optional 1080p mask mode caps the longer mask edge at 1920 pixels. SAM 2
still uses its normal 1024-pixel model input; the smaller setting reduces the
resolution of tracked, stored, and edited masks. Export resizes masks back to
the video's original dimensions using nearest-neighbor interpolation. This
restores the output dimensions, not fine details lost by downsampling; use
original resolution when edges such as hair need maximum detail.

On a 12-frame synthetic 4K Base+ test, process peak RSS measured 1311.8 MiB
with original-size masks and 1283.5 MiB in 1080p mode; tracking took 13.39 s
and 13.36 s, respectively. The model remains the largest memory cost, so
1080p mode primarily reduces mask memory and storage, not inference time. These
figures are not a benchmark for real footage.

PNG sequences and video exports let you choose any writable destination; the
default location is `exports/`. Mask frames are read from their PNG sequence
directly by FFmpeg. B/W MP4 export does not decode the source video;
transparent MOV export lets FFmpeg decode and composite the source, avoiding an
unnecessary RGB round-trip through Python. Manual brush corrections are
composited into temporary masks under the selected project folder. The B/W mask
is exported as H.264 MP4; the transparent subject is exported as ProRes 4444
MOV because standard MP4/H.264 does not preserve alpha.

The image and video predictors share the same Base+ model instance. During
tracking, per-frame masks are still saved immediately, while the project
manifest is written once when the run ends or is cancelled. The live viewer is
refreshed at most about three times per second to avoid competing with
inference; mask output remains per-frame.

The launcher redirects temporary and application cache paths to the application
folder, wherever that folder is located. macOS can still manage its own swap,
logs, and Metal/MPS caches; the application cannot guarantee zero OS-managed
writes. Disk-full errors are surfaced.

## GitHub copy

The repository intentionally excludes the virtual environment, model
checkpoints, runtime caches and logs, local projects, and exports. Keep video
and project data local; download the Base+ checkpoint separately as described
above. Exported video files can be written to any writable disk, including
locations outside the application folder.

## Limits

- SAM 2's upstream installation guidance targets Linux/CUDA, not officially
  supported macOS/MPS. This build was smoke-tested on the stated M4 machine, but
  MPS support is not guaranteed for every operator or project.
- The test clip was synthetic and short. Real footage, occlusions, fast motion,
  variable-frame-rate accuracy, and long-session memory still need project-level
  validation and manual review.
- Propagation is a quality-assistance tool, not perfect tracking. Inspect
  boundaries and correct lost or drifting subjects.
- Subjects are tracked independently and combined for export. The app does not
  yet export separate video files per subject or integrate directly with
  Resolve, Blender, or Nuke timelines. RGBA PNG sequences are suitable for
  compositing and retain transparency.
- Frame preview seeks with FFmpeg per requested frame. It keeps the GUI
  responsive but is not a real-time playback guarantee for every codec.

## Troubleshooting

- **Model not found:** verify the Base+ checkpoint path and SHA-256 above.
- **FFmpeg not found:** install FFmpeg on the external volume if desired and
  ensure its `bin` directory is on `PATH`.
- **MPS operator error:** keep the launcher enabled so PyTorch can fall back
  unsupported operations to CPU; inspect `runtime/logs/app.log`.
- **Tracking or memory issue:** stop propagation, save the project, correct a
  later keyframe, and rerun only the required interval. No quality reduction is
  applied silently.
- **Disk usage:** inspect `runtime/`, `projects/`, and `exports/`; these are the
  only locations Roto Local manages. Remove only caches or projects you no
  longer need.

## Tests

Run the focused standard-library tests with temporary files on the external
volume:

```sh
TMPDIR="$PWD/runtime/tmp" \
PYTHONPYCACHEPREFIX="$PWD/runtime/cache/pycache" \
PYTHONPATH=src \
.venv/bin/python -m unittest discover -s tests -v
```
