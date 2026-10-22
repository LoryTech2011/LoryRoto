from pathlib import Path
import os
import subprocess
import tempfile
from typing import Callable

import numpy as np
from PIL import Image

from .media import decode_frame
from .project import ProjectError, ProjectStore


def export_sequence(
    project: ProjectStore,
    first_frame: int,
    last_frame: int,
    destination: str | Path,
    rgba: bool,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    if first_frame < 0 or last_frame >= project.video_info.frame_count or first_frame > last_frame:
        raise ProjectError("Invalid export range.")
    output = Path(destination).resolve()
    missing = [
        frame_index
        for frame_index in range(first_frame, last_frame + 1)
        if not project.is_processed(frame_index)
    ]
    if missing:
        raise ProjectError(
            f"Missing mask at frame {missing[0]}; "
            "export was not started."
        )
    output.mkdir(parents=True, exist_ok=True)
    total = last_frame - first_frame + 1
    for count, frame_index in enumerate(range(first_frame, last_frame + 1), start=1):
        mask = project.mask_at_video_resolution(frame_index)
        if rgba:
            rgb = decode_frame(project.source, frame_index, project.video_info)
            pixels = np.dstack((rgb, mask.astype(np.uint8) * 255))
            image = Image.fromarray(pixels)
        else:
            image = Image.fromarray(mask.astype(np.uint8) * 255)
        target = output / f"{frame_index:08d}.png"
        temporary = target.with_suffix(".png.tmp")
        image.save(temporary, format="PNG")
        temporary.replace(target)
        if progress:
            progress(count, total)
    return output


def export_video_sequence(
    project: ProjectStore,
    first_frame: int,
    last_frame: int,
    destination: str | Path,
    mode: str,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    if first_frame < 0 or last_frame >= project.video_info.frame_count or first_frame > last_frame:
        raise ProjectError("Invalid export range.")
    if mode not in {"mask", "alpha"}:
        raise ProjectError("Invalid video export format.")
    suffix = ".mp4" if mode == "mask" else ".mov"
    output = Path(destination).resolve()
    if output.suffix.lower() != suffix:
        raise ProjectError(f"This video type must be exported as {suffix}.")
    missing = [
        frame_index
        for frame_index in range(first_frame, last_frame + 1)
        if not project.is_processed(frame_index)
    ]
    if missing:
        raise ProjectError(
            f"Missing mask at frame {missing[0]}; "
            "export was not started."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-",
        suffix=suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    info = project.video_info
    width, height = info.width, info.height
    total = last_frame - first_frame + 1
    mask_folder = project.folder / "masks"
    temporary_masks = None
    manual_frames = set()
    for subject_operations in project.data.get("subject_manual_ops", {}).values():
        manual_frames.update(
            int(frame_index)
            for frame_index, operations in subject_operations.items()
            if operations and first_frame <= int(frame_index) <= last_frame
        )
    manual_frames.update(
        int(frame_index)
        for frame_index, operations in project.data.get("manual_ops", {}).items()
        if operations and first_frame <= int(frame_index) <= last_frame
    )
    if manual_frames:
        try:
            temporary_masks = tempfile.TemporaryDirectory(
                prefix=".roto-export-masks-",
                dir=output.parent,
            )
            mask_folder = Path(temporary_masks.name)
            for frame_index in range(first_frame, last_frame + 1):
                destination_mask = mask_folder / f"{frame_index:08d}.png"
                if frame_index in manual_frames:
                    pixels = project.final_mask(frame_index).astype(np.uint8) * 255
                    Image.fromarray(pixels).save(destination_mask, format="PNG")
                else:
                    destination_mask.symlink_to(
                        project.combined_mask_path(frame_index)
                    )
        except Exception:
            if temporary_masks is not None:
                temporary_masks.cleanup()
            raise
    arguments = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{info.fps:.9f}",
        "-start_number",
        str(first_frame),
        "-i",
        str(mask_folder / "%08d.png"),
    ]
    if mode == "mask":
        arguments.extend(
            [
                "-vf",
                f"scale={width}:{height}:flags=neighbor,format=yuv420p",
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        )
    else:
        arguments.extend(
            [
                "-ss",
                f"{info.timestamp(first_frame):.9f}",
                "-i",
                str(project.source),
                "-filter_complex",
                (
                    f"[1:v:0]setpts=PTS-STARTPTS,"
                    f"fps={info.fps:.9f},trim=end_frame={total},"
                    f"scale={width}:{height}[source];"
                    f"[0:v:0]setpts=PTS-STARTPTS,"
                    f"scale={width}:{height}:flags=neighbor,"
                    f"format=gray[mask];"
                    "[source][mask]alphamerge[output]"
                ),
                "-map",
                "[output]",
                "-map",
                "1:a:0?",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )
        arguments.extend(
            [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",
                "-alpha_bits",
                "16",
                "-pix_fmt",
                "yuva444p10le",
            ]
        )
    arguments.extend(
        [
            "-frames:v",
            str(total),
            "-t",
            f"{total / info.fps:.9f}",
            "-progress",
            "pipe:1",
            "-nostats",
        ]
    )
    arguments.append(str(temporary))

    process = None
    try:
        try:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise ProjectError(f"Could not start FFmpeg: {exc}") from exc
        if process.stdout is None or process.stderr is None:
            raise ProjectError("Could not open FFmpeg streams.")
        for line in process.stdout:
            if progress and line.startswith("frame="):
                try:
                    processed = min(int(line.partition("=")[2].strip()), total)
                except ValueError:
                    continue
                progress(processed, total)
        detail = process.stderr.read().strip()
        return_code = process.wait()
        if return_code:
            raise ProjectError(
                detail or f"FFmpeg failed to export the video (exit code {return_code})."
            )
        temporary.replace(output)
        return output
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if process is not None and process.stderr is not None:
            process.stderr.close()
        temporary.unlink(missing_ok=True)
        if temporary_masks is not None:
            temporary_masks.cleanup()
