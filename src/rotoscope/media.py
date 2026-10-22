import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    codec: str

    def timestamp(self, frame_index: int) -> float:
        return frame_index / self.fps


def _require_tool(name: str) -> str:
    from shutil import which

    executable = which(name)
    if executable is None:
        raise MediaError(f"{name} non trovato. Installalo e riavvia l'app.")
    return executable


def probe_video(path: str | Path) -> VideoInfo:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MediaError(f"File video non trovato: {source}")
    result = subprocess.run(
        [
            _require_tool("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,codec_name",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise MediaError(result.stderr.strip() or "ffprobe non è riuscito a leggere il video.")
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        fps = float(Fraction(rate))
        duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
        frame_count = int(stream.get("nb_frames") or round(duration * fps))
        if fps <= 0 or duration <= 0 or frame_count <= 0:
            raise ValueError("metadati temporali mancanti")
        return VideoInfo(
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps=fps,
            duration=duration,
            frame_count=frame_count,
            codec=str(stream.get("codec_name", "sconosciuto")),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise MediaError(f"Metadati video non validi: {exc}") from exc


def decode_frame_window(
    path: str | Path,
    first_frame: int,
    frame_count: int,
    info: VideoInfo,
    max_dimension: int | None = None,
) -> list[np.ndarray]:
    if frame_count < 1 or not 0 <= first_frame < info.frame_count:
        raise MediaError("Intervallo di decodifica non valido.")
    count = min(frame_count, info.frame_count - first_frame)
    source = Path(path).expanduser().resolve()
    width, height = info.width, info.height
    arguments = [
        _require_tool("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-ss",
        f"{info.timestamp(first_frame):.9f}",
        "-i",
        str(source),
    ]
    if max_dimension is not None and max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        arguments.extend(["-vf", f"scale={width}:{height}"])
    arguments.extend(
        [
            "-frames:v",
            str(count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    try:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            timeout=max(45, count * 2),
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError("Timeout durante la decodifica del video.") from exc
    frame_size = width * height * 3
    expected_size = count * frame_size
    if result.returncode or len(result.stdout) != expected_size:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise MediaError(
            detail
            or f"FFmpeg ha decodificato {len(result.stdout) // frame_size} frame, "
            f"attesi {count}, a partire dal frame {first_frame}."
        )
    decoded = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        count, height, width, 3
    )
    return [frame for frame in decoded]


def decode_frame(path: str | Path, frame_index: int, info: VideoInfo) -> np.ndarray:
    return decode_frame_window(path, frame_index, 1, info)[0]


def iter_video_frames(
    path: str | Path,
    first_frame: int,
    frame_count: int,
    info: VideoInfo,
) -> Iterator[np.ndarray]:
    if frame_count < 1 or not 0 <= first_frame < info.frame_count:
        raise MediaError("Intervallo di decodifica non valido.")
    count = min(frame_count, info.frame_count - first_frame)
    source = Path(path).expanduser().resolve()
    arguments = [
        _require_tool("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-ss",
        f"{info.timestamp(first_frame):.9f}",
        "-i",
        str(source),
        "-frames:v",
        str(count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise MediaError(f"Avvio FFmpeg non riuscito: {exc}") from exc

    frame_size = info.width * info.height * 3
    try:
        if process.stdout is None or process.stderr is None:
            raise MediaError("Impossibile aprire lo stream FFmpeg.")
        for frame_index in range(count):
            buffer = bytearray(frame_size)
            view = memoryview(buffer)
            offset = 0
            while offset < frame_size:
                bytes_read = process.stdout.readinto(view[offset:])
                if not bytes_read:
                    break
                offset += bytes_read
            if offset != frame_size:
                detail = process.stderr.read().decode("utf-8", errors="replace").strip()
                return_code = process.wait()
                raise MediaError(
                    detail
                    or f"FFmpeg ha decodificato {frame_index} frame, attesi {count} "
                    f"dal frame {first_frame} (codice {return_code})."
                )
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(
                info.height,
                info.width,
                3,
            )

        extra_data = process.stdout.read()
        detail = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code or extra_data:
            raise MediaError(
                detail
                or f"FFmpeg non ha terminato correttamente la decodifica "
                f"(codice {return_code})."
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def extract_jpeg_window(
    source: str | Path,
    info: VideoInfo,
    first_frame: int,
    frame_count: int,
    output_dir: str | Path,
) -> list[Path]:
    if frame_count < 1 or first_frame < 0 or first_frame >= info.frame_count:
        raise MediaError("Intervallo di estrazione non valido.")
    count = min(frame_count, info.frame_count - first_frame)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            _require_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{info.timestamp(first_frame):.9f}",
            "-i",
            str(Path(source).resolve()),
            "-frames:v",
            str(count),
            "-fps_mode",
            "passthrough",
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(folder / "%05d.jpg"),
        ],
        check=False,
        capture_output=True,
        timeout=max(60, count * 3),
    )
    if result.returncode:
        raise MediaError(result.stderr.decode("utf-8", errors="replace").strip() or
                         "Estrazione del blocco video non riuscita.")
    frames = sorted(folder.glob("*.jpg"))
    if len(frames) != count:
        raise MediaError(
            f"FFmpeg ha estratto {len(frames)} fotogrammi, attesi {count}; "
            "il video potrebbe avere frame rate variabile o metadati incompleti."
        )
    return frames
