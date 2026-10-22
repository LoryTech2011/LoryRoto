import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageDraw

from .media import VideoInfo


class ProjectError(RuntimeError):
    pass


class ProjectStore:
    SUBJECT_COLORS = (
        "#43e09b",
        "#ff9f43",
        "#63a8ff",
        "#d889ff",
        "#f2d94e",
        "#45d6d0",
        "#ff78bd",
        "#a7e063",
    )

    def __init__(self, folder: str | Path, root: str | Path):
        self.folder = Path(folder).resolve()
        self.root = Path(root).resolve()
        self.manifest_path = self.folder / "project.json"
        if not self.folder.is_relative_to(self.root):
            raise ProjectError("Il progetto deve rimanere nella cartella dell'app.")
        try:
            self.data: dict[str, Any] = json.loads(self.manifest_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError(f"Progetto non leggibile: {exc}") from exc
        self._ensure_subject_data()

    @classmethod
    def create(
        cls,
        root: str | Path,
        source: str | Path,
        info: VideoInfo,
    ) -> "ProjectStore":
        app_root = Path(root).resolve()
        projects_folder = app_root / "projects"
        if not projects_folder.resolve().is_relative_to(app_root):
            raise ProjectError("La cartella progetti deve rimanere nella cartella dell'app.")
        folder = projects_folder / str(uuid.uuid4())
        folder.mkdir(parents=True)
        if not folder.resolve().is_relative_to(app_root):
            raise ProjectError("Il progetto deve rimanere nella cartella dell'app.")
        for name in ("masks", "exports"):
            (folder / name).mkdir()
        data = {
            "version": 1,
            "source": str(Path(source).resolve()),
            "created": datetime.now().astimezone().isoformat(),
            "video": {
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "duration": info.duration,
                "frame_count": info.frame_count,
                "codec": info.codec,
            },
            "points": {},
            "manual_ops": {},
            "status": {},
            "mask_max_dimension": None,
            "subjects": [{"id": "subject-1", "name": "Soggetto 1", "color": cls.SUBJECT_COLORS[0]}],
            "subject_points": {"subject-1": {}},
            "subject_manual_ops": {"subject-1": {}},
        }
        project = cls.__new__(cls)
        project.folder = folder
        project.root = app_root
        project.manifest_path = folder / "project.json"
        project.data = data
        project.save()
        return project

    def _ensure_subject_data(self) -> None:
        subjects = self.data.get("subjects")
        legacy_project = not subjects
        if not subjects:
            subjects = [
                {
                    "id": "subject-1",
                    "name": "Soggetto 1",
                    "color": self.SUBJECT_COLORS[0],
                }
            ]
            self.data["subjects"] = subjects
        self.data.setdefault(
            "subject_points",
            {"subject-1": self.data.get("points", {})},
        )
        self.data.setdefault(
            "subject_manual_ops",
            {"subject-1": self.data.get("manual_ops", {})},
        )
        if legacy_project:
            legacy_masks = list((self.folder / "masks").glob("*.png"))
            if legacy_masks:
                subject_folder = self.folder / "masks" / "subject-1"
                subject_folder.mkdir(parents=True, exist_ok=True)
                for path in legacy_masks:
                    os.replace(path, subject_folder / path.name)
                for path in legacy_masks:
                    if path.stem.isdigit():
                        self._save_combined_ai_mask(int(path.stem))
            self.save()

    @property
    def subjects(self) -> list[dict[str, str]]:
        return [dict(subject) for subject in self.data["subjects"]]

    def add_subject(self) -> dict[str, str]:
        index = 1
        known_ids = {subject["id"] for subject in self.data["subjects"]}
        while f"subject-{index}" in known_ids:
            index += 1
        subject = {
            "id": f"subject-{index}",
            "name": f"Soggetto {index}",
            "color": self.SUBJECT_COLORS[(index - 1) % len(self.SUBJECT_COLORS)],
        }
        self.data["subjects"].append(subject)
        self.data["subject_points"][subject["id"]] = {}
        self.data["subject_manual_ops"][subject["id"]] = {}
        self.save()
        return dict(subject)

    def remove_subject(self, subject_id: str) -> None:
        if len(self.data["subjects"]) <= 1:
            raise ProjectError("Il progetto deve mantenere almeno un soggetto.")
        subject = self._subject(subject_id)
        self.data["subjects"] = [
            item for item in self.data["subjects"] if item["id"] != subject_id
        ]
        self.data["subject_points"].pop(subject_id, None)
        self.data["subject_manual_ops"].pop(subject_id, None)
        mask_folder = self.folder / "masks" / subject["id"]
        if mask_folder.exists():
            for mask_path in mask_folder.glob("*.png"):
                mask_path.unlink()
            mask_folder.rmdir()
        combined_frames = [
            int(path.stem)
            for path in self.combined_mask_path(0).parent.glob("*.png")
            if path.stem.isdigit()
        ]
        for frame_index in combined_frames:
            self._save_combined_ai_mask(frame_index)
        self.save()

    def _subject(self, subject_id: str | None = None) -> dict[str, str]:
        if subject_id is None:
            return self.data["subjects"][0]
        for subject in self.data["subjects"]:
            if subject["id"] == subject_id:
                return subject
        raise ProjectError(f"Soggetto non trovato: {subject_id}")

    @property
    def source(self) -> Path:
        return Path(self.data["source"])

    @property
    def video_info(self) -> VideoInfo:
        return VideoInfo(**self.data["video"])

    @property
    def mask_shape(self) -> tuple[int, int]:
        info = self.video_info
        max_dimension = self.data.get("mask_max_dimension")
        if max_dimension is None or max(info.width, info.height) <= max_dimension:
            return info.height, info.width
        scale = max_dimension / max(info.width, info.height)
        return max(1, round(info.height * scale)), max(1, round(info.width * scale))

    def set_mask_max_dimension(self, max_dimension: int | None) -> None:
        if max_dimension not in {None, 1920}:
            raise ProjectError("La risoluzione della maschera deve essere originale o 1080p.")
        if self.data.get("mask_max_dimension") != max_dimension:
            self.data["mask_max_dimension"] = max_dimension
            self.save()

    def save(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), "utf-8")
        os.replace(temporary, self.manifest_path)

    def points(
        self,
        frame_index: int,
        subject_id: str | None = None,
    ) -> list[dict[str, float | int]]:
        subject = self._subject(subject_id)
        points = self.data["subject_points"].get(subject["id"], {})
        return list(points.get(str(frame_index), []))

    def set_points(
        self,
        frame_index: int,
        points: list[dict[str, float | int]],
        subject_id: str | None = None,
    ) -> None:
        subject = self._subject(subject_id)
        self.data["subject_points"].setdefault(subject["id"], {})[
            str(frame_index)
        ] = points
        self.save()

    def manual_ops(
        self,
        frame_index: int,
        subject_id: str | None = None,
    ) -> list[dict[str, Any]]:
        subject = self._subject(subject_id)
        operations = self.data["subject_manual_ops"].get(subject["id"], {})
        return list(operations.get(str(frame_index), []))

    def set_manual_ops(
        self,
        frame_index: int,
        operations: list[dict[str, Any]],
        subject_id: str | None = None,
    ) -> None:
        subject = self._subject(subject_id)
        self.data["subject_manual_ops"].setdefault(subject["id"], {})[
            str(frame_index)
        ] = operations
        self.data["status"][str(frame_index)] = "review"
        self.save()

    def mask_path(
        self,
        frame_index: int,
        subject_id: str | None = None,
    ) -> Path:
        subject = self._subject(subject_id)
        return self.folder / "masks" / subject["id"] / f"{frame_index:08d}.png"

    def combined_mask_path(self, frame_index: int) -> Path:
        return self.folder / "masks" / f"{frame_index:08d}.png"

    def save_ai_mask(
        self,
        frame_index: int,
        mask: np.ndarray,
        persist_project: bool = True,
        subject_id: str | None = None,
    ) -> None:
        subject = self._subject(subject_id)
        array = np.asarray(mask, dtype=bool)
        video_shape = (self.video_info.height, self.video_info.width)
        expected = self.mask_shape
        if array.shape not in {video_shape, expected}:
            raise ProjectError(
                f"Dimensione maschera {array.shape}, attesa {video_shape} o {expected}."
            )
        if array.shape != expected:
            resampling = (
                Image.Resampling.LANCZOS
                if expected[0] < array.shape[0] or expected[1] < array.shape[1]
                else Image.Resampling.NEAREST
            )
            image = Image.fromarray(array.astype(np.uint8) * 255)
            array = np.asarray(
                image.resize((expected[1], expected[0]), resampling),
                dtype=np.uint8,
            ) >= 128
        destination = self.mask_path(frame_index, subject["id"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".png.tmp")
        Image.fromarray(array.astype(np.uint8) * 255).save(temporary, format="PNG")
        os.replace(temporary, destination)
        self._save_combined_ai_mask(frame_index)
        self.data["status"][str(frame_index)] = (
            "processed" if self.is_processed(frame_index) else "annotated"
        )
        if persist_project:
            self.save()

    def ai_mask(
        self,
        frame_index: int,
        subject_id: str | None = None,
    ) -> np.ndarray:
        subject = self._subject(subject_id)
        path = self.mask_path(frame_index, subject["id"])
        if not path.is_file():
            return np.zeros(self.mask_shape, dtype=bool)
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8) > 0

    def _save_combined_ai_mask(self, frame_index: int) -> None:
        combined = np.zeros(self.mask_shape, dtype=bool)
        for subject in self.data["subjects"]:
            path = self.mask_path(frame_index, subject["id"])
            if not path.is_file():
                self.combined_mask_path(frame_index).unlink(missing_ok=True)
                return
            with Image.open(path) as image:
                subject_mask = np.asarray(
                    image.convert("L"),
                    dtype=np.uint8,
                ) > 0
            if subject_mask.shape != combined.shape:
                resized = Image.fromarray(subject_mask.astype(np.uint8) * 255)
                resized = resized.resize(
                    (combined.shape[1], combined.shape[0]),
                    Image.Resampling.NEAREST,
                )
                subject_mask = np.asarray(resized, dtype=np.uint8) > 0
            combined |= subject_mask
        destination = self.combined_mask_path(frame_index)
        temporary = destination.with_suffix(".png.tmp")
        Image.fromarray(combined.astype(np.uint8) * 255).save(
            temporary,
            format="PNG",
        )
        os.replace(temporary, destination)

    def manual_masks(
        self,
        frame_index: int,
        subject_id: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        subject = self._subject(subject_id)
        shape = self.mask_shape
        ai_path = self.mask_path(frame_index, subject["id"])
        if ai_path.is_file():
            with Image.open(ai_path) as image:
                shape = (image.height, image.width)
        scale_x = shape[1] / self.video_info.width
        scale_y = shape[0] / self.video_info.height
        add = Image.new("1", (shape[1], shape[0]), 0)
        remove = Image.new("1", (shape[1], shape[0]), 0)
        for operation in self.manual_ops(frame_index, subject["id"]):
            points = [
                (
                    int(round(point[0] * scale_x)),
                    int(round(point[1] * scale_y)),
                )
                for point in operation["points"]
            ]
            if not points:
                continue
            radius = max(1, int(round(operation["radius"] * min(scale_x, scale_y))))
            target = add if operation["action"] == "add" else remove
            draw = ImageDraw.Draw(target)
            if len(points) > 1:
                draw.line(points, fill=1, width=radius * 2, joint="curve")
            for x, y in points:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        return np.asarray(add, dtype=bool), np.asarray(remove, dtype=bool)

    def subject_mask(
        self,
        frame_index: int,
        subject_id: str | None = None,
    ) -> np.ndarray:
        subject = self._subject(subject_id)
        ai = self.ai_mask(frame_index, subject["id"])
        if not self.manual_ops(frame_index, subject["id"]):
            return ai
        add, remove = self.manual_masks(frame_index, subject["id"])
        return (ai | add) & ~remove

    def final_mask(self, frame_index: int) -> np.ndarray:
        target_height, target_width = self.mask_shape
        combined = np.zeros((target_height, target_width), dtype=bool)
        for subject in self.data["subjects"]:
            mask = self.subject_mask(frame_index, subject["id"])
            if mask.shape != combined.shape:
                image = Image.fromarray(mask.astype(np.uint8) * 255)
                image = image.resize(
                    (target_width, target_height),
                    Image.Resampling.NEAREST,
                )
                mask = np.asarray(image, dtype=np.uint8) > 0
            combined |= mask
        return combined

    def masks_by_subject(
        self,
        frame_index: int,
    ) -> Iterator[tuple[dict[str, str], np.ndarray]]:
        for subject in self.data["subjects"]:
            yield dict(subject), self.subject_mask(frame_index, subject["id"])

    def mask_at_video_resolution(self, frame_index: int) -> np.ndarray:
        mask = self.final_mask(frame_index)
        expected = (self.video_info.height, self.video_info.width)
        if mask.shape == expected:
            return mask
        image = Image.fromarray(mask.astype(np.uint8) * 255)
        image = image.resize(
            (expected[1], expected[0]),
            Image.Resampling.NEAREST,
        )
        return np.asarray(image, dtype=np.uint8) > 0

    def is_processed(self, frame_index: int) -> bool:
        return all(
            self.mask_path(frame_index, subject["id"]).is_file()
            for subject in self.data["subjects"]
        )

    def clear_ai_mask(
        self,
        frame_index: int,
        subject_id: str | None = None,
    ) -> None:
        subject = self._subject(subject_id)
        path = self.mask_path(frame_index, subject["id"])
        if path.exists():
            path.unlink()
        self._save_combined_ai_mask(frame_index)
        self.data["status"][str(frame_index)] = (
            "annotated" if self.points(frame_index, subject["id"]) else "unprocessed"
        )
        self.save()
