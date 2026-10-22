import tempfile
import threading
from typing import Iterator

import numpy as np

from .config import AppPaths
from .media import extract_jpeg_window
from .project import ProjectStore


class InferenceError(RuntimeError):
    pass


class RotoEngine:
    def __init__(self, paths: AppPaths):
        self.paths = paths
        self.device = "cpu"
        self.torch = None
        self.image_predictor = None
        self.video_predictor = None
        self._active_image_index: int | None = None
        self._active_image: np.ndarray | None = None
        self._active_subject_id: str | None = None
        self._previous_logits: np.ndarray | None = None
        self._previous_point_count = 0

    def load(self) -> str:
        if not self.paths.model.is_file():
            raise InferenceError(f"Checkpoint SAM 2.1 Base+ non trovato: {self.paths.model}")
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.video_predictor = build_sam2_video_predictor(
            self.paths.model_config,
            str(self.paths.model),
            device=self.device,
            apply_postprocessing=False,
        )
        self.image_predictor = SAM2ImagePredictor(self.video_predictor)
        return f"SAM 2.1 Base+ pronto ({self.device.upper()}; fallback CPU PyTorch abilitato)"

    def segment(
        self,
        frame_index: int,
        image: np.ndarray,
        points: list[dict[str, float | int]],
        subject_id: str | None = None,
    ) -> np.ndarray:
        if self.image_predictor is None:
            raise InferenceError("Il modello immagine non è caricato.")
        if not any(point["label"] == 1 for point in points):
            raise InferenceError("Aggiungi almeno un punto positivo prima di segmentare.")
        if (
            frame_index != self._active_image_index
            or image is not self._active_image
            or subject_id != self._active_subject_id
        ):
            self.image_predictor.set_image(np.ascontiguousarray(image))
            self._active_image_index = frame_index
            self._active_image = image
            self._active_subject_id = subject_id
            self._previous_logits = None
        coordinates = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float32)
        labels = np.asarray([p["label"] for p in points], dtype=np.int32)
        kwargs = {}
        if len(points) < self._previous_point_count:
            self._previous_logits = None
        if self._previous_logits is not None:
            kwargs["mask_input"] = self._previous_logits
        multimask_output = len(points) == 1
        masks, scores, low_res_logits = self.image_predictor.predict(
            point_coords=coordinates,
            point_labels=labels,
            multimask_output=multimask_output,
            **kwargs,
        )
        candidate_masks = np.asarray(masks, dtype=bool)
        best = int(np.asarray(scores).argmax()) if multimask_output else 0
        self._previous_logits = np.asarray(low_res_logits[best], dtype=np.float32)[None, :, :]
        self._previous_point_count = len(points)
        return candidate_masks[best]

    def _ensure_video_predictor(self):
        if self.video_predictor is None:
            raise InferenceError("Il modello video SAM 2.1 non è caricato.")
        return self.video_predictor

    def propagate(
        self,
        project: ProjectStore,
        first_frame: int,
        last_frame: int,
        start_frame: int,
        direction: str,
        window_size: int,
        cancel: threading.Event,
        subject_id: str | None = None,
    ) -> Iterator[tuple[int, np.ndarray, int, int]]:
        if (
            first_frame < 0
            or last_frame >= project.video_info.frame_count
            or first_frame > last_frame
        ):
            raise InferenceError("Intervallo di propagazione non valido.")
        if not first_frame <= start_frame <= last_frame:
            raise InferenceError("Il fotogramma iniziale deve essere dentro l'intervallo.")
        if direction not in {"forward", "reverse", "both"}:
            raise InferenceError("Direzione di tracking non valida.")
        if window_size < 12:
            raise InferenceError("La finestra minima verificata per SAM 2 è di 12 fotogrammi.")
        subject_id = subject_id or project.subjects[0]["id"]
        prompts = project.points(start_frame, subject_id)
        if not any(point["label"] == 1 for point in prompts):
            raise InferenceError(
                f"Aggiungi un punto positivo sul fotogramma iniziale {start_frame}."
            )

        if direction == "reverse" and start_frame == first_frame:
            if not cancel.is_set():
                yield start_frame, project.ai_mask(start_frame, subject_id), 1, 1
            return

        if self.image_predictor is not None:
            self.image_predictor.reset_predictor()
        self._active_image = None
        self._active_image_index = None
        self._active_subject_id = None
        self._previous_logits = None
        self._previous_point_count = 0

        predictor = self._ensure_video_predictor()
        total = {
            "forward": last_frame - start_frame + 1,
            "reverse": start_frame - first_frame + 1,
            "both": last_frame - first_frame + 1,
        }[direction]
        processed = 0
        directions = []
        if direction in {"forward", "both"}:
            directions.append((start_frame, last_frame, False))
        if direction in {"reverse", "both"}:
            directions.append((start_frame, first_frame, True))

        for origin, boundary, reverse in directions:
            seed_mask = None
            cursor = origin
            first_chunk = True
            while (cursor >= boundary if reverse else cursor <= boundary):
                if cancel.is_set():
                    return
                if reverse:
                    chunk_end = cursor
                    chunk_start = max(boundary, chunk_end - window_size + 1)
                else:
                    chunk_start = cursor
                    chunk_end = min(boundary, chunk_start + window_size - 1)
                chunk_count = chunk_end - chunk_start + 1
                seed_frame = chunk_count - 1 if reverse else 0

                with tempfile.TemporaryDirectory(
                    prefix="roto-frames-",
                    dir=self.paths.temporary,
                ) as temporary:
                    extract_jpeg_window(
                        project.source,
                        project.video_info,
                        chunk_start,
                        chunk_count,
                        temporary,
                    )
                    state = predictor.init_state(
                        temporary,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=False,
                    )
                    try:
                        mask_height, mask_width = project.mask_shape
                        state["video_height"] = mask_height
                        state["video_width"] = mask_width
                        anchor_frame = chunk_start + seed_frame
                        prompts = project.points(anchor_frame, subject_id)
                        if prompts:
                            predictor.add_new_points_or_box(
                                state,
                                seed_frame,
                                1,
                                points=self._tracking_points(prompts, project),
                                labels=np.asarray(
                                    [p["label"] for p in prompts],
                                    dtype=np.int32,
                                ),
                            )
                        elif seed_mask is not None:
                            predictor.add_new_mask(state, seed_frame, 1, seed_mask)
                        else:
                            raise InferenceError(
                                f"Nessun prompt o maschera seme sul fotogramma {anchor_frame}."
                            )

                        for annotated_frame in range(chunk_start, chunk_end + 1):
                            if annotated_frame == anchor_frame:
                                continue
                            prompts = project.points(annotated_frame, subject_id)
                            if prompts:
                                predictor.add_new_points_or_box(
                                    state,
                                    annotated_frame - chunk_start,
                                    1,
                                    points=self._tracking_points(prompts, project),
                                    labels=np.asarray(
                                        [p["label"] for p in prompts],
                                        dtype=np.int32,
                                    ),
                                )

                        for frame_index, _, logits in predictor.propagate_in_video(
                            state,
                            start_frame_idx=seed_frame,
                            max_frame_num_to_track=chunk_count - 1,
                            reverse=reverse,
                        ):
                            if cancel.is_set():
                                return
                            mask = logits[0, 0].detach().cpu().numpy() > 0
                            global_index = chunk_start + frame_index
                            seed_mask = mask.copy()
                            is_chunk_overlap = not first_chunk and frame_index == seed_frame
                            is_direction_overlap = (
                                direction == "both"
                                and reverse
                                and global_index == start_frame
                            )
                            if not is_chunk_overlap and not is_direction_overlap:
                                processed += 1
                                yield global_index, mask, processed, total
                            del logits, mask
                        first_chunk = False
                    finally:
                        try:
                            predictor.reset_state(state)
                        finally:
                            del state
                            if self.torch is not None and self.device == "mps":
                                self.torch.mps.empty_cache()

                if reverse:
                    if chunk_start == boundary:
                        break
                    cursor = chunk_start
                else:
                    if chunk_end == boundary:
                        break
                    cursor = chunk_end

    @staticmethod
    def _tracking_points(
        points: list[dict[str, float | int]],
        project: ProjectStore,
    ) -> np.ndarray:
        video = project.video_info
        mask_height, mask_width = project.mask_shape
        scale_x = mask_width / video.width
        scale_y = mask_height / video.height
        return np.asarray(
            [[p["x"] * scale_x, p["y"] * scale_y] for p in points],
            dtype=np.float32,
        )
