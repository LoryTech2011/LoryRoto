import threading
from queue import Queue

from PySide6.QtCore import QObject, Signal, Slot

from .config import AppPaths
from .inference import RotoEngine
from .project import ProjectStore


class InferenceWorker(QObject):
    loaded = Signal(str)
    failed = Signal(str)
    mask_ready = Signal(int, object, str, int)
    propagated_frame = Signal(int)
    progress = Signal(int, int)
    propagation_finished = Signal()

    def __init__(self, paths: AppPaths):
        super().__init__()
        self.paths = paths
        self.engine = RotoEngine(paths)
        self.cancel_event = threading.Event()
        self._requests: Queue[tuple | None] = Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="RotoInference",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def load_model(self) -> None:
        self._requests.put(("load",))

    @Slot(int, object, object, str, int)
    def segment(
        self,
        frame_index: int,
        image,
        points,
        subject_id: str,
        generation: int,
    ) -> None:
        self._requests.put(
            ("segment", frame_index, image, points, subject_id, generation)
        )

    @Slot(str, int, int, int, str, int)
    def propagate(
        self,
        project_folder: str,
        first_frame: int,
        last_frame: int,
        start_frame: int,
        direction: str,
        window_size: int,
    ) -> None:
        self.cancel_event.clear()
        self._requests.put(
            (
                "propagate",
                project_folder,
                first_frame,
                last_frame,
                start_frame,
                direction,
                window_size,
            )
        )

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            action, *arguments = request
            try:
                if action == "load":
                    self.loaded.emit(self.engine.load())
                elif action == "segment":
                    frame_index, image, points, subject_id, generation = arguments
                    mask = self.engine.segment(
                        frame_index,
                        image,
                        points,
                        subject_id,
                    )
                    self.mask_ready.emit(frame_index, mask, subject_id, generation)
                elif action == "propagate":
                    self._propagate(*arguments)
            except Exception as exc:
                self.failed.emit(str(exc))

    def _propagate(
        self,
        project_folder: str,
        first_frame: int,
        last_frame: int,
        start_frame: int,
        direction: str,
        window_size: int,
    ) -> None:
        project = None
        try:
            project = ProjectStore(project_folder, self.paths.root)
            subject_ids = [subject["id"] for subject in project.subjects]
            per_subject_total = {
                "forward": last_frame - start_frame + 1,
                "reverse": start_frame - first_frame + 1,
                "both": last_frame - first_frame + 1,
            }[direction]
            total = per_subject_total * len(subject_ids)
            processed = 0
            for subject_id in subject_ids:
                for frame_index, mask, _, _ in self.engine.propagate(
                    project,
                    first_frame,
                    last_frame,
                    start_frame,
                    direction,
                    window_size,
                    self.cancel_event,
                    subject_id,
                ):
                    project.save_ai_mask(
                        frame_index,
                        mask,
                        persist_project=False,
                        subject_id=subject_id,
                    )
                    processed += 1
                    self.propagated_frame.emit(frame_index)
                    self.progress.emit(processed, total)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if project is not None:
                try:
                    project.save()
                except OSError as exc:
                    self.failed.emit(f"Salvataggio progetto dopo il tracking non riuscito: {exc}")
            self.propagation_finished.emit()

    def cancel(self) -> None:
        self.cancel_event.set()

    def stop(self) -> None:
        self.cancel()
        self._requests.put(None)
        self._thread.join(timeout=10)
