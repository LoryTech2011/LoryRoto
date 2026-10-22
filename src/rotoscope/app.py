import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config import AppPaths
from .exporter import export_sequence, export_video_sequence
from .media import MediaError, VideoInfo, decode_frame_window, probe_video
from .project import ProjectError, ProjectStore
from .viewer import ViewerWidget
from .worker import InferenceWorker


class TaskSignals(QObject):
    frame_ready = Signal(int, int, bool, int, object)
    frame_failed = Signal(int, int, str)
    failed = Signal(str)
    export_done = Signal(str)
    export_progress = Signal(int, int)
    export_finished = Signal()


class FrameTask(QRunnable):
    PREVIEW_FRAME_COUNT = 6
    PREVIEW_MAX_DIMENSION = 960
    FULL_FRAME_BATCH_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        path: Path,
        info: VideoInfo,
        frame_index: int,
        generation: int,
        preview: bool,
    ):
        super().__init__()
        self.path = path
        self.info = info
        self.frame_index = frame_index
        self.generation = generation
        self.preview = preview
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            if self.preview:
                count = min(self.PREVIEW_FRAME_COUNT, self.info.frame_count)
                max_dimension = self.PREVIEW_MAX_DIMENSION
            else:
                bytes_per_frame = self.info.width * self.info.height * 3
                count = min(
                    self.PREVIEW_FRAME_COUNT,
                    max(1, self.FULL_FRAME_BATCH_BYTES // bytes_per_frame),
                    self.info.frame_count,
                )
                max_dimension = None
            first_frame = min(
                max(0, self.frame_index - count // 3),
                self.info.frame_count - count,
            )
            frames = decode_frame_window(
                self.path,
                first_frame,
                count,
                self.info,
                max_dimension,
            )
            self.signals.frame_ready.emit(
                self.generation,
                self.frame_index,
                self.preview,
                first_frame,
                frames,
            )
        except Exception as exc:
            self.signals.frame_failed.emit(
                self.generation,
                self.frame_index,
                str(exc),
            )


class ExportTask(QRunnable):
    def __init__(
        self,
        project_folder: Path,
        root: Path,
        first_frame: int,
        last_frame: int,
        destination: Path,
        rgba: bool,
    ):
        super().__init__()
        self.project_folder = project_folder
        self.root = root
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.destination = destination
        self.rgba = rgba
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            project = ProjectStore(self.project_folder, self.root)
            output = export_sequence(
                project,
                self.first_frame,
                self.last_frame,
                self.destination,
                self.rgba,
                self.signals.export_progress.emit,
            )
            self.signals.export_done.emit(str(output))
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class VideoExportTask(QRunnable):
    def __init__(
        self,
        project_folder: Path,
        root: Path,
        first_frame: int,
        last_frame: int,
        destination: Path,
        mode: str,
    ):
        super().__init__()
        self.project_folder = project_folder
        self.root = root
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.destination = destination
        self.mode = mode
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            project = ProjectStore(self.project_folder, self.root)
            output = export_video_sequence(
                project,
                self.first_frame,
                self.last_frame,
                self.destination,
                self.mode,
                self.signals.export_progress.emit,
            )
            self.signals.export_done.emit(str(output))
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.export_finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.paths = AppPaths(root)
        self.paths.ensure()
        self.project: ProjectStore | None = None
        self.active_subject_id = "subject-1"
        self.info: VideoInfo | None = None
        self.frame: np.ndarray | None = None
        self._displayed_frame_index: int | None = None
        self.current_frame = 0
        self.frame_busy = False
        self.pending_frame: tuple[int, bool] | None = None
        self._source_generation = 0
        self._frame_caches: dict[bool, OrderedDict[int, np.ndarray]] = {
            False: OrderedDict(),
            True: OrderedDict(),
        }
        self._frame_cache_bytes = {False: 0, True: 0}
        self._frame_cache_limits = {
            False: 48 * 1024 * 1024,
            True: 24 * 1024 * 1024,
        }
        self._frame_cache_sources: dict[bool, Path | None] = {
            False: None,
            True: None,
        }
        self.progress_started_at: float | None = None
        self.last_propagation_preview_at = 0.0
        self.latest_propagated_frame: int | None = None
        self.undo_operations: dict[int, list[dict]] = {}
        self.redo_operations: dict[int, list[dict]] = {}
        self.propagating = False
        self.video_exporting = False
        self.setWindowTitle("Roto Local")
        self.resize(1420, 900)
        self.setAcceptDrops(True)
        self._configure_logging()
        self._build_ui()
        self._start_worker()
        self._connect_shortcuts()

    def _configure_logging(self) -> None:
        logging.basicConfig(
            filename=self.paths.logs / "app.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #161d25; color: #e5edf5; }
            QPushButton, QToolButton { background: #263441; border: 1px solid #3b4d5c;
                border-radius: 5px; padding: 7px 10px; }
            QPushButton:hover, QToolButton:hover { background: #334657; }
            QPushButton:checked, QToolButton:checked { background: #126b61; border-color: #34c9a1; }
            QSlider::groove:horizontal { height: 5px; background: #344552; }
            QSlider::handle:horizontal { width: 13px; margin: -5px 0; border-radius: 6px; background: #36c9a2; }
            QSpinBox { background: #202b35; border: 1px solid #40515e; padding: 4px; }
            QProgressBar { border: 1px solid #344552; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background: #31b994; }
            """
        )
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 10, 12, 8)

        toolbar = QHBoxLayout()
        self.open_video_button = QPushButton("Open Video")
        self.open_project_button = QPushButton("Open Project")
        self.save_button = QPushButton("Save")
        toolbar.addWidget(self.open_video_button)
        toolbar.addWidget(self.open_project_button)
        toolbar.addWidget(self.save_button)
        toolbar.addStretch(1)
        self.video_info_label = QLabel("No video")
        toolbar.addWidget(self.video_info_label)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.viewer = ViewerWidget()
        splitter.addWidget(self.viewer)
        splitter.addWidget(self._make_tools_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1120, 270])
        layout.addWidget(splitter, 1)

        transport = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.previous_button = QPushButton("◀ Frame")
        self.next_button = QPushButton("Frame ▶")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.frame_label = QLabel("0 / 0")
        transport.addWidget(self.play_button)
        transport.addWidget(self.previous_button)
        transport.addWidget(self.next_button)
        transport.addWidget(self.timeline, 1)
        transport.addWidget(self.frame_label)
        layout.addLayout(transport)

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Range:"))
        self.range_start = QSpinBox()
        self.range_end = QSpinBox()
        self.range_start.setPrefix("From ")
        self.range_end.setPrefix("To ")
        self.range_start.setRange(0, 0)
        self.range_end.setRange(0, 0)
        range_layout.addWidget(self.range_start)
        range_layout.addWidget(self.range_end)
        range_layout.addWidget(QLabel("Direction:"))
        self.tracking_direction = QComboBox()
        self.tracking_direction.addItem("Forward", "forward")
        self.tracking_direction.addItem("Backward", "reverse")
        self.tracking_direction.addItem("Both", "both")
        self.tracking_direction.setCurrentIndex(2)
        self.tracking_direction.setToolTip(
            "Tracking starts from the frame selected on the timeline."
        )
        range_layout.addWidget(self.tracking_direction)
        range_layout.addWidget(QLabel("Memory window:"))
        self.window_size = QComboBox()
        self.window_size.addItem("12 frames — low", 12)
        self.window_size.addItem("24 frames — medium", 24)
        self.window_size.addItem("48 frames — high", 48)
        self.window_size.setCurrentIndex(0)
        self.window_size.setToolTip(
            "Longer windows use more memory but reduce tracking restarts."
        )
        range_layout.addWidget(self.window_size)
        range_layout.addWidget(QLabel("Mask resolution:"))
        self.mask_resolution = QComboBox()
        self.mask_resolution.addItem("Original", None)
        self.mask_resolution.addItem("1080p — less RAM", 1920)
        self.mask_resolution.setToolTip(
            "At 1080p, masks use less memory and storage; "
            "exports are resized to the video's resolution."
        )
        self.mask_resolution.currentIndexChanged.connect(
            self._mask_resolution_changed
        )
        range_layout.addWidget(self.mask_resolution)
        self.propagate_button = QPushButton("Start Tracking")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        range_layout.addWidget(self.propagate_button)
        range_layout.addWidget(self.cancel_button)
        range_layout.addWidget(self.progress, 1)
        layout.addLayout(range_layout)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Loading SAM 2.1 model…")
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance_playback)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(35)
        self.preview_timer.timeout.connect(self._preview_timeline_frame)

        self.open_video_button.clicked.connect(self.open_video)
        self.open_project_button.clicked.connect(self.open_project)
        self.save_button.clicked.connect(self.save_project)
        self.previous_button.clicked.connect(lambda: self._seek(self.current_frame - 1))
        self.next_button.clicked.connect(lambda: self._seek(self.current_frame + 1))
        self.play_button.clicked.connect(self.toggle_play)
        self.timeline.valueChanged.connect(self._timeline_changed)
        self.timeline.sliderReleased.connect(self._timeline_released)
        self.viewer.point_clicked.connect(self._add_point)
        self.viewer.brush_stroke.connect(self._add_brush_stroke)
        self.propagate_button.clicked.connect(self._start_propagation)
        self.cancel_button.clicked.connect(self._cancel_propagation)

    def _make_tools_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(QLabel("<b>Annotation</b>"))
        subject_row = QHBoxLayout()
        self.subject_selector = QComboBox()
        self.add_subject_button = QPushButton("+ Subject")
        self.remove_subject_button = QPushButton("−")
        self.remove_subject_button.setToolTip("Remove selected subject")
        self.remove_subject_button.setEnabled(False)
        subject_row.addWidget(self.subject_selector, 1)
        subject_row.addWidget(self.add_subject_button)
        subject_row.addWidget(self.remove_subject_button)
        layout.addLayout(subject_row)
        self.point_group = QButtonGroup(self)
        self.point_group.setExclusive(False)
        self.positive_button = QToolButton()
        self.positive_button.setText("Positive Point")
        self.positive_button.setCheckable(True)
        self.positive_button.setChecked(True)
        self.negative_button = QToolButton()
        self.negative_button.setText("Negative Point")
        self.negative_button.setCheckable(True)
        self.point_group.addButton(self.positive_button)
        self.point_group.addButton(self.negative_button)
        layout.addWidget(self.positive_button)
        layout.addWidget(self.negative_button)

        layout.addSpacing(10)
        layout.addWidget(QLabel("<b>Manual Brush</b>"))
        self.brush_group = QButtonGroup(self)
        self.brush_group.setExclusive(False)
        self.add_brush_button = QToolButton()
        self.add_brush_button.setText("Add Area")
        self.add_brush_button.setCheckable(True)
        self.remove_brush_button = QToolButton()
        self.remove_brush_button.setText("Subtract Area")
        self.remove_brush_button.setCheckable(True)
        self.brush_group.addButton(self.add_brush_button)
        self.brush_group.addButton(self.remove_brush_button)
        layout.addWidget(self.add_brush_button)
        layout.addWidget(self.remove_brush_button)
        self.brush_size = QSlider(Qt.Orientation.Horizontal)
        self.brush_size.setRange(2, 160)
        self.brush_size.setValue(24)
        self.brush_size_label = QLabel("Brush: 24 px")
        layout.addWidget(self.brush_size_label)
        layout.addWidget(self.brush_size)

        history = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        history.addWidget(self.undo_button)
        history.addWidget(self.redo_button)
        layout.addLayout(history)
        layout.addSpacing(10)
        layout.addWidget(QLabel("<b>Export</b>"))
        self.export_mask_button = QPushButton("Export Mask PNG Sequence")
        self.export_rgba_button = QPushButton("Export RGBA PNG Sequence")
        self.export_mask_video_button = QPushButton("Export B&W Mask MP4")
        self.export_cutout_video_button = QPushButton("Export Transparent Subject MOV")
        self.view_mode = QComboBox()
        self.view_mode.addItem("Video + Overlay", "overlay")
        self.view_mode.addItem("Original", "original")
        self.view_mode.addItem("B&W Mask", "mask")
        self.view_mode.addItem("Isolated Subject", "isolated")
        self.model_status = QLabel("SAM 2.1 Base+ · loading…")
        layout.addWidget(self.export_mask_button)
        layout.addWidget(self.export_rgba_button)
        layout.addWidget(self.export_mask_video_button)
        layout.addWidget(self.export_cutout_video_button)
        layout.addSpacing(8)
        layout.addWidget(QLabel("<b>Viewer</b>"))
        layout.addWidget(self.view_mode)
        layout.addStretch(1)
        layout.addWidget(self.model_status)
        layout.addWidget(QLabel("Zoom: scroll wheel · Pan: middle mouse button"))

        self.positive_button.clicked.connect(lambda: self._set_point_mode(1))
        self.negative_button.clicked.connect(lambda: self._set_point_mode(0))
        self.subject_selector.currentIndexChanged.connect(self._subject_changed)
        self.add_subject_button.clicked.connect(self._add_subject)
        self.remove_subject_button.clicked.connect(self._remove_subject)
        self.add_brush_button.clicked.connect(lambda checked: self._set_brush("add", checked))
        self.remove_brush_button.clicked.connect(lambda checked: self._set_brush("remove", checked))
        self.brush_size.valueChanged.connect(self._set_brush_size)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.export_mask_button.clicked.connect(lambda: self.export_current_range(False))
        self.export_rgba_button.clicked.connect(lambda: self.export_current_range(True))
        self.export_mask_video_button.clicked.connect(
            lambda: self.export_current_video("mask")
        )
        self.export_cutout_video_button.clicked.connect(
            lambda: self.export_current_video("alpha")
        )
        self.view_mode.currentIndexChanged.connect(
            lambda: self.viewer.set_display_mode(self.view_mode.currentData())
        )
        return panel

    def _start_worker(self) -> None:
        self.worker = InferenceWorker(self.paths)
        self.worker.loaded.connect(self._model_loaded)
        self.worker.failed.connect(self._show_error)
        self.worker.mask_ready.connect(self._mask_ready)
        self.worker.propagated_frame.connect(self._propagated_frame)
        self.worker.progress.connect(self._progress_changed)
        self.worker.propagation_finished.connect(self._propagation_finished)
        self.worker.start()
        self.worker.load_model()

    def _connect_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self._seek(self.current_frame - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._seek(self.current_frame + 1))
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo)
        QShortcut(QKeySequence("Meta+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Meta+Shift+Z"), self, activated=self.redo)
        self.segment_request = _QueuedSignal()
        self.propagate_request = _PropagateSignal()
        self.segment_request.requested.connect(self.worker.segment, Qt.ConnectionType.QueuedConnection)
        self.propagate_request.requested.connect(self.worker.propagate, Qt.ConnectionType.QueuedConnection)

    def _model_loaded(self, message: str) -> None:
        self.model_status.setText(message)
        self.statusBar().showMessage(message, 10000)

    def _set_point_mode(self, label: int) -> None:
        self.viewer.point_label = label
        self.viewer.brush_action = None
        self.positive_button.setChecked(label == 1)
        self.negative_button.setChecked(label == 0)
        self.add_brush_button.setChecked(False)
        self.remove_brush_button.setChecked(False)

    def _selected_subject(self) -> dict[str, str]:
        if self.project is None:
            return {"id": self.active_subject_id, "name": "Subject 1", "color": "#43e09b"}
        for subject in self.project.subjects:
            if subject["id"] == self.active_subject_id:
                return subject
        return self.project.subjects[0]

    def _refresh_subject_controls(self) -> None:
        subjects = self.project.subjects if self.project is not None else []
        self.subject_selector.blockSignals(True)
        self.subject_selector.clear()
        for subject in subjects:
            self.subject_selector.addItem(subject["name"], subject["id"])
            self.subject_selector.setItemData(
                self.subject_selector.count() - 1,
                QColor(subject["color"]),
                Qt.ItemDataRole.ForegroundRole,
            )
        selected_index = self.subject_selector.findData(self.active_subject_id)
        if selected_index < 0 and subjects:
            selected_index = 0
            self.active_subject_id = subjects[0]["id"]
        if selected_index >= 0:
            self.subject_selector.setCurrentIndex(selected_index)
        self.subject_selector.setEnabled(bool(subjects) and not self.propagating)
        self.add_subject_button.setEnabled(self.project is not None and not self.propagating)
        self.remove_subject_button.setEnabled(
            self.project is not None and len(subjects) > 1 and not self.propagating
        )
        self.subject_selector.blockSignals(False)
        self.viewer.overlay_color = self._selected_subject()["color"]
        self.viewer.update()

    def _subject_changed(self, index: int) -> None:
        subject_id = self.subject_selector.itemData(index)
        if subject_id:
            self.active_subject_id = subject_id
            self._refresh_subject_controls()
            self._refresh_overlay()

    def _add_subject(self) -> None:
        if self.project is None:
            self.statusBar().showMessage("Open a video or project first.", 4000)
            return
        try:
            subject = self.project.add_subject()
            self.active_subject_id = subject["id"]
            self._refresh_subject_controls()
            self._refresh_overlay()
            self.statusBar().showMessage(
                f"{subject['name']} added. Select it and add its prompts.",
                5000,
            )
        except (ProjectError, OSError) as exc:
            self._show_error(str(exc))

    def _remove_subject(self) -> None:
        if self.project is None or len(self.project.subjects) <= 1:
            return
        subject = self._selected_subject()
        answer = QMessageBox.question(
            self,
            "Remove Subject",
            f"Remove {subject['name']} and all of its masks and annotations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.project.remove_subject(subject["id"])
            for frame_index in list(self.undo_operations):
                self.undo_operations[frame_index] = [
                    action
                    for action in self.undo_operations[frame_index]
                    if action.get("subject_id") != subject["id"]
                ]
            for frame_index in list(self.redo_operations):
                self.redo_operations[frame_index] = [
                    action
                    for action in self.redo_operations[frame_index]
                    if action.get("subject_id") != subject["id"]
                ]
            self.active_subject_id = self.project.subjects[0]["id"]
            self._refresh_subject_controls()
            self._refresh_overlay()
        except (ProjectError, OSError) as exc:
            self._show_error(str(exc))

    def _set_brush(self, action: str, checked: bool) -> None:
        if checked:
            other = self.remove_brush_button if action == "add" else self.add_brush_button
            self.positive_button.setChecked(False)
            self.negative_button.setChecked(False)
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
            self.viewer.brush_action = action
        else:
            self.viewer.brush_action = None

    def _set_brush_size(self, size: int) -> None:
        self.viewer.brush_radius = max(1, size // 2)
        self.brush_size_label.setText(f"Brush: {size} px")

    def open_video(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Video",
                str(self.paths.root),
                "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;All Files (*)",
            )
        if not path:
            return
        try:
            info = probe_video(path)
            self.project = ProjectStore.create(self.paths.root, path, info)
            self.active_subject_id = self.project.subjects[0]["id"]
            self._refresh_subject_controls()
            self.project.set_mask_max_dimension(self.mask_resolution.currentData())
            self._reset_frame_cache()
            self.undo_operations.clear()
            self.redo_operations.clear()
            self.info = info
            self.current_frame = 0
            self.timeline.setRange(0, info.frame_count - 1)
            self.range_start.setRange(0, info.frame_count - 1)
            self.range_end.setRange(0, info.frame_count - 1)
            self.range_start.setValue(0)
            self.range_end.setValue(info.frame_count - 1)
            self.video_info_label.setText(
                f"{info.width}×{info.height} · {info.fps:.3f} fps · "
                f"{info.duration:.2f} s · {info.frame_count} frames"
            )
            self.setWindowTitle(f"Roto Local — {Path(path).name}")
            self._seek(0)
        except (MediaError, ProjectError, OSError) as exc:
            self._show_error(str(exc))

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(self.paths.projects),
            "Roto Local Project (project.json)",
        )
        if not path:
            return
        try:
            project = ProjectStore(Path(path).parent, self.paths.root)
            info = project.video_info
            self.mask_resolution.blockSignals(True)
            mask_resolution_index = self.mask_resolution.findData(
                project.data.get("mask_max_dimension")
            )
            if mask_resolution_index >= 0:
                self.mask_resolution.setCurrentIndex(mask_resolution_index)
            self.mask_resolution.blockSignals(False)
            self.project = project
            self.active_subject_id = project.subjects[0]["id"]
            self._refresh_subject_controls()
            self._reset_frame_cache()
            self.undo_operations = {}
            self.redo_operations = {}
            self.info = info
            self.current_frame = 0
            self.timeline.setRange(0, info.frame_count - 1)
            self.range_start.setRange(0, info.frame_count - 1)
            self.range_end.setRange(0, info.frame_count - 1)
            self.range_start.setValue(0)
            self.range_end.setValue(info.frame_count - 1)
            self.video_info_label.setText(
                f"{info.width}×{info.height} · {info.fps:.3f} fps · "
                f"{info.duration:.2f} s · {info.frame_count} frames"
            )
            self.setWindowTitle(f"Roto Local — {project.source.name}")
            self._seek(0)
        except (ProjectError, OSError, KeyError, TypeError, ValueError) as exc:
            self._show_error(str(exc))

    def save_project(self) -> None:
        if self.project is None:
            return
        try:
            self.project.save()
            self.statusBar().showMessage("Project saved", 4000)
        except OSError as exc:
            self._show_error(f"Could not save project: {exc}")

    def _mask_resolution_changed(self) -> None:
        if self.project is None:
            return
        try:
            self.project.set_mask_max_dimension(self.mask_resolution.currentData())
            if self.viewer.frame is not None:
                self._refresh_overlay()
        except (ProjectError, OSError) as exc:
            self._show_error(str(exc))

    def _reset_frame_cache(self) -> None:
        self._source_generation += 1
        for cache in self._frame_caches.values():
            cache.clear()
        for preview in self._frame_cache_bytes:
            self._frame_cache_bytes[preview] = 0
            self._frame_cache_sources[preview] = None
        self.pending_frame = None
        self.frame = None
        self._displayed_frame_index = None
        self.viewer.set_frame(None)
        self.viewer.set_overlay(None, [])

    def _set_frame_label(self, frame_index: int) -> None:
        state = "✓" if self.project and self.project.is_processed(frame_index) else "·"
        if self.info:
            self.frame_label.setText(
                f"{state} {frame_index + 1} / {self.info.frame_count}"
            )

    def _timeline_changed(self, frame_index: int) -> None:
        if self.info is None:
            return
        frame_index = min(max(frame_index, 0), self.info.frame_count - 1)
        self.current_frame = frame_index
        if self._displayed_frame_index != frame_index:
            self.frame = None
        self._set_frame_label(frame_index)
        if self.timeline.isSliderDown():
            self.preview_timer.start()
        else:
            self._seek(frame_index)

    def _timeline_released(self) -> None:
        self.preview_timer.stop()
        self._seek(self.current_frame)

    def _preview_timeline_frame(self) -> None:
        if self.timeline.isSliderDown():
            self._request_frame(self.current_frame, preview=True)

    def _seek(self, frame_index: int, preview: bool = False) -> None:
        if self.info is None:
            return
        frame_index = min(max(frame_index, 0), self.info.frame_count - 1)
        if self._displayed_frame_index != frame_index:
            self.frame = None
        self.current_frame = frame_index
        if self.timeline.value() != frame_index:
            self.timeline.blockSignals(True)
            self.timeline.setValue(frame_index)
            self.timeline.blockSignals(False)
        self._set_frame_label(frame_index)
        self._request_frame(frame_index, preview=preview)

    def _cache_frame(
        self,
        frame_index: int,
        frame: np.ndarray,
        preview: bool,
        source: Path,
    ) -> None:
        if self._frame_cache_sources[preview] != source:
            self._frame_caches[preview].clear()
            self._frame_cache_bytes[preview] = 0
            self._frame_cache_sources[preview] = source
        cache = self._frame_caches[preview]
        previous = cache.pop(frame_index, None)
        if previous is not None:
            self._frame_cache_bytes[preview] -= previous.nbytes
        cache[frame_index] = frame
        self._frame_cache_bytes[preview] += frame.nbytes
        while (
            self._frame_cache_bytes[preview] > self._frame_cache_limits[preview]
            and len(cache) > 1
        ):
            _, discarded = cache.popitem(last=False)
            self._frame_cache_bytes[preview] -= discarded.nbytes

    def _cached_frame(self, frame_index: int, preview: bool) -> np.ndarray | None:
        cache = self._frame_caches[preview]
        frame = cache.get(frame_index)
        if frame is not None:
            cache.move_to_end(frame_index)
        return frame

    def _request_frame(self, frame_index: int, preview: bool) -> None:
        if self.info is None or self.project is None:
            return
        source = self.project.source.resolve()
        if self._frame_cache_sources[preview] != source:
            self._frame_caches[preview].clear()
            self._frame_cache_bytes[preview] = 0
            self._frame_cache_sources[preview] = source
        cached = self._cached_frame(frame_index, preview)
        if cached is None and preview:
            cached = self._cached_frame(frame_index, preview=False)
            if cached is not None:
                preview = False
        if cached is not None:
            self.pending_frame = None
            self._show_frame(frame_index, cached, preview)
            return
        if not preview:
            preview_frame = self._cached_frame(frame_index, preview=True)
            if preview_frame is not None:
                self._show_frame(frame_index, preview_frame, preview=True)
        if self.frame_busy:
            self.pending_frame = (frame_index, preview)
            return
        self.frame_busy = True
        task = FrameTask(
            source,
            self.info,
            frame_index,
            self._source_generation,
            preview,
        )
        task.signals.frame_ready.connect(self._frame_batch_ready)
        task.signals.frame_failed.connect(self._frame_failed)
        QThreadPool.globalInstance().start(task)

    def _frame_batch_ready(
        self,
        generation: int,
        requested_frame: int,
        preview: bool,
        first_frame: int,
        frames: list[np.ndarray],
    ) -> None:
        self.frame_busy = False
        if generation == self._source_generation and self.project is not None:
            source = self.project.source.resolve()
            for offset, frame in enumerate(frames):
                self._cache_frame(first_frame + offset, frame, preview, source)
            if requested_frame == self.current_frame and (
                not preview or self.timeline.isSliderDown()
            ):
                frame = self._cached_frame(requested_frame, preview)
                if frame is not None:
                    self._show_frame(requested_frame, frame, preview)
            elif preview and requested_frame == self.current_frame:
                self.pending_frame = (requested_frame, False)
        if self.pending_frame is not None:
            next_frame, next_preview = self.pending_frame
            self.pending_frame = None
            self._request_frame(next_frame, next_preview)

    def _frame_failed(
        self,
        generation: int,
        frame_index: int,
        message: str,
    ) -> None:
        self.frame_busy = False
        if generation == self._source_generation and frame_index == self.current_frame:
            self._show_error(message)
        if self.pending_frame is not None:
            next_frame, next_preview = self.pending_frame
            self.pending_frame = None
            self._request_frame(next_frame, next_preview)

    def _show_frame(self, frame_index: int, frame: np.ndarray, preview: bool) -> None:
        if frame_index != self.current_frame:
            return
        self.frame = None if preview else frame
        self._displayed_frame_index = frame_index
        self.viewer.set_frame(frame)
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        if self.project is None or self.info is None or self.viewer.frame is None:
            self.viewer.set_overlay(None, [])
            return
        display_height, display_width = self.viewer.frame.shape[:2]
        scale_x = display_width / self.info.width
        scale_y = display_height / self.info.height
        overlays = []
        for subject, mask in self.project.masks_by_subject(self.current_frame):
            if mask.shape != (display_height, display_width):
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (display_width, display_height),
                    Image.Resampling.NEAREST,
                )
                mask = np.asarray(mask_image, dtype=np.uint8) > 0
            points = [
                {
                    **point,
                    "x": point["x"] * scale_x,
                    "y": point["y"] * scale_y,
                }
                for point in self.project.points(self.current_frame, subject["id"])
            ]
            overlays.append(
                {
                    "mask": mask,
                    "points": points,
                    "color": subject["color"],
                }
            )
        self.viewer.set_subject_overlays(overlays)

    def _add_point(self, x: float, y: float, label: int) -> None:
        if self.project is None:
            return
        if self.frame is None:
            self.statusBar().showMessage(
                "Wait for the full-resolution frame to load.", 3000
            )
            return
        subject = self._selected_subject()
        points = self.project.points(self.current_frame, subject["id"])
        if label == 0 and not any(point["label"] == 1 for point in points):
            self.statusBar().showMessage(
                "Add a positive point on the subject first.", 4000
            )
            return
        point = {"x": round(x, 2), "y": round(y, 2), "label": label}
        points.append(point)
        self.undo_operations.setdefault(self.current_frame, []).append(
            {"kind": "point", "value": point, "subject_id": subject["id"]}
        )
        self.redo_operations[self.current_frame] = []
        self.project.set_points(self.current_frame, points, subject["id"])
        self._refresh_overlay()
        self.segment_request.requested.emit(
            self.current_frame,
            self.frame,
            points,
            subject["id"],
            self._source_generation,
        )
        self.statusBar().showMessage(
            "Updating mask… " + ("include" if label else "exclude"),
            3000,
        )

    def _mask_ready(
        self,
        frame_index: int,
        mask: np.ndarray,
        subject_id: str,
        generation: int,
    ) -> None:
        if (
            self.project is None
            or generation != self._source_generation
            or subject_id not in {subject["id"] for subject in self.project.subjects}
        ):
            return
        try:
            self.project.save_ai_mask(frame_index, mask, subject_id=subject_id)
            if frame_index == self.current_frame:
                self._refresh_overlay()
            self.statusBar().showMessage("Mask updated", 3000)
        except ProjectError as exc:
            self._show_error(str(exc))

    def _add_brush_stroke(self, points, action: str, radius: int) -> None:
        if self.project is None or self.frame is None:
            if self.project is not None:
                self.statusBar().showMessage(
                    "Wait for the full-resolution frame to load.", 3000
                )
            return
        frame_index = self.current_frame
        subject = self._selected_subject()
        operations = self.project.manual_ops(frame_index, subject["id"])
        operation = {
            "action": action,
            "radius": radius,
            "points": [[round(x, 2), round(y, 2)] for x, y in points],
        }
        operations.append(operation)
        self.undo_operations.setdefault(frame_index, []).append(
            {"kind": "brush", "value": operation, "subject_id": subject["id"]}
        )
        self.redo_operations[frame_index] = []
        self.project.set_manual_ops(frame_index, operations, subject["id"])
        self._refresh_overlay()

    def undo(self) -> None:
        self._change_history(undo=True)

    def redo(self) -> None:
        self._change_history(undo=False)

    def _change_history(self, undo: bool) -> None:
        if self.project is None:
            return
        frame_index = self.current_frame
        if self.frame is None and self.undo_operations.get(frame_index):
            self.statusBar().showMessage(
                "Wait for the full-resolution frame to load.", 3000
            )
            return
        undo_stack = self.undo_operations.setdefault(frame_index, [])
        redo_stack = self.redo_operations.setdefault(frame_index, [])
        if undo:
            if not undo_stack:
                return
            action = undo_stack.pop()
            redo_stack.append(action)
        else:
            if not redo_stack:
                return
            action = redo_stack.pop()
            undo_stack.append(action)

        if action["kind"] == "point":
            subject_id = action["subject_id"]
            points = self.project.points(frame_index, subject_id)
            if undo:
                for index in range(len(points) - 1, -1, -1):
                    if points[index] == action["value"]:
                        points.pop(index)
                        break
            else:
                points.append(action["value"])
            self.project.set_points(frame_index, points, subject_id)
            if any(point["label"] == 1 for point in points) and self.frame is not None:
                self.segment_request.requested.emit(
                    frame_index,
                    self.frame,
                    points,
                    subject_id,
                    self._source_generation,
                )
            else:
                self.project.clear_ai_mask(frame_index, subject_id)
        else:
            subject_id = action["subject_id"]
            operations = self.project.manual_ops(frame_index, subject_id)
            if undo:
                for index in range(len(operations) - 1, -1, -1):
                    if operations[index] == action["value"]:
                        operations.pop(index)
                        break
            else:
                operations.append(action["value"])
            self.project.set_manual_ops(frame_index, operations, subject_id)
        self._refresh_overlay()

    def _start_propagation(self) -> None:
        if self.project is None or self.propagating:
            return
        first, last = self.range_start.value(), self.range_end.value()
        if first > last:
            self._show_error("The first frame in the range must come before the last frame.")
            return
        start = self.current_frame
        if not first <= start <= last:
            self._show_error(
                "Select a starting frame within the chosen range."
            )
            return
        for subject in self.project.subjects:
            prompts = self.project.points(start, subject["id"])
            if not any(point["label"] == 1 for point in prompts):
                self._show_error(
                    f"Add at least one positive point for {subject['name']} "
                    "on the selected frame."
                )
                self.subject_selector.setCurrentIndex(
                    self.subject_selector.findData(subject["id"])
                )
                return
        direction = self.tracking_direction.currentData()
        try:
            self.project.set_mask_max_dimension(self.mask_resolution.currentData())
        except (ProjectError, OSError) as exc:
            self._show_error(str(exc))
            return
        self.propagating = True
        self._refresh_subject_controls()
        self.progress_started_at = time.monotonic()
        self.last_propagation_preview_at = 0.0
        self.latest_propagated_frame = None
        self.cancel_button.setEnabled(True)
        self.propagate_button.setEnabled(False)
        self.progress.setValue(0)
        self.propagate_request.requested.emit(
            str(self.project.folder),
            first,
            last,
            start,
            direction,
            int(self.window_size.currentData()),
        )
        direction_label = self.tracking_direction.currentText().lower()
        self.statusBar().showMessage(
            f"Tracking {direction_label} from frame {start + 1} — "
            "manual corrections recommended"
        )

    def _cancel_propagation(self) -> None:
        self.worker.cancel()
        self.statusBar().showMessage("Cancellation requested…")

    def _progress_changed(self, processed: int, total: int) -> None:
        percent = int(100 * processed / max(total, 1))
        self.progress.setValue(percent)
        remaining = ""
        if self.progress_started_at is not None and processed > 0:
            elapsed = time.monotonic() - self.progress_started_at
            rate = processed / max(elapsed, 0.001)
            seconds_left = int(max(total - processed, 0) / rate)
            remaining = f" · ~{seconds_left}s remaining"
        self.statusBar().showMessage(
            f"Tracking: {percent}% · {processed}/{total} frames{remaining}"
        )

    def _propagated_frame(self, frame_index: int) -> None:
        self.latest_propagated_frame = frame_index
        now = time.monotonic()
        if now - self.last_propagation_preview_at >= 0.35:
            self.last_propagation_preview_at = now
            self._seek(frame_index, preview=True)

    def _propagation_finished(self) -> None:
        latest_frame = self.latest_propagated_frame
        self.propagating = False
        self._refresh_subject_controls()
        self.progress_started_at = None
        self.cancel_button.setEnabled(False)
        self.propagate_button.setEnabled(True)
        if latest_frame is not None:
            self._seek(latest_frame)
        self.latest_propagated_frame = None

    def export_current_range(self, rgba: bool) -> None:
        if self.project is None:
            self._show_error("Open a video or project first.")
            return
        start, end = self.range_start.value(), self.range_end.value()
        kind = "rgba" if rgba else "masks"
        suggested = self.paths.exports / self.project.folder.name / kind
        destination_text = QFileDialog.getExistingDirectory(
            self,
            "Choose an export folder",
            str(suggested),
        )
        if not destination_text:
            return
        destination = Path(destination_text)
        task = ExportTask(
            self.project.folder,
            self.paths.root,
            start,
            end,
            destination,
            rgba,
        )
        task.signals.failed.connect(self._show_error)
        task.signals.export_done.connect(
            lambda output: self.statusBar().showMessage(f"Exported: {output}", 10000)
        )
        task.signals.export_progress.connect(
            lambda count, total: self.progress.setValue(int(100 * count / max(total, 1)))
        )
        QThreadPool.globalInstance().start(task)
        self.statusBar().showMessage("Exporting…")

    def export_current_video(self, mode: str) -> None:
        if self.video_exporting:
            self.statusBar().showMessage("A video export is already in progress.")
            return
        if self.project is None:
            self._show_error("Open a video or project first.")
            return
        if mode not in {"mask", "alpha"}:
            self._show_error("Invalid video export format.")
            return
        start, end = self.range_start.value(), self.range_end.value()
        suffix = ".mp4" if mode == "mask" else ".mov"
        name = "mask_bw" if mode == "mask" else "subject_alpha"
        suggested = (
            self.paths.exports
            / f"{name}_{start + 1:08d}-{end + 1:08d}{suffix}"
        )
        title = (
            "Export B&W Mask MP4"
            if mode == "mask"
            else "Export Transparent Subject MOV"
        )
        file_filter = (
            "MP4 Video (*.mp4)" if mode == "mask" else "MOV Video (*.mov)"
        )
        destination_text, _ = QFileDialog.getSaveFileName(
            self,
            title,
            str(suggested),
            file_filter,
        )
        if not destination_text:
            return
        destination = Path(destination_text).expanduser()
        if destination.suffix.lower() != suffix:
            if destination.suffix:
                self._show_error(f"The file must use the {suffix} extension.")
                return
            destination = destination.with_suffix(suffix)
        destination = destination.resolve()
        task = VideoExportTask(
            self.project.folder,
            self.paths.root,
            start,
            end,
            destination,
            mode,
        )
        task.signals.failed.connect(self._show_error)
        task.signals.export_done.connect(
            lambda output: self.statusBar().showMessage(f"Exported: {output}", 10000)
        )
        task.signals.export_progress.connect(
            lambda count, total: self.progress.setValue(
                int(100 * count / max(total, 1))
            )
        )
        task.signals.export_finished.connect(self._video_export_finished)
        self.video_exporting = True
        self.export_mask_video_button.setEnabled(False)
        self.export_cutout_video_button.setEnabled(False)
        QThreadPool.globalInstance().start(task)
        self.statusBar().showMessage("Exporting video…")

    def _video_export_finished(self) -> None:
        self.video_exporting = False
        self.export_mask_video_button.setEnabled(True)
        self.export_cutout_video_button.setEnabled(True)

    def toggle_play(self) -> None:
        if self.info is None:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play")
        else:
            self.play_timer.start(max(1, round(1000 / self.info.fps)))
            self.play_button.setText("Pause")

    def _advance_playback(self) -> None:
        end = self.range_end.value() if self.info else 0
        if self.current_frame >= end:
            self.play_timer.stop()
            self.play_button.setText("Play")
        else:
            self._seek(self.current_frame + 1)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.open_video(urls[0].toLocalFile())

    def _show_error(self, message: str) -> None:
        logging.error(message)
        self.statusBar().showMessage(message, 12000)
        QMessageBox.critical(self, "Roto Local", message)

    def closeEvent(self, event) -> None:
        self.save_project()
        self.worker.stop()
        QThreadPool.globalInstance().waitForDone()
        super().closeEvent(event)


class _QueuedSignal(QObject):
    requested = Signal(int, object, object, str, int)


class _PropagateSignal(QObject):
    requested = Signal(str, int, int, int, str, int)


def main(root: Path) -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Roto Local")
    window = MainWindow(root)
    window.show()
    return app.exec()
