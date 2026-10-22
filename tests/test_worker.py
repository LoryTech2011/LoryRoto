import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rotoscope.config import AppPaths
from rotoscope.media import VideoInfo
from rotoscope.project import ProjectStore
from rotoscope.worker import InferenceWorker


ROOT = Path(__file__).resolve().parents[1]


class _FakeEngine:
    def propagate(self, project, first_frame, last_frame, *args):
        for frame_index in range(first_frame, last_frame + 1):
            yield (
                frame_index,
                np.full(
                    (project.video_info.height, project.video_info.width),
                    frame_index % 2,
                    dtype=bool,
                ),
                frame_index - first_frame + 1,
                last_frame - first_frame + 1,
            )


class InferenceWorkerTests(unittest.TestCase):
    def test_masks_save_per_frame_but_manifest_saves_once(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            paths = AppPaths(root)
            paths.ensure()
            source = root / "source.mp4"
            source.write_bytes(b"synthetic source")
            info = VideoInfo(16, 12, 24.0, 1.0, 3, "h264")
            project = ProjectStore.create(root, source, info)

            worker = InferenceWorker(paths)
            worker.engine = _FakeEngine()
            original_save = ProjectStore.save
            manifest_writes = []

            def counted_save(store):
                manifest_writes.append(store.manifest_path)
                original_save(store)

            with patch.object(ProjectStore, "save", counted_save):
                worker._propagate(
                    str(project.folder),
                    0,
                    2,
                    0,
                    "forward",
                    12,
                )

            self.assertEqual(len(manifest_writes), 1)
            reopened = ProjectStore(project.folder, root)
            self.assertEqual(
                [reopened.is_processed(index) for index in range(3)],
                [True, True, True],
            )
            self.assertEqual(
                [reopened.data["status"][str(index)] for index in range(3)],
                ["processed", "processed", "processed"],
            )


if __name__ == "__main__":
    unittest.main()
