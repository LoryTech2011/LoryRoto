import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from rotoscope.config import AppPaths
from rotoscope.media import VideoInfo
from rotoscope.project import ProjectError, ProjectStore


ROOT = Path(__file__).resolve().parents[1]


class ProjectStoreTests(unittest.TestCase):
    def test_subject_masks_and_manual_corrections_are_independent(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source reference only")
            info = VideoInfo(16, 12, 24.0, 1.0, 24, "h264")
            project = ProjectStore.create(root, source, info)
            second = project.add_subject()
            first_mask = np.zeros((12, 16), dtype=bool)
            first_mask[2:5, 2:5] = True
            second_mask = np.zeros((12, 16), dtype=bool)
            second_mask[7:10, 10:13] = True
            project.save_ai_mask(0, first_mask)
            project.save_ai_mask(0, second_mask, subject_id=second["id"])
            project.set_manual_ops(
                0,
                [{"action": "add", "radius": 1, "points": [[5.0, 5.0]]}],
                second["id"],
            )

            self.assertTrue(project.subject_mask(0)[3, 3])
            self.assertFalse(project.subject_mask(0)[5, 5])
            self.assertTrue(project.subject_mask(0, second["id"])[5, 5])
            self.assertTrue(project.final_mask(0)[3, 3])
            self.assertTrue(project.final_mask(0)[8, 11])
            self.assertEqual(len(list(project.masks_by_subject(0))), 2)
            self.assertTrue(project.is_processed(0))

            project.remove_subject(second["id"])
            self.assertFalse(project.final_mask(0)[8, 11])
            self.assertTrue(project.final_mask(0)[3, 3])
            self.assertTrue(project.is_processed(0))

    def test_legacy_single_mask_project_is_migrated_to_first_subject(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source reference only")
            info = VideoInfo(16, 12, 24.0, 1.0, 24, "h264")
            project = ProjectStore.create(root, source, info)
            mask = np.zeros((12, 16), dtype=bool)
            mask[3:8, 5:10] = True
            project.save_ai_mask(0, mask)
            (project.folder / "masks" / "00000000.png").unlink()
            project.mask_path(0).unlink()
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                project.folder / "masks" / "00000000.png"
            )
            for key in ("subjects", "subject_points", "subject_manual_ops"):
                project.data.pop(key)
            project.save()

            reopened = ProjectStore(project.folder, root)

            np.testing.assert_array_equal(reopened.ai_mask(0), mask)
            self.assertTrue(reopened.mask_path(0).is_file())
            self.assertTrue(reopened.combined_mask_path(0).is_file())

    def test_clearing_subject_mask_removes_stale_combined_mask(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source reference only")
            info = VideoInfo(16, 12, 24.0, 1.0, 24, "h264")
            project = ProjectStore.create(root, source, info)
            mask = np.ones((12, 16), dtype=bool)
            project.save_ai_mask(0, mask)

            project.clear_ai_mask(0)

            self.assertFalse(project.mask_path(0).exists())
            self.assertFalse(project.combined_mask_path(0).exists())
            self.assertFalse(project.ai_mask(0).any())

    def test_combines_subject_masks_saved_at_different_resolutions(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source reference only")
            info = VideoInfo(3840, 2160, 24.0, 1.0, 24, "h264")
            project = ProjectStore.create(root, source, info)
            first = np.zeros((2160, 3840), dtype=bool)
            first[100:300, 100:300] = True
            project.save_ai_mask(0, first)
            project.set_mask_max_dimension(1920)
            second = project.add_subject()
            second_mask = np.zeros((2160, 3840), dtype=bool)
            second_mask[1600:1900, 3000:3300] = True
            project.save_ai_mask(0, second_mask, subject_id=second["id"])

            combined = project.final_mask(0)

            self.assertEqual(combined.shape, (1080, 1920))
            self.assertTrue(combined[100, 100])
            self.assertTrue(combined[875, 1575])

    def test_ai_mask_save_can_defer_manifest_write(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            source = root / "source.mov"
            source.write_bytes(b"source reference only")
            info = VideoInfo(16, 12, 24.0, 1.0, 24, "h264")
            project = ProjectStore.create(root, source, info)
            mask = np.ones((12, 16), dtype=bool)

            with patch.object(project, "save", wraps=project.save) as save:
                project.save_ai_mask(0, mask, persist_project=False)
                save.assert_not_called()
                self.assertTrue(project.is_processed(0))
                project.save()
                save.assert_called_once()

    def test_final_mask_skips_manual_rasterization_without_corrections(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            source = root / "source.mov"
            source.write_bytes(b"source reference only")
            info = VideoInfo(16, 12, 24.0, 1.0, 24, "h264")
            project = ProjectStore.create(root, source, info)
            mask = np.zeros((12, 16), dtype=bool)
            mask[3:8, 4:9] = True
            project.save_ai_mask(0, mask)

            with patch.object(project, "manual_masks") as manual_masks:
                np.testing.assert_array_equal(project.final_mask(0), mask)
                manual_masks.assert_not_called()

    def test_1080p_mask_storage_and_full_resolution_export_mask(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source reference only")
            info = VideoInfo(3840, 2160, 24.0, 1.0, 24, "h264")
            project = ProjectStore.create(root, source, info)
            project.set_mask_max_dimension(1920)

            mask = np.zeros((2160, 3840), dtype=bool)
            mask[400:1600, 800:2800] = True
            project.save_ai_mask(0, mask)

            with Image.open(project.mask_path(0)) as saved:
                self.assertEqual(saved.size, (1920, 1080))
            self.assertEqual(project.ai_mask(0).shape, (1080, 1920))
            project.set_manual_ops(
                0,
                [
                    {
                        "action": "add",
                        "radius": 50,
                        "points": [[3500.0, 2000.0]],
                    }
                ],
            )
            final_mask = project.final_mask(0)
            self.assertEqual(final_mask.shape, (1080, 1920))
            self.assertTrue(final_mask[1000, 1750])
            self.assertEqual(
                project.mask_at_video_resolution(0).shape,
                (2160, 3840),
            )
            self.assertTrue(project.mask_at_video_resolution(0)[2000, 3500])

    def test_masks_and_manual_corrections_survive_reload(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            source = root / "source.mov"
            source.write_bytes(b"source reference only")
            info = VideoInfo(16, 12, 24.0, 1.0, 24, "h264")
            project = ProjectStore.create(root, source, info)

            ai_mask = np.zeros((12, 16), dtype=bool)
            ai_mask[2:7, 3:9] = True
            project.save_ai_mask(0, ai_mask)
            project.set_points(0, [{"x": 5.0, "y": 4.0, "label": 1}])
            project.set_manual_ops(
                0,
                [{"action": "add", "radius": 1, "points": [[12.0, 9.0]]}],
            )

            reopened = ProjectStore(project.folder, root)
            final_mask = reopened.final_mask(0)
            self.assertTrue(final_mask[4, 5])
            self.assertTrue(final_mask[9, 12])
            self.assertEqual(reopened.points(0)[0]["label"], 1)
            self.assertEqual(reopened.data["status"]["0"], "review")
            self.assertEqual(reopened.source, source)
            self.assertFalse((project.folder / source.name).exists())

    def test_project_rejects_paths_outside_application_root(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            project_dir = root / "projects" / "outside"
            project_dir.mkdir(parents=True)
            with self.assertRaises(ProjectError):
                ProjectStore(project_dir, root / "different-root")

    def test_app_paths_reject_symlinked_runtime_outside_root(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            base = Path(temporary)
            root = base / "app"
            root.mkdir()
            external = base / "outside"
            external.mkdir()
            (root / "runtime").symlink_to(external, target_is_directory=True)

            with self.assertRaises(OSError):
                AppPaths(root).ensure()
            self.assertEqual(list(external.iterdir()), [])

    def test_project_creation_rejects_symlinked_projects_folder(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            base = Path(temporary)
            root = base / "app"
            root.mkdir()
            external = base / "outside"
            external.mkdir()
            (root / "projects").symlink_to(external, target_is_directory=True)
            source = base / "source.mov"
            source.write_bytes(b"source reference only")
            info = VideoInfo(16, 12, 24.0, 1.0, 24, "h264")

            with self.assertRaises(ProjectError):
                ProjectStore.create(root, source, info)
            self.assertEqual(list(external.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
