import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rotoscope.config import AppPaths
from rotoscope.inference import RotoEngine
from rotoscope.media import VideoInfo
from rotoscope.project import ProjectStore


ROOT = Path(__file__).resolve().parents[1]


class _Logits:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return _Logits(self.values[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakePredictor:
    def __init__(self):
        self.calls = []
        self.seed_masks = 0
        self.states = []
        self.points = []

    def init_state(self, frame_folder, **kwargs):
        state = {
            "num_frames": len(list(Path(frame_folder).glob("*.jpg"))),
            "video_height": 0,
            "video_width": 0,
        }
        self.states.append(state)
        return state

    def add_new_points_or_box(self, state, frame_index, object_id, points, labels):
        self.anchor = frame_index
        self.points.append(np.asarray(points))

    def add_new_mask(self, state, frame_index, object_id, mask):
        self.seed_masks += 1

    def propagate_in_video(
        self,
        state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
    ):
        self.calls.append((state["num_frames"], start_frame_idx, reverse))
        if reverse:
            indices = range(start_frame_idx, max(start_frame_idx - max_frame_num_to_track, 0) - 1, -1)
        else:
            end = min(
                start_frame_idx + max_frame_num_to_track,
                state["num_frames"] - 1,
            )
            indices = range(start_frame_idx, end + 1)
        for frame_index in indices:
            logits = _Logits(np.ones((1, 1, 12, 16), dtype=np.float32))
            yield frame_index, [1], logits

    def reset_state(self, state):
        pass


class _FakeImagePredictor:
    def __init__(self):
        self.reset_count = 0

    def reset_predictor(self):
        self.reset_count += 1


class _CandidateImagePredictor:
    def __init__(self, masks, scores):
        self.masks = masks
        self.scores = scores
        self.calls = []

    def set_image(self, image):
        pass

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.masks, self.scores, np.arange(len(self.masks))[:, None, None]


class InferenceDirectionTests(unittest.TestCase):
    def test_segment_uses_single_mask_for_multiple_prompts(self):
        masks = np.zeros((3, 20, 20), dtype=bool)
        masks[0, 8, 3] = True
        masks[0, 8, 10] = True
        masks[1, 8, 3] = True
        masks[2, 8, 10] = True
        engine = RotoEngine(AppPaths(ROOT))
        predictor = _CandidateImagePredictor(masks[:1], np.asarray([0.8]))
        engine.image_predictor = predictor
        points = [
            {"x": 3.0, "y": 8.0, "label": 1},
            {"x": 10.0, "y": 8.0, "label": 1},
            {"x": 16.0, "y": 8.0, "label": 0},
        ]

        result = engine.segment(0, np.zeros((20, 20, 3), dtype=np.uint8), points)

        self.assertTrue(result[8, 3])
        self.assertTrue(result[8, 10])
        self.assertFalse(predictor.calls[0]["multimask_output"])
        self.assertEqual(engine._previous_logits.item(), 0.0)

    def test_segment_uses_iou_score_for_single_prompt_candidates(self):
        masks = np.zeros((3, 20, 20), dtype=bool)
        masks[2, 8, 3] = True
        engine = RotoEngine(AppPaths(ROOT))
        predictor = _CandidateImagePredictor(masks, np.asarray([0.2, 0.6, 0.9]))
        engine.image_predictor = predictor

        result = engine.segment(
            0,
            np.zeros((20, 20, 3), dtype=np.uint8),
            [{"x": 3.0, "y": 8.0, "label": 1}],
        )

        self.assertTrue(result[8, 3])
        self.assertTrue(predictor.calls[0]["multimask_output"])
        self.assertEqual(engine._previous_logits.item(), 2.0)

    def test_propagation_uses_only_selected_subject_prompts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            paths = AppPaths(root)
            paths.ensure()
            source = root / "source.mp4"
            source.write_bytes(b"synthetic source")
            info = VideoInfo(16, 12, 24.0, 0.5, 12, "h264")
            project = ProjectStore.create(root, source, info)
            second = project.add_subject()
            project.set_points(
                0,
                [{"x": 3.0, "y": 4.0, "label": 1}],
                "subject-1",
            )
            project.set_points(
                0,
                [{"x": 12.0, "y": 8.0, "label": 1}],
                second["id"],
            )
            predictor = _FakePredictor()
            engine = RotoEngine(paths)
            engine.video_predictor = predictor

            def extract_window(source, video_info, first_frame, frame_count, output_dir):
                folder = Path(output_dir)
                for index in range(frame_count):
                    (folder / f"{index:05d}.jpg").write_bytes(b"frame")

            with patch("rotoscope.inference.extract_jpeg_window", extract_window):
                list(
                    engine.propagate(
                        project,
                        0,
                        11,
                        0,
                        "forward",
                        12,
                        threading.Event(),
                        second["id"],
                    )
                )

            np.testing.assert_array_equal(predictor.points[0], [[12.0, 8.0]])

    def test_1080p_tracking_scales_state_and_prompt_coordinates(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            paths = AppPaths(root)
            paths.ensure()
            source = root / "source.mp4"
            source.write_bytes(b"synthetic source")
            info = VideoInfo(3840, 2160, 24.0, 0.5, 12, "h264")
            project = ProjectStore.create(root, source, info)
            project.set_mask_max_dimension(1920)
            project.set_points(
                0,
                [{"x": 1920.0, "y": 1080.0, "label": 1}],
            )
            predictor = _FakePredictor()
            engine = RotoEngine(paths)
            engine.video_predictor = predictor

            def extract_window(source, video_info, first_frame, frame_count, output_dir):
                folder = Path(output_dir)
                for index in range(frame_count):
                    (folder / f"{index:05d}.jpg").write_bytes(b"frame")

            with patch("rotoscope.inference.extract_jpeg_window", extract_window):
                list(
                    engine.propagate(
                        project,
                        0,
                        11,
                        0,
                        "forward",
                        12,
                        threading.Event(),
                    )
                )

            self.assertEqual(
                (predictor.states[0]["video_height"], predictor.states[0]["video_width"]),
                (1080, 1920),
            )
            np.testing.assert_array_equal(predictor.points[0], [[960.0, 540.0]])

    def test_tracking_directions_start_at_selected_frame_and_cover_range(self):
        expected_frames = {
            "forward": list(range(14, 28)),
            "reverse": list(range(14, -1, -1)),
            "both": list(range(14, 28)) + list(range(13, -1, -1)),
        }
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            paths = AppPaths(root)
            paths.ensure()
            source = root / "source.mp4"
            source.write_bytes(b"synthetic source")
            info = VideoInfo(16, 12, 24.0, 28 / 24, 28, "h264")
            project = ProjectStore.create(root, source, info)
            project.set_points(14, [{"x": 5.0, "y": 5.0, "label": 1}])

            for direction, expected in expected_frames.items():
                with self.subTest(direction=direction):
                    predictor = _FakePredictor()
                    engine = RotoEngine(paths)
                    engine.video_predictor = predictor
                    image_predictor = _FakeImagePredictor()
                    engine.image_predictor = image_predictor
                    engine._active_image = np.zeros((2, 2, 3), dtype=np.uint8)
                    engine._active_image_index = 14
                    engine._previous_logits = np.ones((1, 1, 1), dtype=np.float32)

                    def extract_window(source, video_info, first_frame, frame_count, output_dir):
                        folder = Path(output_dir)
                        for index in range(frame_count):
                            (folder / f"{index:05d}.jpg").write_bytes(b"frame")

                    with patch("rotoscope.inference.extract_jpeg_window", extract_window):
                        results = list(
                            engine.propagate(
                                project,
                                0,
                                27,
                                14,
                                direction,
                                12,
                                threading.Event(),
                            )
                        )

                    self.assertEqual(
                        [frame_index for frame_index, _, _, _ in results],
                        expected,
                    )
                    self.assertEqual([progress for _, _, progress, _ in results], list(range(1, len(expected) + 1)))
                    self.assertTrue(all(total == len(expected) for _, _, _, total in results))
                    self.assertTrue(all(mask.shape == (12, 16) for _, mask, _, _ in results))
                    self.assertGreater(predictor.seed_masks, 0)
                    self.assertEqual(
                        any(reverse for _, _, reverse in predictor.calls),
                        direction in {"reverse", "both"},
                    )
                    self.assertEqual(image_predictor.reset_count, 1)
                    self.assertIsNone(engine._active_image)
                    self.assertIsNone(engine._previous_logits)

    def test_reverse_tracking_at_first_frame_preserves_seed_mask(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            root = Path(temporary)
            paths = AppPaths(root)
            paths.ensure()
            source = root / "source.mp4"
            source.write_bytes(b"synthetic source")
            info = VideoInfo(16, 12, 24.0, 1.0, 24, "h264")
            project = ProjectStore.create(root, source, info)
            project.set_points(0, [{"x": 5.0, "y": 5.0, "label": 1}])
            seed = np.zeros((12, 16), dtype=bool)
            seed[2:7, 3:9] = True
            project.save_ai_mask(0, seed)

            engine = RotoEngine(paths)
            result = list(
                engine.propagate(
                    project,
                    0,
                    23,
                    0,
                    "reverse",
                    12,
                    threading.Event(),
                )
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][0], 0)
            np.testing.assert_array_equal(result[0][1], seed)
            self.assertEqual((result[0][2], result[0][3]), (1, 1))


if __name__ == "__main__":
    unittest.main()
