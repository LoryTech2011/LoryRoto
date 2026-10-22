import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rotoscope.exporter import export_sequence, export_video_sequence
from rotoscope.media import decode_frame, decode_frame_window, probe_video
from rotoscope.project import ProjectStore


ROOT = Path(__file__).resolve().parents[1]


class MediaExportTests(unittest.TestCase):
    def test_ffmpeg_decode_and_rgba_export(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            temp_root = Path(temporary)
            frames_dir = temp_root / "input"
            frames_dir.mkdir()
            for index in range(4):
                image = Image.new("RGB", (32, 24), (10, 20, 30))
                ImageDraw.Draw(image).rectangle((index * 3, 4, index * 3 + 8, 18), fill=(240, 80, 40))
                image.save(frames_dir / f"{index:05d}.png")
            source = temp_root / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    "24",
                    "-i",
                    str(frames_dir / "%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            info = probe_video(source)
            self.assertEqual(info.width, 32)
            self.assertEqual(info.height, 24)
            decoded = decode_frame(source, 2, info)
            self.assertEqual(decoded.shape, (24, 32, 3))
            window = decode_frame_window(source, 0, info.frame_count, info)
            self.assertEqual(len(window), info.frame_count)
            self.assertTrue(all(frame.shape == (24, 32, 3) for frame in window))
            self.assertFalse(window[0].flags.owndata)
            preview = decode_frame_window(source, 0, info.frame_count, info, 16)
            self.assertTrue(all(frame.shape == (12, 16, 3) for frame in preview))

            project = ProjectStore.create(temp_root, source, info)
            alpha = np.zeros((24, 32), dtype=bool)
            alpha[4:19, 6:16] = True
            for index in range(info.frame_count):
                project.save_ai_mask(index, alpha)
            destination = temp_root / "exports" / "rgba"
            export_sequence(project, 0, info.frame_count - 1, destination, rgba=True)
            with Image.open(destination / "00000000.png") as output:
                self.assertEqual(output.mode, "RGBA")
                output_alpha = np.asarray(output.getchannel("A"))
                self.assertEqual(int(output_alpha.max()), 255)
                self.assertEqual(int(output_alpha.min()), 0)
            outside_sequence = temp_root.parent / f"{temp_root.name}-outside-sequence"
            export_sequence(
                project,
                0,
                info.frame_count - 1,
                outside_sequence,
                rgba=False,
            )
            self.assertTrue((outside_sequence / "00000000.png").is_file())
            for path in outside_sequence.iterdir():
                path.unlink()
            outside_sequence.rmdir()

    def test_video_export_creates_bw_mp4_and_alpha_prores_mov(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runtime" / "tmp") as temporary:
            temp_root = Path(temporary)
            frames_dir = temp_root / "input"
            frames_dir.mkdir()
            for index in range(4):
                image = Image.new("RGB", (32, 24), (10, 20, 30))
                ImageDraw.Draw(image).rectangle(
                    (index * 2 + 6, 4, index * 2 + 17, 18),
                    fill=(240, 80, 40),
                )
                image.save(frames_dir / f"{index:05d}.png")
            source = temp_root / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    "24",
                    "-i",
                    str(frames_dir / "%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            info = probe_video(source)
            project = ProjectStore.create(temp_root, source, info)
            for index in range(info.frame_count):
                mask = np.zeros((info.height, info.width), dtype=bool)
                mask[4:19, 6 + index * 2:18 + index * 2] = True
                project.save_ai_mask(index, mask)
            project.set_manual_ops(
                0,
                [
                    {
                        "action": "remove",
                        "radius": 2,
                        "points": [[10.0, 10.0]],
                    }
                ],
            )

            chosen_folder = temp_root / "exports" / "custom-location"
            mask_video = chosen_folder / "mask.mp4"
            alpha_video = chosen_folder / "subject.mov"
            export_video_sequence(project, 0, 3, mask_video, "mask")
            export_video_sequence(project, 0, 3, alpha_video, "alpha")
            self.assertTrue(mask_video.is_file())
            self.assertTrue(alpha_video.is_file())
            outside_video = temp_root.parent / f"{temp_root.name}-outside.mp4"
            export_video_sequence(project, 0, 3, outside_video, "mask")
            self.assertTrue(outside_video.is_file())
            outside_video.unlink()

            def probe(path):
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,pix_fmt,width,height",
                        "-of",
                        "json",
                        str(path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return json.loads(result.stdout)["streams"][0]

            mask_stream = probe(mask_video)
            self.assertEqual(mask_stream["codec_name"], "h264")
            self.assertEqual((mask_stream["width"], mask_stream["height"]), (32, 24))
            decoded_mask = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(mask_video),
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            )
            grayscale = np.frombuffer(decoded_mask.stdout, dtype=np.uint8).reshape(24, 32)
            self.assertLessEqual(int(grayscale[10, 10]), 10)
            self.assertGreaterEqual(int(grayscale[10, 7]), 240)
            self.assertLessEqual(int(grayscale[1, 1]), 10)
            alpha_stream = probe(alpha_video)
            self.assertEqual(alpha_stream["codec_name"], "prores")
            self.assertIn(
                alpha_stream["pix_fmt"],
                {"yuva444p10le", "yuva444p12le"},
            )
            self.assertEqual((alpha_stream["width"], alpha_stream["height"]), (32, 24))

            decoded_alpha = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(alpha_video),
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgba",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            )
            rgba = np.frombuffer(decoded_alpha.stdout, dtype=np.uint8).reshape(24, 32, 4)
            self.assertLessEqual(int(rgba[10, 10, 3]), 5)
            self.assertGreaterEqual(int(rgba[10, 7, 3]), 250)
            self.assertLessEqual(int(rgba[1, 1, 3]), 5)


if __name__ == "__main__":
    unittest.main()
