import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import check_training_setup as setup


class TestTrainingSetup(unittest.TestCase):
    def make_layout(self, root, omit=None, empty=None):
        config = {"data_root": str(root / "ffpp/c23"), "cross_dataset_root": str(root / "celebdf/test")}
        folders = []
        for split in ("train", "val", "test"):
            folders.append(root / "ffpp/c23" / split / "real")
            folders.extend(root / "ffpp/c23" / split / "fake" / method for method in setup.FFPP_METHODS)
        folders.extend((root / "celebdf/test/0", root / "celebdf/test/1"))
        for folder in folders:
            if folder == omit:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            if folder != empty:
                (folder / "0").mkdir()
        return config

    def test_accepts_all_split_method_and_celebdf_video_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            counts = setup.check_data_layout(self.make_layout(Path(tmp)))
        self.assertEqual(len(counts), 20)
        self.assertTrue(all(count == 1 for _, count in counts))

    def test_missing_method_is_reported_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_layout(root, omit=root / "ffpp/c23/val/fake/FaceShifter")
            with self.assertRaisesRegex(ValueError, "Missing data directories:[\\s\\S]*FaceShifter"):
                setup.check_data_layout(config)

    def test_empty_class_is_reported_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_layout(root, empty=root / "celebdf/test/1")
            with self.assertRaisesRegex(ValueError, "No video folders in"):
                setup.check_data_layout(config)

    def test_relative_dataset_roots_match_training_working_directory(self):
        self.assertEqual(setup.config_path({"data_root": "faces/ffpp"}, "data_root"), setup.TRAINING_DIR / "faces/ffpp")

    def test_cuda_request_cannot_silently_fall_back_to_cpu(self):
        torch = SimpleNamespace(
            __version__="test-cpu", version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.object(setup.importlib, "import_module", side_effect=lambda name: torch if name == "torch" else Mock()):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                setup.check_runtime({"device": "cuda"})


if __name__ == "__main__":
    unittest.main()
