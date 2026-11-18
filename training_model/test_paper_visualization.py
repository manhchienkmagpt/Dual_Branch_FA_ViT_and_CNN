import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from datasets import get_eval_transform, get_normalization_stats
import visualize_paper as visualization


class TestPaperNormalization(unittest.TestCase):
    def test_denormalization_recovers_rgb_for_each_backbone(self):
        image = (np.arange(48).reshape(4, 4, 3) * 5).astype(np.uint8)
        for backbone in ("favit_cnn", "redesigned_favit", "resnet50"):
            with self.subTest(backbone=backbone):
                mean, std = get_normalization_stats(backbone)
                tensor = get_eval_transform(4, mean=mean, std=std)(image=image)["image"]
                rgb = visualization._denormalize(tensor, mean, std)
                np.testing.assert_allclose(rgb, image / 255.0, atol=3e-7)

    def test_visualization_main_uses_clip_stats_for_images_and_overlays(self):
        config = {
            "backbone": "favit_cnn",
            "vit_backbone_name": "vit_base_patch16_clip_224",
            "cross_dataset_root": "unused/test",
            "image_size": 4,
            "device": "cpu",
        }
        args = SimpleNamespace(
            config="unused.yaml", checkpoint="unused.pth", output_dir=str(Path.cwd()),
            num_heatmaps=2, max_samples=20, eval_json=None,
        )
        model = nn.Linear(1, 1)
        with (
            patch.object(visualization, "parse_args", return_value=args),
            patch.object(visualization, "load_config", return_value=config),
            patch.object(visualization, "build_model_from_config", return_value=model),
            patch.object(visualization, "load_checkpoint", return_value={"model_state_dict": model.state_dict()}),
            patch.object(visualization, "DeepfakeFrameDataset") as dataset,
            patch.object(visualization, "plot_patch_heatmaps") as heatmaps,
            patch.object(visualization, "plot_tsne") as tsne,
            patch.object(Path, "mkdir"),
            redirect_stdout(io.StringIO()),
        ):
            visualization.main()

        mean, std = get_normalization_stats("favit_cnn")
        image = np.full((4, 4, 3), 128, dtype=np.uint8)
        actual = dataset.call_args.kwargs["eval_transform"](image=image)["image"]
        expected = (torch.full((3, 4, 4), 128 / 255.0) - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(heatmaps.call_args.kwargs["mean"], mean)
        self.assertEqual(heatmaps.call_args.kwargs["std"], std)
        self.assertIs(heatmaps.call_args.args[1], dataset.return_value)
        self.assertIs(tsne.call_args.args[1], dataset.return_value)


if __name__ == "__main__":
    unittest.main()
