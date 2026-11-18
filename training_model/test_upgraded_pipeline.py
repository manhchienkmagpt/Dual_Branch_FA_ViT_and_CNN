import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import timm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import DeepfakeFrameDataset, get_normalization_stats  # noqa: E402
from models import (  # noqa: E402
    FALoss,
    FAViTCNN,
    SpectralConsistencyLoss,
    build_model,
    perturb_batch_for_debiasing,
)
from utils.metrics import (  # noqa: E402
    aggregate_video_predictions,
    compute_eer,
    extract_video_id,
)


class TestUpgradedFAViTPipeline(unittest.TestCase):
    def test_favit_cnn_forward_backward_and_peft(self):
        model = build_model(
            backbone="favit_cnn",
            pretrained=False,
            image_size=224,
            vit_backbone_name="vit_base_patch16_clip_224",
        )
        norm_outputs = []
        handle = model.classifier[0].register_forward_hook(
            lambda module, args, output: norm_outputs.append(output)
        )
        x = torch.randn(2, 3, 224, 224)
        logits, features, aux_logits = model(x)
        handle.remove()
        self.assertEqual(len(norm_outputs), 1)
        torch.testing.assert_close(features, norm_outputs[0])

        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(features.shape, (2, 768))
        self.assertIsNotNone(aux_logits)
        self.assertEqual(aux_logits.shape, (2, 1))
        self.assertIsNotNone(model.last_patch_anomaly_map)
        self.assertEqual(model.last_patch_anomaly_map.shape, (2, 14, 14))

        # Verify PEFT: first 10 blocks Attention/MLP frozen, last 2 blocks & adapters trainable
        early_attn_params = [
            p for b in model.vit.blocks[:-2] for n, p in b.named_parameters() if "attn" in n or "mlp" in n
        ]
        self.assertTrue(len(early_attn_params) > 0)
        self.assertTrue(all(not p.requires_grad for p in early_attn_params))
        self.assertTrue(any(p.requires_grad for p in model.freq_extractor.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.fusions.parameters()))

        loss = logits.sum() + aux_logits.sum() + FALoss()(
            features, torch.tensor([0.0, 1.0]), model.classifier[2].weight
        )
        loss.backward()
        self.assertIsNotNone(model.freq_extractor.fft_filter.band_weights.grad)
        self.assertTrue(torch.isfinite(model.classifier[0].weight.grad).all())

    def test_normalization_matches_pretrained_backbones(self):
        for backbone, model_name in [
            ("favit_cnn", "vit_base_patch16_clip_224"),
            ("redesigned_favit", "vit_base_patch16_224"),
            ("resnet50", "resnet50"),
        ]:
            with self.subTest(backbone=backbone):
                cfg = timm.models.get_pretrained_cfg(model_name)
                mean, std = get_normalization_stats(backbone, model_name)
                self.assertEqual(mean, cfg.mean)
                self.assertEqual(std, cfg.std)

    def test_ablation_flags(self):
        x = torch.randn(2, 3, 224, 224)
        for kwargs in [
            {"use_freq": False, "use_cnn_branch": False, "use_learnable_fft": False},
            {"use_learnable_fft": False},
            {"use_bidirectional_fusion": False},
            {"use_aux_head": False},
        ]:
            model = FAViTCNN(
                backbone_name="vit_base_patch16_clip_224",
                pretrained=False,
                **kwargs,
            )
            if kwargs.get("use_cnn_branch") is False:
                self.assertIsNone(model.freq_extractor)
            logits, feat, aux = model(x)
            self.assertEqual(logits.shape, (2, 1))
            self.assertEqual(feat.shape[0], 2)
            if kwargs.get("use_aux_head") is False:
                self.assertIsNone(aux)

    def test_baseline_backbones_convnext_and_resnet(self):
        x = torch.randn(2, 3, 224, 224)
        for name in ("resnet50", "convnext_tiny"):
            model = build_model(backbone=name, pretrained=False, image_size=224)
            out = model(x)
            self.assertEqual(out.shape, (2,))

    def test_losses_and_perturbation(self):
        x = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        feat = torch.randn(4, 64, requires_grad=True)
        feat_p = torch.randn(4, 64, requires_grad=True)
        logits = torch.randn(4, 1, requires_grad=True)
        logits_p = torch.randn(4, 1, requires_grad=True)
        cls_w = torch.randn(1, 64, requires_grad=True)

        fa_loss = FALoss()(feat, labels, cls_w)
        # Correctly classified features: real aligned with -w, fake aligned with +w -> low FALoss
        w_dir = torch.zeros(1, 64)
        w_dir[0, 0] = 1.0
        aligned_feat = torch.zeros(4, 64)
        aligned_feat[:2, 0] = -1.0  # real -> -w
        aligned_feat[2:, 0] = 1.0   # fake -> +w
        aligned_fa = FALoss()(aligned_feat, labels, w_dir)
        self.assertLess(aligned_fa.item(), 1e-3)
        spec_loss = SpectralConsistencyLoss()(feat, feat_p, logits, logits_p)
        pert = perturb_batch_for_debiasing(x)

        self.assertTrue(torch.isfinite(fa_loss))
        self.assertTrue(torch.isfinite(spec_loss))
        self.assertEqual(pert.shape, x.shape)
        (fa_loss + spec_loss).backward()

    def test_video_level_metrics_and_eer(self):
        paths = [
            "ffpp/c23/test/real/001/0001.jpg",
            "ffpp/c23/test/real/001/0002.jpg",
            "ffpp/c23/test/real/002/0001.jpg",
            "ffpp/c23/test/fake/Deepfakes/001_002/0001.jpg",
            "ffpp/c23/test/fake/Deepfakes/001_002/0002.jpg",
            "ffpp/c23/test/fake/Deepfakes/003_004/0001.jpg",
        ]
        self.assertEqual(extract_video_id(paths[0]), "ffpp/c23/test/real/001")
        self.assertEqual(extract_video_id(paths[1]), "ffpp/c23/test/real/001")
        # Same video folder name in different methods must stay distinct.
        self.assertNotEqual(
            extract_video_id("ffpp/c23/test/fake/Deepfakes/001_002/0001.jpg"),
            extract_video_id("ffpp/c23/test/fake/FaceSwap/001_002/0001.jpg"),
        )
        clip_mean, _ = get_normalization_stats("favit_cnn", "vit_base_patch16_clip_224")
        res_mean, _ = get_normalization_stats("resnet50", "vit_base_patch16_clip_224")
        self.assertNotEqual(clip_mean, res_mean)

        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.15, 0.85, 0.9, 0.8]
        metrics, vids, vlabels, vprobs = aggregate_video_predictions(paths, labels, probs, threshold=0.5)
        self.assertEqual(metrics["num_videos"], 4)
        self.assertAlmostEqual(metrics["auc"], 1.0, places=5)
        self.assertAlmostEqual(metrics["eer"], 0.0, places=5)
        self.assertAlmostEqual(compute_eer(vlabels, vprobs), 0.0, places=5)

    def test_numeric_video_folders_ignore_frame_names(self):
        folders = [
            ("ffpp/c23/test/real/0", 0, [0.1, 0.3]),
            ("ffpp/c23/test/real/1", 0, [0.2, 0.4]),
            ("ffpp/c23/test/fake/Deepfakes/0", 1, [0.7, 0.9]),
            ("ffpp/c23/test/fake/FaceSwap/0", 1, [0.6, 1.0]),
        ]
        paths, labels, probs = [], [], []
        for folder, label, frame_probs in folders:
            paths.extend([
                f"{folder}/first_sample000000_frame000000.jpg",
                f"{folder}/different_name.png",
            ])
            labels.extend([label, label])
            probs.extend(frame_probs)

        metrics, ids, video_labels, video_probs = aggregate_video_predictions(paths, labels, probs)
        self.assertEqual(metrics["num_videos"], len(folders))
        actual = dict(zip(ids, zip(video_labels, video_probs)))
        for folder, label, frame_probs in folders:
            self.assertEqual(actual[folder][0], label)
            self.assertAlmostEqual(actual[folder][1], float(np.mean(frame_probs)))
        self.assertAlmostEqual(metrics["auc"], 1.0)

    def test_video_folder_ids_preserve_dataset_split_class_and_method(self):
        folders = [
            "ffpp/c23/train/fake/Deepfakes/0",
            "ffpp/c23/val/fake/Deepfakes/0",
            "ffpp/c23/test/fake/Deepfakes/0",
            "ffpp/c23/test/fake/FaceSwap/0",
            "ffpp/c23/test/real/0",
            "celebdf/test/0/0",
            "celebdf/test/1/0",
        ]
        ids = [extract_video_id(f"{folder}/0001.jpg") for folder in folders]
        self.assertEqual(ids, folders)
        self.assertEqual(len(set(ids)), len(folders))

    def test_nested_ffpp_and_celebdf_directory_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))

            # FF++ nested layout: train/real/<vid>/0.jpg and train/fake/<method>/<vid>/0.jpg
            for p in [
                root / "ffpp/c23/train/real/vid_r1/0001.jpg",
                root / "ffpp/c23/train/real/vid_r1/0002.jpg",
                root / "ffpp/c23/train/fake/Deepfakes/vid_df1/0001.jpg",
                root / "ffpp/c23/train/fake/Face2Face/vid_f2f1/0001.jpg",
                root / "celebdf/test/0/id0_0001/0001.jpg",
                root / "celebdf/test/0/id0_0001/0002.jpg",
                root / "celebdf/test/1/id0_id1_0001/0001.jpg",
            ]:
                p.parent.mkdir(parents=True, exist_ok=True)
                img.save(p)

            ffpp_all = DeepfakeFrameDataset(
                root_dir=str(root / "ffpp/c23"),
                split="train",
                dataset_type="ffpp",
                mode="train",
            )
            self.assertEqual(len(ffpp_all), 4)

            ffpp_loo = DeepfakeFrameDataset(
                root_dir=str(root / "ffpp/c23"),
                split="train",
                dataset_type="ffpp",
                mode="train",
                exclude_classes=["deepfakes"],
            )
            self.assertEqual(len(ffpp_loo), 3)

            celeb = DeepfakeFrameDataset(
                root_dir=str(root / "celebdf/test"),
                split=None,
                dataset_type="cross",
                mode="test",
            )
            self.assertEqual(len(celeb), 3)
            self.assertEqual([int(s[1]) for s in celeb.samples], [0, 0, 1])

            vid_0 = extract_video_id(str(root / "celebdf/test/0/id0_0001/0001.jpg"))
            vid_1 = extract_video_id(str(root / "celebdf/test/0/id0_0001/0002.jpg"))
            self.assertEqual(vid_0, (root / "celebdf/test/0/id0_0001").as_posix())
            self.assertEqual(vid_0, vid_1)


if __name__ == "__main__":
    unittest.main()
