import copy
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

import train


class TestAUCCheckpointSelection(unittest.TestCase):
    def run_training(self, validation_metrics, checkpoint=None, **overrides):
        start_epoch = checkpoint["epoch"] if checkpoint else 0
        config = {
            "image_size": 1,
            "backbone": "resnet50",
            "pretrained": False,
            "epochs": start_epoch + len(validation_metrics),
            "lr": 1e-3,
            "use_amp": False,
            "selection_metric": "auc",
            "early_stopping_patience": 10,
            "lr_scheduler_patience": 2,
        }
        config.update(overrides)
        remaining = iter(validation_metrics)
        saved = []

        def run_epoch(**kwargs):
            values = (0.5, 0.5) if kwargs.get("optimizer") is not None else next(remaining)
            metrics = {"auc": values[0], "accuracy": values[1], "loss": 0.5}
            if len(values) > 2:
                metrics["video_auc"] = values[2]
            return metrics

        def save(**kwargs):
            state = {
                key: kwargs[key]
                for key in (
                    "epoch", "best_auc", "best_accuracy", "config",
                    "epochs_without_improvement",
                )
            }
            state["model_state_dict"] = kwargs["model"].state_dict()
            state["optimizer_state_dict"] = kwargs["optimizer"].state_dict()
            state["scheduler_state_dict"] = kwargs["scheduler"].state_dict()
            saved.append((Path(kwargs["save_path"]).name, copy.deepcopy(state)))

        loader = SimpleNamespace(dataset=None)
        with (
            patch.object(train, "build_model", return_value=nn.Linear(1, 1)),
            patch.object(train, "run_one_epoch", side_effect=run_epoch),
            patch.object(train, "save_checkpoint", side_effect=save),
            redirect_stdout(io.StringIO()),
        ):
            train.run_training_loop(
                config, loader, loader, Path("best.pth"), torch.device("cpu"),
                resume_path="memory.pth" if checkpoint else None,
                checkpoint_data=checkpoint,
            )
        return saved

    def test_auc_selects_checkpoint_when_accuracy_disagrees(self):
        saved = self.run_training([
            (0.70, 0.95), (0.80, 0.90), (0.80, 0.99),
            (0.75, 1.00), (0.81, 0.80),
        ])
        best = [state for name, state in saved if name == "best.pth"]
        last = [state for name, state in saved if name == "last_best.pth"]
        self.assertEqual([state["epoch"] for state in best], [1, 2, 5])
        self.assertEqual(len(last), 5)
        self.assertEqual(best[-1]["best_auc"], 0.81)
        self.assertEqual(best[-1]["best_accuracy"], 1.0)
        self.assertEqual(last[3]["epochs_without_improvement"], 2)
        # Plateau scheduling must follow AUC even while accuracy improves.
        self.assertAlmostEqual(
            last[3]["optimizer_state_dict"]["param_groups"][0]["lr"], 5e-4
        )
        self.assertEqual(best[1]["scheduler_state_dict"]["last_epoch"], 2)
        self.assertEqual(best[1]["scheduler_state_dict"]["best"], 0.80)

    def test_nan_and_tied_auc_do_not_improve_or_reset_early_stopping(self):
        saved = self.run_training(
            [(0.90, 0.60), (float("nan"), 0.70), (0.90, 0.80), (0.95, 0.90)],
            early_stopping_patience=2,
        )
        self.assertEqual(
            [state["epoch"] for name, state in saved if name == "best.pth"], [1]
        )
        final = saved[-1][1]
        self.assertEqual(final["epoch"], 3)
        self.assertEqual(final["best_auc"], 0.90)
        self.assertEqual(final["epochs_without_improvement"], 2)

    def test_video_auc_selects_checkpoint_independently_of_frame_auc(self):
        saved = self.run_training(
            [(0.90, 0.95, 0.70), (0.80, 0.90, 0.80), (0.95, 0.99, 0.75)],
            selection_metric="video_auc",
        )
        best = [state for name, state in saved if name == "best.pth"]
        self.assertEqual([state["epoch"] for state in best], [1, 2])
        self.assertEqual(best[-1]["best_auc"], 0.80)
        self.assertEqual(saved[-1][1]["scheduler_state_dict"]["best"], 0.80)

    def test_validation_aggregates_frame_probabilities_by_video(self):
        samples = [
            (0.1, 0, "val/real/r1/001.jpg"),
            (0.1, 0, "val/real/r1/002.jpg"),
            (0.6, 0, "val/real/r2/001.jpg"),
            (0.4, 1, "val/fake/Deepfakes/f1/001.jpg"),
            (0.9, 1, "val/fake/Deepfakes/f1/002.jpg"),
            (0.9, 1, "val/fake/Deepfakes/f1/003.jpg"),
            (0.55, 1, "val/fake/Deepfakes/f2/001.jpg"),
        ]
        data = [
            (torch.logit(torch.tensor([prob])), torch.tensor(float(label)), path)
            for prob, label, path in samples
        ]
        metrics = train.run_one_epoch(
            nn.Identity(), DataLoader(data, batch_size=3),
            nn.BCEWithLogitsLoss(), torch.device("cpu"),
        )
        self.assertAlmostEqual(metrics["auc"], 10 / 12)
        self.assertAlmostEqual(metrics["video_auc"], 0.75)

    def test_resume_preserves_auc_and_scheduler_state(self):
        metrics = [(0.70, 0.95), (0.80, 0.90), (0.80, 0.99), (0.75, 1.0), (0.81, 0.80)]
        uninterrupted = self.run_training(metrics)
        for name in ("best.pth", "last_best.pth"):
            with self.subTest(checkpoint=name):
                checkpoint = next(
                    state for filename, state in uninterrupted
                    if filename == name and state["epoch"] == 2
                )
                resumed = self.run_training(metrics[2:], checkpoint=checkpoint)
                expected = uninterrupted[-1][1]
                actual = resumed[-1][1]
                for key in (
                    "epoch", "best_auc", "best_accuracy",
                    "epochs_without_improvement", "scheduler_state_dict",
                ):
                    self.assertEqual(actual[key], expected[key])
                self.assertEqual(
                    actual["optimizer_state_dict"]["param_groups"],
                    expected["optimizer_state_dict"]["param_groups"],
                )

    def test_resume_rejects_changed_selection_metric(self):
        for old_metric, new_metric in [("auc", "video_auc"), ("video_auc", "auc")]:
            with self.subTest(old_metric=old_metric):
                checkpoint = self.run_training(
                    [(0.95, 0.9, 0.85)], selection_metric=old_metric,
                )[-1][1]
                with self.assertRaisesRegex(ValueError, "selection_metric changed"):
                    self.run_training(
                        [(0.8, 0.9, 0.7)], checkpoint=checkpoint, selection_metric=new_metric,
                    )

    def test_legacy_resume_resets_unknown_monitor_history(self):
        checkpoint = self.run_training([(0.95, 0.9)])[-1][1]
        del checkpoint["config"]["selection_metric"]
        checkpoint["epochs_without_improvement"] = 20
        saved = self.run_training(
            [(0.5, 0.6, 0.8)], checkpoint=checkpoint, selection_metric="video_auc",
        )
        self.assertEqual([name for name, _ in saved], ["best.pth", "last_best.pth"])
        final = saved[-1][1]
        self.assertEqual(final["best_auc"], 0.8)
        self.assertEqual(final["epochs_without_improvement"], 0)
        self.assertEqual(final["scheduler_state_dict"]["best"], 0.8)
        self.assertEqual(final["scheduler_state_dict"]["last_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
