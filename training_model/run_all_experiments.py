import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


def build_scenarios() -> List[Dict[str, Any]]:
    """Define all End-to-End paper scenarios grouped by role."""
    scenarios: List[Dict[str, Any]] = [
        # --- Group 1: Proposed Model (3 seeds for mean +- std) ---
        {
            "id": "proposed_unified_favit_seed42",
            "group": "main",
            "run_robustness": True,
            "run_visualize": True,
            "overrides": {"backbone": "favit_cnn", "seed": 42},
        },
        {
            "id": "proposed_unified_favit_seed123",
            "group": "main",
            "run_robustness": False,
            "run_visualize": False,
            "overrides": {"backbone": "favit_cnn", "seed": 123},
        },
        {
            "id": "proposed_unified_favit_seed2025",
            "group": "main",
            "run_robustness": False,
            "run_visualize": False,
            "overrides": {"backbone": "favit_cnn", "seed": 2025},
        },
        # --- Group 2: Baselines (Table 1 & Table 2) ---
        {
            "id": "baseline_resnet50",
            "group": "baselines",
            "run_robustness": True,
            "run_visualize": False,
            "overrides": {"backbone": "resnet50", "lambda_fal": 0.0, "lambda_freq_cons": 0.0, "aux_weight": 0.0},
        },
        {
            "id": "baseline_convnext_tiny",
            "group": "baselines",
            "run_robustness": True,
            "run_visualize": False,
            "overrides": {"backbone": "convnext_tiny", "lambda_fal": 0.0, "lambda_freq_cons": 0.0, "aux_weight": 0.0},
        },
        {
            "id": "baseline_efficientnetb4",
            "group": "baselines",
            "run_robustness": False,
            "run_visualize": False,
            "overrides": {"backbone": "efficientnetb4", "lambda_fal": 0.0, "lambda_freq_cons": 0.0, "aux_weight": 0.0},
        },
        {
            "id": "baseline_swin_small",
            "group": "baselines",
            "run_robustness": False,
            "run_visualize": False,
            "overrides": {"backbone": "swin_small", "lambda_fal": 0.0, "lambda_freq_cons": 0.0, "aux_weight": 0.0},
        },
        {
            "id": "baseline_redesigned_favit",
            "group": "baselines",
            "run_robustness": False,
            "run_visualize": False,
            "overrides": {"backbone": "redesigned_favit", "lambda_fal": 0.0, "lambda_freq_cons": 0.0, "aux_weight": 0.0},
        },
        # --- Group 3: Ablation Study (Table 4) ---
        {
            "id": "ablation_1_vit_gam",
            "group": "ablation",
            "run_robustness": False,
            "run_visualize": False,
            "overrides": {
                "backbone": "favit_cnn",
                "use_freq": False,
                "use_cnn_branch": False,
                "use_learnable_fft": False,
                "use_bidirectional_fusion": False,
                "use_aux_head": False,
                "lambda_fal": 0.0,
                "lambda_freq_cons": 0.0,
                "aux_weight": 0.0,
            },
        },
        {
            "id": "ablation_2_plus_freq",
            "group": "ablation",
            "run_robustness": False,
            "run_visualize": False,
            "overrides": {
                "backbone": "favit_cnn",
                "use_freq": True,
                "use_learnable_fft": True,
                "use_bidirectional_fusion": False,
                "use_aux_head": False,
                "lambda_fal": 0.0,
                "lambda_freq_cons": 0.0,
                "aux_weight": 0.0,
            },
        },
        {
            "id": "ablation_3_plus_bidirectional",
            "group": "ablation",
            "run_robustness": False,
            "run_visualize": False,
            "overrides": {
                "backbone": "favit_cnn",
                "use_freq": True,
                "use_learnable_fft": True,
                "use_bidirectional_fusion": True,
                "use_aux_head": False,
                "lambda_fal": 0.0,
                "lambda_freq_cons": 0.0,
                "aux_weight": 0.0,
            },
        },
        {
            "id": "ablation_4_plus_spectral_debias",
            "group": "ablation",
            "run_robustness": False,
            "run_visualize": False,
            "overrides": {
                "backbone": "favit_cnn",
                "use_freq": True,
                "use_learnable_fft": True,
                "use_bidirectional_fusion": True,
                "use_aux_head": False,
                "lambda_fal": 0.0,
                "lambda_freq_cons": 0.5,
                "aux_weight": 0.0,
            },
        },
    ]

    # --- Group 4: Cross-Manipulation Leave-One-Out (Table 3) ---
    for holdout in ("deepfakes", "face2face", "faceswap", "neuraltextures"):
        scenarios.append(
            {
                "id": f"loo_holdout_{holdout}",
                "group": "loo",
                "run_robustness": False,
                "run_visualize": False,
                "overrides": {
                    "backbone": "favit_cnn",
                    "exclude_manipulations": [holdout, "faceshifter"],
                },
            }
        )
    return scenarios


def load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"scenarios": {}}


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    tmp_path = state_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(state_path)


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def aggregate_csvs(runs_dir: Path, scenario_ids: List[str]) -> Optional[Path]:
    frames = []
    for sid in scenario_ids:
        csv_file = runs_dir / sid / "eval" / "eval_summary.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df.insert(0, "scenario", sid)
            frames.append(df)
    if not frames:
        return None
    out_path = runs_dir / "all_experiments_summary.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable End-to-End Paper Experiment Runner")
    parser.add_argument("--base-config", type=str, default="configs/unified_favit_rtx5060ti.yaml")
    parser.add_argument("--runs-dir", type=str, default="experiments_runs")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["main", "baselines", "ablation", "loo"],
        choices=["main", "baselines", "ablation", "loo"],
        help="Scenario groups to execute",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Run only specific scenario IDs")
    parser.add_argument("--extra-cross", nargs="*", default=[], help="Extra cross-dataset name=path for eval_suite.py")
    parser.add_argument("--dry-run", action="store_true", help="Print execution & resume plan without running")
    return parser.parse_args()


def main():
    args = parse_args()
    workdir = Path(__file__).resolve().parent
    base_cfg_path = (workdir / args.base_config) if not Path(args.base_config).is_absolute() else Path(args.base_config)
    with base_cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    runs_dir = (workdir / args.runs_dir) if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    state_path = runs_dir / "runner_state.json"
    state = load_state(state_path)

    all_scenarios = build_scenarios()
    selected = [
        s for s in all_scenarios
        if s["group"] in args.groups and (args.only is None or s["id"] in args.only)
    ]
    if not selected:
        print("No scenarios matched the selection.")
        return

    print(f"Loaded {len(selected)} scenarios | State file: {state_path}")
    for s in selected:
        sid = s["id"]
        s_state = state["scenarios"].setdefault(
            sid, {"status": "pending", "train_done": False, "eval_done": False, "viz_done": False}
        )
        scenario_dir = runs_dir / sid
        ckpt_dir = scenario_dir / "checkpoints"
        eval_dir = scenario_dir / "eval"
        viz_dir = scenario_dir / "figures"
        cfg_path = scenario_dir / "config.yaml"

        cfg = copy.deepcopy(base_cfg)
        cfg.update(s["overrides"])
        cfg["save_dir"] = str(ckpt_dir)
        cfg["checkpoint_name"] = "best_model.pth"
        cfg["resume_from"] = None

        best_ckpt = ckpt_dir / "best_model.pth"
        last_ckpt = ckpt_dir / "last_best_model.pth"

        if args.dry_run:
            resume_note = f" (will resume from {last_ckpt.name})" if (not s_state["train_done"] and last_ckpt.exists()) else ""
            print(
                f"  - [{s['group']}] {sid}: status={s_state['status']} | "
                f"train_done={s_state['train_done']}{resume_note} | "
                f"eval_done={s_state['eval_done']} | viz_done={s_state['viz_done']}"
            )
            continue

        if s_state["status"] == "completed" and s_state["train_done"] and s_state["eval_done"] and (not s["run_visualize"] or s_state["viz_done"]):
            print(f"\n[SKIP] {sid} already completed.")
            continue

        scenario_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        s_state["status"] = "in_progress"
        save_state(state_path, state)

        try:
            # Step 1: Training (auto-resumes from last_best_model.pth if interrupted)
            if not s_state["train_done"]:
                train_cmd = [sys.executable, "train.py", "--config", str(cfg_path)]
                if last_ckpt.exists():
                    print(f"[RESUME] Resuming training for {sid} from {last_ckpt}")
                    train_cmd.extend(["--resume", str(last_ckpt)])
                run_cmd(train_cmd, cwd=workdir)
                s_state["train_done"] = True
                save_state(state_path, state)

            # Step 2: Evaluation Suite
            if not s_state["eval_done"]:
                eval_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
                if not eval_ckpt.exists():
                    raise FileNotFoundError(f"No checkpoint found for {sid} in {ckpt_dir}")
                eval_cmd = [
                    sys.executable,
                    "eval_suite.py",
                    "--config",
                    str(cfg_path),
                    "--checkpoint",
                    str(eval_ckpt),
                    "--output-dir",
                    str(eval_dir),
                ]
                if not s["run_robustness"]:
                    eval_cmd.append("--skip-robustness")
                if args.extra_cross:
                    eval_cmd.extend(["--extra-cross", *args.extra_cross])
                run_cmd(eval_cmd, cwd=workdir)
                s_state["eval_done"] = True
                save_state(state_path, state)

            # Step 3: Paper Figures (for main proposed model)
            if s["run_visualize"] and not s_state["viz_done"]:
                eval_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
                viz_cmd = [
                    sys.executable,
                    "visualize_paper.py",
                    "--config",
                    str(cfg_path),
                    "--checkpoint",
                    str(eval_ckpt),
                    "--eval-json",
                    str(eval_dir / "eval_summary.json"),
                    "--output-dir",
                    str(viz_dir),
                ]
                run_cmd(viz_cmd, cwd=workdir)
                s_state["viz_done"] = True
                save_state(state_path, state)

            s_state["status"] = "completed"
            save_state(state_path, state)
            print(f"[DONE] {sid} completed.")

        except KeyboardInterrupt:
            s_state["status"] = "interrupted"
            save_state(state_path, state)
            print(f"\n[INTERRUPTED] Saved state at {sid}. Re-run the same command to resume from exact checkpoint.")
            sys.exit(130)
        except Exception as exc:
            s_state["status"] = "failed"
            s_state["error"] = str(exc)
            save_state(state_path, state)
            print(f"\n[FAILED] {sid}: {exc}. State saved; fix the issue and re-run to resume.")
            raise

    if not args.dry_run:
        master_csv = aggregate_csvs(runs_dir, [s["id"] for s in all_scenarios])
        if master_csv is not None:
            print(f"\nMaster summary CSV updated: {master_csv}")


if __name__ == "__main__":
    main()
