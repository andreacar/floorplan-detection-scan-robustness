"""Utilities to run the embedding-focused experiments sequentially and summarize their outputs."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "experiment_results"
SUMMARY_PATH = RESULTS_DIR / "embedding_suite_summary.csv"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_training(env_overrides: Dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(env_overrides)
    subprocess.run([sys.executable, str(TRAIN_SCRIPT)], cwd=PROJECT_ROOT, env=env, check=True)


def run_embedding_suite(
    *,
    base_checkpoint: str,
    dist_score_file: str,
    distance_train_txt: str,
    paired_layouts_txt: Optional[str] = None,
    subset: int = 400,
    lr: str = "1e-5",
    epochs: str = "1",
) -> List[str]:
    """Run the risk-guided, distance curriculum, and (optional) paired invariance experiments.

    Args:
        base_checkpoint: path to the scanned best checkpoint (e.g. runs/.../exp2_scanned/checkpoints/best).
        dist_score_file: layout_path,score mapping (scores normalized to [0,1]).
        distance_train_txt: TRAIN_TXT override containing the layouts to target.
        paired_layouts_txt: optional TRAIN_TXT pointing to the paired clean+scanned layouts.
        subset: SUBSET_TRAIN.
        lr: fine-tune learning rate.
        epochs: number of epochs to run (default `1`).

    Returns:
        Ordered list of RUN_NAME values that were executed.
    """
    checkpoint_path = Path(base_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {base_checkpoint}")

    runs: List[str] = []

    experiments = [
        {
            "name": f"embedding_risk_alpha1_{_timestamp()}",
            "env": {
                "RUN_EXPERIMENTS": "scanned",
                "SCANNED_INIT_WEIGHTS_DIR": str(checkpoint_path),
                "SUBSET_TRAIN": str(subset),
                "EPOCHS": epochs,
                "LR": lr,
                "DISTANCE_SCORE_FILE": dist_score_file,
                "DISTANCE_SCORE_ALPHA": "1.0",
            },
        },
        {
            "name": f"embedding_risk_alpha2_{_timestamp()}",
            "env": {
                "RUN_EXPERIMENTS": "scanned",
                "SCANNED_INIT_WEIGHTS_DIR": str(checkpoint_path),
                "SUBSET_TRAIN": str(subset),
                "EPOCHS": epochs,
                "LR": lr,
                "DISTANCE_SCORE_FILE": dist_score_file,
                "DISTANCE_SCORE_ALPHA": "2.0",
            },
        },
        {
            "name": f"embedding_distance_control_{_timestamp()}",
            "env": {
                "RUN_EXPERIMENTS": "scanned",
                "TRAIN_TXT": distance_train_txt,
                "SCANNED_INIT_WEIGHTS_DIR": str(checkpoint_path),
                "SUBSET_TRAIN": str(subset),
                "EPOCHS": epochs,
                "LR": lr,
                "DISTANCE_SCORE_FILE": dist_score_file,
                "DISTANCE_CURRICULUM_ENABLE": "1",
                "DISTANCE_CURRICULUM_THRESHOLDS": "0.5,0.85",
                "DISTANCE_CURRICULUM_LOW_SIGMA": "0.0,0.8",
                "DISTANCE_CURRICULUM_LOW_KERNELS": "0,1",
                "DISTANCE_CURRICULUM_MID_SIGMA": "0.4,1.6",
                "DISTANCE_CURRICULUM_MID_KERNELS": "1,2,3",
                "DISTANCE_CURRICULUM_HIGH_SIGMA": "1.0,2.2",
                "DISTANCE_CURRICULUM_HIGH_KERNELS": "2,3,4",
            },
        },
    ]

    if paired_layouts_txt:
        experiments.append(
            {
                "name": f"embedding_paired_inv_{_timestamp()}",
                "env": {
                    "RUN_EXPERIMENTS": "scanned",
                    "TRAIN_TXT": paired_layouts_txt,
                    "SCANNED_INIT_WEIGHTS_DIR": str(checkpoint_path),
                    "SUBSET_TRAIN": str(subset),
                    "EPOCHS": epochs,
                    "LR": lr,
                    "INVARIANCE_LAMBDA": "0.05",
                },
            }
        )

    for exp in experiments:
        run_name = exp["name"]
        env = exp["env"].copy()
        env["RUN_NAME"] = run_name
        _run_training(env)
        runs.append(run_name)

    return runs


def summarize_results(run_names: Iterable[str], *, output_path: Path = SUMMARY_PATH) -> Path:
    """Collect the scanned metrics from each run and write a comparison CSV."""
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_name",
        "val_scanned_recall85",
        "val_scanned_ap85",
        "val_scanned_ap75",
        "epochs_ran",
    ]

    rows = []
    for name in run_names:
        summary_file = RUNS_DIR / name / "exp2_scanned" / "experiment_summary.json"
        if not summary_file.exists():
            continue
        data = json.loads(summary_file.read_text())
        rows.append(
            {
                "run_name": name,
                "val_scanned_recall85": data.get("best_recall85"),
                "val_scanned_ap85": data.get("best_ap85"),
                "val_scanned_ap75": data.get("best_ap75"),
                "epochs_ran": data.get("epochs_ran"),
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_path
