#!/usr/bin/env python3
import argparse
import gc
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _check(cond: bool, label: str, details: str = "") -> Tuple[bool, str]:
    msg = f"[{'OK' if cond else 'FAIL'}] {label}"
    if details:
        msg += f" :: {details}"
    return cond, msg


def _resolve_public_dataset_fallback() -> Tuple[Path, Path, str]:
    try:
        import config as active_config  # type: ignore

        base_dir = Path(str(getattr(active_config, "BASE_DIR", "")))
        test_txt = Path(str(getattr(active_config, "TEST_TXT", "")))
        return base_dir, test_txt, "active config fallback"
    except Exception:
        return Path(""), Path(""), "no fallback"


def _import_checks() -> List[Tuple[bool, str]]:
    required = [
        "torch",
        "torchvision",
        "transformers",
        "pycocotools",
        "PIL",
        "numpy",
        "matplotlib",
        "yaml",
    ]
    out: List[Tuple[bool, str]] = []
    for mod in required:
        try:
            importlib.import_module(mod)
            out.append(_check(True, f"import {mod}"))
        except Exception as exc:
            out.append(_check(False, f"import {mod}", str(exc)))
    return out


def _file_checks() -> List[Tuple[bool, str]]:
    required_paths = [
        ROOT / "stable_runs/20260115_134958/exp1_clean/checkpoints/best/model.safetensors",
        ROOT / "stable_runs/20260115_134958/exp2_scanned/checkpoints/best/model.safetensors",
        ROOT / "AB_on_CD_results/ALL_AB_ON_CD_RESULTS.json",
        ROOT / "paper_experiments/out/error_compare/summary.json",
        ROOT / "paper_experiments/out/size_compare/summary.json",
        ROOT / "paper_experiments/out/factorized_pinned/summary.json",
        ROOT / "artifacts/figures/fig_pairing_schematic.png",
        ROOT / "artifacts/figures/fig_annotation_invariance.png",
        ROOT / "artifacts/figures/fig_iou_bands.png",
        ROOT / "artifacts/figures/fig_abcd_grid.png",
        ROOT / "artifacts/figures/fig_failure_taxonomy.png",
        ROOT / "artifacts/figures/fig_size_distribution.png",
        ROOT / "artifacts/tables/table_manifest.json",
        ROOT / "artifacts/tables/table_subset_softmax.csv",
        ROOT / "artifacts/tables/table_roi_ab_on_cd.csv",
        ROOT / "artifacts/tables/table_error_decomposition.csv",
        ROOT / "artifacts/tables/table_size_success.csv",
        ROOT / "artifacts/tables/table_factorized_degradation.csv",
    ]
    return [_check(p.exists(), f"path exists", str(p.relative_to(ROOT))) for p in required_paths]


def _dataset_checks() -> List[Tuple[bool, str]]:
    out: List[Tuple[bool, str]] = []
    for run_dir in [
        ROOT / "stable_runs/20260115_134958/exp1_clean",
        ROOT / "stable_runs/20260115_134958/exp2_scanned",
    ]:
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            out.append(_check(False, "pinned config exists", str(cfg_path.relative_to(ROOT))))
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        base_dir = Path(str(cfg.get("BASE_DIR", "")))
        test_txt = Path(str(cfg.get("TEST_TXT", "")))
        source = "pinned config"
        if not (base_dir.exists() and test_txt.exists()):
            fallback_base, fallback_test, fallback_source = _resolve_public_dataset_fallback()
            if fallback_base.exists() and fallback_test.exists():
                base_dir = fallback_base
                test_txt = fallback_test
                source = fallback_source
        out.append(_check(base_dir.exists(), "BASE_DIR reachable", f"{base_dir} [{source}]"))
        out.append(_check(test_txt.exists(), "TEST_TXT reachable", f"{test_txt} [{source}]"))
    return out


def _cli_checks() -> List[Tuple[bool, str]]:
    commands = [
        ("render_4_variants --help", [sys.executable, "diagnostic/render_4_variants.py", "--help"]),
        ("training/train.py --help", [sys.executable, "training/train.py", "--help"]),
        ("evaluation.test_AB_on_CD --help", [sys.executable, "-m", "evaluation.test_AB_on_CD", "--help"]),
    ]
    out: List[Tuple[bool, str]] = []
    for label, cmd in commands:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        ok = proc.returncode == 0
        details = "ok" if ok else (proc.stderr.strip() or proc.stdout.strip())
        out.append(_check(ok, "cli", f"{label} :: {details}"))
    return out


def _checkpoint_load_checks() -> List[Tuple[bool, str]]:
    out: List[Tuple[bool, str]] = []
    try:
        from transformers import RTDetrForObjectDetection  # type: ignore
    except Exception as exc:
        return [_check(False, "checkpoint loader import", str(exc))]

    ckpts = [
        ROOT / "stable_runs/20260115_134958/exp1_clean/checkpoints/best",
        ROOT / "stable_runs/20260115_134958/exp2_scanned/checkpoints/best",
    ]
    for ckpt in ckpts:
        try:
            model = RTDetrForObjectDetection.from_pretrained(str(ckpt), local_files_only=True)
            del model
            gc.collect()
            out.append(_check(True, "checkpoint load", str(ckpt.relative_to(ROOT))))
        except Exception as exc:
            out.append(_check(False, "checkpoint load", f"{ckpt.relative_to(ROOT)} :: {exc}"))
    return out


def _runner_check() -> Tuple[bool, str]:
    cmd = [sys.executable, "tools/run_paper.py", "paper_all", "--dry"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    ok = proc.returncode == 0
    details = "run_paper dry ok" if ok else (proc.stderr.strip() or proc.stdout.strip())
    return _check(ok, "paper_all dry-run", details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submission quickcheck.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any failure.")
    args = parser.parse_args()

    checks: List[Tuple[bool, str]] = []
    checks.extend(_import_checks())
    checks.extend(_cli_checks())
    checks.extend(_file_checks())
    checks.extend(_dataset_checks())
    checks.extend(_checkpoint_load_checks())
    checks.append(_runner_check())

    failures = 0
    for ok, msg in checks:
        print(msg)
        if not ok:
            failures += 1

    print(f"\nSummary: {len(checks) - failures}/{len(checks)} checks passed")
    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
