import argparse
import shutil
from pathlib import Path

# Files to preserve for reproducibility
KEEP = {
    "command.txt",
    "config.json",
    "augment_config.json",
    "experiment_overrides.json",
    "env.txt",
    "pip_freeze.txt",
    "latest_epoch.json",
    "history.jsonl",
    "preprocessor_config.json",   # 🔴 CRITICAL FIX
}

KEEP_DIRS = {
    "metrics",
    "splits",
    "visuals",
    "epochs",
    "checkpoints",   # we will filter inside
}

def copy_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            if item.name in KEEP_DIRS:
                if item.name == "checkpoints":
                    # Only copy best checkpoint
                    best = item / "best"
                    if best.exists():
                        shutil.copytree(best, dst / "checkpoints" / "best", dirs_exist_ok=True)
                else:
                    shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            if item.name in KEEP:
                shutil.copy2(item, dst / item.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_path", help="e.g. runs/20260201_213408/exp1_clean")
    ap.add_argument("--out", default="stable_runs", help="Output root")
    args = ap.parse_args()

    src = Path(args.run_path)
    if not src.exists():
        raise SystemExit(f"[ERROR] Not found: {src}")

    run_id = src.parent.name
    exp = src.name
    dst = Path(args.out) / run_id / exp

    if dst.exists():
        raise SystemExit(f"[ERROR] Destination exists: {dst} (remove it first)")

    copy_tree(src, dst)
    print(f"[OK] Pinned run -> {dst}")

if __name__ == "__main__":
    main()
