#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


REPO = "andreacar/floorplan-detection-scan-robustness"
TAG = "v1.0.0"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

ASSETS = {
    "exp1_clean_stable_run.tar.gz": {
        "url": f"{BASE_URL}/exp1_clean_stable_run.tar.gz",
        "extract_to": Path("stable_runs/20260115_134958"),
    },
    "exp2_scanned_stable_run.tar.gz": {
        "url": f"{BASE_URL}/exp2_scanned_stable_run.tar.gz",
        "extract_to": Path("stable_runs/20260115_134958"),
    },
    "ab_on_cd_results.tar.gz": {
        "url": f"{BASE_URL}/ab_on_cd_results.tar.gz",
        "extract_to": Path("."),
    },
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _check_remote() -> None:
    for name, meta in ASSETS.items():
        req = urllib.request.Request(meta["url"], method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as response:
            print(f"{name}:")
            print(f"  status: {response.status}")
            print(f"  size: {response.headers.get('Content-Length', '')}")
            print(f"  type: {response.headers.get('Content-Type', '')}")
            print(f"  url: {meta['url']}")


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract GitHub release assets required for the public quickcheck path."
    )
    parser.add_argument(
        "--assets-dir",
        default="release_assets",
        help="Directory used to cache downloaded release asset tarballs.",
    )
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help="Verify that the GitHub release asset URLs are reachable without downloading them.",
    )
    args = parser.parse_args()

    if args.check_remote:
        _check_remote()
        return

    root = Path(__file__).resolve().parents[1]
    assets_dir = (root / args.assets_dir).resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)

    for name, meta in ASSETS.items():
        archive_path = assets_dir / name
        if not archive_path.exists():
            print(f"Downloading {name}")
            _download(meta["url"], archive_path)
        else:
            print(f"Using cached {name}")
        extract_to = (root / meta["extract_to"]).resolve()
        print(f"Extracting {name} -> {extract_to}")
        _extract(archive_path, extract_to)

    print("Release assets restored.")
    print("Expected paths now available:")
    print("  stable_runs/20260115_134958/exp1_clean/checkpoints/best")
    print("  stable_runs/20260115_134958/exp2_scanned/checkpoints/best")
    print("  AB_on_CD_results/ALL_AB_ON_CD_RESULTS.json")


if __name__ == "__main__":
    main()
