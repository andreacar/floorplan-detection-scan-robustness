#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset_release import (
    DATASET_CACHE_DIR_ENV,
    default_cache_dir,
    ensure_dataset_available,
    release_metadata,
    verify_remote_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract the published CubiCasa5k-ScanShift dataset release."
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Override the dataset cache directory. Defaults to ~/.cache/floorplan-detection-scan-robustness/datasets.",
    )
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help="Verify that the published Zenodo release endpoints are reachable without downloading the archive.",
    )
    args = parser.parse_args()

    if args.check_remote:
        checks = verify_remote_release()
        for name, info in checks.items():
            print(f"{name}:")
            print(f"  status: {info['status']}")
            print(f"  size: {info['content_length']}")
            print(f"  type: {info['content_type']}")
            print(f"  url: {info['url']}")
        return

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else default_cache_dir()
    dataset_root = ensure_dataset_available(cache_dir)
    meta = release_metadata()

    print(f"Downloaded/extracted release {meta['version']}")
    print(f"Dataset root: {dataset_root}")
    print(f"Archive cache: {cache_dir / meta['archive_name']}")
    print(f"Summary file: {cache_dir / meta['summary_name']}")
    print(f"Checksums file: {cache_dir / meta['sha256sums_name']}")
    print("Use this dataset explicitly with:")
    print(f"  export DATASET_BASE_DIR={dataset_root}")
    print("Or rely on the default cache automatically and optionally override")
    print(f"  {DATASET_CACHE_DIR_ENV}={cache_dir}")


if __name__ == "__main__":
    main()
