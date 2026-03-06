#!/usr/bin/env python3
import os
import subprocess
import sys
from datetime import datetime


def main() -> None:
    if len(sys.argv) > 1:
        detectors = [arg.strip().lower() for arg in sys.argv[1:] if arg.strip()]
    else:
        detectors = ["rtdetr", "fasterrcnn", "retinanet"]

    script_path = os.path.join(os.path.dirname(__file__), "train.py")
    base_env = os.environ.copy()

    for det in detectors:
        run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{det}"
        env = base_env.copy()
        print(f"\n=== Running DETECTOR={det} (RUN_NAME={run_name}) ===\n")
        subprocess.run(
            [sys.executable, script_path, "--preset", "clean_scan", "--run-name", run_name, "--detector", det],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
