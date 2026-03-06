#!/usr/bin/env python3
import os
import subprocess
import sys


def main() -> None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Back-compat wrapper; the maintained script is in this repo.
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir
    cmd = [sys.executable, "-m", "paper_experiments.degradation_sweep"] + sys.argv[1:]
    subprocess.run(cmd, check=True, cwd=root_dir, env=env)


if __name__ == "__main__":
    main()
