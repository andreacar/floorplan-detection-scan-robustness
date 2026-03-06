#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def main() -> None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Back-compat wrapper (older repos called this "ROI variants" eval).
    # The maintained implementation lives in `evaluation/test_AB_on_CD.py`.
    #
    # Historical CLI: `python -m evaluation.eval_roi_variants --run-dir <.../exp2_scanned>`
    # We translate that into explicit model checkpoint dirs for A (clean) + B (scan).

    ap = argparse.ArgumentParser(description="ROI variants eval (compat wrapper).")
    ap.add_argument(
        "--run-dir",
        default="",
        help="Either master run dir containing exp1_clean/exp2_scanned or an exp dir itself.",
    )
    ap.add_argument("--model-a-dir", default="", help="Override model A checkpoint dir (HF format).")
    ap.add_argument("--model-b-dir", default="", help="Override model B checkpoint dir (HF format).")
    ap.add_argument("--out-dir", default="AB_on_CD_results", help="Output directory.")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional variant image filenames (defaults to C/D).",
    )
    args, unknown = ap.parse_known_args()

    model_a = args.model_a_dir
    model_b = args.model_b_dir
    if args.run_dir and (not model_a or not model_b):
        run_dir = os.path.abspath(args.run_dir)
        base = run_dir
        leaf = os.path.basename(run_dir.rstrip(os.sep))
        if leaf in {"exp1_clean", "exp2_scanned"}:
            base = os.path.dirname(run_dir)
        cand_a = os.path.join(base, "exp1_clean", "checkpoints", "best")
        cand_b = os.path.join(base, "exp2_scanned", "checkpoints", "best")
        if not model_a:
            model_a = cand_a
        if not model_b:
            model_b = cand_b

    if not model_a or not model_b:
        raise SystemExit(
            "Missing model checkpoint dirs.\n"
            "Provide either:\n"
            "- --run-dir <...> (with exp1_clean/exp2_scanned inside), or\n"
            "- --model-a-dir and --model-b-dir."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir
    cmd = [
        sys.executable,
        "-m",
        "evaluation.test_AB_on_CD",
        "--model-a-dir",
        model_a,
        "--model-b-dir",
        model_b,
        "--out-dir",
        args.out_dir,
    ]
    if args.variants:
        cmd += ["--variants"] + list(args.variants)
    cmd += unknown

    subprocess.run(cmd, check=True, cwd=root_dir, env=env)


if __name__ == "__main__":
    main()
