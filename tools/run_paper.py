import argparse, subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PINNED = {
    "clean": "stable_runs/20260115_134958/exp1_clean",
    "scanned": "stable_runs/20260115_134958/exp2_scanned",
}

ENABLE_MITIGATION = os.environ.get("ENABLE_MITIGATION", "0") == "1"

def _pinned_root() -> str:
    # stable_runs/<RUN>/exp{...} -> stable_runs/<RUN>
    return str(Path(PINNED["clean"]).resolve().parent)


def _ckpt_dir(run_dir: str) -> str:
    return str((Path(run_dir) / "checkpoints" / "best").as_posix())


def cmd(*args):
    return list(args)

SUITES = {

    # ---------------- TRAINING ----------------
    "table1": lambda mode: [
        cmd("python", "training/train.py", "--preset", "clean_scan"),
    ] if mode == "train" else [],

    # ---------------- EVALUATIONS ----------------
    "subset": lambda mode: [
        cmd("python", "-m", "paper_experiments.subset_softmax_eval",
            "--run-dir", PINNED["clean"]),
        cmd("python", "-m", "paper_experiments.subset_softmax_eval",
            "--run-dir", PINNED["scanned"]),
    ],

    "roi": lambda mode: [
        cmd(
            "python",
            "-m",
            "evaluation.test_AB_on_CD",
            "--model-a-dir",
            _ckpt_dir(PINNED["clean"]),
            "--model-b-dir",
            _ckpt_dir(PINNED["scanned"]),
            "--out-dir",
            "AB_on_CD_results",
        ),
    ],

    "error_decomp": lambda mode: [
        cmd(
            "python",
            "-m",
            "paper_experiments.error_decomposition",
            "--run",
            f"name=clean,ckpt={_ckpt_dir(PINNED['clean'])},image=model_baked.png",
            "--run",
            f"name=scanned,ckpt={_ckpt_dir(PINNED['scanned'])},image=F1_scaled.png",
            "--out-dir",
            "paper_experiments/out/error_compare",
        ),
    ],

    "size_success": lambda mode: [
        cmd(
            "python",
            "-m",
            "paper_experiments.size_success",
            "--run",
            f"name=clean,ckpt={_ckpt_dir(PINNED['clean'])},image=model_baked.png",
            "--run",
            f"name=scanned,ckpt={_ckpt_dir(PINNED['scanned'])},image=F1_scaled.png",
            "--out-dir",
            "paper_experiments/out/size_compare",
        ),
    ],

    "factorized": lambda mode: [
        cmd(
            "python",
            "-m",
            "paper_experiments.factorized_degradation",
            "--ckpt",
            _ckpt_dir(PINNED["clean"]),
            "--image-name",
            "model_baked.png",
            "--out-dir",
            "paper_experiments/out/factorized_pinned",
        ),
    ],

    "mitigation": lambda mode: [
        cmd("bash", "mitigation/run_mechanism_mitigation.sh", _pinned_root()),
    ] if (mode == "train" and ENABLE_MITIGATION) else [],

    "cross_arch": lambda mode: [
        cmd("python", "training/run_all_detectors.py") if mode == "train" else [],
        cmd("python", "-m", "paper_experiments.make_cross_arch_signature"),
    ],

    "all_figures": lambda mode: [
        cmd("python", "-m", "paper_experiments.make_visual_suite"),
        cmd("python", "-m", "paper_experiments.make_paper_visuals"),
    ],

    "all_tables": lambda mode: [
        cmd("python", "-m", "paper_experiments.make_paper_tables"),
    ],

    # ---------------- FULL PAPER ----------------
    "paper_all": lambda mode: (
        SUITES["table1"](mode)
        + SUITES["subset"](mode)
        + SUITES["roi"](mode)
        + SUITES["error_decomp"](mode)
        + SUITES["size_success"](mode)
        + SUITES["factorized"](mode)
        + SUITES["mitigation"](mode)
        + SUITES["cross_arch"](mode)
        + SUITES["all_figures"](mode)
        + SUITES["all_tables"](mode)
    ),
}

def run_cmd(cmd, dry=False):
    if not cmd:
        return 0
    if cmd[:2] == ["bash", "mitigation/run_mechanism_mitigation.sh"] and not ENABLE_MITIGATION:
        print("\n[SKIP] mitigation (set ENABLE_MITIGATION=1 to enable training)")
        return 0
    print("\n[RUN] " + " ".join(cmd))
    if dry:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    p = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return p.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=SUITES.keys())
    ap.add_argument("--mode", choices=["pinned", "train"], default="pinned")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    commands = SUITES[args.suite](args.mode)

    for cmd in commands:
        rc = run_cmd(cmd, dry=args.dry)
        if rc != 0:
            print(f"[ERROR] Command failed: {' '.join(cmd)}")
            sys.exit(rc)

    print(f"\n[OK] Suite finished: {args.suite} ({args.mode})")

if __name__ == "__main__":
    main()
