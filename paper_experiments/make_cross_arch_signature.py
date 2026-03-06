#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Cross-architecture robustness signature plot.")
    parser.add_argument("--out-dir", default="paper_experiments/out/cross_arch_signature")
    parser.add_argument("--prefix", default="cross_arch_signature_delta")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Values from Table 5 (clean vs scanned)
    models = ["RT-DETR", "Faster R-CNN", "RetinaNet"]
    clean_ap50 = np.array([0.897, 0.900, 0.853])
    clean_ap75 = np.array([0.867, 0.866, 0.745])
    scan_ap50 = np.array([0.577, 0.541, 0.488])
    scan_ap75 = np.array([0.412, 0.352, 0.271])

    delta_ap50 = clean_ap50 - scan_ap50
    delta_ap75 = clean_ap75 - scan_ap75

    print("=== Cross-architecture ΔAP (clean - scanned) ===")
    for name, d50, d75 in zip(models, delta_ap50, delta_ap75):
        print(f"{name:12s}  ΔAP50={d50:.3f}  ΔAP75={d75:.3f}")

    # Style: LaTeX-like serif
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    x = np.arange(len(models))
    width = 0.34

    fig, ax = plt.subplots(figsize=(3.3, 2.2), dpi=300)
    ax.bar(x - width / 2, delta_ap50, width, label="ΔAP50", color="#4C78A8")
    ax.bar(x + width / 2, delta_ap75, width, label="ΔAP75", color="#F58518")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12, ha="right")
    ax.set_ylabel("Clean − Scanned")
    ax.set_ylim(0, max(delta_ap50.max(), delta_ap75.max()) + 0.05)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Robustness signature across architectures", pad=6)

    fig.tight_layout()
    pdf_path = os.path.join(out_dir, f"{args.prefix}.pdf")
    svg_path = os.path.join(out_dir, f"{args.prefix}.svg")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
