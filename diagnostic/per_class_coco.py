#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def mean_ignore_neg(x):
    x = x[x > -1]
    return float(np.mean(x)) if x.size else float("nan")


def main(ann, pred):
    cocoGt = COCO(ann)
    cocoDt = cocoGt.loadRes(pred)

    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.maxDets = [1, 10, 100]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    cats = cocoGt.loadCats(cocoGt.getCatIds())
    cat_ids = [c["id"] for c in cats]
    cat_names = [c["name"] for c in cats]

    precision = cocoEval.eval["precision"]  # [T,R,K,A,M]
    recall = cocoEval.eval["recall"]        # [T,K,A,M]
    ious = cocoEval.params.iouThrs
    maxdets = cocoEval.params.maxDets

    # area index 0 = "all", maxDet index for 100
    a = 0
    m = maxdets.index(100)

    # IoU indices for 0.50 and 0.75
    t50 = int(np.where(np.isclose(ious, 0.50))[0][0])
    t75 = int(np.where(np.isclose(ious, 0.75))[0][0])

    print("\n=== Per-class COCO (area=all, maxDet=100) ===")
    for k, name in enumerate(cat_names):
        ap = mean_ignore_neg(precision[:, :, k, a, m])
        ap50 = mean_ignore_neg(precision[t50, :, k, a, m])
        ap75 = mean_ignore_neg(precision[t75, :, k, a, m])
        ar = float(recall[:, k, a, m].mean())
        print(f"{name:>10s} | AP={ap:.3f} AP50={ap50:.3f} AP75={ap75:.3f} AR={ar:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True)
    p.add_argument("--pred", required=True)
    args = p.parse_args()
    main(args.ann, args.pred)
