from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from config import COCO_MAX_DETS
import numpy as np


def coco_evaluate(dataset, anno_json_path, pred_json_path, iou_thrs=None, return_coco=False):
    coco_gt = COCO(anno_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)

    cocoEval = COCOeval(coco_gt, coco_dt, "bbox")
    cocoEval.params.maxDets = COCO_MAX_DETS
    if iou_thrs is not None:
        cocoEval.params.iouThrs = np.array(iou_thrs, dtype=np.float64)

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if return_coco:
        return cocoEval.stats, cocoEval

    return cocoEval.stats
