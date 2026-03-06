import torch
from torch_geometric.data import HeteroData

NODE_TYPE = "STRUCT"
REL_NAMES = ["contains", "overlaps", "touches", "near", "opening_on", "room_adjacent", "knn"]
EDGE_DIM = 9
EPS = 1e-6


def _xyxy_to_cxcywh_norm(boxes_xyxy: torch.Tensor, w: int, h: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
    cx = (x1 + x2) * 0.5 / float(w)
    cy = (y1 + y2) * 0.5 / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    x = torch.stack([cx, cy, bw, bh], dim=1)
    x[:, 0:2] = x[:, 0:2].clamp(0.0, 1.0)
    x[:, 2:4] = x[:, 2:4].clamp(1e-4, 1.0)
    return x


def _cxcywh_norm_to_xyxy_px(x: torch.Tensor, w: int, h: int) -> torch.Tensor:
    cx, cy, bw, bh = x.unbind(dim=1)
    x1 = (cx - bw * 0.5) * float(w)
    y1 = (cy - bh * 0.5) * float(h)
    x2 = (cx + bw * 0.5) * float(w)
    y2 = (cy + bh * 0.5) * float(h)
    out = torch.stack([x1, y1, x2, y2], dim=1)
    out[:, 0] = out[:, 0].clamp(0.0, float(w))
    out[:, 2] = out[:, 2].clamp(0.0, float(w))
    out[:, 1] = out[:, 1].clamp(0.0, float(h))
    out[:, 3] = out[:, 3].clamp(0.0, float(h))
    return out


def _edge_attr_batch(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, is_reverse_flag: float) -> torch.Tensor:
    # x: [N,4] normalized cxcywh
    src = x[u]
    dst = x[v]
    scx, scy, sw, sh = src.unbind(dim=1)
    tcx, tcy, tw, th = dst.unbind(dim=1)

    sw = sw.clamp(min=1e-4); sh = sh.clamp(min=1e-4)
    tw = tw.clamp(min=1e-4); th = th.clamp(min=1e-4)

    dx = tcx - scx
    dy = tcy - scy
    dist = torch.sqrt(dx * dx + dy * dy + EPS)

    sx1 = scx - sw * 0.5
    sy1 = scy - sh * 0.5
    sx2 = scx + sw * 0.5
    sy2 = scy + sh * 0.5

    tx1 = tcx - tw * 0.5
    ty1 = tcy - th * 0.5
    tx2 = tcx + tw * 0.5
    ty2 = tcy + th * 0.5

    ix1 = torch.maximum(sx1, tx1)
    iy1 = torch.maximum(sy1, ty1)
    ix2 = torch.minimum(sx2, tx2)
    iy2 = torch.minimum(sy2, ty2)
    iw = (ix2 - ix1).clamp(min=0.0)
    ih = (iy2 - iy1).clamp(min=0.0)
    inter = iw * ih

    area_s = (sx2 - sx1).clamp(min=0.0) * (sy2 - sy1).clamp(min=0.0)
    area_t = (tx2 - tx1).clamp(min=0.0) * (ty2 - ty1).clamp(min=0.0)
    union = area_s + area_t - inter
    iou = inter / (union + EPS)

    log_w = torch.log((tw + EPS) / (sw + EPS))
    log_h = torch.log((th + EPS) / (sh + EPS))

    t_contains_s_center = ((tx1 <= scx) & (scx <= tx2) & (ty1 <= scy) & (scy <= ty2)).float()
    s_contains_t_center = ((sx1 <= tcx) & (tcx <= sx2) & (sy1 <= tcy) & (tcy <= sy2)).float()

    geom8 = torch.stack([dx, dy, dist, iou, log_w, log_h, t_contains_s_center, s_contains_t_center], dim=1)
    is_rev = torch.full((geom8.size(0), 1), float(is_reverse_flag), device=x.device, dtype=torch.float32)
    return torch.cat([is_rev, geom8], dim=1)  # [E,9]


def _knn_edges(x: torch.Tensor, k: int) -> torch.Tensor:
    # returns edge_index [2,E] directed
    N = x.size(0)
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    centers = x[:, 0:2]
    dist = torch.cdist(centers, centers)  # [N,N]
    # include self then drop
    nn = torch.topk(dist, k=min(k + 1, N), largest=False).indices  # [N,k+1]

    u_list = []
    v_list = []
    for u in range(N):
        for v in nn[u].tolist():
            if v == u:
                continue
            u_list.append(u)
            v_list.append(v)

    return torch.tensor([u_list, v_list], dtype=torch.long, device=x.device)


def _support_from_edges(N: int, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # undirected support
    sup = torch.zeros((N,), device=u.device, dtype=torch.float32)
    sup.index_add_(0, u, w)
    sup.index_add_(0, v, w)
    return sup


def _subset_reindex(keep: torch.Tensor) -> torch.Tensor:
    # keep: [N] bool -> map old_idx -> new_idx, -1 for dropped
    N = keep.numel()
    mapping = torch.full((N,), -1, dtype=torch.long, device=keep.device)
    mapping[keep] = torch.arange(int(keep.sum().item()), device=keep.device, dtype=torch.long)
    return mapping


class GraphPostprocessOption1:
    """
    Option 1:
      (1) high-recall detector boxes -> nodes
      (2) predict edges (Stage-B) on KNN candidates
      (3) structural support filter to remove junk
      (4) predict edges again on filtered nodes
      (5) refine geometry (Stage-B2) using predicted edges, confidence-weighted aggregation
    """
    def __init__(
        self,
        edge_pred,
        geom_refiner,
        device: str = "cuda",
        knn_k: int = 10,
        edge_thr: float = 0.45,
        support_thr: float = 0.8,
        keep_score_thr: float = 0.25,
        add_reverse_edges_for_encoder: bool = True,
    ):
        self.edge_pred = edge_pred.to(device).eval()
        self.geom_refiner = geom_refiner.to(device).eval()
        for p in self.edge_pred.parameters():
            p.requires_grad = False
        for p in self.geom_refiner.parameters():
            p.requires_grad = False

        self.device = device
        self.knn_k = int(knn_k)
        self.edge_thr = float(edge_thr)
        self.support_thr = float(support_thr)
        self.keep_score_thr = float(keep_score_thr)
        self.add_reverse = bool(add_reverse_edges_for_encoder)

    @torch.no_grad()
    def __call__(self, boxes_xyxy: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, w: int, h: int):
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy, scores, labels

        # move to device
        boxes_xyxy = boxes_xyxy.to(self.device)
        scores = scores.to(self.device)
        labels = labels.to(self.device)

        # nodes
        x0 = _xyxy_to_cxcywh_norm(boxes_xyxy, w, h)
        sem_id0 = labels.long().clamp(min=0, max=63)

        # initial graph with KNN edges for encoder context
        data0 = HeteroData()
        data0[NODE_TYPE].x = x0
        data0[NODE_TYPE].sem_id = sem_id0

        ei_knn = _knn_edges(x0, self.knn_k)
        if ei_knn.numel() == 0:
            return boxes_xyxy.detach().cpu(), scores.detach().cpu(), labels.detach().cpu()

        u_knn = ei_knn[0]
        v_knn = ei_knn[1]
        ea_knn = _edge_attr_batch(x0, u_knn, v_knn, is_reverse_flag=0.0)

        data0[(NODE_TYPE, "knn", NODE_TYPE)].edge_index = ei_knn
        data0[(NODE_TYPE, "knn", NODE_TYPE)].edge_attr = ea_knn

        # candidates = KNN edges (fast, good enough)
        exist_logits, type_logits = self.edge_pred(data0, u_knn, v_knn, ea_knn)
        p_exist = torch.sigmoid(exist_logits).float()

        keep_e = p_exist >= self.edge_thr
        if int(keep_e.sum().item()) == 0:
            # nothing structurally confident -> keep only high score boxes
            keep_n = scores >= self.keep_score_thr
            if int(keep_n.sum().item()) == 0:
                # fallback: keep top-1
                keep_n[scores.argmax()] = True
            boxes_out = boxes_xyxy[keep_n]
            scores_out = scores[keep_n]
            labels_out = labels[keep_n]
            return boxes_out.detach().cpu(), scores_out.detach().cpu(), labels_out.detach().cpu()

        u_e = u_knn[keep_e]
        v_e = v_knn[keep_e]
        p_e = p_exist[keep_e]
        rel_id = type_logits[keep_e].argmax(dim=1).long()

        # structural support filter
        N0 = x0.size(0)
        support = _support_from_edges(N0, u_e, v_e, p_e)
        keep_n = (support >= self.support_thr) | (scores >= self.keep_score_thr)

        if int(keep_n.sum().item()) < 2:
            # keep at least top-2 by score
            top = torch.argsort(scores, descending=True)[: min(2, scores.numel())]
            keep_n = torch.zeros_like(keep_n)
            keep_n[top] = True

        # subset nodes
        map_old2new = _subset_reindex(keep_n)
        x1 = x0[keep_n]
        sem1 = sem_id0[keep_n]
        scores1 = scores[keep_n]
        labels1 = labels[keep_n]

        # rebuild KNN on filtered nodes
        data1 = HeteroData()
        data1[NODE_TYPE].x = x1
        data1[NODE_TYPE].sem_id = sem1

        ei1 = _knn_edges(x1, self.knn_k)
        if ei1.numel() == 0:
            boxes_out = _cxcywh_norm_to_xyxy_px(x1, w, h)
            return boxes_out.detach().cpu(), scores1.detach().cpu(), labels1.detach().cpu()

        u1 = ei1[0]; v1 = ei1[1]
        ea1 = _edge_attr_batch(x1, u1, v1, 0.0)
        data1[(NODE_TYPE, "knn", NODE_TYPE)].edge_index = ei1
        data1[(NODE_TYPE, "knn", NODE_TYPE)].edge_attr = ea1

        # predict edges again on filtered nodes (cleaner)
        exist_logits2, type_logits2 = self.edge_pred(data1, u1, v1, ea1)
        p2 = torch.sigmoid(exist_logits2).float()
        keep2 = p2 >= self.edge_thr

        if int(keep2.sum().item()) == 0:
            boxes_out = _cxcywh_norm_to_xyxy_px(x1, w, h)
            return boxes_out.detach().cpu(), scores1.detach().cpu(), labels1.detach().cpu()

        u2 = u1[keep2]
        v2 = v1[keep2]
        ea2 = ea1[keep2]
        rel2 = type_logits2[keep2].argmax(dim=1).long()
        w2 = p2[keep2]  # confidence weights

        # add predicted relation edges into data1 for encoder context (optional but helps)
        if self.add_reverse:
            # create reverse edges for each predicted edge for message passing
            u_rev = v2
            v_rev = u2
            ea_rev = _edge_attr_batch(x1, u_rev, v_rev, 1.0)
        else:
            u_rev = v_rev = ea_rev = None

        # populate hetero edges by relation
        for rid, relname in enumerate(REL_NAMES):
            mask = rel2 == rid
            if int(mask.sum().item()) == 0:
                continue
            ei_rel = torch.stack([u2[mask], v2[mask]], dim=0)
            ea_rel = ea2[mask]
            data1[(NODE_TYPE, relname, NODE_TYPE)].edge_index = ei_rel
            data1[(NODE_TYPE, relname, NODE_TYPE)].edge_attr = ea_rel

            if self.add_reverse:
                ei_rel_r = torch.stack([u_rev[mask], v_rev[mask]], dim=0)
                ea_rel_r = ea_rev[mask]
                # append reverse edges to same relation type
                data1[(NODE_TYPE, relname, NODE_TYPE)].edge_index = torch.cat(
                    [data1[(NODE_TYPE, relname, NODE_TYPE)].edge_index, ei_rel_r], dim=1
                )
                data1[(NODE_TYPE, relname, NODE_TYPE)].edge_attr = torch.cat(
                    [data1[(NODE_TYPE, relname, NODE_TYPE)].edge_attr, ea_rel_r], dim=0
                )

        # geometry refinement deltas on predicted edges
        delta_e = self.geom_refiner(data1, u2, v2, ea2, rel2)  # [E,4]

        # confidence-weighted aggregation to destination nodes v2
        N1 = x1.size(0)
        sum_delta = torch.zeros((N1, 4), device=self.device, dtype=torch.float32)
        cnt = torch.zeros((N1, 1), device=self.device, dtype=torch.float32)
        sum_delta.index_add_(0, v2, delta_e * w2[:, None])
        cnt.index_add_(0, v2, w2[:, None])

        x_hat = x1 + sum_delta / cnt.clamp(min=1e-6)
        x_hat[:, 0:2] = x_hat[:, 0:2].clamp(0.0, 1.0)
        x_hat[:, 2:4] = x_hat[:, 2:4].clamp(1e-4, 1.0)

        boxes_out = _cxcywh_norm_to_xyxy_px(x_hat, w, h)
        return boxes_out.detach().cpu(), scores1.detach().cpu(), labels1.detach().cpu()
