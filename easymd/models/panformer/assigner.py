import torch
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core import mask
from mmdet.utils import util_mixins


@BBOX_ASSIGNERS.register_module()
class HungarianAssignerMultiSample(BaseAssigner):
    """Computes Hungarian assignment for multiple samples efficiently.
    """

    def __init__(self,
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            mask_cost=dict(type='DiceCost', weight=2.0),
            ):
        build_ = lambda cfg: build_match_cost(cfg) if cfg is not None else None

        self.cls_cost = build_(cls_cost)
        self.reg_cost = build_(reg_cost)
        self.iou_cost = build_(iou_cost)
        self.mask_cost = build_(mask_cost)

    @torch.no_grad()
    def assign(self,
            bbox_pred,
            cls_pred,
            mask_pred,
            gt_bboxes,
            gt_labels,
            gt_masks,
            img_meta,
            gt_bboxes_ignore=None,
            eps=1e-5
            ):
        """Computes Hungarian assignment for multiple samples efficiently.

        Args:
            bbox_pred (Tensor | List[Tensor]): predicted bboxes (cx, cy, w, h) with 
                normalized coorinates in range [0, 1].
                Shape [bsz, num_query, 4] or List[[num_query, 4]]
            cls_pred (Tensor | List[Tensor]): predicted classification logits.
                Shape [bsz, num_query, num_classes] or List[[num_query, num_classes]]
            gt_bboxes (List[Tensor]): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape List[[num_gt, 4]]
            gt_labels (List[Tensor]): Ground truth class labels. Shape List[[num_gt,]]
            img_meta List[(dict)]: Meta information for the images.
            gt_bboxes_ignore ([Tensor], optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.

        Returns:
            List[ AssignResult ]: The assigned result.
        """
        assert gt_bboxes_ignore is None
        has_mask = mask_pred is not None
        if has_mask:
            assert self.mask_cost is not None
        else:
            assert self.mask_cost is None

        # flatten all the data
        sizes_src = [len(x) for x in cls_pred]
        sizes_tgt = [len(x) for x in gt_labels]
        bbox_pred = torch.cat(list(bbox_pred))
        cls_pred = torch.cat(list(cls_pred))
        gt_bboxes = torch.cat(list(gt_bboxes))
        gt_labels = torch.cat(list(gt_labels))

        if has_mask:
            mask_pred = torch.cat(list(mask_pred))
            gt_masks = torch.cat(list(gt_masks))

        # compute costs 
        factor = torch.as_tensor([x['img_shape'][:2][::-1] for x in img_meta],
                dtype=bbox_pred.dtype, device=bbox_pred.device).repeat(1, 2) # [w, h, w, h]
        factor_src = torch.cat([x[None].expand(size, -1, -1) for x, size in zip(factor, sizes_src)])
        factor_tgt = torch.cat([x[None].expand(size, -1, -1) for x, size in zip(factor, sizes_tgt)])

        cls_cost = self.cls_cost(cls_pred, gt_labels)
        reg_cost = self.reg_cost(bbox_pred, gt_bboxes / factor_src)
        iou_cost = self.iou_cost(bbox_cxcywh_to_xyxy(bbox_pred) * factor_tgt, gt_bboxes)
        cost = cls_cost + reg_cost + iou_cost

        if has_mask:
            mask_cost = self.mask_cost(mask_pred, gt_mask)
            cost = cost + mask_cost

        # Hungarian matching
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        cost = cost.detach().data.cpu()
        results = []
        gt_inds_list = torch.full((bbox_pred.shape[0],), -1, dtype=torch.long, device=bbox_pred.device)
        assign_labels = gt_inds_list.clone().split(sizes_src)
        gt_inds_list = gt_inds_list.split(sizes_src)

        gt_labels = gt_labels.split(sizes_tgt)
        if has_mask:
            gt_masks = gt_masks.split(sizes_tgt)

        for i, cost_ in enumerate(cost.split(sizes_tgt, 1)):
            gt_inds = gt_inds_list[i]
            labels = assign_labels[i]

            if sizes_src[i] == 0:
                results.append(AssignResult(sizes_tgt[i], gt_inds, None, labels))
            elif sizes_tgt[i] == 0:
                gt_inds[:] = 0
                results.append(AssignResult(sizes_tgt[i], gt_inds, None, labels))
            else:
                row_idx, col_idx = linear_sum_assignment(cost_[i])
                row_idx = torch.from_numpy(row_idx).to(bbox_pred.device)
                col_idx = torch.from_numpy(col_idx).to(bbox_pred.device)

                gt_inds[:] = 0
                gt_inds[row_idx] = col_idx + 1
                labels[row_idx] = gt_labels[i][col_idx]
                results.append(AssignResult(sizes_tgt[i], gt_inds, None, labels))

            # NOTE: hack to store masks
            if has_mask:
                results[-1].masks = gt_masks[i][col_idx]
        return results

