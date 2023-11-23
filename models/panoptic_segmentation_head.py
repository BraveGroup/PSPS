import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (Linear, build_activation_layer,
                      bias_init_with_prob, constant_init)
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.dense_heads.detr_head import DETRHead
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.core import (build_assigner, build_sampler, 
                                    multi_apply, reduce_mean,
                                    bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.models.builder import HEADS, build_loss
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
import copy

from .mask_detr_head import MaskDETRHead
from .utils import cprint

@HEADS.register_module()
class PanopticSegmentationHead(MaskDETRHead):
    def __init__(self,
                 num_thing_classes,
                 num_stuff_classes,
                 in_channels,
                 num_query=100,
                 transformer=None,
                 thing_quality_threshold=0.25,
                 stuff_quality_threshold=0.25,
                 thing_overlap_threshold=0.4,
                 stuff_overlap_threshold=0.2,
                 thing_mask_decoder=dict(
                    type='MaskTransformerDecoder',
                    num_layers=4,
                    self_attn=False),
                 stuff_mask_decoder=dict(
                     type='MaskTransformerDecoder',
                     num_layers=6,
                     self_attn=True),
                 loss_mask=dict(type='DiceLoss', loss_weight=2.0, activate=False),
                 **kwargs):
        super(PanopticSegmentationHead, self).__init__(
            num_thing_classes=num_thing_classes,
            num_stuff_classes=num_stuff_classes,
            in_channels=in_channels,
            num_query=num_query,
            transformer=transformer,
            **kwargs)
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        #self.num_all_classes = num_thing_classes + num_stuff_classes
        self.thing_quality_threshold = thing_quality_threshold
        self.stuff_quality_threshold = stuff_quality_threshold
        self.thing_overlap_threshold = thing_overlap_threshold
        self.stuff_overlap_threshold = stuff_overlap_threshold

        self.loss_mask = build_loss(loss_mask)

        self.thing_mask_decoder = build_transformer(thing_mask_decoder)
        self.stuff_mask_decoder = build_transformer(stuff_mask_decoder)
        #self.semantic_mask_decoder = build_transformer(stuff_mask_decoder)

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        super(PanopticSegmentationHead, self)._init_layers()

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        #self.semantic_query = nn.Embedding(
        #    self.num_all_classes, self.embed_dims * 2)
        self.stuff_query = nn.Embedding(
            self.num_stuff_classes, self.embed_dims * 2)

        fc_cls = Linear(self.embed_dims, 1)
        self.stuff_cls_branches = _get_clones(
            fc_cls, self.stuff_mask_decoder.num_layers)
        #self.semantic_cls_branches = _get_clones(
        #    fc_cls, self.semantic_mask_decoder.num_layers)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        super(PanopticSegmentationHead, self).init_weights()

    @force_fp32(apply_to=('x',))
    def forward_train(self,
                      img,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """Forward function for training mode
        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        *outs, outs_for_mask = self.forward(x, img_metas)
        if gt_labels is None:
            loss_inputs = tuple(outs) + (gt_bboxes, img_metas)
        else:
            loss_inputs = tuple(outs) + (gt_bboxes, gt_labels, img_metas)

        # Deformable DETR's box loss
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # Mask losses
        outputs_classes, outputs_coords, \
            enc_outputs_class, enc_outputs_coord = outs
        memory, memory_mask, memory_pos, \
            query, query_pos, hw_lvl = outs_for_mask

        losses_thing, num_thing = self.loss_thing_masks(outputs_classes[-1],
                                             outputs_coords[-1],
                                             memory,
                                             memory_mask,
                                             query,
                                             hw_lvl,
                                             gt_bboxes,
                                             gt_labels,
                                             gt_masks,
                                             img_metas)
        losses_stuff, num_stuff = self.loss_stuff_masks(memory,
                                             memory_mask,
                                             hw_lvl,
                                             gt_semantic_seg,
                                             include_thing_classes=False)

        thing_weight = num_thing / (num_thing + num_stuff)
        stuff_weight = num_stuff / (num_thing + num_stuff)

        losses = {k: v * thing_weight for k, v in losses.items()}
        losses.update({k: v * thing_weight for k, v in losses_thing.items()})
        losses.update({k: v * stuff_weight for k, v in losses_stuff.items()})
        return losses

    def loss_thing_masks(self,
                  cls_scores,
                  bbox_preds,
                  memory,
                  memory_mask,
                  query,
                  hw_lvl,
                  gt_bboxes_list,
                  gt_labels_list,
                  gt_masks_list,
                  img_metas,
                  ):
        # Only matched queries are used to predict masks.
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, None)
        labels_list, label_weights_list, \
        bbox_targets_list, bbox_weights_list, \
        num_total_pos, num_total_neg, \
        pos_inds_list, gt_inds_list = cls_reg_targets

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = cls_scores.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        mask_targets = [mask[inds] for mask, inds in zip(gt_masks_list, gt_inds_list)]
        mask_targets = torch.cat(mask_targets).float()
        mask_weights = [inds.new_ones(inds.shape[0]) for inds in gt_inds_list]
        mask_weights = torch.cat(mask_weights)

        # thing mask decoding
        max_query_num = max(len(pos_inds) for pos_inds in pos_inds_list)
        bs, _, dims = query.shape
        thing_query = query.new_zeros([bs, max_query_num, dims])
        for i, pos_inds in enumerate(pos_inds_list):
            thing_query[i, :len(pos_inds)] = query[i, pos_inds]

        thing_query_list, thing_mask_list = self.thing_mask_decoder(
            memory=memory,
            memory_mask=memory_mask,
            query=thing_query,
            query_pos=None,
            hw_lvl=hw_lvl)

        thing_masks = []
        for masks in thing_mask_list:
            assert masks.shape == masks.squeeze(-1).shape, masks.shape
            masks = [x[:len(pos_inds)].view(-1, *hw_lvl[0]) for \
                     x, pos_inds in zip(masks, pos_inds_list)]
            thing_masks.append(torch.cat(masks))
        thing_masks = torch.stack(thing_masks)  # [nDecoder, nQuery*nImg, h, w]
        thing_masks = F.interpolate(thing_masks, mask_targets.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # compute loss
        losses = dict()
        for i, masks in enumerate(thing_masks):
            loss_mask = self.loss_mask(masks,
                                       mask_targets,
                                       mask_weights,
                                       avg_factor=num_total_pos)
            losses[f'd{i}.loss_thing_mask'] = loss_mask

        # to make pytorch happy with unused parameters
        losses['loss_dummy'] = thing_query_list[-1].sum() * 0
        return losses, num_total_pos

    def loss_stuff_masks(self,
                         memory,
                         memory_mask,
                         hw_lvl,
                         gt_semantic_seg,
                         include_thing_classes=False):
        bs, _, dims = memory.shape

        # prepare targets
        num_classes = self.num_thing_classes + self.num_stuff_classes
        stuff_targets = F.one_hot(gt_semantic_seg.long(), 256)[..., :num_classes]
        if not include_thing_classes:
            stuff_targets = stuff_targets[..., self.num_thing_classes:]
        stuff_targets = stuff_targets.permute(0, 3, 1, 2).float().flatten(0, 1)
        stuff_weights = stuff_targets.max(2)[0].max(1)[0]
        num_total_pos = stuff_weights.sum()
        num_total_pos = stuff_weights.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        stuff_cls_targets = (1 - stuff_weights).long()

        # get prediction
        stuff_query, stuff_query_pos = torch.split(
            self.stuff_query.weight[None].expand(bs, -1, -1),
            dims, -1)
        stuff_query_list, stuff_mask_list = self.stuff_mask_decoder(
            memory=memory,
            memory_mask = memory_mask,
            query=stuff_query,
            query_pos=stuff_query_pos,
            hw_lvl=hw_lvl)

        # masks
        stuff_masks = []
        for masks in stuff_mask_list:
            assert masks.shape == masks.squeeze(-1).shape, masks.shape
            stuff_masks.append(masks.reshape(-1, *hw_lvl[0]))
        stuff_masks = torch.stack(stuff_masks) # [nDecoder, nQuery*nImg, h, w]
        stuff_masks = F.interpolate(stuff_masks, stuff_targets.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # classes
        stuff_clses = []
        for i, query in enumerate(stuff_query_list):
            query_cls = self.stuff_cls_branches[i](query).view(-1, 1)
            stuff_clses.append(query_cls)

        # compute loss
        losses = dict()
        for i, masks in enumerate(stuff_masks):
            loss_mask = self.loss_mask(masks,
                                       stuff_targets,
                                       stuff_weights,
                                       avg_factor=num_total_pos)
            losses[f'd{i}.loss_stuff_mask'] = loss_mask

        for i, stuff_cls in enumerate(stuff_clses):
            loss_cls = self.loss_cls(stuff_cls,
                                     stuff_cls_targets,
                                     avg_factor=num_total_pos)
            losses[f'd{i}.loss_stuff_cls'] = loss_cls * 2

        return losses, num_total_pos

    @force_fp32(apply_to=('x',))
    def simple_test(self, x, img_metas, rescale=False):
        *outs, outs_for_mask = self.forward(x, img_metas)

        outputs_classes, outputs_coords, \
            enc_outputs_class, enc_outputs_coord = outs

        memory, memory_mask, memory_pos, \
            query, query_pos, hw_lvl = outs_for_mask

        # get boxes
        cls_scores = outputs_classes[-1]
        bbox_preds = outputs_coords[-1]

        det_result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            det_result_list.append(proposals[:2])

            # decode thing masks
            bch_shape = img_metas[img_id]['batch_input_shape']
            ori_shape = img_metas[img_id]['ori_shape'][:2]

            det_bboxes, det_labels, bbox_index = proposals

            thing_query = query[img_id:img_id+1, bbox_index]
            thing_query_list, thing_mask_list = self.thing_mask_decoder(
                memory=memory,
                memory_mask=memory_mask,
                query=thing_query,
                query_pos=None,
                hw_lvl=hw_lvl)

            # decode stuff masks
            stuff_query, stuff_query_pos = self.stuff_query.weight[None].split(self.embed_dims, -1)
            stuff_query_list, stuff_mask_list = self.stuff_mask_decoder(
                memory=memory,
                memory_mask = memory_mask,
                query=stuff_query,
                query_pos=stuff_query_pos,
                hw_lvl=hw_lvl)
            
            # merge masks
            pred_masks = torch.cat([thing_mask_list[-1],
                                    stuff_mask_list[-1]], 1)
            pred_masks = pred_masks.view(-1, *hw_lvl[0])
            pred_masks = F.interpolate(pred_masks[None], bch_shape,
                                       mode='bilinear',
                                       align_corners=False)[0]
            pred_masks = pred_masks[..., :img_shape[0], :img_shape[1]]
            pred_masks = F.interpolate(pred_masks[None], ori_shape,
                                       mode='bilinear',
                                       align_corners=False)[0]

            # merge scores
            stuff_score = self.stuff_cls_branches[-1](stuff_query_list[-1])
            stuff_score = stuff_score.sigmoid().view(-1)
            thing_score = det_bboxes[..., -1]
            scores_cls = torch.cat([thing_score, stuff_score])

            binary_masks = pred_masks > 0.5
            mask_sizes = binary_masks.sum((1, 2)).float()
            scores_msk = (pred_masks * binary_masks).sum((1, 2)) / \
                            (mask_sizes + 1)

            pred_scores = scores_cls * (scores_msk**2)

            # masks' labels
            thing_labels = det_labels
            stuff_labels = torch.arange(self.num_stuff_classes,
                                        device=det_labels.device) + self.num_thing_classes
            pred_labels = torch.cat([thing_labels, stuff_labels])

            # post processing, greedy filling
            num_classes = self.num_thing_classes + self.num_stuff_classes
            scores_sorted, index = torch.sort(pred_scores, descending=True)
            filled = binary_masks.new_zeros(pred_masks.shape[-2:]).bool()
            panoptic_map = torch.full(pred_masks.shape[-2:],
                                      num_classes,
                                      device=filled.device).long()
            panoptic_id = 1

            for i, score in zip(index, scores_sorted):
                L = pred_labels[i]
                # assume things are always before stuff
                is_thing = L < self.num_thing_classes
                if is_thing:
                    score_threshold = self.thing_quality_threshold 
                    overlap_threshold = self.thing_overlap_threshold
                else:
                    score_threshold = self.stuff_quality_threshold
                    overlap_threshold = self.stuff_overlap_threshold

                if score < score_threshold:
                    continue

                area = mask_sizes[i]
                if area == 0:
                    continue
                
                overlap_area = (binary_masks[i] & filled).sum()
                if (overlap_area / area) > overlap_threshold:
                    continue

                mask = binary_masks[i] & (~filled)
                filled[mask] = True
                panoptic_map[mask] = panoptic_id * INSTANCE_OFFSET + L
                panoptic_id += 1

        result_list = [dict(
            pan_results=panoptic_map.data.cpu().numpy())]
        return result_list

