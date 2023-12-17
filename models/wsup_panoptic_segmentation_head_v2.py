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

from .panoptic_segmentation_head import PanopticSegmentationHead

from .utils import (expand_target_masks, partial_cross_entropy_loss, 
                    color_prior_loss, neighbour_diff, image_to_boundary)
from pydijkstra import dijkstra_image
import numpy as np

from .utils import cprint
from .wsup_panoptic_segmentation_head import FeatureProjection,\
                                             WSupPanopticSegmentationHead
from .panoptic_segmentation_head import PanopticSegmentationHead

__all__ = ['WSupPanopticSegmentationHeadV2']

def ASSERT_EQ(x, y):
    assert x == y, f'Not Equal! {x} vs. {y}'

@HEADS.register_module()
class WSupPanopticSegmentationHeadV2(WSupPanopticSegmentationHead):
    """This is the share head version, where the 'stuff head' and
       the 'semantic head' are merged.
    """
    def __init__(self,
                 *args,
                 point_expand_size=17,
                 lambda_boundary=0.1,
                 lambda_embedding=0.1,
                 warmup_iter=700,
                 **kwargs):
        super(WSupPanopticSegmentationHead, self).__init__(*args, **kwargs)

        # additional hyperparameters
        self.point_expand_size = point_expand_size
        self.lambda_boundary = lambda_boundary
        self.lambda_embedding = lambda_embedding
        self.warmup_iter = warmup_iter

        # additional semantic queries, only thing queries are required
        self._semantic_query = nn.Embedding(self.num_thing_classes, self.embed_dims * 2)
        num_classes = self.num_thing_classes + self.num_stuff_classes
        self.semantic_proj = nn.Conv2d(num_classes, num_classes, 1, groups=num_classes)

        # additional feature embedding branch
        self.feature_proj = FeatureProjection(self.embed_dims, 128, 'l2')
        self.iter_count = 0

    # def init_weights(self):
    #     super(WSupPanopticSegmentationHead, self).init_weights()

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
        """Override to support for weakly supervised learning.
        Here, 'gt_masks', 'gt_sematic_seg' are all weak labels, with only
        sparse point annotations.
        'gt_bboxes' is never used.
        """

        *outs, outs_for_mask = self.forward(x, img_metas)

        outputs_classes, outputs_coords, \
            enc_outputs_class, enc_outputs_coord = outs
        memory, memory_mask, memory_pos, \
            query, query_pos, hw_lvl = outs_for_mask

        # generate pseudo labels
        losses_pl, pl_bboxes, pl_masks, pl_semantics = \
            self.forward_pseudo_label(img,
                                      memory,
                                      memory_mask,
                                      memory_pos,
                                      hw_lvl,
                                      gt_labels,
                                      gt_masks,
                                      gt_semantic_seg,
                                      img_metas)

        # Box loss
        if gt_labels is None:
            loss_inputs = tuple(outs) + (pl_bboxes, img_metas)
        else:
            loss_inputs = tuple(outs) + (pl_bboxes, gt_labels, img_metas)

        # Deformable DETR's box loss
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # Mask losses
        losses_thing, num_thing = self.loss_thing_masks(outputs_classes[-1],
                                             outputs_coords[-1],
                                             memory,
                                             memory_mask,
                                             query,
                                             hw_lvl,
                                             pl_bboxes,
                                             gt_labels,
                                             pl_masks,
                                             img_metas)
        # losses_stuff, num_stuff = self.loss_stuff_masks(memory,
        #                                      memory_mask,
        #                                      hw_lvl,
        #                                      pl_semantics,
        #                                      include_thing_classes=False)

        #num_stuff = self.num_stuff_classes + self.num_thing_classes
        #thing_weight = num_thing / (num_thing + num_stuff)
        #stuff_weight = num_stuff / (num_thing + num_stuff)
        thing_weight = stuff_weight = 1

        losses = {k: v * thing_weight for k, v in losses.items()}
        losses.update({k: v * thing_weight for k, v in losses_thing.items()})
        #losses.update({k: v * stuff_weight for k, v in losses_stuff.items()})

        # Finally, the new pseudo-label related losses
        # Adjust the losses by the warm-up strategy
        weight = max(min(self.iter_count / self.warmup_iter, 1), 0)
        self.iter_count += 1
        losses = {k: v * weight for k, v in losses.items()}

        # losses_pl mixes pl-loss and stuff-loss
        for k, v in losses_pl.items():
            if 'stuff' in k:
                losses[k] = v * (weight * stuff_weight)
            else:
                losses[k] = v

        return losses

    def forward_pseudo_label(self,
                             images,
                             memory,
                             memory_mask,
                             memory_pos,
                             hw_lvl,
                             gt_labels,
                             gt_masks,
                             gt_semantic_seg,
                             img_metas):
        pred_semantic, given_semantic, semantic_existence, \
            loss_sem_pre = self.get_semantic(images, memory, memory_mask,
                                             hw_lvl, gt_semantic_seg)
        feat_embedding = self.get_embedding(memory, memory_pos,
                                            memory_mask, hw_lvl)

        pl_bboxes, pl_masks, pl_semantics = \
            self.get_pseudo_label(images,
                                  pred_semantic,
                                  given_semantic,
                                  semantic_existence,
                                  gt_labels,
                                  gt_masks,
                                  feat_embedding,
                                  img_metas)
        
        # compute losses related to the new pseudo label branches
        losses = dict()

        # semantic segmentation loss, which do not need pl_masks and pl_bboxes
        losses.update(loss_sem_pre)

        # embedding loss
        num_classes = self.num_thing_classes + self.num_stuff_classes
        pl_stuff_masks = F.one_hot(pl_semantics, 256)
        pl_stuff_masks = pl_stuff_masks[..., self.num_thing_classes:num_classes]
        pl_stuff_masks = pl_stuff_masks.permute(0, 3, 1, 2)

        loss_embedding = self.loss_embedding(feat_embedding,
                                             gt_masks,
                                             gt_semantic_seg,
                                             pl_masks,
                                             pl_stuff_masks)
        losses.update(loss_embedding)

        return losses, pl_bboxes, pl_masks, pl_semantics

    def get_semantic(self,
                     images,
                     memory,
                     memory_mask,
                     hw_lvl,
                     gt_semantic_seg):
        bs, _, dims = memory.shape
        semantic_query = torch.cat([self._semantic_query.weight, self.stuff_query.weight], dim=0)
        sem_query, sem_query_pos = torch.split(
            semantic_query[None].expand(bs, -1, -1),
            dims, -1)
        sem_query_list, sem_mask_list = self.stuff_mask_decoder(
            memory=memory,
            memory_mask=memory_mask,
            query=sem_query,
            query_pos=sem_query_pos,
            hw_lvl=hw_lvl)
        
        # post process mask prediction
        sem_masks = []
        for masks in sem_mask_list:
            sem_masks.append(masks.reshape(-1, *hw_lvl[0]))
        sem_masks = torch.stack(sem_masks) # [nDec, bs * nClass, h, w]

        # post process cls prediction
        sem_clses = []
        for i, query in enumerate(sem_query_list):
            sem_cls = self.stuff_cls_branches[i](query).view(-1, 1)
            sem_clses.append(sem_cls)
        sem_clses = torch.stack(sem_clses) # [nDec, bs * nClass, 1]

        nDec, _, h, w = sem_masks.shape
        sem_masks_proj = self.semantic_proj(sem_masks.view(nDec * bs, -1, h, w))
        sem_masks_proj = sem_masks_proj.view(nDec, -1, h, w)
        sem_masks_logit = sem_masks_proj * sem_clses.sigmoid()[..., None]

        # prepare losses
        num_classes = self.num_thing_classes + self.num_stuff_classes
        target_mask = F.one_hot(gt_semantic_seg.long(), 256)[..., :num_classes]
        target_mask = target_mask.permute(0, 3, 1, 2).float()
        target_mask = expand_target_masks(target_mask, self.point_expand_size)
        target_mask = target_mask.flatten(0, 1)

        target_weight = target_mask.max(2)[0].max(1)[0]
        target_cls = (1 - target_weight).long()

        num_total_pos = target_weight.sum()
        num_total_pos = target_weight.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # cls loss
        losses = dict()
        for i, sem_cls in enumerate(sem_clses):
            loss_cls = self.loss_cls(sem_cls,
                                     target_cls,
                                     avg_factor=num_total_pos)
            losses[f'd{i}.loss_semantic_cls'] = loss_cls

        # mask loss
        target_mask_ds = F.interpolate(target_mask[None], (h, w),
                                       mode='nearest').view(bs, -1, h, w)
        images_ds = F.interpolate(images, (h, w),
                                  mode='bilinear', align_corners=False)

        for i, sem_logit in enumerate(sem_masks_logit):
            loss_mask = partial_cross_entropy_loss(
                sem_logit.view(bs, -1, h, w),
                target_mask_ds)
            loss_color = color_prior_loss(
                sem_logit.view(bs, -1, h, w),
                images_ds,
                kernel_size=5)
            losses[f'd{i}.loss_semantic_mask'] = loss_mask
            losses[f'd{i}.loss_semantic_color'] = loss_color * 3

        pred_semantic = F.interpolate(sem_masks_logit[-1].view(bs, -1, h, w),
                                      target_mask.shape[-2:],
                                      mode='bilinear',
                                      align_corners=False)
        given_semantic = target_mask.unflatten(0, (bs, num_classes))
        semantic_existence = target_weight.view(bs, -1)

        # we merge 'loss_stuff_masks' here
        with torch.no_grad():
            semantic_probs = F.interpolate(sem_masks_logit[-1:],
                                           target_mask.shape[-2:],
                                           mode='bilinear',
                                           align_corners=False)[0]
            semantic_probs = torch.maximum(semantic_probs, target_mask).unflatten(0, (bs, -1))
            semantic_probs = semantic_probs.softmax(1)
            confidence, pl_semantics = semantic_probs.max(1)
            pl_semantics[confidence < 0.5] = 255

            pl_targets = F.one_hot(pl_semantics.long(), 256)[..., self.num_thing_classes:num_classes]
            pl_targets = pl_targets.permute(0, 3, 1, 2).float().flatten(0, 1)

        stuff_masks = sem_masks.view(nDec, bs, num_classes, h, w)[:, :, self.num_thing_classes:].flatten(1, 2)
        stuff_masks = F.interpolate(stuff_masks, pl_targets.shape[-2:],
                                  mode='bilinear', align_corners=False)
        pl_weights = target_weight.view(bs, num_classes)[:, self.num_thing_classes:].flatten()
        num_total_pos = bs
        for i, masks in enumerate(stuff_masks):
            loss_mask = self.loss_mask(masks,
                                       pl_targets,
                                       pl_weights,
                                       avg_factor=num_total_pos)
            losses[f'd{i}.loss_stuff_mask'] = loss_mask

        return pred_semantic, given_semantic, semantic_existence, losses
    