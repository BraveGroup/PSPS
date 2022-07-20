import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
import mmcv
from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.utils import build_transformer
from easymd.models.panformer import DETRHeadv2

from .utils import *

from sklearn.decomposition import PCA
import cv2
import os
import pickle

from pydijkstra import dijkstra2d, dijkstra_image

import alphashape
from descartes import PolygonPatch

_USE_CENTER = False
_USE_IMPLICIT_CENTER = False
_USE_FEATURE_EMBEDS = True
_COLLECT_INTERS = False
_INTERS_FOLDER = '/home/junsong_fan/Panformer/analysis_feats/inters'
if _COLLECT_INTERS:
    os.makedirs(_INTERS_FOLDER, exist_ok=True)

_USE_BOUNDARY = False
DIJKSTRA = True
#WARMUP_ITER = 1324 * 5
WARMUP_ITER = 14659  # used in coco
#WARMUP_ITER = 662 # used in voc


#EXPAND_SIZE = 21
#EXPAND_SIZE = 17
COLOR_PRIOR_LOSS = True

VISUALIZATION_ONLY = False

ALPHA_SHAPE = False

def get_alpha_shape(H, W, points):
    #assert isinstance(points, np.ndarray), type(points)
    N = len(points)
    out = np.zeros((N, H, W), np.uint8)
    #points = points.astype(np.int32)
    for i, pnts in enumerate(points):
        #assert pnts.ndim == 2, points.shape
        assert pnts.ndim == 2, (pnts.ndim, len(points))
        pnts = pnts.astype(np.int64)
        try:
            alpha_shape = alphashape.alphashape(pnts, 0)
            verts = PolygonPatch(alpha_shape).get_verts()
            verts = verts.reshape(-1, 1, 2).astype(np.int32)[..., ::-1].copy()
            cv2.fillPoly(out[i], [verts], color=1)
        except:
            pass
        #try:
        #    alpha_shape = alphashape.alphashape(pnts)
        #except:
        #    alpha_shape = alphashape.alphashape(pnts, 0)
        #try:
        #    verts = PolygonPatch(alpha_shape).get_verts()
        #    verts = verts.reshape(-1, 1, 2).astype(np.int32)[..., ::-1].copy()
        #    cv2.fillPoly(out[i], [verts], color=1)
        #except:
        #    pass
    return out

def _get_rgb_image(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
        assert image.ndim == 3 and image.shape[2] == 3, image.shape
        return image

    assert image.ndim == 3 and image.shape[0] == 3, image.shape
    if isinstance(image, torch.Tensor):
        image = image.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image * std + mean
    if image.max() > 1.5:
        image = np.clip(image, a_min=0, a_max=255)
    else:
        image = np.clip(image, a_min=0, a_max=1) * 255
    image = image.astype(np.uint8)
    return image.copy()

class _MaskTransformerWrapper(nn.Module):
    def __init__(self, num_layers, self_attn=False):
        super().__init__()
        cfg = dict(
                type='PanformerMaskDecoder',
                d_model=256,
                num_heads=8,
                num_layers=num_layers,
                self_attn=self_attn)
        self.model = build_transformer(cfg)

    def forward(self, memory, memory_mask, placeholder1, query, placeholder2, query_pos, hw_lvl):
        assert memory_mask.shape[0] == memory.shape[0], (memory_mask.shape, memory.shap)

        memory = memory.transpose(0, 1)
        query = query.transpose(0, 1)
        if query_pos is not None:
            query_pos = query_pos.transpose(0, 1)
        assert len(hw_lvl) == 4, (hw_lvl)
        hw_lvl = hw_lvl[:3]

        # query-first memory
        #   memory,         # [sum(H_i*W_i), bs, embed_dims]
        #   memory_pos,     # [sum(H_i*W_i), bs, embed_dims]
        #   memory_mask,    # [bs, sum(H_i*W_i)]
        #   query,          # [num_query, bs, embed_dims]
        #   query_pos,      # [num_query, bs, embed_dims]
        # out: [num_query, bs, embed_dims], [bs, num_query, h, w]
        all_query, all_mask = self.model(
                memory=memory,
                #memory_pos=memory_pos,
                memory_mask=memory_mask,
                query=query,
                query_pos=query_pos,
                hw_lvl=hw_lvl)

        assert all_query[0].shape[1] == memory.shape[1], ([x.shape for x in all_query], memory.shape)
        all_query = [x.transpose(0, 1) for x in all_query]
        mask_final = all_mask[-1]
        inter_masks = all_mask[:-1]
        return mask_final, inter_masks, all_query

class _MemoryProj(nn.Module):
    def __init__(self, in_channels, out_channels, norm='l2'):
        super().__init__()
        self.layer = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 1))
        self.norm = norm

    def forward(self, memory, memory_pos, memory_mask, hw_lvl):
        bsz, nMem, embed_dims = memory.shape
        memory_with_pos = memory + memory_pos
        memory_with_pos = memory_with_pos.detach()

        hw_lvl = hw_lvl[:3]
        assert nMem == sum(h * w for h, w in hw_lvl), (memory.shape, hw_lvl)

        begin = 0
        memory_out = []
        for i, (h, w) in enumerate(hw_lvl):
            m2d = memory_with_pos[:, begin:begin+h*w, :].view(bsz, h, w, embed_dims)
            m2d = m2d.permute(0, 3, 1, 2).contiguous()
            begin = begin + h * w

            m2d = F.interpolate(m2d, hw_lvl[0], mode='bilinear', align_corners=False)
            memory_out.append(m2d)

        memory_out = sum(memory_out)
        out = self.layer(memory_out)
        if self.norm == 'l2':
            out = F.normalize(out, p=2, dim=1)
        elif self.norm is None:
            pass
        elif self.norm == 'tanh':
            out = F.tanh(out)
        else:
            raise ValueError(self.norm)
        return out

@HEADS.register_module()
class WsupPanformerHead(DETRHeadv2):
    """
    Head of Panoptic SegFormer

    Code is modified from the `official github repo
    <https://github.com/open-mmlab/mmdetection>`_.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(
            self,
            *args,
            with_box_refine=False,
            as_two_stage=False,
            transformer=None,
            quality_threshold_things=0.25,
            quality_threshold_stuff=0.25,
            overlap_threshold_things=0.4,
            overlap_threshold_stuff=0.2,
            use_argmax=False,
            datasets='coco',  # MDS
            thing_transformer_head=dict(
                type='TransformerHead',  # mask decoder for things
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            stuff_transformer_head=dict(
                type='TransformerHead',  # mask decoder for stuff
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            loss_mask=dict(type='DiceLoss', weight=2.0),
            train_cfg=dict(
                assigner=dict(type='HungarianAssigner',
                              cls_cost=dict(type='ClassificationCost',
                                            weight=1.),
                              reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                              iou_cost=dict(type='IoUCost',
                                            iou_mode='giou',
                                            weight=2.0)),
                sampler=dict(type='PseudoSampler'),
            ),
            **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.quality_threshold_things = quality_threshold_things
        self.quality_threshold_stuff = quality_threshold_stuff
        self.overlap_threshold_things = overlap_threshold_things
        self.overlap_threshold_stuff = overlap_threshold_stuff
        self.use_argmax = use_argmax
        self.datasets = datasets
        self.fp16_enabled = False

        self.lambda_diff_prob = kwargs['lambda_diff_prob']
        self.lambda_diff_bond = kwargs['lambda_diff_bond']
        self.lambda_diff_feat = kwargs['lambda_diff_feat']
        self.lambda_color_prior = kwargs['lambda_color_prior']
        self.EXPAND_SIZE = kwargs['expand_size']

        # MDS: id_and_category_maps is the category_dict
        if datasets == 'coco':
            from easymd.datasets.coco_panoptic import id_and_category_maps
            self.cat_dict = id_and_category_maps
        else:
            self.cat_dict = None
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.num_dec_things = thing_transformer_head['num_decoder_layers']
        self.num_dec_stuff = stuff_transformer_head['num_decoder_layers']
        super(WsupPanformerHead, self).__init__(*args,
                                            transformer=transformer,
                                            train_cfg=train_cfg,
                                            **kwargs)
        if train_cfg:
            sampler_cfg = train_cfg['sampler_with_mask']
            self.sampler_with_mask = build_sampler(sampler_cfg, context=self)
            assigner_cfg = train_cfg['assigner_with_mask']
            self.assigner_with_mask = build_assigner(assigner_cfg)
            self.assigner_filter = build_assigner(
                dict(
                    type='HungarianAssigner_filter',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost',
                                  weight=5.0,
                                  box_format='xywh'),
                    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                    max_pos=
                    3  # Depends on GPU memory, setting it to 1, model can be trained on 1080Ti
                ), )

        self.loss_mask = build_loss(loss_mask)
        self.things_mask_head = _MaskTransformerWrapper(4, False)
        self.stuff_mask_head = _MaskTransformerWrapper(6, True)
        self.semantic_mask_head = _MaskTransformerWrapper(6, True)
        num_classes = self.num_things_classes + self.num_stuff_classes
        self.semantic_proj = nn.Conv2d(num_classes, num_classes, 1, groups=num_classes)
        self.count = 0
        self.tmp_state = {}
        self.warmup_niter = 0

        if _USE_FEATURE_EMBEDS:
            self.memory_proj = _MemoryProj(self.embed_dims, 128, 'l2')
        if _USE_CENTER:
            self.center_proj = _MemoryProj(self.embed_dims, 2, 'tanh')
        if _USE_IMPLICIT_CENTER:
            self.implicit_center_proj = _MemoryProj(self.embed_dims, self.embed_dims, None)
        if _USE_BOUNDARY:
            self.boundary_proj = _MemoryProj(self.embed_dims, 1, None)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        fc_cls_stuff = Linear(self.embed_dims, 1)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        self.stuff_query = nn.Embedding(self.num_stuff_classes,
                                        self.embed_dims * 2)
        self.semantic_query = nn.Embedding(self.num_things_classes + self.num_stuff_classes,
                self.embed_dims * 2)
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff) # used in mask deocder
        self.cls_semantic_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_stuff_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

        #self.semantic_proj.weight.copy_(torch.eye(self.num_things_classes+self.num_stuff_classes, dtype=torch.float32))
        num_classes = self.num_things_classes + self.num_stuff_classes
        #nn.init.eye_(self.semantic_proj.weight.view(num_classes, num_classes))
        nn.init.constant_(self.semantic_proj.weight, 1.0)
        nn.init.constant_(self.semantic_proj.bias, 0.0)

    @force_fp32(apply_to=('mlvl_feats', ))
    def forward(self, mlvl_feats, img_metas=None):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
       
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']

        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        (memory, memory_pos, memory_mask, query_pos), hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
        )

        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        len_last_feat = hw_lvl[-1][0] * hw_lvl[-1][1]

        # we should feed these to mask deocder.
        args_tuple = (memory[:, :-len_last_feat, :],
                      memory_mask[:, :-len_last_feat],
                      memory_pos[:, :-len_last_feat, :], query, None,
                      query_pos, hw_lvl)

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                   enc_outputs_class, \
                   enc_outputs_coord.sigmoid(), args_tuple, reference
        else:
            return outputs_classes, outputs_coords, \
                   None, None, args_tuple, reference

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list',
                          'args_tuple', 'reference'))
    def loss(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list=None,
        gt_semantics_list=None,
        img_metas=None,
        gt_bboxes_ignore=None,
    ):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            args_tuple (Tuple) several args
            reference (Tensor) reference from location decoder
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        bch_shape = img_metas[0]['batch_input_shape']

        #isthing = [gt_labels < self.num_things_classes for gt_labels in gt_labels_list]
        #gt_th_labels_list = [data[mask] for data, mask in zip(gt_labels_list, isthing)]
        #gt_th_bboxes_list = [data[mask] for data, mask in zip(gt_bboxes_list, isthing)]
        #gt_th_masks_list = [data[mask] for data, mask in zip(gt_masks_list, isthing)]
        gt_bboxes_list = None

        loss_dict = {}

        # batch-size first
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple
        bsz, _, embed_dims = memory_pos.shape

        loss_semantic, semantic_info = self.loss_semantic(args_tuple,
                gt_semantics_list,
                img_metas,
                gt_bboxes_ignore)
        loss_dict.update(loss_semantic)

        # semantic prediction
        # all_sem_cls, all_sem_masks_logit = self.forward_semantic(args_tuple)
        # n_dec, _, h, w = all_sem_masks_logit.shape

        # intial semantic targets, point-level
        #sem_target_mask, sem_target_mask_weight, sem_target_cls = self.get_point_semantic_targets(gt_semantics_list)
        #H, W = sem_target_mask.shape[-2:]

        #semantic_info = (
        #        F.interpolate(all_sem_masks_logit[-1].view(bsz, -1, h, w), (H, W), mode='bilinear', align_corners=False),
        #        sem_target_mask.view(bsz, -1, H, W),
        #        sem_target_mask_weight.view(bsz, -1))

        ## initial semantic loss
        #loss_dict.update(self.loss_sem_cls(all_sem_cls, sem_target_cls, sem_target_mask_weight))
        #loss_dict.update(self.loss_sem_mask(all_sem_masks_logit, sem_target_mask, sem_target_mask_weight, True))
        #loss_dict.update(self.loss_sem_color(all_sem_masks_logit))

        # others

        if _USE_FEATURE_EMBEDS:
            memory_proj = self.memory_proj(memory, memory_pos, memory_mask, hw_lvl) # [bsz, dim, h, w]
        else:
            memory_proj = None

        if _USE_CENTER:
            center_proj = self.center_proj(memory, memory_pos, memory_mask, hw_lvl)
        else:
            center_proj = None

        if _USE_IMPLICIT_CENTER:
            implicit_center_proj = self.implicit_center_proj(memory, memory_pos, memory_mask, hw_lvl)
            memory_pos2d = memory_pos[:, :hw_lvl[0][0]*hw_lvl[0][1]].view(bsz, *hw_lvl[0], embed_dims)
            memory_pos2d = memory_pos2d.permute(0, 3, 1, 2).contiguous()
        else:
            implicit_center_proj, memory_pos2d = None, None

        if _USE_BOUNDARY:
            boundary_proj = self.boundary_proj(memory, memory_pos, memory_mask, hw_lvl).squeeze(1)
        else:
            boundary_proj = None

        if DIJKSTRA:
            pl_labels_list, pl_bboxes_list, pl_masks_list, pl_semantics_list = \
                    self.get_dijkstra_pseudo_label(*semantic_info, gt_labels_list, gt_masks_list, boundary_proj, memory_proj, img_metas)
        else:
            pl_labels_list, pl_bboxes_list, pl_masks_list, pl_semantics_list = \
                self.get_pseudo_gt_from_semantic(*semantic_info, gt_labels_list, gt_masks_list, memory_proj, center_proj, implicit_center_proj, memory_pos2d, img_metas)

        # additional semantic loss
        #pl_sem_mask = F.one_hot(pl_semantics_list, 256)[..., :self.num_things_classes+self.num_stuff_classes]
        #pl_sem_mask = pl_sem_mask.permute(0, 3, 1, 2).float()
        #loss_sem2 = self.loss_sem_mask(all_sem_masks_logit, pl_sem_mask.flatten(0, 1), None, True)
        #loss_dict.update({f'{k}(2)': v for k, v in loss_sem2.items()})

        # location decoder loss
        for i, (cls_scores, bbox_preds) in enumerate(zip(all_cls_scores, all_bbox_preds)):
            if i == len(all_cls_scores) - 1:
                continue
            loss_cls, loss_bbox, loss_iou = self.loss_single(
                    cls_scores, bbox_preds,
                    pl_bboxes_list, pl_labels_list,
                    img_metas, gt_bboxes_ignore,)
            loss_dict.update({
                f'd{i}.loss_cls': loss_cls,
                f'd{i}.loss_bbox': loss_bbox,
                f'd{i}.loss_iou': loss_iou, })

        if enc_cls_scores is not None:
            binary_labels_list = [torch.zeros_like(x) for x in pl_labels_list]
            enc_losses_cls, enc_losses_bbox, enc_losses_iou = self.loss_single(
                    enc_cls_scores, enc_bbox_preds,
                    pl_bboxes_list, binary_labels_list,
                    img_metas, gt_bboxes_ignore)
            loss_dict.update({
                'enc_loss_cls': enc_losses_cls,
                'enc_loss_bbox': enc_losses_bbox,
                'enc_loss_iou': enc_losses_iou})

        # mask decoder loss
        loss_mask_decoder, thing_ratio, stuff_ratio, all_th_masks, all_st_masks = self.loss_single_panoptic_simplified(
                all_cls_scores[-1],
                all_bbox_preds[-1],
                args_tuple,
                pl_bboxes_list,
                pl_labels_list,
                pl_masks_list,
                pl_semantics_list,
                img_metas,
                gt_bboxes_ignore)
        loss_dict.update(loss_mask_decoder)

        for k in loss_dict.keys():
            ratio = stuff_ratio if 'st' in k else thing_ratio
            loss_dict[k] = loss_dict[k] * ratio

        # feature metric distance loss
        pl_stuff_masks = F.one_hot(pl_semantics_list.long(), 256)[..., self.num_things_classes:self.num_things_classes+self.num_stuff_classes].permute(0, 3, 1, 2).to(pl_masks_list[0])

        if _USE_FEATURE_EMBEDS:
            loss_cl = self.loss_metric(memory_proj, gt_labels_list, gt_masks_list, gt_semantics_list, pl_masks_list, pl_stuff_masks, img_metas)
            loss_dict.update(loss_cl)

        # geometric centers
        if _USE_CENTER:
            loss_geo = self.loss_geometric(center_proj, pl_masks_list, pl_stuff_masks, img_metas)
            loss_dict.update(loss_geo)

        if _USE_IMPLICIT_CENTER:
            loss_impgeo = self.loss_implicit_geometric(implicit_center_proj, memory_pos2d, pl_masks_list, pl_stuff_masks, img_metas)
            loss_dict.update(loss_impgeo)

        if _USE_BOUNDARY:
            loss_bond = self.loss_boundary(boundary_proj, pl_masks_list, pl_stuff_masks, img_metas)
            loss_dict.update(loss_bond)

        # tmp states
        self.tmp_state.update({
            'gt_bboxes_list': pl_bboxes_list,
            'gt_labels_list': pl_labels_list,
            'gt_masks_list': pl_masks_list,
            'gt_semantics_list': pl_semantics_list,
            'th_cls_scores': all_cls_scores[-1],
            'th_bbox_preds': all_bbox_preds[-1],
            })

        if _COLLECT_INTERS:
            pred_semantic_masks, sup_semantic_masks, class_labels = semantic_info
            self.collect_inters(img_metas, memory_proj, center_proj,
                    gt_labels_list, gt_masks_list, gt_semantics_list,
                    pred_semantic_masks, sup_semantic_masks, class_labels)

        # TODO: hard code for pseudo-label warmup
        warmup = max(min(float(self.warmup_niter) / WARMUP_ITER, 1), 0)
        self.warmup_niter += 1
        k_contains = lambda k: any([x in k for x in ['sem', 'loss_cl']])
        #loss_dict = {k: v * (1 if 'sem' in k else warmup) for k, v in loss_dict.items()}
        loss_dict = {k: v * (1 if k_contains(k) else warmup) for k, v in loss_dict.items()}

        if VISUALIZATION_ONLY:
            loss_dict = {k : v * 0 for k, v in loss_dict.items()}
        return loss_dict

    def forward_semantic(self, args_tuple):
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple
        bsz, _, embed_dims = query.shape

        # predictions of masks
        sem_query, sem_query_pos = torch.split(
                self.semantic_query.weight[None].expand(bsz, -1, -1),
                embed_dims, -1)
        mask_sem, mask_inter_sem, query_inter_sem = self.semantic_mask_head(
                memory, memory_mask, None, sem_query, None, sem_query_pos, hw_lvl=hw_lvl)
        all_sem_masks = []
        for sem_masks in mask_inter_sem + [mask_sem]:
            sem_masks = sem_masks.reshape(-1, *hw_lvl[0])
            all_sem_masks.append(sem_masks)
        all_sem_masks = torch.stack(all_sem_masks) # [n_dec, bsz*cls, h, w]

        n_dec, _, h, w = all_sem_masks.shape
        all_sem_masks = self.semantic_proj(all_sem_masks.view(n_dec*bsz, -1, h, w)).view(n_dec, -1, h, w)

        # predictions of classes
        all_sem_cls = []
        for i, query_sem in enumerate(query_inter_sem):
            sem_cls = self.cls_semantic_branches[i](query_sem).view(-1, 1)
            all_sem_cls.append(sem_cls)
        all_sem_cls = torch.stack(all_sem_cls) # [n_dec, bsz*cls, 1]

        # merge
        all_sem_cls_prob = all_sem_cls[..., None].sigmoid() # [n_dec, bsz*cls, 1, 1]
        all_sem_masks_logit = all_sem_masks * all_sem_cls_prob # [n_dec, bsz*cls, h, w]

        self.tmp_state.update(dict(semantic_pred_logit=all_sem_masks_logit[-1].view(bsz, -1, h, w)))
        return all_sem_cls, all_sem_masks_logit

    def loss_sem_cls(self, all_sem_cls, target_cls, target_mask_weight):
        loss_dict = {}
        for i, sem_cls in enumerate(all_sem_cls):
            loss_cls = self.loss_cls(
                    sem_cls,
                    target_cls,
                    avg_factor=target_mask_weight.sum())
            loss_dict[f'd{i}.loss_sem_cls'] = loss_cls
        return loss_dict

    def loss_sem_mask(self, all_sem_masks_logit, target_mask, target_mask_weight, smaller=True):
        loss_dict = {}
        n_dec, bsz_cls, h, w = all_sem_masks_logit.shape
        bsz = bsz_cls // (self.num_things_classes + self.num_stuff_classes)
        _, H, W = target_mask.shape

        target_mask = target_mask.view(bsz, -1, H, W)
        if smaller:
            target_mask = F.interpolate(target_mask.float(), (h, w), mode='nearest')
        else:
            all_sem_masks_logit = F.interpolate(all_sem_masks_logit, (H, W), mode='bilinear', align_corners=False)

        H, W = target_mask.shape[-2:]
        for i, sem_mask_logit in enumerate(all_sem_masks_logit):
            loss_mask = partial_cross_entropy_loss(
                    sem_mask_logit.view(bsz, -1, H, W),
                    target_mask)
            loss_dict[f'd{i}.loss_sem_mask'] = loss_mask
        return loss_dict

    def loss_sem_color(self, all_sem_masks_logit):
        loss_dict = {}
        n_dec, bsz_cls, h, w = all_sem_masks_logit.shape
        bsz = bsz_cls // (self.num_things_classes + self.num_stuff_classes)
        assert bsz * (self.num_things_classes + self.num_stuff_classes) == bsz_cls

        images = self.tmp_state['img']
        images = F.interpolate(images, (h, w), mode='bilinear', align_corners=False)
        for i, sem_mask_logit in enumerate(all_sem_masks_logit):
            loss_color = color_prior_loss(
                    sem_mask_logit.view(bsz, -1, h, w),
                    images,
                    kernel_size=5)
            loss_dict[f'd{i}loss_sem_color'] = loss_color * self.lambda_color_prior
        return loss_dict

    #def loss_sem_crf(self, all_sem_masks_logit):
    #    loss_dict = {}
    #    n_dec, bsz_cls, h, w = all_sem_masks_logit.shape
    #    bsz = bsz_cls // (self.num_things_classes + self.num_stuff_classes)
    #    assert bsz * (self.num_things_classes + self.num_stuff_classes) == bsz_cls
    #    images = self.tmp_state['img']
    #    images = F.interpolate(images, (h, w), mode='bilinear', align_corners=False)
    #    boundary = image_to_boundary(images)
    #    boundary = F.max_pool2d(boundary[None], 3, 1, 1)[0]
    #    boundary = (boundary > 0.1).float()

    def get_point_semantic_targets(self, gt_semantics_list):
        target_mask = F.one_hot(gt_semantics_list.long(), 256)
        num_classes = self.num_things_classes + self.num_stuff_classes
        target_mask = target_mask[..., :num_classes]
        target_mask = target_mask.permute(0, 3, 1, 2).float() # [bsz, cls, h, w]
        bsz, _, h, w = target_mask.shape
        target_mask = expand_target_masks(target_mask, self.EXPAND_SIZE)
        target_mask = target_mask.flatten(0, 1) # [bsz*cls, h, w]
        target_mask_weight = target_mask.max(2)[0].max(1)[0] # [bsz*cls]
        target_cls = (1 - target_mask_weight).long() # [bsz*cls]
        self.tmp_state.update(dict(semantic_pred_target=target_mask.view(bsz, -1, h, w)))
        return target_mask, target_mask_weight, target_cls

    @torch.no_grad()
    def collect_inters(self, img_metas, memory_projs, center_projs, 
            gt_labels_list, gt_masks_list, gt_semantics_list,
            pred_semantic_logits, sup_semantic_masks, class_labels):
        assert gt_semantics_list.ndim == 3, gt_semantics_list.shape
        assert pred_semantic_logits.ndim == 4, pred_semantic_logits.shape

        bch_shape = img_metas[0]['batch_input_shape']
        memory_projs = F.interpolate(memory_projs, bch_shape, mode='bilinear', align_corners=False)
        center_projs = F.interpolate(center_projs, bch_shape, mode='bilinear', align_corners=False)
        pred_semantic_logits = F.interpolate(pred_semantic_logits, bch_shape, mode='bilinear', align_corners=False)
        H, W = bch_shape
        assert gt_masks_list[0].shape[-2:] == (H, W), (gt_masks_list[0].shape, bch_shape)

        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W))).to(center_projs)
        coords = coords / coords.max(1, keepdim=True)[0].max(2, keepdim=True)[0]
        pred_centers = (center_projs + coords[None]).clip(min=0, max=1)

        for i in range(len(img_metas)):
            img_meta = img_metas[i]
            name = img_meta['ori_filename'].rsplit('.')[0]
            img_shape = img_meta['img_shape'][:2]
            ori_shape = img_meta['ori_shape'][:2]
            fea_shape = ori_shape

            memory_proj = memory_projs[i, :, :img_shape[0], :img_shape[1]] # [c, h, w]
            memory_proj = F.interpolate(memory_proj[None], fea_shape, mode='bilinear', align_corners=False)[0]

            center_proj = center_projs[i, :, :img_shape[0], :img_shape[1]] # [c, h, w]
            center_proj = F.interpolate(center_proj[None], fea_shape, mode='bilinear', align_corners=False)[0]

            pred_center = pred_centers[i, :, :img_shape[0], :img_shape[1]] # [c, h, w]
            pred_center = F.interpolate(pred_center[None], fea_shape, mode='bilinear', align_corners=False)[0]

            gt_labels = gt_labels_list[i]
            gt_masks = gt_masks_list[i][:, :img_shape[0], :img_shape[1]]
            gt_semantics = gt_semantics_list[i, :img_shape[0], :img_shape[1]]

            pred_semantic_logit = pred_semantic_logits[i, :, :img_shape[0], :img_shape[1]]
            class_label = class_labels[i]

            data = (
                img_meta,
                memory_proj.data.cpu().numpy().astype(np.float16),
                center_proj.data.cpu().numpy().astype(np.float16),
                pred_center.data.cpu().numpy().astype(np.float16),
                gt_labels.data.cpu().numpy().astype(np.int16),
                gt_masks.data.cpu().numpy().astype(np.uint8),
                gt_semantics.data.cpu().numpy().astype(np.uint8),
                pred_semantic_logit.data.cpu().numpy().astype(np.float16),
                class_label.data.cpu().numpy().astype(np.uint8))

            with open(os.path.join(_INTERS_FOLDER, name + '.pkl'), 'wb') as f:
                pickle.dump(data, f)

    def loss_semantic(self,
            args_tuple,
            gt_semantics_list,
            img_metas=None,
            gt_bboxes_ignore=None):

        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple
        bsz, _, embed_dims = query.shape

        # get predictions of semantic masks
        sem_query, sem_query_pos = torch.split(
                self.semantic_query.weight[None].expand(bsz, -1, -1),
                embed_dims, -1)
        mask_sem, mask_inter_sem, query_inter_sem = self.semantic_mask_head(
                memory, memory_mask, None, sem_query, None, sem_query_pos, hw_lvl=hw_lvl)
        all_sem_masks = []
        for sem_masks in mask_inter_sem + [mask_sem]:
            sem_masks = sem_masks.squeeze(-1)
            sem_masks = sem_masks.reshape(-1, *hw_lvl[0])
            all_sem_masks.append(sem_masks)
        all_sem_masks = torch.stack(all_sem_masks) # [n_dec, bsz*cls, h, w]

        n_dec, _, h, w = all_sem_masks.shape
        all_sem_masks_proj = self.semantic_proj(all_sem_masks.view(n_dec*bsz, -1, h, w)).view(n_dec, -1, h, w)

        # get predictions of semantic cls
        all_sem_cls = []
        for i, query_sem in enumerate(query_inter_sem):
            sem_cls = self.cls_semantic_branches[i](query_sem).view(-1, 1)
            all_sem_cls.append(sem_cls)
        all_sem_cls = torch.stack(all_sem_cls) # [n_dec, bsz*cls, 1]

        # target mask
        target_mask = F.one_hot(gt_semantics_list.long(), max(256, gt_semantics_list.max()+1))
        num_classes = self.num_things_classes + self.num_stuff_classes
        target_mask = target_mask[..., :num_classes]
        target_mask = target_mask.permute(0, 3, 1, 2).float() # [bsz, cls, h, w]
        target_mask = expand_target_masks(target_mask, self.EXPAND_SIZE)
        target_mask = target_mask.flatten(0, 1) # [bsz*cls, h, w]
        target_mask_weight = target_mask.max(2)[0].max(1)[0] # [bsz*cls]
        target_cls = (1 - target_mask_weight).long()

        # NOTE: dice loss is unavailable because of partial annotations
        loss_dict = {}
        for i, sem_cls in enumerate(all_sem_cls):
            loss_cls = self.loss_cls(
                    sem_cls,
                    target_cls,
                    avg_factor=target_mask_weight.sum())
            loss_dict[f'd{i}.loss_sem_cls'] = loss_cls

        all_sem_cls_prob  = all_sem_cls[..., None].sigmoid() # [n_dec, bsz*cls, 1, 1]
        all_sem_masks_logit_scaled = all_sem_masks_proj * all_sem_cls_prob
        all_sem_masks_logit = F.interpolate(all_sem_masks_logit_scaled, target_mask.shape[-2:],
                mode='bilinear', align_corners=False)

        images = self.tmp_state['img']

        H, W = all_sem_masks_logit_scaled.shape[-2:]
        images_ = F.interpolate(images, (H, W), mode='bilinear', align_corners=False)
        target_mask_small = F.interpolate(target_mask[None], (H, W), mode='nearest').view(bsz, -1, H, W)

        for i, sem_mask_logit in enumerate(all_sem_masks_logit_scaled):
            loss_mask = partial_cross_entropy_loss(
                    sem_mask_logit.view(bsz, -1, H, W),
                    target_mask_small)
            loss_sem_color = color_prior_loss(
                    sem_mask_logit.view(bsz, -1, H, W),
                    images_,
                    kernel_size=5)
            loss_dict[f'd{i}.loss_sem_mask'] = loss_mask
            loss_dict[f'd{i}.loss_sem_color'] = loss_sem_color * self.lambda_color_prior

        H2, W2 = all_sem_masks_logit.shape[-2:]
        self.tmp_state.update({
            'semantic_pred_logit': all_sem_masks_logit[-1].view(bsz, -1, H2, W2),
            'semantic_pred_target': target_mask.view(bsz, -1, H2, W2)
            })

        semantic_info = (
                all_sem_masks_logit[-1].view(bsz, -1, H2, W2),
                target_mask.view(bsz, -1, H2, W2),
                target_mask_weight.view(bsz, -1),
                )
        return loss_dict, semantic_info

    def loss_metric(self,
            memory_proj,
            gt_labels_list,
            gt_masks_list,
            gt_semantics_list,
            pred_masks_list,
            pred_stuff_masks_list,
            img_metas,
            use_stuff=True):

        # points' masks, including stuff classes
        if use_stuff:
            gt_stuff_masks = F.one_hot(gt_semantics_list.long(), 256)[..., self.num_things_classes:self.num_things_classes+self.num_stuff_classes].permute(0, 3, 1, 2).to(gt_masks_list[0])
            gt_masks_list = [torch.cat([th_masks, st_masks]) for th_masks, st_masks in zip(gt_masks_list, gt_stuff_masks)]
            pred_masks_list = [torch.cat([th_masks, st_masks]) for th_masks, st_masks in zip(pred_masks_list, pred_stuff_masks_list)]

        # two sizes
        h, w = memory_proj.shape[-2:]
        H, W = gt_masks_list[0].shape[-2:]
        coord_factor = memory_proj.new_tensor([1, h / H, w / W]).view(1, -1)

        #scale = 1. / memory_proj.shape[1]**.5
        scale = 0.07

        # compute loss
        loss_cl = []
        for i, gt_masks in enumerate(gt_masks_list):
            coords = torch.nonzero(gt_masks) # [N, 3], {insID, ih, iw}
            coords = (coords * coord_factor).long()

            query = memory_proj[i][:, coords[:, 1], coords[:, 2]] # [dim, N]
            pred_masks = F.interpolate(pred_masks_list[i][None].detach().float(), (h, w), mode='bilinear', align_corners=False)[0]
            reference = torch.einsum('dhw,nhw->dn', memory_proj[i], pred_masks) / pred_masks.sum((1,2)).clip(min=1)[None] # [dim, n]

            dot = (query.T @ reference) / scale # [N, n]

            ins_indices = coords[:, 0] # [N,], value in {0, 1, ..., n-1}
            pos_mask = torch.zeros_like(dot)
            pos_mask = torch.zeros_like(dot)
            pos_mask[torch.arange(dot.shape[0], device=dot.device), ins_indices] = 1
            loss_cl_i = - (torch.log_softmax(dot, 1) * pos_mask).sum(1)
            pos_mask_valid = (pos_mask.sum(1) < pos_mask.shape[1]).float()
            loss_cl.append((loss_cl_i * pos_mask_valid).sum() / pos_mask_valid.sum().clip(min=1))
        loss_cl = sum(loss_cl) / len(loss_cl)
        loss_dict = dict(
                loss_cl=loss_cl
                )
        self.tmp_state.update(dict(memory_proj=memory_proj))
        return loss_dict

    def loss_geometric(self, center_proj, th_masks_list, st_masks, img_metas):
        #st_masks = F.one_hot(st_masks.long(), 256)[..., self.num_things_classes:self.num_things_classes+self.num_stuff_classes].permute(0, 3, 1, 2).to(th_masks_list[0]) # [bsz, nStuff, H, W]

        H, W = th_masks_list[0].shape[-2:]
        center_proj = F.interpolate(center_proj, (H, W), mode='bilinear', align_corners=False)
        self.tmp_state.update({'center_proj': center_proj})

        # for thing classes, predict their center.
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W))).to(center_proj) # [2, H, W]
        coords = coords / coords.max(1, keepdim=True)[0].max(2, keepdim=True)[0]

        pred_centers_all = (coords[None] + center_proj).clip(min=0, max=1)

        bsz = len(center_proj)
        loss_centers = []
        for i, th_masks in enumerate(th_masks_list):
            th_masks = th_masks.detach()

            areas = th_masks.sum((1, 2))
            centers = (th_masks[:, None] * coords[None]).sum((2, 3)) / areas[:, None].clip(min=1) # [n, 2]
            centers = centers[:, :, None, None].expand(-1, -1, H, W)
            pred_centers = pred_centers_all[i:i+1].expand(len(th_masks), -1, -1, -1) # [n, 2, H, W]

            loss_center = F.smooth_l1_loss(pred_centers, centers, reduction='none')
            loss_center = (loss_center * th_masks[:, None]).sum((1, 2, 3)) / (H * W) # [n]

            loss_centers.append(loss_center * (areas > 10))

        loss_centers = torch.cat(loss_centers).mean()
        #loss_centers = sum(l_cnt.mean() for l_cnt in loss_centers) / len(loss_centers)

        # for stuff classes, predict zero delta
        st_regions = st_masks.max(1)[0]
        loss_bg = (center_proj.pow(2).sum(1, keepdim=True) * st_regions).mean()

        loss = {
                'loss_geo_th': loss_centers * 10,
                'loss_geo_st': loss_bg * 10}
        return loss

    def loss_implicit_geometric(self, center_proj, ref_pos, th_masks_list, st_masks, img_metas):
        #H, W = th_masks_list[0].shape[-2:]
        #center_proj = F.interpolate(center_proj, (H, W), mode='bilinear', align_corners=False)
        assert center_proj.shape == ref_pos.shape, (center_proj.shape, ref_pos.shape)
        H, W = center_proj.shape[-2:]
        self.tmp_state.update({'implicit_center_proj': center_proj})

        # for thing classes, predict their center.
        ref_pos = ref_pos.detach()
        pred_centers_all = center_proj + ref_pos

        loss_centers = []
        for i, th_masks in enumerate(th_masks_list):
            th_masks = F.interpolate(th_masks.detach()[None].float(), (H, W), mode='bilinear', align_corners=False)[0]
            areas = th_masks.sum((1, 2))
            centers = (th_masks[:, None] * ref_pos[i:i+1]).sum((2, 3)) / areas[:, None].clip(min=1) # [n, 128]
            centers = centers[:, :, None, None].expand(-1, -1, H, W)
            pred_centers = pred_centers_all[i:i+1].expand(len(th_masks), -1, -1, -1) # [n, 128, H, W]

            loss_center = ((centers - pred_centers)**2).sum(1)
            loss_center = (loss_center * th_masks).sum((1, 2)) / (H * W)

            loss_centers.append(loss_center * (areas > 10))

        loss_centers = torch.cat(loss_centers).mean()

        # for stuff classes, predict zero delta
        st_masks = F.interpolate(st_masks.float(), (H, W), mode='bilinear', align_corners=False)
        st_regions = st_masks.max(1)[0]
        loss_bg = (center_proj.pow(2).sum(1, keepdim=True) * st_regions).mean()

        loss = {
                'loss_impgeo_th': loss_centers * 10,
                'loss_impgeo_st': loss_bg * 10 }
        return loss

    def loss_boundary(self, boundary_proj, pl_masks_list, pl_stuff_masks, img_metas):
        images = self.tmp_state['img']
        H, W = images.shape[-2:]
        boundary_proj = F.interpolate(boundary_proj[None], (H, W), mode='bilinear', align_corners=False)[0]

        #
        target_boundary = image_to_boundary(images)
        #target_boundary = (target_boundary > 0.5).float()
        target_boundary = target_boundary / target_boundary[:, 5:-5, 5:-5].max().view(-1, 1, 1).clip(min=1e-5)
        target_boundary.clip_(min=0, max=1) # [n, h, w]
        target_boundary = F.max_pool2d(target_boundary[None], 3, 1, 1)[0]

        #
        pl_masks = [torch.cat([tmsk, smsk]).float() for tmsk, smsk in zip(pl_masks_list, pl_stuff_masks)]
        pl_masks_avg = [F.avg_pool2d(pmsk[None], 3, 1, 1)[0] for pmsk in pl_masks]
        pl_edge = torch.stack([(pmsk - pmsk_avg).abs().max(0)[0] > 1e-3 for pmsk, pmsk_avg in zip(pl_masks, pl_masks_avg)]).float() # [n, H, W]
        pl_edge = torch.maximum(pl_edge, target_boundary)

        # loss
        alpha = 0.1
        losses = {}
        for i, tgt in enumerate([target_boundary, pl_edge]):
            pt = (boundary_proj.sigmoid() * tgt) + ((-boundary_proj).sigmoid() * (1 - tgt))
            at = alpha * tgt + (1 - alpha) * (1 - tgt)
            loss = - F.logsigmoid(boundary_proj) * tgt - F.logsigmoid(-boundary_proj) * (1 - tgt)
            loss = (1 - pt).pow(2) * at * loss
            losses[f'loss_bnd{i}'] = loss.mean() * 100
        self.tmp_state.update({'pred_boundary': boundary_proj, 'target_boundarys': [target_boundary, pl_edge]})
        return losses

    @torch.no_grad()
    def get_pseudo_gt_from_semantic(self, 
            pred_masks, gt_masks, cls_labels,
            gt_labels_list, gt_masks_list,
            memory_proj, center_proj, implicit_center_proj, memory_pos, img_metas):
        """
        Input:
            pred_masks:     [bsz, c_th+c_st, h, w], logits before softmax.
            gt_masks:       [bsz, c_th+c_st, h, w], in value {0, 1}, semantic gts, typically point-level labels.
            cls_labels:     [bsz, c_th+c_st], in value {0, 1}, indicating whether the class exits.
            gt_labels_list: List<bsz>[ [n,] ], in value {0, 1, ..., C-1}, class label of each thing class, 'n' is the number of instances in the sample.
            gt_masks_list:  List<bsz>[ [n, h, w] ], in value {0, 1}, 'n' is the number of instances in the sample.
        """
        # semantic masks for stuff classes.
        pred_mask_probs = torch.maximum(pred_masks.softmax(1) * cls_labels[:, :, None, None], gt_masks)
        out_semantic_probs, out_semantics_list = pred_mask_probs.max(1)
        out_semantics_list[out_semantic_probs < 0.5] = 255

        # thing masks
        H, W = gt_masks_list[0].shape[-2:]
        assert pred_masks.shape[-2:] == (H, W), (pred_masks.shape, gt_masks.shape, [x.shape for x in gt_masks_list])

        coords_raw = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), -1).to(pred_masks) # [H, W, 2]
        coords = coords_raw / coords_raw.new_tensor([H, W]).view(1, 1, 2)
        bsz = len(gt_masks_list)

        if _USE_CENTER:
            coords_ref = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W))).to(pred_masks) # [2, H, W]
            coords_ref = coords_ref / coords_ref.max(1, keepdim=True)[0].max(2, keepdim=True)[0]
            center_proj = F.interpolate(center_proj, (H, W), mode='bilinear', align_corners=False)
            center_coords = (coords_ref[None] + center_proj).clip(min=0, max=1)

        if _USE_IMPLICIT_CENTER:
            implicit_center_coords = memory_pos + implicit_center_proj
            ih, iw = implicit_center_coords.shape[-2:]
            ifactor = implicit_center_coords.new_tensor([1, ih / H, iw / W])

        # feat metric distance
        #memory_proj = F.interpolate(memory_proj, (H, W), mode='bilinear', align_corners=False)
        if _USE_FEATURE_EMBEDS:
            mh, mw = memory_proj.shape[-2:]
            mfactor = pred_mask_probs.new_tensor([1, mh / H, mw / W])
            #self.tmp_state.update({'memory_proj': memory_proj})

        out_bboxes_list, out_masks_list = [], []
        out_labels_list = gt_labels_list
        bbox_valid_list = []

        # geo_scale = 16
        # geo_measure_tensor = F.interpolate(pred_mask_probs, (H//geo_scale, W//geo_scale), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()
        # gh, gw = geo_measure_tensor.shape[1:3]
        # geo_measure_tensor_coords = torch.stack(torch.meshgrid(torch.arange(gh), torch.arange(gw)), -1).to(geo_measure_tensor)
        # geo_measure_tensor_coords = geo_measure_tensor_coords / geo_measure_tensor_coords.new_tensor([gh, gw]).view(1, 1, 2)
        # geo_measure_tensor = torch.cat([geo_measure_tensor, geo_measure_tensor_coords[None].expand(bsz, -1, -1, -1)], -1)
        # np_geo_measure_tensor = geo_measure_tensor.data.cpu().numpy()
        # gfactor = pred_mask_probs.new_tensor([1, gh / H, gw / W])

        for i in range(bsz):
            point_masks = gt_masks_list[i]
            if len(point_masks) == 0:
                raise RuntimeError

            # geometric distance
            # geo_distances = []
            # for point_mask in point_masks:
            #     point_coords = coords[point_mask > 0] # [nPnts, 2]
            #     if len(point_coords) == 0:
            #         raise RuntimeError(f'Empty masks.')
            #     distances = ((coords[:, :, None] - point_coords[None, None])**2).sum(-1) # [H, W, nPnts]
            #     distances = distances.min(-1)[0]
            #     geo_distances.append(distances)
            # geo_distances = torch.stack(geo_distances) # [n, H, W]

            # # dijkstra-based geometric distance
            # pnt_indices = torch.nonzero(point_masks) # [N, 3]
            # pnt_indices = (pnt_indices * gfactor).long()
            # pnt_indices[:, 1].clip_(min=0, max=gh-1)
            # pnt_indices[:, 2].clip_(min=0, max=gw-1)
            # geo_distances = dijkstra2d(np_geo_measure_tensor[i], pnt_indices[:, 1:].data.cpu().numpy())
            # geo_distances = torch.as_tensor(geo_distances, device=pnt_indices.device)
            # ins_ind_masks = [pnt_indices[:, 0] == i for i in range(len(point_masks))]
            # geo_distances = torch.stack([geo_distances[ind_mask].min(0)[0] for ind_mask in ins_ind_masks])
            # geo_distances = F.interpolate(geo_distances[None], (H, W), mode='bilinear', align_corners=False)[0]
            #geo_distances = geo_distances / geo_distances.max(1, keepdim=True)[0].max(2, keepdim=True)[0].clip(min=1)

            # # feature distance
            # pnt_indices = torch.nonzero(point_masks) # [N, 3]
            # pnt_indices = (pnt_indices * mfactor).long()
            # pnt_indices[:, 1].clip_(min=0, max=mh-1)
            # pnt_indices[:, 2].clip_(min=0, max=mw-1)
            # pnt_embeds = memory_proj[i][:, pnt_indices[:, 1], pnt_indices[:, 2]] # [dim, N]
            # pnt_distances = 1 - torch.einsum('dhw,dn->nhw', memory_proj[i], pnt_embeds)
            # pnt_distances = F.interpolate(pnt_distances[None], (H, W), mode='bilinear', align_corners=False)[0]
            # pnt_distances = [pnt_distances[pnt_indices[:, 0]==i].min(0)[0] for i in range(len(point_masks))]
            # pnt_distances = torch.stack(pnt_distances) # [n, H, W]
            # #pnt_distances = pnt_distances - pnt_distances.min(2, keepdim=True)[0].min(1, keepdim=True)[0]
            # #pnt_distances = pnt_distances / pnt_distances.max(2, keepdim=True)[0].max(1, keepdim=True)[0].clip(min=1e-5)

            # class distance
            pred_classes = pred_mask_probs[i][:self.num_things_classes] # [C, H, W]
            gt_classes = F.one_hot(gt_labels_list[i], 256)[..., :self.num_things_classes].float() # [n, C]
            cls_distances = 1 - torch.einsum('chw,nc->nhw', pred_classes, gt_classes)
            all_distance = cls_distances * 10

            # center distance
            if _USE_CENTER:
                pnt_indices = torch.nonzero(point_masks) # [N, 3]
                ins_ind_masks = [pnt_indices[:, 0] == i for i in range(len(point_masks))]
                pnt_cen_coords = center_coords[i][:, pnt_indices[:, 1], pnt_indices[:, 2]].T # [N, 2]
                cen_distances = ((pnt_cen_coords[..., None, None] - center_coords[i:i+1])**2).sum(1) # [N, H, W]
                cen_distances = torch.stack([cen_distances[ind_mask].min(0)[0] for ind_mask in ins_ind_masks])
                all_distance = all_distance + cen_distances * 5
            
            if _USE_IMPLICIT_CENTER:
                pnt_indices = torch.nonzero(point_masks) # [N, 3]
                pnt_indices = (pnt_indices * ifactor.view(1, 3)).long()
                pnt_indices[:, 1].clip_(min=0, max=ih-1)
                pnt_indices[:, 2].clip_(min=0, max=iw-1)
                ins_ind_masks = [pnt_indices[:, 0] == i for i in range(len(point_masks))]
                pnt_cen_coords = implicit_center_coords[i][:, pnt_indices[:, 1], pnt_indices[:, 2]].T # [N, 2]
                cen_distances = ((pnt_cen_coords[..., None, None] - implicit_center_coords[i:i+1])**2).sum(1) # [N, H, W]
                cen_distances = torch.stack([cen_distances[ind_mask].min(0)[0] for ind_mask in ins_ind_masks])
                cen_distances = F.interpolate(cen_distances[None], (H, W), mode='bilinear', align_corners=False)[0]
                all_distance = all_distance + cen_distances * 5

            #all_distance = geo_distances + cls_distances * 10 + cen_distances * 5
            #all_distance = geo_distances + cls_distances * 10
            #all_distance = cls_distances * 10 + cen_distances * 5

            num_ins = all_distance.shape[0]
            pseudo_gt = all_distance.argmin(0)
            pseudo_gt[pred_mask_probs[i].argmax(0) >= self.num_things_classes] = num_ins
            pseudo_gt = F.one_hot(pseudo_gt, num_ins+1)[..., :num_ins].permute(2, 0, 1).contiguous()

            pseudo_bboxes = []
            img_h, img_w = img_metas[i]['img_shape'][:2]

            mask_sizes = []
            for pgt in pseudo_gt:
                mask_coords = coords_raw[pgt > 0]
                mask_sizes.append(len(mask_coords))

                if len(mask_coords) == 0:
                    bboxes = [img_w//4, img_h//4, img_w*3//4, img_h*3//4]
                else:
                    y0, x0 = mask_coords.min(0)[0]
                    y1, x1 = mask_coords.max(0)[0]
                    bboxes = [x0, y0, x1+1, y1+1]
                pseudo_bboxes.append(bboxes)
            pseudo_bboxes = coords_raw.new_tensor(pseudo_bboxes)

            pseudo_bboxes[:, 0::2].clip_(min=0, max=img_w)
            pseudo_bboxes[:, 1::2].clip_(min=0, max=img_h)

            out_bboxes_list.append(pseudo_bboxes)
            out_masks_list.append(pseudo_gt)

            #bbox_valid_list.append(pseudo_gt.new_tensor(mask_sizes) >= 10)

        #out_labels_list = [x[mask] for x, mask in zip(out_labels_list, bbox_valid_list)]
        #out_bboxes_list = [x[mask] for x, mask in zip(out_masks_list, bbox_valid_list)]
        #out_masks_list = [x[mask] for x, mask in zip(out_masks_list, bbox_valid_list)]
        return out_labels_list, out_bboxes_list, out_masks_list, out_semantics_list

    @torch.no_grad()
    def get_dijkstra_pseudo_label(self,
            pred_semantic_masks,
            sup_semantic_masks,
            cls_labels,
            gt_labels_list,
            gt_masks_list,
            boundary_proj,
            memory_proj,
            img_metas):
        """
        Input:
            pred_semantic_masks:     [bsz, c_th+c_st, h, w], logits before softmax.
            sup_semantic_masks:       [bsz, c_th+c_st, h, w], in value {0, 1}, semantic gts, typically point-level labels.
            cls_labels:     [bsz, c_th+c_st], in value {0, 1}, indicating whether the class exits.
            gt_labels_list: List<bsz>[ [n,] ], in value {0, 1, ..., C-1}, class label of each thing class, 'n' is the number of instances in the sample.
            gt_masks_list:  List<bsz>[ [n, h, w] ], in value {0, 1}, 'n' is the number of instances in the sample.
        """
        #images = self.tmp_state['img']
        #H, W = all_sem_masks_logit_scaled.shape[-2:]
        #images_ = F.interpolate(images, (H, W), mode='bilinear', align_corners=False)
        # semantic results
        pred_mask_probs = torch.maximum(pred_semantic_masks.softmax(1) * cls_labels[:, :, None, None], sup_semantic_masks)
        out_semantic_probs, out_semantics_list = pred_mask_probs.max(1)
        out_semantics_list[out_semantic_probs < 0.5] = 255
        pl_semantics_list = out_semantics_list

        # 
        H, W = gt_masks_list[0].shape[-2:]
        downsample = 16
        h, w = H // downsample, W // downsample
        dfactor = pred_mask_probs.new_tensor([1, 1./downsample, 1./downsample])

        resize = lambda x: F.interpolate(x, (h, w), mode='bilinear', align_corners=False)

        # class prob
        diff_prob = neighbour_diff(resize(pred_mask_probs), 'l1') # [n, 8, h, w]
        
        # euclidean coords
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w))).to(pred_mask_probs)
        coords = coords / torch.as_tensor([h, w]).to(coords.device).view(2, 1, 1)
        diff_euc = neighbour_diff(coords[None], 'l2') # [1, 8, h, w]

        # boundary
        boundary_proj = None
        if boundary_proj is None:
            images = self.tmp_state['img']
            img_edge = image_to_boundary(resize(images))
            diff_bond = neighbour_diff(img_edge[:, None], 'max')
        else:
            images = self.tmp_state['img']
            img_edge = image_to_boundary(resize(images))
            boundary_proj = resize(boundary_proj.sigmoid()[None])[0]
            #boundary_proj = boundary_proj - boundary_proj.min()
            #boundary_proj = boundary_proj / boundary_proj.max()
            #diff_bond = (neighbour_diff(img_edge[:, None], 'max') + neighbour_diff(boundary_proj[:, None], 'max')) / 2
            diff_bond = neighbour_diff(boundary_proj[:, None], 'max')

        # embed feature
        if _USE_FEATURE_EMBEDS:
            assert memory_proj is not None
        if memory_proj is not None:
            memory_proj = resize(memory_proj)
            diff_feat = neighbour_diff(memory_proj, 'dot')
            diff_feat = diff_feat.clip(min=0)
        else:
            diff_feat = 0

        #diff_all = diff_prob + diff_bond * 0.1 + diff_feat * 0.1
        diff_all = diff_prob * 1 + diff_bond * self.lambda_diff_bond + diff_feat * self.lambda_diff_feat

        # [n, h, w, 8]
        diff_np = diff_all.permute(0, 2, 3, 1).data.cpu().numpy()

        coords_raw = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), -1).to(pred_mask_probs) # [H, W, 2]

        out_bboxes_list, out_masks_list = [], []
        likeli_list = []
        for i, gt_masks in enumerate(gt_masks_list):
            # query point coordinates
            pnt_coords = torch.nonzero(gt_masks)
            if len(pnt_coords) == 0:
                pseudo_gt = torch.zeros_like(gt_masks)
                likeli_list.append(None)
            else:
                pnt_coords_raw = pnt_coords
                pnt_coords = (pnt_coords * dfactor.view(1, 3)).long()
                pnt_coords[:, 1].clip_(min=0, max=h-1)
                pnt_coords[:, 2].clip_(min=0, max=w-1)
                pnt_coords_np = pnt_coords.data.cpu().numpy()
                pnt_indices = [pnt_coords_np[:, 0] == ii for ii in range(len(gt_masks))]

                # min distance
                mindist = dijkstra_image(diff_np[i], pnt_coords_np[:, 1:])
                #mindist = np.array([mindist[pnt_coords_np[:, 0] == i].min(0) for i in range(len(gt_masks))])
                mindist_max = 10
                mindist_list = []
                for ii in range(len(gt_masks)):
                    #mindist_i = mindist[pnt_coords_np[:, 0] == ii]
                    mindist_i = mindist[pnt_indices[ii]]
                    if len(mindist_i) > 0:
                        mindist_list.append(mindist_i.min(0))
                    else:
                        mindist_list.append(np.full((h, w), mindist_max, np.float32))
                mindist = np.array(mindist_list)
                mindist = torch.as_tensor(mindist, dtype=torch.float32, device=pnt_coords.device)
                mindist /= mindist.max() + 1e-5
                mindist = F.interpolate(mindist[None], (H, W), mode='bilinear', align_corners=False)[0]
                dist_likeli = 1 - mindist # [n, H, W]

                # class comp
                gt_labels = F.one_hot(gt_labels_list[i], 256)[..., :self.num_things_classes].float() # [n, c]
                gt_labels = torch.cat([gt_labels,cls_labels[i:i+1, self.num_things_classes:].expand(len(gt_labels), -1).float()], -1) # [n, C]
                clas_likeli = torch.einsum('nc,chw->nhw', gt_labels, pred_mask_probs[i])

                # instance masks
                likeli = dist_likeli * clas_likeli
                likeli_list.append([dist_likeli, clas_likeli])
                
                pseudo_gt = likeli.argmax(0)
                pseudo_gt[pred_mask_probs[i].argmax(0) >= self.num_things_classes] = len(likeli)
                pseudo_gt = F.one_hot(pseudo_gt, len(likeli)+1)[..., :len(likeli)].permute(2, 0, 1).contiguous() # [n, h, w]

                if ALPHA_SHAPE:
                    if True:
                        # full resoulution
                        pnt_coords_raw_np = pnt_coords_raw.data.cpu().numpy()
                        alpha_mask = get_alpha_shape(H, W, [pnt_coords_raw_np[pnt_indices[ii]][:, 1:] for ii in range(len(gt_masks))]).astype(np.float32)
                        alpha_mask = torch.as_tensor(alpha_mask, dtype=torch.float32, device=pnt_coords_raw.device)
                    else:
                        # downsampled resolution
                        alpha_mask = get_alpha_shape(h, w, [pnt_coords_np[pnt_indices[ii]][:, 1:] for ii in range(len(gt_masks))]).astype(np.float32)
                        alpha_mask = torch.as_tensor(alpha_mask, dtype=torch.float32, device=pnt_coords.device)
                        alpha_mask = F.interpolate(alpha_mask[None], (H, W), mode='bilinear', align_corners=False)[0]

                    alpha_mask_conflict = alpha_mask.sum(0) > 1.001
                    alpha_mask = alpha_mask * (1 - alpha_mask_conflict.float())[None] # [n, h, w]
                    alpha_mask_exclude = alpha_mask.max(0, keepdim=True)[0] - alpha_mask # [n, h, w]

                    assert pseudo_gt.shape == alpha_mask.shape
                    pseudo_gt[alpha_mask.bool()] = 1
                    pseudo_gt[alpha_mask_exclude.bool()] = 0

            # revise semantic masks
            if ALPHA_SHAPE:
                sem_from_things_mask, sem_from_things_index = pseudo_gt.bool().max(0)
                sem_from_things = gt_labels_list[i][sem_from_things_index.flatten()]
                sem_from_things = sem_from_things.reshape(sem_from_things_mask.shape)
                pl_semantics_list[i][sem_from_things_mask] = sem_from_things[sem_from_things_mask]

            # instance bboxes
            pseudo_bboxes = []
            img_h, img_w = img_metas[i]['img_shape'][:2]
            mask_sizes = []
            for pgt in pseudo_gt:
                mask_coords = coords_raw[pgt > 0]
                mask_sizes.append(len(mask_coords))
                if len(mask_coords) == 0:
                    bboxes = [img_w//4, img_h//4, img_w*3//4, img_h*3//4]
                else:
                    y0, x0 = mask_coords.min(0)[0]
                    y1, x1 = mask_coords.max(0)[0]
                    bboxes = [x0, y0, x1+1, y1+1]
                pseudo_bboxes.append(bboxes)
            pseudo_bboxes = coords_raw.new_tensor(pseudo_bboxes)
            pseudo_bboxes[:, 0::2].clip_(min=0, max=img_w)
            pseudo_bboxes[:, 1::2].clip_(min=0, max=img_h)

            out_bboxes_list.append(pseudo_bboxes)
            out_masks_list.append(pseudo_gt)

        self.tmp_state.update({'likeli_list': likeli_list})

        pl_labels_list = gt_labels_list
        pl_bboxes_list = out_bboxes_list
        pl_masks_list = out_masks_list
        return pl_labels_list, pl_bboxes_list, pl_masks_list, pl_semantics_list

    def simplified_filter_and_loss(self,
            cls_scores,
            bbox_preds,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_inds_list, gt_inds_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bboxes,
                                 bboxes_gt,
                                 bbox_weights,
                                 avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos)

        assert len(gt_inds_list) == len(gt_masks_list) == num_imgs, (len(gt_inds_list), len(gt_masks_list), num_imgs)
        mask_targets = [mask[inds] for mask, inds in zip(gt_masks_list, gt_inds_list)]
        mask_weights = [mask.new_ones(mask.shape[0]) for mask in mask_targets]

        return loss_cls, loss_bbox, loss_iou, pos_inds_list, num_total_pos, mask_targets, mask_weights


    def loss_single_panoptic_simplified(self,
            cls_scores,
            bbox_preds,
            args_tuple,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            gt_semantics_list,
            img_metas,
            gt_bboxes_ignore_list=None):
        loss_dict = {}

        (loss_cls, loss_iou, loss_bbox,
         pos_inds_mask_list, num_total_pos_thing,
         mask_targets_list, mask_weights_list) = self.simplified_filter_and_loss(
                 cls_scores, bbox_preds,
                 gt_bboxes_list, gt_labels_list, gt_masks_list,
                 img_metas, gt_bboxes_ignore_list)
        loss_dict.update({
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_bbox': loss_bbox })

        # batch first args
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple
        bsz, _, embed_dims = query.shape

        # thing masks & loss
        max_query_num = max(len(pos_inds) for pos_inds in pos_inds_mask_list)
        thing_query = query.new_zeros([bsz, max_query_num, embed_dims])
        for i, pos_inds in enumerate(pos_inds_mask_list):
            thing_query[i, :len(pos_inds)] = query[i, pos_inds]

        mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory, memory_mask, None, thing_query, None, None, hw_lvl=hw_lvl)
        # dummy loss
        loss_dict['loss_cls'] = loss_dict['loss_cls'] + sum(x.sum() * 0 for x in query_inter_things)

        all_th_masks = []
        for th_masks in mask_inter_things + [mask_things]:
            th_masks = th_masks.squeeze(-1)
            th_masks = [mask[:len(pos_inds)].view(-1, *hw_lvl[0]) for mask, pos_inds in \
                    zip(th_masks, pos_inds_mask_list)]
            all_th_masks.append(th_masks)

        self.tmp_state.update({
            'th_masks': all_th_masks[-1],
            'th_pos_inds_list': pos_inds_mask_list, })

        th_mask_targets = torch.cat(mask_targets_list).float()
        th_mask_weights = torch.cat(mask_weights_list)

        all_th_masks = [torch.cat(th_masks) for th_masks in all_th_masks]
        all_th_masks = F.interpolate(torch.stack(all_th_masks), th_mask_targets.shape[-2:],
                mode='bilinear', align_corners=False)
        for i, th_masks in enumerate(all_th_masks):
            loss_mask = self.loss_mask(th_masks,
                    th_mask_targets,
                    th_mask_weights,
                    avg_factor=num_total_pos_thing)
            loss_dict.update({f'd{i}.loss_mask': loss_mask})

        #loss_dict.update({'loss_dummy': sum(x.sum()*0 for x in query_inter_things)})

        # stuff masks & loss
        stuff_query, stuff_query_pos = torch.split(
                self.stuff_query.weight[None].expand(bsz, -1, -1),
                embed_dims, -1)
        
        mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory, memory_mask, None, stuff_query, None, stuff_query_pos, hw_lvl=hw_lvl)

        all_st_masks = []
        for st_masks in mask_inter_stuff + [mask_stuff]:
            st_masks = st_masks.squeeze(-1)
            st_masks = st_masks.reshape(-1, *hw_lvl[0])
            all_st_masks.append(st_masks)

        all_st_cls = []
        for i, query_st in enumerate(query_inter_stuff):
            st_cls = self.cls_stuff_branches[i](query_st).view(-1, 1)
            all_st_cls.append(st_cls)

        self.tmp_state.update({
            'st_masks': all_st_masks[-1].view(bsz, -1, *hw_lvl[0]),
            'st_cls': all_st_cls[-1].view(bsz, -1), })
        
        if isinstance(gt_semantics_list, list):
            gt_semantics_list = torch.stack(gt_semantics_list)
        target_st = F.one_hot(gt_semantics_list.long(), max(256, gt_semantics_list.max()+1))
        target_st = target_st[..., self.num_things_classes:self.num_things_classes+self.num_stuff_classes]
        target_st = target_st.permute(0, 3, 1, 2).float().flatten(0, 1)

        all_st_masks = F.interpolate(torch.stack(all_st_masks), target_st.shape[-2:],
                mode='bilinear', align_corners=False)

        target_st_weight = target_st.max(2)[0].max(1)[0]
        num_total_pos_stuff = target_st_weight.sum()
        if num_total_pos_stuff > 0:
            for i, st_masks in enumerate(all_st_masks):
                loss_mask = self.loss_mask(st_masks,
                        target_st,
                        target_st_weight,
                        avg_factor=num_total_pos_stuff)
                loss_dict.update({f'd{i}.loss_st_mask': loss_mask})

            target_st_label = (1 - target_st_weight).long()
            for i, st_cls in enumerate(all_st_cls):
                loss_cls = self.loss_cls(
                        st_cls,
                        target_st_label,
                        avg_factor=num_total_pos_stuff) * 2
                loss_dict.update({f'd{i}.loss_st_cls': loss_cls})
        else:
            loss_dict.update({f'd{i}.loss_st_mask': (st_masks*0).sum() for i, st_masks in enumerate(all_st_masks)})
            loss_dict.update({f'd{i}.loss_st_cls': (st_cls*0).sum() for i, st_cls in enumerate(all_st_cls)})

        thing_ratio = num_total_pos_thing / (num_total_pos_thing + num_total_pos_stuff)
        stuff_ratio = 1 - thing_ratio
        return loss_dict, thing_ratio, stuff_ratio, all_th_masks, all_st_masks


    @force_fp32(apply_to=('x', ))
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_semantic_seg=None,
                      **kwargs):
        """
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)

        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            if gt_masks is None:
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            else:
                loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, gt_semantic_seg,
                                      img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_things_classes
            bbox_index = indexes // self.num_things_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return bbox_index, det_bboxes, det_labels

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list',
                          'args_tuple'))
    @torch.no_grad()
    def get_bboxes(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        img_metas,
        rescale=False,
    ):
        """
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        det_results = []
        pan_results = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id][..., :self.num_things_classes]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape'][:2]
            ori_shape = img_metas[img_id]['ori_shape'][:2]
            bch_shape = img_metas[img_id]['batch_input_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            proposals= self._get_bboxes_single(
                    cls_score, bbox_pred, img_shape, scale_factor, rescale)
            det_results.append(proposals)

            bbox_index, det_bboxes, det_labels = proposals

            thing_query = query[img_id:img_id+1, bbox_index]
            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                    memory[img_id:img_id+1],
                    memory_mask[img_id:img_id+1],
                    None,
                    thing_query,
                    None,
                    None,
                    hw_lvl=hw_lvl)

            stuff_query, stuff_query_pos = self.stuff_query.weight[None].split(self.embed_dims, -1)
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                    memory[img_id:img_id+1],
                    memory_mask[img_id:img_id+1],
                    None,
                    stuff_query,
                    None,
                    stuff_query_pos,
                    hw_lvl=hw_lvl)

            mask_pred = torch.cat([mask_things, mask_stuff], 1).view(-1, *hw_lvl[0])
            mask_pred = F.interpolate(mask_pred[None], bch_shape,
                    mode='bilinear', align_corners=False)[0]
            mask_pred = mask_pred[..., :img_shape[0], :img_shape[1]]
            mask_pred = F.interpolate(mask_pred[None], ori_shape,
                    mode='bilinear', align_corners=False)[0]

            stuff_query = query_inter_stuff[-1]
            scores_stuff = self.cls_stuff_branches[-1]( stuff_query).sigmoid().view(-1)

            binary_masks = mask_pred > 0.5
            mask_sizes = binary_masks.sum((1, 2)).float()

            scores_cls = torch.cat([det_bboxes[..., -1], scores_stuff])
            scores_msk = (mask_pred * binary_masks).sum((1, 2)) / (mask_sizes + 1)
            scores_all = scores_cls * (scores_msk**2)
            labels_all = torch.cat([det_labels,
                torch.arange(self.num_stuff_classes).to(det_labels) + self.num_things_classes])

            scores_all_, index = torch.sort(scores_all, descending=True)
            filled = binary_masks.new_zeros(mask_pred.shape[-2:]).bool()
            pan_result = torch.full(mask_pred.shape[-2:], 
                    self.num_things_classes+self.num_stuff_classes, device=mask_pred.device).long()
            pan_id = 1

            for i, score in zip(index, scores_all_):
                L = labels_all[i]
                isthing = L < self.num_things_classes

                score_threshold = self.quality_threshold_things \
                    if isthing else self.quality_threshold_stuff
                if score < score_threshold:
                    continue

                area = mask_sizes[i]
                if area == 0:
                    continue

                intersect_area = (binary_masks[i] & filled).sum()
                inter_threshold = self.overlap_threshold_things \
                        if isthing else self.overlap_threshold_stuff
                if (intersect_area / area) > inter_threshold:
                    continue

                mask = binary_masks[i] & (~filled)
                filled[mask] = True
                pan_result[mask] = pan_id * INSTANCE_OFFSET + L
                pan_id += 1

            # semantic prediction
            sem_query, sem_query_pos = self.semantic_query.weight[None].split(self.embed_dims, -1)
            mask_sem, mask_inter_sem, query_inter_sem = self.semantic_mask_head(
                    memory[img_id:img_id+1],
                    memory_mask[img_id:img_id+1],
                    None,
                    sem_query,
                    None,
                    sem_query_pos,
                    hw_lvl=hw_lvl)
            mask_sem = mask_sem.reshape(1, -1, *hw_lvl[0])
            mask_sem = self.semantic_proj(mask_sem).squeeze(0)
            sem_cls = self.cls_semantic_branches[-1](query_inter_sem[-1]).view(-1, 1, 1)
            sem_mask_pred = mask_sem * sem_cls.sigmoid()
            sem_mask_pred = F.interpolate(sem_mask_pred[None], bch_shape,
                    mode='bilinear', align_corners=False)[0]
            sem_mask_pred = sem_mask_pred[..., :img_shape[0], :img_shape[1]]
            sem_mask_pred = F.interpolate(sem_mask_pred[None], ori_shape,
                    mode='bilinear', align_corners=False)[0]
            sem_result = sem_mask_pred.argmax(0)

            pan_results.append(dict(
                pan_results=pan_result.data.cpu().numpy(),
                semantic=sem_result.data.cpu().numpy()))

        if VISUALIZATION_ONLY:
            #self.tmp_state['test_pan_results'] = pan_results
            dirname = './coco_test_results'
            os.makedirs(dirname, exist_ok=True)
            for i in range(len(pan_results)):
                img = self.tmp_state['img'][i]
                img_meta = img_metas[i]
                name = img_meta['ori_filename'].rsplit('.', 1)[0]
                with open(os.path.join(dirname, name+'.pkl'), 'wb') as f:
                    pickle.dump((img_meta, _get_rgb_image(img), pan_results[i]['pan_results']), f)
        return pan_results


    def get_visualization_panresult(self, thing_cls_scores, thing_masks, stuff_cls_scores, stuff_masks):
        assert thing_masks.ndim == 3, thing_masks.shape
        assert stuff_masks.ndim == 3, stuff_masks.shape
        assert thing_cls_scores.ndim == 2, thing_cls_scores.shape
        assert len(thing_cls_scores) == len(thing_masks), (thing_cls_scores.shape, thing_masks.shape)
        assert len(stuff_cls_scores) == len(stuff_masks), (stuff_cls_scores.shape, stuff_masks.shape)

        masks = torch.cat([thing_masks, stuff_masks])
        thing_cls_scores, thing_labels = thing_cls_scores.max(1)
        cls_scores = torch.cat([thing_cls_scores, stuff_cls_scores])
        stuff_labels = torch.arange(self.num_stuff_classes).to(thing_labels) + self.num_things_classes
        labels = torch.cat([thing_labels, stuff_labels])
        
        bmasks = masks > .5
        mask_scores = (masks * bmasks).sum((1, 2)) / bmasks.sum((1, 2)).clip(min=1)
        scores = cls_scores * mask_scores**2

        num_classes = self.num_things_classes + self.num_stuff_classes
        pan_result = torch.full(masks.shape[-2:], num_classes, device=masks.device, dtype=torch.long)
        pan_scores = torch.zeros(masks.shape[-2:], device=masks.device, dtype=torch.float32)
        filled = bmasks.new_zeros(masks.shape[-2:]).bool()

        scores_, index = torch.sort(scores, descending=True)
        pan_id = 1
        for i, score in zip(index, scores_):
            L = labels[i]

            M = bmasks[i] & (~filled)
            if M.sum() == 0: continue

            pan_result[M] = pan_id * INSTANCE_OFFSET + L
            filled[M] = True
            pan_scores[M] = masks[i][M]**2 * score
            pan_id += 1
        return pan_result, pan_scores

    @torch.no_grad()
    def get_visualization_single(self, i):
        # gt
        img = self.tmp_state['img'][i]
        img_meta = self.tmp_state['img_metas'][i]
        gt_bboxes = self.tmp_state['gt_bboxes_list'][i]
        gt_labels = self.tmp_state['gt_labels_list'][i]
        gt_masks = self.tmp_state['gt_masks_list'][i]
        gt_semantics = self.tmp_state['gt_semantics_list'][i]

        gt_stuff_masks = F.one_hot(gt_semantics.long())[..., self.num_things_classes:self.num_things_classes+self.num_stuff_classes]
        gt_stuff_masks = gt_stuff_masks.permute(2, 0, 1).float()

        gt_thing_scores = F.one_hot(gt_labels.long()).float()
        gt_stuff_scores = gt_stuff_masks.new_ones(gt_stuff_masks.shape[0])

        gt_pan, _ = self.get_visualization_panresult(
                gt_thing_scores, gt_masks, gt_stuff_scores, gt_stuff_masks)

        # pred things
        th_pos_inds = self.tmp_state['th_pos_inds_list'][i]
        th_cls_all = self.tmp_state['th_cls_scores'][i]
        th_bboxes_all = self.tmp_state['th_bbox_preds'][i]
        th_cls = th_cls_all[th_pos_inds].sigmoid()
        th_bboxes = th_bboxes_all[th_pos_inds]
        th_bboxes = bbox_cxcywh_to_xyxy(th_bboxes)
        H, W = img_meta['img_shape'][:2]
        th_bboxes[:, 0::2] = (th_bboxes[:, 0::2] * W).clip(min=0, max=W)
        th_bboxes[:, 1::2] = (th_bboxes[:, 1::2] * H).clip(min=0, max=H)
        th_masks = self.tmp_state['th_masks'][i]

        # pred stuff
        st_cls = self.tmp_state['st_cls'][i].sigmoid()
        st_masks = self.tmp_state['st_masks'][i]

        # pred
        H, W = img.shape[-2:]
        th_masks = F.interpolate(th_masks[None], (H, W), mode='bilinear', align_corners=False)[0]
        st_masks = F.interpolate(st_masks[None], (H, W), mode='bilinear', align_corners=False)[0]
        pred_pan, pred_scores = self.get_visualization_panresult(
                th_cls, th_masks, st_cls, st_masks)
        
        # pred semantic
        pred_semantic = self.tmp_state['semantic_pred_logit'][i].softmax(0)
        pred_semantic = F.interpolate(pred_semantic[None], (H, W), mode='bilinear', align_corners=False)[0]
        pred_semantic_prob, pred_semantic = pred_semantic.max(0)
        gt_semantics_expand = self.tmp_state['semantic_pred_target'][i]
        valid, gt_semantics_expand = gt_semantics_expand.max(0)
        gt_semantics_expand[valid < 1e-3] = 255

        # memory projection
        raw_imgs = []
        if 'memory_proj' in self.tmp_state:
            memory_proj = self.tmp_state['memory_proj'][i] # [c, h, w]
            _, h, w = memory_proj.shape
            vis_memory_proj = PCA(n_components=3).fit_transform(memory_proj.data.cpu().numpy().reshape(-1, h*w).T)
            vis_memory_proj = vis_memory_proj.reshape(h, w, 3)
            vis_memory_proj -= vis_memory_proj.min()
            vis_memory_proj /= vis_memory_proj.max() + 1e-5
            raw_imgs.append(vis_memory_proj)

        # center prediction
        if 'center_proj' in self.tmp_state:
            center_proj = self.tmp_state['center_proj'][i] # [2, h, w]
            center_proj = center_proj.data.cpu().numpy().astype(np.float32)
            mag, ang = cv2.cartToPolar(center_proj[0], center_proj[1])
            hsv = np.zeros(mag.shape+(3,), np.uint8)
            hsv[..., 0] = (ang * 180 / np.pi / 2).clip(min=0, max=180)
            hsv[..., 1] = (mag / mag.max().clip(min=1e-5) * 255).clip(min=0, max=255)
            hsv[..., 2] = 255
            vis_center_proj = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            raw_imgs.append(vis_center_proj)

        # implicit center prediction
        if 'implicit_center_proj' in self.tmp_state:
            center_proj = self.tmp_state['implicit_center_proj'][i]
            _, h, w = center_proj.shape
            vis_center_proj = PCA(n_components=3).fit_transform(center_proj.data.cpu().numpy().reshape(-1, h*w).T)
            vis_center_proj = vis_center_proj.reshape(h, w, 3)
            vis_center_proj -= vis_center_proj.min()
            vis_center_proj /= vis_center_proj.max() + 1e-5
            raw_imgs.append(vis_center_proj)

        if 'pred_boundary' in self.tmp_state:
            pred_boundary = self.tmp_state['pred_boundary'][i].sigmoid()
            target_boundary = [x[i] for x in self.tmp_state['target_boundarys']]
            #target_boundary += [(x > 0.1).float() for x in target_boundary]
            for bnd in [pred_boundary] + target_boundary:
                bnd = bnd[..., None].expand(-1, -1, 3)
                vis_bnd = (bnd.data.cpu().numpy() * 255).astype(np.uint8)
                raw_imgs.append(vis_bnd)

        out_dict = {
                'image': img,
                'pan_results': [gt_pan, pred_pan, gt_semantics_expand, pred_semantic],
                'bboxes': [gt_bboxes, th_bboxes],
                'labels': [gt_labels, th_cls.max(1)[1]],
                'heatmaps': [pred_scores, pred_semantic_prob],}

        if len(raw_imgs) > 0:
            out_dict.update({'raw': raw_imgs})

        return out_dict

    def get_visualization(self):
        """for visualization during training
        Return:
            Dict contains or List[ Dict ], each dict contains:
                'image':        tensor, required, [H, W, 3]
                'pan_results':  tensor | List[tensor], opt., [H, W]
                'bboxes':       tensor | List[tensor], opt., [N, 4] | [N, 5]
                'heatmaps':     tensor | List[tensor], opt., [H, W]
        """
        try:
            out_dict = self.get_visualization_single(0)
        except:
            out_dict = {}
        return out_dict

    def get_test_visualization(self):
        images = self.tmp_state['img']
        bsz = len(images)
        out_list = []
        for i in range(bsz):
            img_meta = self.tmp_state['img_metas'][i]
            name = img_meta['ori_filename'].rsplit('.')[0]
            pred_pan = self.tmp_state['test_pan_results'][i]['pan_results']
            img = self.tmp_state['img'][i]
            out_list.append(dict(image=img, name=name, pred_pan=pred_pan))
        return out_list

    def get_result_visualization(self):
        """ name
            image
            pan_results
            sem_results
            feature_proj
            heatmaps
        """
        images = self.tmp_state['img']
        bsz = len(images)

        out_list = []
        for i in range(bsz):
            img = self.tmp_state['img'][i]
            img_meta = self.tmp_state['img_metas'][i]
            name = img_meta['ori_filename'].rsplit('.')[0]

            img_shape = img_meta['img_shape'][:2]
            ori_shape = img_meta['ori_shape'][:2]

            # pseudo label
            gt_labels = self.tmp_state['gt_labels_list'][i]
            gt_masks = self.tmp_state['gt_masks_list'][i]
            gt_semantics = self.tmp_state['gt_semantics_list'][i]

            gt_stuff_masks = F.one_hot(gt_semantics.long())[..., self.num_things_classes:self.num_things_classes+self.num_stuff_classes]
            gt_stuff_masks = gt_stuff_masks.permute(2, 0, 1).float()
            gt_thing_scores = F.one_hot(gt_labels.long()).float()
            gt_stuff_scores = gt_stuff_masks.new_ones(gt_stuff_masks.shape[0])
            gt_pan, _ = self.get_visualization_panresult(
                    gt_thing_scores, gt_masks, gt_stuff_scores, gt_stuff_masks)

            # pred panoptic
            H, W = img_meta['img_shape'][:2]
            th_pos_inds = self.tmp_state['th_pos_inds_list'][i]
            th_cls_all = self.tmp_state['th_cls_scores'][i]
            th_cls = th_cls_all[th_pos_inds].sigmoid()
            th_masks = self.tmp_state['th_masks'][i]

            st_cls = self.tmp_state['st_cls'][i].sigmoid()
            st_masks = self.tmp_state['st_masks'][i]

            th_masks = F.interpolate(th_masks[None], (H, W), mode='bilinear', align_corners=False)[0]
            st_masks = F.interpolate(st_masks[None], (H, W), mode='bilinear', align_corners=False)[0]
            pred_pan, pred_scores = self.get_visualization_panresult(
                    th_cls, th_masks, st_cls, st_masks)

            # pred semantic
            pred_semantic = self.tmp_state['semantic_pred_logit'][i].softmax(0)
            pred_semantic = F.interpolate(pred_semantic[None], (H, W), mode='bilinear', align_corners=False)[0]
            pred_semantic_prob, pred_semantic = pred_semantic.max(0)

            # raw point label
            gt_semantics_expand = self.tmp_state['semantic_pred_target'][i]
            valid, gt_semantics_expand = gt_semantics_expand.max(0)
            gt_semantics_expand[valid < 1e-3] = 255

            # distance
            dist, clas = self.tmp_state['likeli_list'][i]

            # feature
            memory_proj = self.tmp_state['memory_proj'][i] # [c, h, w]
            _, h, w = memory_proj.shape
            vis_memory_proj = PCA(n_components=3).fit_transform(memory_proj.data.cpu().numpy().reshape(-1, h*w).T)
            vis_memory_proj = vis_memory_proj.reshape(h, w, 3)
            vis_memory_proj -= vis_memory_proj.min()
            vis_memory_proj /= vis_memory_proj.max() + 1e-5

            out_dict = dict(
                    name=name,
                    image=img,
                    pan_results=[gt_pan, pred_pan],
                    sem_results=[gt_semantics_expand, pred_semantic],
                    feature_proj = vis_memory_proj,
                    heatmaps=list(dist) + list(clas) + list(dist * clas),
                    )
            out_list.append(out_dict)
        return out_list


