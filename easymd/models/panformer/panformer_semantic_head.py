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
        super(PanformerHead, self).__init__(*args,
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
        self.semantic_proj = nn.Conv2d(num_classes, num_classes, 1)
        self.count = 0
        self.tmp_state = {}

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

        isthing = [gt_labels < self.num_things_classes for gt_labels in gt_labels_list]
        gt_th_labels_list = [data[mask] for data, mask in zip(gt_labels_list, isthing)]
        gt_th_bboxes_list = [data[mask] for data, mask in zip(gt_bboxes_list, isthing)]
        gt_th_masks_list = [data[mask] for data, mask in zip(gt_masks_list, isthing)]

        loss_dict = {}

        # location decoder loss
        for i, (cls_scores, bbox_preds) in enumerate(zip(all_cls_scores, all_bbox_preds)):
            if i == len(all_cls_scores) - 1:
                continue
            loss_cls, loss_bbox, loss_iou = self.loss_single(
                    cls_scores, bbox_preds,
                    gt_th_bboxes_list, gt_th_labels_list,
                    img_metas, gt_bboxes_ignore,)
            loss_dict.update({
                f'd{i}.loss_cls': loss_cls,
                f'd{i}.loss_bbox': loss_bbox,
                f'd{i}.loss_iou': loss_iou, })

        if enc_cls_scores is not None:
            binary_labels_list = [torch.zeros_like(x) for x in gt_th_labels_list]
            enc_losses_cls, enc_losses_bbox, enc_losses_iou = self.loss_single(
                    enc_cls_scores, enc_bbox_preds,
                    gt_th_bboxes_list, binary_labels_list,
                    img_metas, gt_bboxes_ignore)
            loss_dict.update({
                'enc_loss_cls': enc_losses_cls,
                'enc_loss_bbox': enc_losses_bbox,
                'enc_loss_iou': enc_losses_iou})

        # mask decoder loss
        loss_mask_decoder, thing_ratio, stuff_ratio = self.loss_single_panoptic_simplified(
                all_cls_scores[-1],
                all_bbox_preds[-1],
                args_tuple,
                gt_th_bboxes_list,
                gt_th_labels_list,
                gt_th_masks_list,
                gt_semantics_list,
                img_metas,
                gt_bboxes_ignore)
        loss_dict.update(loss_mask_decoder)

        for k in loss_dict.keys():
            ratio = stuff_ratio if 'st' in k else thing_ratio
            loss_dict[k] = loss_dict[k] * ratio

        self.tmp_state.update({
            'gt_bboxes_list': gt_bboxes_list,
            'gt_labels_list': gt_labels_list,
            'gt_masks_list': gt_masks_list,
            'gt_semantics_list': gt_semantics_list,
            'th_cls_scores': all_cls_scores[-1],
            'th_bbox_preds': all_bbox_preds[-1], })
        return loss_dict

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

        # get predictions of semantic cls
        all_sem_cls = []
        for i, query_sem in enumerate(query_inter_sem):
            sem_cls = self.cls_semantic_branches[i](query_sem).view(-1, 1)
            all_sem_cls.append(sem_cls)
        all_sem_cls = torch.stack(all_sem_cls) # [n_dec, bsz*cls, 1]

        all_sem_masks_proj = self.semantic_proj(all_sem_masks)

        # target mask
        target_mask = F.one_hot(gt_semantics_list.long(), max(256, gt_semantics_list.max()+1))
        target_mask = target_mask.permute(0, 2, 1, 2).float() # [bsz, cls, h, w]
        target_mask = expand_target_masks(target_mask)


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

        thing_ratio = num_total_pos_thing / (num_total_pos_thing + num_total_pos_stuff)
        stuff_ratio = 1 - thing_ratio
        return loss_dict, thing_ratio, stuff_ratio


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
            pan_id = 0

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

            pan_results.append(dict(pan_results=pan_result.data.cpu().numpy()))
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
        pan_id = 0
        for i, score in zip(index, scores_):
            L = labels[i]

            M = bmasks[i] & (~filled)
            if M.sum() == 0: continue

            pan_result[M] = pan_id * INSTANCE_OFFSET + L
            filled[M] = True
            pan_scores[M] = masks[i][M]**2 * score
            pan_id += 1
        return pan_result, pan_scores

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
        
        out_dict = {
                'image': img,
                'pan_results': [gt_pan, pred_pan],
                'bboxes': [gt_bboxes, th_bboxes],
                'labels': [gt_labels, th_cls.max(1)[1]],
                'heatmaps': pred_scores, }
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
        out_dict = self.get_visualization_single(0)
        return out_dict

