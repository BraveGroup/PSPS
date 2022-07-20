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
class PanformerHead(DETRHeadv2):
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
        #self.things_mask_head = build_transformer(thing_transformer_head)
        #self.stuff_mask_head = build_transformer(stuff_transformer_head)
        self.things_mask_head = _MaskTransformerWrapper(4, False)
        self.stuff_mask_head = _MaskTransformerWrapper(6, True)
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
        #self.reg_branches2 = _get_clones(reg_branch, self.num_dec_things) # used in mask decoder
        #self.cls_thing_branches = _get_clones(fc_cls, self.num_dec_things) # used in mask decoder
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff) # used in mask deocder

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
            #for m in self.cls_thing_branches:
            #    nn.init.constant_(m.bias, bias_init)
            for m in self.cls_stuff_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        #for m in self.reg_branches2:
        #    constant_init(m[-1], 0, bias=0)
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

        #gt_bboxes_list,
        #gt_labels_list,
        #gt_masks_list=None,
        #gt_semantics_list=None,

        self.tmp_state.update({
            'gt_bboxes_list': gt_bboxes_list,
            'gt_labels_list': gt_labels_list,
            'gt_masks_list': gt_masks_list,
            'gt_semantics_list': gt_semantics_list,
            'th_cls_scores': all_cls_scores[-1],
            'th_bbox_preds': all_bbox_preds[-1], })
        return loss_dict

        ### seprate things and stuff
        gt_things_lables_list = []
        gt_things_bboxes_list = []
        gt_things_masks_list = []
        gt_stuff_labels_list = []
        gt_stuff_masks_list = []
        
        #for i, each in enumerate(gt_labels_list):   
        #    # MDS: for coco, id<80 (Continuous id) is things. This is not true for other data sets
        #    things_selected = each < self.num_things_classes

        #    stuff_selected = things_selected == False

        #    gt_things_lables_list.append(gt_labels_list[i][things_selected])
        #    gt_things_bboxes_list.append(gt_bboxes_list[i][things_selected])
        #    gt_things_masks_list.append(gt_masks_list[i][things_selected])

        #    gt_stuff_labels_list.append(gt_labels_list[i][stuff_selected])
        #    gt_stuff_masks_list.append(gt_masks_list[i][stuff_selected])

        #num_dec_layers = len(all_cls_scores)
        #all_gt_bboxes_list = [
        #    gt_things_bboxes_list for _ in range(num_dec_layers - 1)
        #]
        #all_gt_labels_list = [
        #    gt_things_lables_list for _ in range(num_dec_layers - 1)
        #]
        ## all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers-1)]
        #all_gt_bboxes_ignore_list = [
        #    gt_bboxes_ignore for _ in range(num_dec_layers - 1)
        #]
        #img_metas_list = [img_metas for _ in range(num_dec_layers - 1)]

        # if the location decoder codntains L layers, we compute the losses of the first L-1 layers
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores[:-1], all_bbox_preds[:-1],
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        losses_cls_f, losses_bbox_f, losses_iou_f, losses_masks_things_f, losses_masks_stuff_f, loss_mask_things_list_f, loss_mask_stuff_list_f, loss_iou_list_f, loss_bbox_list_f, loss_cls_list_f, loss_cls_stuff_list_f, things_ratio, stuff_ratio = self.loss_single_panoptic(
            all_cls_scores[-1], all_bbox_preds[-1], args_tuple, None, #reference,
            gt_things_bboxes_list, gt_things_lables_list, gt_things_masks_list,
            (gt_stuff_labels_list, gt_stuff_masks_list), img_metas,
            gt_bboxes_ignore)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_things_lables_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_things_bboxes_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls * things_ratio
            loss_dict['enc_loss_bbox'] = enc_losses_bbox * things_ratio
            loss_dict['enc_loss_iou'] = enc_losses_iou * things_ratio
            # loss_dict['enc_loss_mask'] = enc_losses_mask
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls_f * things_ratio
        loss_dict['loss_bbox'] = losses_bbox_f * things_ratio
        loss_dict['loss_iou'] = losses_iou_f * things_ratio
        loss_dict['loss_mask_things'] = losses_masks_things_f * things_ratio
        loss_dict['loss_mask_stuff'] = losses_masks_stuff_f * stuff_ratio
        # loss from other decoder layers
        num_dec_layer = 0
        for i in range(len(loss_mask_things_list_f)):
            loss_dict[f'd{i}.loss_mask_things_f'] = loss_mask_things_list_f[
                i] * things_ratio
            loss_dict[f'd{i}.loss_iou_f'] = loss_iou_list_f[i] * things_ratio
            loss_dict[f'd{i}.loss_bbox_f'] = loss_bbox_list_f[i] * things_ratio
            loss_dict[f'd{i}.loss_cls_f'] = loss_cls_list_f[i] * things_ratio
        for i in range(len(loss_mask_stuff_list_f)):
            loss_dict[f'd{i}.loss_mask_stuff_f'] = loss_mask_stuff_list_f[
                i] * stuff_ratio
            loss_dict[f'd{i}.loss_cls_stuff_f'] = loss_cls_stuff_list_f[
                i] * stuff_ratio
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                losses_cls,
                losses_bbox,
                losses_iou,
        ):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * things_ratio
            loss_dict[
                f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i * things_ratio
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i * things_ratio

            num_dec_layer += 1
        # print(loss_dict)
        return loss_dict


    
    def filter_query(self,
                     cls_scores_list,
                     bbox_preds_list,
                     gt_bboxes_list,
                     gt_labels_list,
                     img_metas,
                     gt_bboxes_ignore_list=None):
        '''
        This function aims to using the cost from the location decoder to filter out low-quality queries.
        '''
        raise NotImplementedError
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (pos_inds_mask_list, neg_inds_mask_list, labels_list,
         label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._filter_query_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return pos_inds_mask_list, neg_inds_mask_list, labels_list, label_weights_list, bbox_targets_list, \
               bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list, neg_inds_list

    def _filter_query_single(self,
                             cls_score,
                             bbox_pred,
                             gt_bboxes,
                             gt_labels,
                             img_meta,
                             gt_bboxes_ignore=None):
        raise NotImplementedError
        num_bboxes = bbox_pred.size(0)
        pos_ind_mask, neg_ind_mask, assign_result = self.assigner_filter.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta,
            gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_things_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        return (pos_ind_mask, neg_ind_mask, labels, label_weights,
                bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets_with_mask(self,
                              cls_scores_list,
                              bbox_preds_list,
                              masks_preds_list_thing,
                              gt_bboxes_list,
                              gt_labels_list,
                              gt_masks_list,
                              img_metas,
                              gt_bboxes_ignore_list=None):
        raise NotImplementedError
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            masks_preds_list_thing  (list[Tensor]):
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single_with_mask,
                                      cls_scores_list, bbox_preds_list,
                                      masks_preds_list_thing, gt_bboxes_list,
                                      gt_labels_list, gt_masks_list, img_metas,
                                      gt_bboxes_ignore_list)
        num_total_pos_thing = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg_thing = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, mask_weights_list,
                num_total_pos_thing, num_total_neg_thing, pos_inds_list)

    def _get_target_single_with_mask(self,
                                     cls_score,
                                     bbox_pred,
                                     masks_preds_things,
                                     gt_bboxes,
                                     gt_labels,
                                     gt_masks,
                                     img_meta,
                                     gt_bboxes_ignore=None):
        raise NotImplementedError
        """
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        gt_masks = gt_masks.float()

        assign_result = self.assigner_with_mask.assign(bbox_pred, cls_score,
                                                       masks_preds_things,
                                                       gt_bboxes, gt_labels,
                                                       gt_masks, img_meta,
                                                       gt_bboxes_ignore)
        sampling_result = self.sampler_with_mask.sample(
            assign_result, bbox_pred, gt_bboxes, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_things_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        mask_weights = masks_preds_things.new_zeros(num_bboxes)
        mask_weights[pos_inds] = 1.0
        pos_gt_masks = sampling_result.pos_gt_masks
        _, w, h = pos_gt_masks.shape
        mask_target = masks_preds_things.new_zeros([num_bboxes, w, h])
        mask_target[pos_inds] = pos_gt_masks

        return (labels, label_weights, bbox_targets, bbox_weights, mask_target,
                mask_weights, pos_inds, neg_inds)

    def get_filter_results_and_loss(self, cls_scores, bbox_preds,
                                    cls_scores_list, bbox_preds_list,
                                    gt_bboxes_list, gt_labels_list, img_metas,
                                    gt_bboxes_ignore_list):
        raise NotImplementedError


        pos_inds_mask_list, neg_inds_mask_list, labels_list, label_weights_list, bbox_targets_list, \
        bbox_weights_list, num_total_pos_thing, num_total_neg_thing, pos_inds_list, neg_inds_list = self.filter_query(
            cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list,
            img_metas, gt_bboxes_ignore_list)
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos_thing * 1.0 + \
                         num_total_neg_thing * self.bg_cls_weight
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

        num_total_pos_thing = loss_cls.new_tensor([num_total_pos_thing])
        num_total_pos_thing = torch.clamp(reduce_mean(num_total_pos_thing),
                                          min=1).item()

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
                                 avg_factor=num_total_pos_thing)

        # regression L1 loss
        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos_thing)
        return loss_cls, loss_iou, loss_bbox,\
            pos_inds_mask_list, num_total_pos_thing

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

        #(loss_cls, loss_iou, loss_bbox,
        # pos_inds_mask_list, num_total_pos_thing) = self.get_filter_results_and_loss(
        #         cls_scores, bbox_preds,
        #         list(cls_scores), list(bbox_preds),
        #         gt_bboxes_list, gt_labels_list,
        #         img_metas, gt_bboxes_ignore_list)
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

        #cls_scores_matched = [data[pos_inds] for data, pos_inds in zip(cls_scores, pos_inds_mask_list)]
        #bbox_preds_matched = [data[pos_inds] for data, pos_inds in zip(bbox_preds, pos_inds_mask_list)]
        #gt_targets = self.get_targets_with_mask(
        #        cls_scores_matched, bbox_preds_matched, all_th_masks[-1],
        #        gt_bboxes_list, gt_labels_list, gt_masks_list,
        #        img_metas, gt_bboxes_ignore_list)  
        #(label_targets_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        # mask_targets_list, mask_weights_list, _, _, pos_inds_list) = gt_targets

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


    def loss_single_panoptic(self,
                             cls_scores,
                             bbox_preds,
                             args_tuple,
                             reference,
                             gt_bboxes_list,
                             gt_labels_list,
                             gt_masks_list,
                             gt_panoptic_list,
                             img_metas,
                             gt_bboxes_ignore_list=None):
        raise NotImplementedError
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            args_tuple:
            reference:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        gt_stuff_labels_list, gt_stuff_masks_list = gt_panoptic_list
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        loss_cls, loss_iou, loss_bbox, pos_inds_mask_list, num_total_pos_thing = self.get_filter_results_and_loss(
            cls_scores, bbox_preds, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)

        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        BS, _, dim_query = query.shape[0], query.shape[1], query.shape[-1]

        len_query = max([len(pos_ind) for pos_ind in pos_inds_mask_list])
        thing_query = torch.zeros([BS, len_query, dim_query],
                                  device=query.device)

        stuff_query, stuff_query_pos = torch.split(self.stuff_query.weight,
                                                   self.embed_dims,
                                                   dim=1)
        stuff_query_pos = stuff_query_pos.unsqueeze(0).expand(BS, -1, -1)
        stuff_query = stuff_query.unsqueeze(0).expand(BS, -1, -1)

        for i in range(BS):
            thing_query[i, :len(pos_inds_mask_list[i])] = query[
                i, pos_inds_mask_list[i]]

        mask_preds_things = []
        mask_preds_stuff = []
        # mask_preds_inter = [[],[],[]]
        mask_preds_inter_things = [[] for _ in range(self.num_dec_things)]
        mask_preds_inter_stuff = [[] for _ in range(self.num_dec_stuff)]
        cls_thing_preds = [[] for _ in range(self.num_dec_things)]
        cls_stuff_preds = [[] for _ in range(self.num_dec_stuff)]
        BS, NQ, L = bbox_preds.shape
        new_bbox_preds = [
            torch.zeros([BS, len_query, L]).to(bbox_preds.device)
            for _ in range(self.num_dec_things)
        ]

        mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
            memory, memory_mask, None, thing_query, None, None, hw_lvl=hw_lvl)

        mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
            memory,
            memory_mask,
            None,
            stuff_query,
            None,
            stuff_query_pos,
            hw_lvl=hw_lvl)

        mask_things = mask_things.squeeze(-1)
        mask_inter_things = torch.stack(mask_inter_things, 0).squeeze(-1)

        mask_stuff = mask_stuff.squeeze(-1)
        mask_inter_stuff = torch.stack(mask_inter_stuff, 0).squeeze(-1)

        for i in range(BS):

            tmp_i = mask_things[i][:len(pos_inds_mask_list[i])].reshape(
                -1, *hw_lvl[0])
            mask_preds_things.append(tmp_i)
            pos_ind = pos_inds_mask_list[i]
            reference_i = reference[i:i + 1, pos_ind, :]

            for j in range(self.num_dec_things):
                tmp_i_j = mask_inter_things[j][i][:len(pos_inds_mask_list[i]
                                                       )].reshape(
                                                           -1, *hw_lvl[0])
                mask_preds_inter_things[j].append(tmp_i_j)

                # mask_preds_inter_things[j].append(mask_inter_things[j].reshape(-1, *hw_lvl[0]))
                query_things = query_inter_things[j]
                t1, t2, t3 = query_things.shape
                tmp = self.reg_branches2[j](query_things.reshape(t1 * t2, t3)).reshape(t1, t2, 4)
                if len(pos_ind) == 0:
                    tmp = tmp.sum(
                    ) + reference_i  # for reply bug of pytorch broadcast
                elif reference_i.shape[-1] == 4:
                    tmp += reference_i
                else:
                    assert reference_i.shape[-1] == 2
                    tmp[..., :2] += reference_i

                outputs_coord = tmp.sigmoid()

                new_bbox_preds[j][i][:len(pos_inds_mask_list[i])] = outputs_coord
                cls_thing_preds[j].append(self.cls_thing_branches[j](
                    query_things.reshape(t1 * t2, t3)))

            # stuff
            tmp_i = mask_stuff[i].reshape(-1, *hw_lvl[0])
            mask_preds_stuff.append(tmp_i)
            for j in range(self.num_dec_stuff):
                tmp_i_j = mask_inter_stuff[j][i].reshape(-1, *hw_lvl[0])
                mask_preds_inter_stuff[j].append(tmp_i_j)

                query_stuff = query_inter_stuff[j]
                s1, s2, s3 = query_stuff.shape
                cls_stuff_preds[j].append(self.cls_stuff_branches[j](
                    query_stuff.reshape(s1 * s2, s3)))

        masks_preds_list_thing = [
            mask_preds_things[i] for i in range(num_imgs)
        ]
        mask_preds_things = torch.cat(mask_preds_things, 0)
        mask_preds_inter_things = [
            torch.cat(each, 0) for each in mask_preds_inter_things
        ]
        cls_thing_preds = [torch.cat(each, 0) for each in cls_thing_preds]
        cls_stuff_preds = [torch.cat(each, 0) for each in cls_stuff_preds]
        mask_preds_stuff = torch.cat(mask_preds_stuff, 0)
        mask_preds_inter_stuff = [
            torch.cat(each, 0) for each in mask_preds_inter_stuff
        ]
        cls_scores_list = [
            cls_scores_list[i][pos_inds_mask_list[i]] for i in range(num_imgs)
        ]

        bbox_preds_list = [
            bbox_preds_list[i][pos_inds_mask_list[i]] for i in range(num_imgs)
        ]

        gt_targets = self.get_targets_with_mask(cls_scores_list,
                                                bbox_preds_list,
                                                masks_preds_list_thing,
                                                gt_bboxes_list, gt_labels_list,
                                                gt_masks_list, img_metas,
                                                gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, _, _,
         pos_inds_list) = gt_targets

        thing_labels = torch.cat(labels_list, 0)
        things_weights = torch.cat(label_weights_list, 0)

        bboxes_taget = torch.cat(bbox_targets_list)
        bboxes_weights = torch.cat(bbox_weights_list)

        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds_list):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bboxes_gt = bbox_cxcywh_to_xyxy(bboxes_taget) * factors

        mask_things_gt = torch.cat(mask_targets_list, 0).to(torch.float)

        mask_weight_things = torch.cat(mask_weights_list,
                                       0).to(thing_labels.device)

        mask_stuff_gt = []
        mask_weight_stuff = []
        stuff_labels = []
        num_total_pos_stuff = 0
        for i in range(BS):
            num_total_pos_stuff += len(gt_stuff_labels_list[i])  ## all stuff

            select_stuff_index = gt_stuff_labels_list[
                i] - self.num_things_classes
            mask_weight_i_stuff = torch.zeros([self.num_stuff_classes])
            mask_weight_i_stuff[select_stuff_index] = 1
            stuff_masks = torch.zeros(
                (self.num_stuff_classes, *mask_targets_list[i].shape[-2:]),
                device=mask_targets_list[i].device).to(torch.bool)
            stuff_masks[select_stuff_index] = gt_stuff_masks_list[i].to(
                torch.bool)
            mask_stuff_gt.append(stuff_masks)
            select_stuff_index = torch.cat([
                select_stuff_index,
                torch.tensor([self.num_stuff_classes],
                             device=select_stuff_index.device)
            ])

            stuff_labels.append(1 - mask_weight_i_stuff)
            mask_weight_stuff.append(mask_weight_i_stuff)

        mask_weight_stuff = torch.cat(mask_weight_stuff,
                                      0).to(thing_labels.device)
        stuff_labels = torch.cat(stuff_labels, 0).to(thing_labels.device)
        mask_stuff_gt = torch.cat(mask_stuff_gt, 0).to(torch.float)

        num_total_pos_stuff = loss_cls.new_tensor([num_total_pos_stuff])
        num_total_pos_stuff = torch.clamp(reduce_mean(num_total_pos_stuff),
                                          min=1).item()
        if mask_preds_things.shape[0] == 0:
            loss_mask_things = (0 * mask_preds_things).sum()
        else:
            mask_preds = F.interpolate(mask_preds_things.unsqueeze(0),
                                       scale_factor=2.0,
                                       mode='bilinear').squeeze(0)
            mask_targets_things = F.interpolate(mask_things_gt.unsqueeze(0),
                                                size=mask_preds.shape[-2:],
                                                mode='bilinear').squeeze(0)
            loss_mask_things = self.loss_mask(mask_preds,
                                              mask_targets_things,
                                              mask_weight_things,
                                              avg_factor=num_total_pos_thing)
        if mask_preds_stuff.shape[0] == 0:
            loss_mask_stuff = (0 * mask_preds_stuff).sum()
        else:
            mask_preds = F.interpolate(mask_preds_stuff.unsqueeze(0),
                                       scale_factor=2.0,
                                       mode='bilinear').squeeze(0)
            mask_targets_stuff = F.interpolate(mask_stuff_gt.unsqueeze(0),
                                               size=mask_preds.shape[-2:],
                                               mode='bilinear').squeeze(0)

            loss_mask_stuff = self.loss_mask(mask_preds,
                                             mask_targets_stuff,
                                             mask_weight_stuff,
                                             avg_factor=num_total_pos_stuff)

        loss_mask_things_list = []
        loss_mask_stuff_list = []
        loss_iou_list = []
        loss_bbox_list = []
        for j in range(len(mask_preds_inter_things)):
            mask_preds_this_level = mask_preds_inter_things[j]
            if mask_preds_this_level.shape[0] == 0:
                loss_mask_j = (0 * mask_preds_this_level).sum()
            else:
                mask_preds_this_level = F.interpolate(
                    mask_preds_this_level.unsqueeze(0),
                    scale_factor=2.0,
                    mode='bilinear').squeeze(0)
                loss_mask_j = self.loss_mask(mask_preds_this_level,
                                             mask_targets_things,
                                             mask_weight_things,
                                             avg_factor=num_total_pos_thing)
            loss_mask_things_list.append(loss_mask_j)
            bbox_preds_this_level = new_bbox_preds[j].reshape(-1, 4)
            bboxes_this_level = bbox_cxcywh_to_xyxy(
                bbox_preds_this_level) * factors
            # We let this loss be 0. We didn't predict bbox in our mask decoder. Predicting bbox in the mask decoder is basically useless
            loss_iou_j = self.loss_iou(bboxes_this_level,
                                       bboxes_gt,
                                       bboxes_weights,
                                       avg_factor=num_total_pos_thing) * 0
            if bboxes_taget.shape[0] != 0:
                loss_bbox_j = self.loss_bbox(
                    bbox_preds_this_level,
                    bboxes_taget,
                    bboxes_weights,
                    avg_factor=num_total_pos_thing) * 0
            else:
                loss_bbox_j = bbox_preds_this_level.sum() * 0
            loss_iou_list.append(loss_iou_j)
            loss_bbox_list.append(loss_bbox_j)
        for j in range(len(mask_preds_inter_stuff)):
            mask_preds_this_level = mask_preds_inter_stuff[j]
            if mask_preds_this_level.shape[0] == 0:
                loss_mask_j = (0 * mask_preds_this_level).sum()
            else:
                mask_preds_this_level = F.interpolate(
                    mask_preds_this_level.unsqueeze(0),
                    scale_factor=2.0,
                    mode='bilinear').squeeze(0)
                loss_mask_j = self.loss_mask(mask_preds_this_level,
                                             mask_targets_stuff,
                                             mask_weight_stuff,
                                             avg_factor=num_total_pos_stuff)
            loss_mask_stuff_list.append(loss_mask_j)

        loss_cls_thing_list = []
        loss_cls_stuff_list = []
        thing_labels = thing_labels.reshape(-1)
        for j in range(len(mask_preds_inter_things)):
            # We let this loss be 0. When using "query-filter", only partial thing queries are feed to the mask decoder. This will cause imbalance when supervising these queries.
            cls_scores = cls_thing_preds[j]

            if cls_scores.shape[0] == 0:
                loss_cls_thing_j = cls_scores.sum() * 0
            else:
                loss_cls_thing_j = self.loss_cls(
                    cls_scores,
                    thing_labels,
                    things_weights,
                    avg_factor=num_total_pos_thing) * 2 * 0
            loss_cls_thing_list.append(loss_cls_thing_j)

        for j in range(len(mask_preds_inter_stuff)):
            if cls_scores.shape[0] == 0:
                loss_cls_stuff_j = cls_stuff_preds[j].sum() * 0
            else:
                loss_cls_stuff_j = self.loss_cls(
                    cls_stuff_preds[j],
                    stuff_labels.to(torch.long),
                    avg_factor=num_total_pos_stuff) * 2
            loss_cls_stuff_list.append(loss_cls_stuff_j)

        ## dynamic adjusting the weights
        things_ratio, stuff_ratio = num_total_pos_thing / (
            num_total_pos_stuff + num_total_pos_thing), num_total_pos_stuff / (
                num_total_pos_stuff + num_total_pos_thing)

        return loss_cls, loss_bbox, loss_iou, loss_mask_things, loss_mask_stuff, loss_mask_things_list, loss_mask_stuff_list, loss_iou_list, loss_bbox_list, loss_cls_thing_list, loss_cls_stuff_list, things_ratio, stuff_ratio

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

        ## append stuff masks
        #num_classes = self.num_things_classes + self.num_stuff_classes
        #if gt_semantic_seg is not None:
        #    for i in range(len(gt_semantic_seg)):
        #        stuff_masks, stuff_labels = [], []
        #        labels = torch.unique(gt_semantic_seg[i])
        #        stuff_index = (labels >= self.num_things_classes) & (labels < num_classes)
        #        for L in labels[stuff_index]:
        #            _mask = gt_semantic_seg[i] == L
        #            stuff_masks.append(_mask)
        #            stuff_labels.append(L)
        #        if len(stuff_labels) > 0:
        #            gt_labels[i] = torch.cat([gt_labels[i], torch.as_tensor(stuff_labels).to(gt_labels[i])], 0)
        #            gt_masks[i] = torch.cat([gt_masks[i], torch.cat(stuff_masks).to(gt_masks[i])], 0)
        #            gt_bboxes[i] = torch.cat([gt_bboxes[i], torch.zeros((len(stuff_labels), 4)).to(gt_bboxes[i])], 0)

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

            pan_results.append(dict(pan_results=pan_result.data.cpu().numpy()))
        return pan_results


        seg_list = []
        panoptic_list = []
        bbox_list = []
        labels_list = []
        pan_results_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            index, bbox, labels = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale)

            i = img_id
            thing_query = query[i:i + 1, index, :]
            thing_query_pos = query_pos[i:i + 1, index, :]
            joint_query = torch.cat([
                thing_query, self.stuff_query.weight[None, :, :self.embed_dims]
            ], 1)

            stuff_query_pos = self.stuff_query.weight[None, :,
                                                      self.embed_dims:]

            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory[i:i + 1],
                memory_mask[i:i + 1],
                None,
                joint_query[:, :-self.num_stuff_classes],
                None,
                None,
                hw_lvl=hw_lvl)
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory[i:i + 1],
                memory_mask[i:i + 1],
                None,
                joint_query[:, -self.num_stuff_classes:],
                None,
                stuff_query_pos,
                hw_lvl=hw_lvl)

            attn_map = torch.cat([mask_things, mask_stuff], 1)
            attn_map = attn_map.squeeze(-1)  # BS, NQ, N_head,LEN

            stuff_query = query_inter_stuff[-1]
            scores_stuff = self.cls_stuff_branches[-1](
                stuff_query).sigmoid().reshape(-1)

            mask_pred = attn_map.reshape(-1, *hw_lvl[0])

            mask_pred = F.interpolate(mask_pred.unsqueeze(0),
                                      size=ori_shape[:2],
                                      mode='bilinear').squeeze(0)

            masks_all = mask_pred
            seg_all = masks_all > 0.5
            sum_seg_all = seg_all.sum((1, 2)).float() + 1
            scores_all = torch.cat([bbox[:, -1], scores_stuff], 0)
            bboxes_all = torch.cat([
                bbox,
                torch.zeros([self.num_stuff_classes, 5], device=labels.device)
            ], 0)

            # MDS: concat stuff id for coco
            labels_all = torch.cat([labels, 
                torch.arange(self.num_things_classes, self.num_things_classes+self.num_stuff_classes).to(labels.device)], 0)

            ## mask wise merging
            seg_scores = (masks_all * seg_all.float()).sum(
                (1, 2)) / sum_seg_all
            scores_all *= (seg_scores**2)

            scores_all, index = torch.sort(scores_all, descending=True)

            masks_all = masks_all[index]
            labels_all = labels_all[index]
            bboxes_all = bboxes_all[index]
            seg_all = seg_all[index]

            bboxes_all[:, -1] = scores_all

            # MDS: select things for instance segmeantion
            things_selected = labels_all < self.num_things_classes
            stuff_selected = labels_all >= self.num_things_classes
            bbox_th = bboxes_all[things_selected][:100]
            labels_th = labels_all[things_selected][:100]
            seg_th = seg_all[things_selected][:100]
            labels_st = labels_all[stuff_selected]
            scores_st = scores_all[stuff_selected]
            masks_st = masks_all[stuff_selected]

            results = torch.zeros((2, *mask_pred.shape[-2:]),
                                  device=mask_pred.device).to(torch.long)
            num_classes = self.num_things_classes + self.num_stuff_classes
            assert num_classes == 21, num_classes
            pan_results = torch.full(mask_pred.shape[-2:], num_classes, device=mask_pred.device, dtype=torch.long)

            id_unique = 1
            pan_unique_id = 0

            unique_scores = np.zeros((num_classes+1,), np.float32)

            for i, scores in enumerate(scores_all):
                # MDS: things and sutff have different threholds may perform a little bit better
                _l = labels_all[i]
                unique_scores[_l] = max(unique_scores[_l], scores)

                if labels_all[i] < self.num_things_classes and scores < self.quality_threshold_things:
                    continue
                elif labels_all[i] >= self.num_things_classes and scores < self.quality_threshold_stuff:
                    continue
                _mask = masks_all[i] > 0.5
                mask_area = _mask.sum().item()
                intersect = _mask & (results[0] > 0)
                intersect_area = intersect.sum().item()
                if labels_all[i] < self.num_things_classes:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area
                                          ) > self.overlap_threshold_things:
                        continue
                else:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area
                                          ) > self.overlap_threshold_stuff:
                        continue
                if intersect_area > 0:
                    _mask = _mask & (results[0] == 0)
                    
                if self.cat_dict is None:
                    results[0, _mask] = 1
                else:
                    results[0, _mask] = self.cat_dict[labels_all[i]]['id']

                if labels_all[i] < self.num_things_classes:
                    results[1, _mask] = id_unique
                    id_unique += 1

                pan_results[_mask] = pan_unique_id * INSTANCE_OFFSET + labels_all[i]
                pan_unique_id += 1

            file_name = img_metas[img_id]['filename'].split('/')[-1].split(
                '.')[0]
            panoptic_list.append(
                (results.permute(1, 2, 0).cpu().numpy(), file_name, ori_shape))
            pan_results_list.append(dict(pan_results=pan_results.data.cpu().numpy()))

            bbox_list.append(bbox_th)
            labels_list.append(labels_th)
            seg_list.append(seg_th)
        return pan_results_list

        results = {
            'bbox': bbox_list,
            'segm': seg_list,
            'labels': labels_list,
            'panoptic': panoptic_list
        }
        return results

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

