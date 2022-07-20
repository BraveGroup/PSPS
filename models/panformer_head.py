import copy

import torch
from torch.distributed.distributed_c10d import get_rank
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS, build_head, build_loss
from mmdet.models.dense_heads.detr_head import DETRHead
from mmdet.models.dense_heads.deformable_detr_head import DeformableDETRHead
from mmdet.models.utils import build_transformer
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET


def _print_shapes(*args):
    print([x.shape if x is not None else 'None' for x in args])

@HEADS.register_module()
class PanformerHead(AnchorFreeHead):
    """
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                 *args,
                 num_thing_classes=0,
                 num_stuff_classes=0,
                 num_classes=0,
                 deformable_detr_head=None,
                 thing_mask_head=None,
                 stuff_mask_head=None,
                 loss_mask=dict(type='DiceLoss', weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)

        if num_classes > 0:
            assert num_thing_classes + num_stuff_classes == num_classes
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_thing_classes + num_stuff_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Isolate the env of deformable-detr from our model,
        deformable_detr_head['num_classes'] = self.num_classes
        deformable_detr_head.update(train_cfg=train_cfg)
        deformable_detr_head.update(test_cfg=dict(max_per_img=test_cfg['max_per_img']))
        self.deform_detr = build_head(deformable_detr_head)

        # Mask decoder for things and stuff.
        self.embed_dims = self.deform_detr.embed_dims
        self.thing_mask_transformer = build_transformer(thing_mask_head)
        self.stuff_mask_transformer = build_transformer(stuff_mask_head)
        self.stuff_query = nn.Embedding(self.num_stuff_classes, self.embed_dims * 2)

        self.loss_mask = build_loss(loss_mask)
        self.loss_cls = self.deform_detr.loss_cls

        self._init_layers()
        self._visualization_stats = {}

    def _init_layers(self):
        fc_cls = Linear(self.embed_dims, 1)
        self.stuff_cls_branches = nn.ModuleList([copy.deepcopy(fc_cls) \
                for _ in range(self.stuff_mask_transformer.num_layers)])

    def init_weights(self):
        self.deform_detr.init_weights()
        for m in self.stuff_cls_branches:
            nn.init.constant_(m.bias, bias_init_with_prob(0.01))


    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

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
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)

        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, gt_semantic_seg, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses


    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Hack the 'DeformableDETRHead' to provide additional mask decoding results.

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
        detr = self.deform_detr

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                detr.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not detr.as_two_stage:
            query_embeds = detr.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord, memory_info = detr.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=detr.reg_branches if detr.with_box_refine else None,  # noqa:E501
                    cls_branches=detr.cls_branches if detr.as_two_stage else None  # noqa:E501
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = detr.cls_branches[lvl](hs[lvl])
            tmp = detr.reg_branches[lvl](hs[lvl])
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

        hw_lvl = [x.shape[-2:] for x in mlvl_feats]
        memory_info = *memory_info, hw_lvl

        if detr.as_two_stage:
            return outputs_classes, outputs_coords, \
                enc_outputs_class, \
                enc_outputs_coord.sigmoid(), memory_info
        else:
            return outputs_classes, outputs_coords, \
                None, None, memory_info

    def loss(self,
            all_cls_scores,
            all_bbox_preds,
            enc_cls_scores,
            enc_bbox_preds,
            memory_info,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            gt_semantic_seg,
            img_metas,
            gt_bboxes_ignore=None):
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

        detr = self.deform_detr

        # split thing and stuff classes
        # NOTE: assume the first 'num_thing_classes' are things, this may be WRONG for some datasets.
        isthing = [gt_labels < self.num_thing_classes for gt_labels in gt_labels_list]
        gt_labels_list_th = [x[idx] for x, idx in zip(gt_labels_list, isthing)]
        gt_bboxes_list_th = [x[idx] for x, idx in zip(gt_bboxes_list, isthing)]
        gt_masks_list_th = [x[idx] for x, idx in zip(gt_masks_list, isthing)]

        # loc-decoder loss
        loss_dict = detr.loss(all_cls_scores, all_bbox_preds, enc_cls_scores, enc_bbox_preds,
                gt_bboxes_list_th, gt_labels_list_th, img_metas, gt_bboxes_ignore)

        # mask-decoder loss
        # get matching results
        num_imgs = all_cls_scores[-1].size(0)
        cls_scores_list = [all_cls_scores[-1][i] for i in range(num_imgs)]
        bbox_preds_list = [all_bbox_preds[-1][i] for i in range(num_imgs)]
        #cls_scores_list = list(all_cls_scores[-1])
        #bbox_preds_list = list(all_bbox_preds[-1])

        # cls_reg_targets = detr.get_targets(
        #         cls_scores_list, bbox_preds_list,
        #         gt_bboxes_list_th, gt_labels_list_th,
        #         img_metas, gt_bboxes_ignore)
        # (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        #  num_total_pos, num_total_neg) = cls_reg_targets
        # pos_inds_list = [label < self.num_thing_classes for label in labels_list]
        cls_reg_targets = self.get_targets(
                cls_scores_list, bbox_preds_list,
                gt_bboxes_list_th, gt_labels_list_th, gt_masks_list_th,
                img_metas, gt_bboxes_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, num_total_pos, num_total_neg,
         pos_inds_list, neg_inds_list, pos_gt_inds_list) = cls_reg_targets

        #   memory,         # [sum(H_i*W_i), bs, embed_dims]
        #   memory_pos,     # [sum(H_i*W_i), bs, embed_dims]
        #   memory_mask,    # [bs, sum(H_i*W_i)]
        #   query,          # [num_query, bs, embed_dims]
        #   query_pos,      # [num_query, bs, embed_dims]
        memory, memory_pos, memory_mask, query, query_pos, hw_lvl = memory_info

        # prepare thing queries for the mask decoder, only those matched are utilized
        #num_pos_inds_list = [x.sum() for x in pos_inds_list]
        num_pos_inds_list = [x.numel() for x in pos_inds_list]
        max_query_num = max(num_pos_inds_list)

        bsz = len(isthing)
        assert query.shape[1] == bsz, (bsz, query.shape)
        _, bsz, embed_dims = query.shape
        thing_query = query.new_zeros((max_query_num, bsz, embed_dims))
        thing_query_pos = query_pos.new_zeros((max_query_num, bsz, embed_dims))
        thing_query_mask = query.new_ones((bsz, max_query_num)).bool()
        for i, (qsize, pos_inds) in enumerate(zip(num_pos_inds_list, pos_inds_list)):
            #thing_query[:qsize, i].data.copy_(query[qmask, i])
            assert qsize > 0, num_pos_inds_list
            thing_query_mask[i, :qsize] = False
            thing_query[:qsize, i] = query[pos_inds, i]
            #thing_query_pos[:qsize, i] = query_pos[qmask, i]
            #print(query.shape, query_pos.shape, thing_query_pos.shape, num_pos_inds_list)

        # thing masks
        all_query_thing, all_mask_thing = self.thing_mask_transformer(
                memory=memory,
                memory_pos=memory_pos,
                memory_mask=memory_mask,
                query=thing_query,
                #query_pos=thing_query_pos,
                query_mask=thing_query_mask,
                hw_lvl=hw_lvl)

        # only keep masks for those selected queries
        all_mask_thing = [[masks[i, :qsize] for i, qsize in enumerate(num_pos_inds_list)] \
                for masks in all_mask_thing]


        # stuff masks
        stuff_query = self.stuff_query.weight.unsqueeze(1).expand(-1, bsz, -1)
        stuff_query, stuff_query_pos = stuff_query.split(embed_dims, -1)
        all_query_stuff, all_mask_stuff = self.stuff_mask_transformer(
                memory=memory,
                memory_pos=memory_pos,
                memory_mask=memory_mask,
                query=stuff_query,
                query_pos=stuff_query_pos,
                hw_lvl=hw_lvl)
        
        # stuff classes, List[ [bs, nQuery] ]
        all_cls_stuff = []
        for query_stuff, cls_branch in zip(all_query_stuff, self.stuff_cls_branches):
            all_cls_stuff.append(cls_branch(query_stuff).transpose(0, 1))


        # NOTE: there is another matching with (cls, box, mask) in the original implementation,
        # however, I think it may be useless because the predicted masks are obtained by fixed
        # matched queries with (cls, box). New assignment may lead to mismatching box prediction
        # and mask prediction from a single query.
        #assert len(gt_masks_list_th) == len(num_pos_inds_list)
        #gt_masks_th = torch.cat(gt_masks_list_th)
        #assert len(mask_targets_list) == len(num_pos_inds_list)
        #gt_masks_th = torch.cat(mask_targets_list)
        gt_masks_th = torch.cat([gt_masks[inds] for gt_masks, inds in zip(gt_masks_list_th, pos_gt_inds_list)])

        for i, mask_thing in enumerate(all_mask_thing):
            pred_mask_th = torch.cat(mask_thing)
            #pred_mask_th = torch.cat([mask[:qsize] for mask, qsize in zip(mask_thing, num_pos_inds_list)])
            assert pred_mask_th.shape[0] == gt_masks_th.shape[0], (pred_mask_th.shape, gt_masks_th.shape)
            #pred_mask_th = F.interpolate(pred_mask_th[None], gt_masks_th.shape[-2:],
            #        mode='bilinear', align_corners=False)[0]
            pred_mask_th = F.interpolate(pred_mask_th[None], bch_shape,
                    mode='bilinear', align_corners=False)[0]
            assert pred_mask_th.shape == gt_masks_th.shape, (pred_mask_th.shape, gt_masks_th.shape)
            loss_mask = self.loss_mask(pred_mask_th, gt_masks_th, avg_factor=num_total_pos)
            loss_dict[f'd{i}.loss_th_mask'] = loss_mask

        # prepare stuff targets
        # NOTE: assumes the first 'num_thing_classes' are thing classes.
        gt_masks_st = F.one_hot(gt_semantic_seg.long(), num_classes=max(256, self.num_classes))
        gt_masks_st = gt_masks_st.permute(0, 3, 1, 2).float()
        gt_masks_st = gt_masks_st[:, self.num_thing_classes:self.num_classes]
        for i, mask_stuff in enumerate(all_mask_stuff):
            mask_stuff = F.interpolate(mask_stuff, gt_masks_st.shape[-2:],
                    mode='bilinear', align_corners=False)
            loss_mask = self.loss_mask(mask_stuff.flatten(0, 1), gt_masks_st.flatten(0, 1))
                    #avg_factor=gt_masks_st.flatten(2).max(2)[0].sum())
            loss_dict[f'd{i}.loss_st_mask'] = loss_mask

        gt_cls_st = 1 - gt_masks_st.flatten(2).max(2)[0]
        for i, cls_stuff in enumerate(all_cls_stuff):
            loss_cls = self.loss_cls(
                    cls_stuff.view(-1, 1),
                    gt_cls_st.view(-1).long(),
                    avg_factor=(1 - gt_cls_st).sum())
            loss_dict[f'd{i}.loss_st_cls'] = loss_cls

        # dummy loss to prevent unused params.
        loss_dict['loss_dummy'] = sum(x.sum() for x in all_query_thing) * 0

        # for visualization
        self._visualization_stats.update({
            'pred_bboxes': bbox_preds_list[0],
            'pred_scores': cls_scores_list[0],
            'pred_masks': all_mask_thing[-1][0],
            'pred_stuff_masks': all_mask_stuff[-1][0],
            'pred_stuff_scores': all_cls_stuff[-1][0].squeeze(1),
            'pred_pos_inds': pos_inds_list[0],
            'gt_bboxes': gt_bboxes_list[0],
            'gt_labels': gt_labels_list[0],
            'gt_masks': gt_masks_list[0],
            'gt_stuff_masks': gt_masks_st[0], })
        return loss_dict

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Copy-Paste from mmdet.models.dense_heads.detr_head, except that it also
        returns indices.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        detr = self.deform_detr

        assert len(cls_score) == len(bbox_pred)
        max_per_img = min(detr.test_cfg.get('max_per_img', detr.num_query), len(cls_score))
        # exclude background
        if detr.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % detr.num_classes
            bbox_index = indexes // detr.num_classes
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
    
    def _get_masks_single(self, masks, scores, labels):
        binary_masks = masks > 0.5
        masks_size = binary_masks.sum((1, 2))

        cls_scores = scores
        msk_scores = (masks * binary_masks).sum((1, 2)) / masks_size.clip(min=1)
        all_scores = cls_scores * msk_scores**2

        assert len(all_scores) == len(masks) == len(labels), (all_scores.shape, masks.shape, labels.shape)
        filled = torch.zeros(masks.shape[-2:], device=masks.device, dtype=torch.bool)
        pan_result = torch.full(masks.shape[-2:], self.num_classes, device=masks.device, dtype=torch.long)
        mask_scores = torch.zeros(masks.shape[-2:], device=masks.device, dtype=torch.float32)
        pan_id = 0
        
        rank_scores, indices = torch.sort(all_scores, descending=True)
        for i, score in zip(indices, rank_scores):
            L = labels[i]
            isthing = L < self.num_thing_classes
            if score < self.test_cfg['mask_score_threshold']:
                continue
            area = masks_size[i]
            if area == 0:
                continue

            intersect_area = (binary_masks[i] & filled).sum()
            overlap_th = self.test_cfg[f"{'thing' if isthing else 'stuff'}_overlap_threshold"]
            if (intersect_area * 1. / area) > overlap_th:
                continue

            mask = binary_masks[i] & (~filled)
            filled[mask] = True
            pan_result[mask] = pan_id * INSTANCE_OFFSET + L
            mask_scores[mask] = masks[i][mask]
            pan_id += 1
        return pan_result, mask_scores

    def get_bboxes(self,
            all_cls_scores,
            all_bbox_preds,
            enc_output_class,
            enc_outputs_coord,
            memory_info,
            img_metas,
            rescale=False):
        """Get predictions.
        Args:
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
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        """

        # cls & bbox predictions, [bs, nQuery, nCls], [bs, nQuery, 4]
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        # masks
        #   memory,         # [sum(H_i*W_i), bs, embed_dims]
        #   memory_pos,     # [sum(H_i*W_i), bs, embed_dims]
        #   memory_mask,    # [bs, sum(H_i*W_i)]
        #   query,          # [num_query, bs, embed_dims]
        #   query_pos,      # [num_query, bs, embed_dims]
        memory, memory_pos, memory_mask, query, query_pos, hw_lvl = memory_info

        det_results = []
        pan_results = []
        for img_id in range(len(img_metas)):
            # get bboxes
            cls_score = cls_scores[img_id][..., :self.num_thing_classes]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape'][:2]
            ori_shape = img_metas[img_id]['ori_shape'][:2]
            bch_shape = img_metas[img_id]['batch_input_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            bbox_index, det_bboxes, det_labels = proposals
            det_results.append(proposals)

            # get masks
            thing_query = query[bbox_index, img_id:img_id+1]
            _, all_mask_thing = self.thing_mask_transformer(
                    memory=memory[:, img_id:img_id+1],
                    memory_pos=memory_pos[:, img_id:img_id+1],
                    memory_mask=memory_mask[img_id:img_id+1],
                    query=thing_query,
                    hw_lvl=hw_lvl)
            pred_mask_th = all_mask_thing[-1] # [bsz, nMask, h, w]

            stuff_query = self.stuff_query.weight.unsqueeze(1)
            stuff_query, stuff_query_pos = stuff_query.split(self.embed_dims, -1)
            all_query_stuff, all_mask_stuff = self.stuff_mask_transformer(
                    memory=memory[:, img_id:img_id+1],
                    memory_pos=memory_pos[:, img_id:img_id+1],
                    memory_mask=memory_mask[img_id:img_id+1],
                    query=stuff_query,
                    query_pos=stuff_query_pos,
                    hw_lvl=hw_lvl)
            pred_mask_st = all_mask_stuff[-1] # [bsz, nStuff, h, w]

            stuff_scores = self.stuff_cls_branches[-1](all_query_stuff[-1]).view(-1).sigmoid()
            assert len(stuff_scores) == self.num_stuff_classes, stuff_scores.shape

            # resize to the original image size, [nMask+nStuff, h, w]
            pred_masks = torch.cat([pred_mask_th, pred_mask_st], 1)
            pred_masks = F.interpolate(pred_masks, bch_shape, mode='bilinear', align_corners=False)
            pred_masks = pred_masks[..., :img_shape[0], :img_shape[1]]
            pred_masks = F.interpolate(pred_masks, ori_shape, mode='bilinear', align_corners=False)
            pred_masks = pred_masks.squeeze(0)#.sigmoid()
            pred_masks_binary = pred_masks > 0.5
            pred_masks_size = pred_masks_binary.sum((1, 2))

            # scores for masks
            pred_cls_scores = torch.cat([det_bboxes[:, -1], stuff_scores])
            pred_msk_scores = (pred_masks * pred_masks_binary).sum((1, 2)) / pred_masks_size.clip(min=1)
            pred_scores = pred_cls_scores * pred_msk_scores**2

            # labels for masks
            pred_labels = torch.cat([det_labels,
                torch.arange(self.num_stuff_classes).to(det_labels) + self.num_thing_classes])

            # assemble results
            assert len(pred_scores) == len(pred_labels) == len(pred_masks), (pred_scores.shape, pred_labels.shape, pred_masks.shape)

            rank_scores, indices = torch.sort(pred_scores, descending=True)
            filled = torch.zeros(pred_masks.shape[-2:], device=pred_masks.device, dtype=torch.bool)
            pan_result = torch.full(pred_masks.shape[-2:], self.num_classes,
                    device=pred_masks.device, dtype=torch.long)
            pan_id = 0
            for i, score in zip(indices, rank_scores):
                L = pred_labels[i]
                isthing = L < self.num_thing_classes

                if score < self.test_cfg['mask_score_threshold']:
                    continue

                area = pred_masks_size[i]
                if area == 0:
                    continue

                intersect_area = (pred_masks_binary[i] & filled).sum()
                overlap_th = self.test_cfg[f"{'thing' if isthing else 'stuff'}_overlap_threshold"]
                if (intersect_area*1. / area) > overlap_th:
                    continue

                mask = pred_masks_binary[i] & (~filled)
                filled[mask] = True
                pan_result[mask] = pan_id * INSTANCE_OFFSET + L
                pan_id += 1

            pan_results.append(dict(pan_results=pan_result.data.cpu().numpy()))
        return pan_results


    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Overwrite because we need to feed 'img_metas' to self.forward()'
        """
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, mask_targets_list, pos_inds_list, neg_inds_list,
         pos_gt_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, gt_masks_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, num_total_pos, num_total_neg,
                pos_inds_list, neg_inds_list, pos_gt_inds_list)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        detr = self.deform_detr
        
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = detr.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = detr.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    detr.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # mask targets
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, mask_targets, pos_inds,
                neg_inds, sampling_result.pos_assigned_gt_inds)


    def get_visualization(self):
        """
        pan_results: [h, w], 2D panoptic segmentation results.
        bboxes: [N, 4|5] boxes in [tlx, tly, brx, bry]
        labels: [N,]
        """
        out_list = []
        img = self._visualization_stats['img']
        img_meta = self._visualization_stats['img_metas']

        
        pos_inds = self._visualization_stats['pred_pos_inds']
        pred_bboxes = self._visualization_stats['pred_bboxes'][pos_inds]
        pred_scores = self._visualization_stats['pred_scores'][pos_inds]
        pred_masks = self._visualization_stats['pred_masks']

        assert len(pred_bboxes) == len(pred_scores) == len(pred_masks), (pred_bboxes.shape, pred_scores.shape, pred_masks.shape)

        # bbox results
        img_shape = img_meta['img_shape'][:2]
        scale_factor = img_meta['scale_factor']
        bch_shape = img_meta['batch_input_shape']
        rescale = False
        bbox_index, det_bboxes, det_labels = self._get_bboxes_single(
                pred_scores, pred_bboxes, img_shape, scale_factor, rescale)

        # panoptic results
        thing_masks = pred_masks[bbox_index]
        stuff_masks = self._visualization_stats['pred_stuff_masks']
        pred_masks = torch.cat([thing_masks, stuff_masks])[None]
        pred_masks = F.interpolate(pred_masks, bch_shape, mode='bilinear', align_corners=False)
        #pred_masks = pred_masks[..., :img_shape[0], :img_shape[1]]
        #pred_masks = F.interpolate(pred_masks, ori_shape, mode='bilinear', align_corners=False)
        pred_masks = pred_masks.squeeze(0)#.sigmoid()

        thing_labels = det_labels
        stuff_labels = torch.arange(self.num_stuff_classes).to(det_labels) + self.num_thing_classes
        pred_labels = torch.cat([thing_labels, stuff_labels])

        thing_scores = det_bboxes[:, -1]
        stuff_scores = self._visualization_stats['pred_stuff_scores'].sigmoid()
        pred_scores = torch.cat([thing_scores, stuff_scores])

        if get_rank() == 0:
            print(pred_scores.view(-1), pred_labels.view(-1),
                    pred_masks.max(2)[0].max(1)[0],
                    pred_masks.min(2)[0].min(1)[0])
        assert pred_scores.min() >= 0 and pred_scores.max() <= 1, (pred_scores.min(), pred_scores.max())

        pan_result, pan_scores = self._get_masks_single(pred_masks, pred_scores, pred_labels)
        assert pan_result.shape == img.shape[-2:], (pan_result.shape, img.shape, img_meta)

        out_list.append(dict(
            image=img,
            bboxes=det_bboxes,
            labels=det_labels,
            pan_results=pan_result,
            heatmaps=pan_scores))

        gt_bboxes = self._visualization_stats['gt_bboxes']
        gt_labels = self._visualization_stats['gt_labels']
        gt_thing_masks = self._visualization_stats['gt_masks']
        gt_stuff_masks = self._visualization_stats['gt_stuff_masks']
        gt_masks = torch.cat([gt_thing_masks, gt_stuff_masks])
        gt_masks_labels = torch.cat([gt_labels, stuff_labels])
        gt_masks_scores = torch.ones_like(gt_masks_labels).float()
        gt_pan_results, _ = self._get_masks_single(gt_masks, gt_masks_scores, gt_masks_labels)
        assert gt_pan_results.shape == img.shape[-2:], (gt_pan_results.shape, img.shape, img_meta)

        out_list.append(dict(
            image=img,
            bboxes=gt_bboxes,
            labels=gt_labels,
            pan_results=gt_pan_results,))

        return out_list
