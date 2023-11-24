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

__all__ = ['WSupPanopticSegmentationHead']

class FeatureProjection(nn.Module):
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
class WSupPanopticSegmentationHead(PanopticSegmentationHead):
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

        # additional semantic branches
        num_classes = self.num_thing_classes + self.num_stuff_classes
        self.semantic_mask_decoder = build_transformer(kwargs['stuff_mask_decoder'])
        self.semantic_query = nn.Embedding(num_classes, self.embed_dims * 2)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        fc_cls = Linear(self.embed_dims, 1)

        self.semantic_cls_branches = _get_clones(fc_cls, self.semantic_mask_decoder.num_layers)
        self.semantic_proj = nn.Conv2d(num_classes, num_classes, 1, groups=num_classes)

        # additional feature embedding branch
        self.feature_proj = FeatureProjection(self.embed_dims, 128, 'l2')
        self.iter_count = 0


    def init_weights(self):
        super(WSupPanopticSegmentationHead, self).init_weights()
        nn.init.constant_(self.semantic_proj.weight, 1.0)
        nn.init.constant_(self.semantic_proj.bias, 0.0)

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
        loss_pl, pl_bboxes, pl_masks, pl_semantics = \
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
        losses_stuff, num_stuff = self.loss_stuff_masks(memory,
                                             memory_mask,
                                             hw_lvl,
                                             pl_semantics,
                                             include_thing_classes=False)

        thing_weight = num_thing / (num_thing + num_stuff)
        stuff_weight = num_stuff / (num_thing + num_stuff)

        losses = {k: v * thing_weight for k, v in losses.items()}
        losses.update({k: v * thing_weight for k, v in losses_thing.items()})
        losses.update({k: v * stuff_weight for k, v in losses_stuff.items()})

        # Finally, the new pseudo-label related losses
        # Adjust the losses by the warm-up strategy
        weight = max(min(self.iter_count / self.warmup_iter, 1), 0)
        self.iter_count += 1
        losses = {k: v * weight for k, v in losses.items()}
        losses.update(loss_pl)
        return losses

    def get_semantic(self,
                     images,
                     memory,
                     memory_mask,
                     hw_lvl,
                     gt_semantic_seg):
        bs, _, dims = memory.shape
        sem_query, sem_query_pos = torch.split(
            self.semantic_query.weight[None].expand(bs, -1, -1),
            dims, -1)
        sem_query_list, sem_mask_list = self.semantic_mask_decoder(
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

        nDec, _, h, w = sem_masks.shape
        sem_masks_proj = self.semantic_proj(sem_masks.view(nDec * bs, -1, h, w))
        sem_masks_proj = sem_masks_proj.view(nDec, -1, h, w)

        # post process class prediction
        sem_cls_list = []
        for i, query in enumerate(sem_query_list):
            sem_cls = self.semantic_cls_branches[i](query).view(-1, 1)
            sem_cls_list.append(sem_cls)

        sem_cls_score = torch.stack(sem_cls_list)[..., None].sigmoid() # [nDec, bs * nClass, 1, 1]
        sem_masks_logit = sem_masks_proj * sem_cls_score

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
        for i, sem_cls in enumerate(sem_cls_list):
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

        return pred_semantic, given_semantic, semantic_existence, losses


    def get_embedding(self,
                      memory,
                      memory_pos,
                      memory_mask,
                      hw_lvl):
        embedding = self.feature_proj(memory, memory_pos, memory_mask, hw_lvl)
        return embedding

    @torch.no_grad()
    def get_pseudo_label(self,
                         images,
                         semantic_masks,
                         semantic_masks_gt,
                         semantic_labels,
                         gt_labels_list,
                         gt_masks_list,
                         embedding,
                         img_metas):
        ds_rate = 16
        H, W = gt_masks_list[0].shape[-2:]
        h, w = H // ds_rate, W // ds_rate
        ds_factor = semantic_masks.new_tensor([1, 1./ds_rate, 1./ds_rate])
        downsample = lambda x: F.interpolate(x, (h, w),
                                             mode='bilinear',
                                             align_corners=False)
        device = semantic_masks.device

        # semantic pseudo label
        semantic_probs = semantic_masks.softmax(1) * semantic_labels[..., None, None]
        semantic_probs = torch.maximum(semantic_probs, semantic_masks_gt)

        confidence, semantic_map = semantic_probs.max(1)
        semantic_map[confidence < 0.5] = 255
        pl_semantics = semantic_map

        # semantic distance [n, 8, h, w]
        semantic_dist = neighbour_diff(downsample(semantic_probs), 'l1')

        # geometry distance
        #coords = torch.stack(torch.mesh_grid(
        #    torch.arange(h, device=device, dtype=torch.float32) / h,
        #    torch.arange(w, device=device, dtype=torch.float32) / w
        #))
        #geometry_dist = neighbour_diff(coords[None], 'l2')

        # boundary distance
        images_edge = image_to_boundary(downsample(images))
        boundary_dist = neighbour_diff(images_edge[:, None], 'max')

        # embedding distance
        embedding_dist = neighbour_diff(downsample(embedding), 'dot').clip(min=0)

        # merge, and inference
        dist = semantic_dist + \
                boundary_dist * self.lambda_boundary + \
                embedding_dist * self.lambda_embedding

        dist_np = dist.permute(0, 2, 3, 1).data.cpu().numpy() # [n, h, w, 8]
        coords_raw = torch.stack(torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device)
        ), -1) # [H, W, 2]

        # find instance masks and boxes
        pl_masks_list, pl_bboxes_list = [], []
        for i, gt_masks in enumerate(gt_masks_list):
            num_ins = len(gt_masks)
            pnt_coords = torch.nonzero(gt_masks)
            if len(pnt_coords) == 0:
                pl_masks = torch.zeros_like(gt_masks)
            else:
                pnt_coords = (pnt_coords * ds_factor.view(1, 3)).long()
                pnt_coords[:, 1].clip_(min=0, max=h-1)
                pnt_coords[:, 2].clip_(min=0, max=w-1)
                pnt_coords_np = pnt_coords.data.cpu().numpy()
                pnt_indices = [pnt_coords_np[:, 0] == ii for ii in range(num_ins)]

                # find min distance
                min_dist = dijkstra_image(dist_np[i], pnt_coords_np[:, 1:])
                min_dist_list = []
                for ii in range(num_ins):
                    # if the ii's instance has some point annotations, extract the corresponding distance maps, and use the minimum value.
                    # if not, use a constant max value
                    min_dist_i = min_dist[pnt_indices[ii]]
                    if len(min_dist_i) > 0:
                        min_dist_list.append(min_dist_i.min(0))
                    else:
                        min_dist_list.append(np.full((h, w), 10, np.float32))
                # [nIns, h, w]
                mindist = torch.as_tensor(np.array(min_dist_list),
                                           dtype=torch.float32, device=device)
                mindist /= mindist.max() + 1e-5
                mindist = F.interpolate(mindist[None], (H, W),
                                        mode='bilinear', align_corners=False)[0]
                dist_evidence = 1 - mindist

                # and classes
                thing_labels = F.one_hot(gt_labels_list[i], self.num_thing_classes)
                stuff_labels = semantic_labels[i:i+1, self.num_thing_classes:]
                stuff_labels = stuff_labels.expand(num_ins, -1)
                gt_labels = torch.cat([thing_labels, stuff_labels], -1).float()
                cls_evidence = torch.einsum('nc,chw->nhw', gt_labels, semantic_probs[i])

                sim = dist_evidence * cls_evidence

                pl_masks = sim.argmax(0)
                non_thing_region = semantic_probs[i].argmax(0) >= self.num_thing_classes
                pl_masks[non_thing_region] = num_ins
                pl_masks = F.one_hot(pl_masks, num_ins+1)[..., :num_ins]
                pl_masks = pl_masks.permute(2, 0, 1).contiguous() # [nIns, h, w]


            # estimate bboxes
            pl_bboxes = []
            imgH, imgW = img_metas[i]['img_shape'][:2]

            for mask in pl_masks:
                mask_coords = coords_raw[mask > 0]
                if len(mask_coords) == 0:
                    # empty mask, make a dummy box
                    bboxes = [imgW//4, imgH//4, imgW*3//4, imgH*3//4]
                else:
                    y0, x0 = mask_coords.min(0)[0]
                    y1, x1 = mask_coords.max(0)[0]
                    bboxes = [x0, y0, x1+1, y1+1]
                pl_bboxes.append(bboxes)
            pl_bboxes = coords_raw.new_tensor(pl_bboxes)
            pl_bboxes[:, 0::2].clip_(min=0, max=imgW)
            pl_bboxes[:, 1::2].clip_(min=0, max=imgH)

            pl_bboxes_list.append(pl_bboxes)
            pl_masks_list.append(pl_masks)
        
        return pl_bboxes_list, pl_masks_list, pl_semantics

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


    def loss_embedding(self,
                       embedding,
                       gt_masks_list,
                       gt_semantic_seg,
                       pl_masks_list,
                       pl_stuff_masks):

        # merge stuff masks
        num_classes = self.num_thing_classes + self.num_stuff_classes
        gt_stuff_masks = F.one_hot(gt_semantic_seg.long(), 256)
        gt_stuff_masks = gt_stuff_masks[..., self.num_thing_classes:num_classes]
        gt_stuff_masks = gt_stuff_masks.permute(0, 3, 1, 2)

        gt_masks_list = [torch.cat([thing_masks, stuff_masks]) for \
                    thing_masks, stuff_masks in zip(gt_masks_list, gt_stuff_masks)]
        pl_masks_list = [torch.cat([thing_masks, stuff_masks]) for \
                    thing_masks, stuff_masks in zip(pl_masks_list, pl_stuff_masks)]

        # contrastive learning between gt points and pl masks
        h, w = embedding.shape[-2:]
        H, W = gt_masks_list[0].shape[-2:]
        coord_factor = embedding.new_tensor([1, h/H, w/W]).view(1, -1)

        scale = 0.07

        losses = []
        for i, (gl_masks, pl_masks) in enumerate(zip(gt_masks_list, pl_masks_list)):
            # [N, 3], each row contains [insID, h, w]
            pl_masks = F.interpolate(pl_masks[None].detach().float(), (h, w),
                                     mode='bilinear', align_corners=False)[0]

            coords = (torch.nonzero(gl_masks) * coord_factor).long()
            query = embedding[i][:, coords[:, 1], coords[:, 2]] # [dim, N]
            reference = torch.einsum('dhw,nhw->dn', embedding[i], pl_masks) /\
                            pl_masks.sum((1, 2)).clip(min=1)[None] # [dim, n]
            dot = (query.T @ reference) / scale

            ins_indices = coords[:, 0]
            pos_mask = torch.zeros_like(dot)
            pos_mask[torch.arange(dot.shape[0], device=dot.device), ins_indices]=1
            pos_mask_valid = (pos_mask.sum(1) < pos_mask.shape[1]).float()
            loss_i = - (torch.log_softmax(dot, 1) * pos_mask).sum(1)
            loss_i = (loss_i * pos_mask_valid).sum() / pos_mask_valid.sum().clip(min=1)
            losses.append(loss_i)

        losses = sum(losses) / len(losses)
        return dict(loss_embed = losses)

