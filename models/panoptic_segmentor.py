import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import SingleStageDetector


@DETECTORS.register_module()
class PanopticSegmentor(SingleStageDetector):
    """
    Directly and densely predict masks (maybe and boxes) on the output features of the backbone+neck.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PanopticSegmentor, self).__init__(
            backbone,
            neck,
            bbox_head,
            train_cfg,
            test_cfg,
            pretrained,
            init_cfg)
        
        self.state_record = {}

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None
                      ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas) 
        x = self.extract_feat(img)

        # prepare targets
        H, W = img.shape[-2:]

        gt_masks_list = []
        for mask in gt_masks:
            mask = torch.tensor(mask.to_ndarray(), device=x[0].device)
            _, h, w = mask.shape
            mask = F.pad(mask, (0, W - w, 0, H - h))
            gt_masks_list.append(mask)

        if gt_semantic_seg is not None:
            assert gt_semantic_seg.shape[-2:] == (H, W), gt_semantic_seg.shape
            gt_semantic_seg = gt_semantic_seg.squeeze(1)
            gt_semantic_seg_pad = torch.full_like(gt_semantic_seg, 255)
            for i, img_meta in enumerate(img_metas):
                h, w = img_meta['img_shape'][:2]
                gt_semantic_seg_pad[i, :h, :w] = gt_semantic_seg[i, :h, :w]

        losses = self.bbox_head.forward_train(
            img, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore,
            gt_masks_list, gt_semantic_seg_pad)
        return losses

        BS,C,H,W = img.shape
        new_gt_masks = []
        for each in gt_masks:
            mask =torch.tensor(each.to_ndarray(),device=x[0].device)
            _,h,w = mask.shape
            padding = (
                0,W-w,
                0,H-h
            )
            mask = F.pad(mask,padding)
            new_gt_masks.append(mask)
        gt_masks = new_gt_masks

        if gt_semantic_seg is not None:
            assert gt_semantic_seg.shape[-2:] == img.shape[-2:], (gt_semantic_seg.shape, img.shape)
            gt_semantic_seg = gt_semantic_seg.squeeze(1)
            gt_semantic_new = torch.full_like(gt_semantic_seg, 255)
            for i, img_meta in enumerate(img_metas):
                h, w = img_meta['img_shape'][:2]
                gt_semantic_new[i, :h, :w] = gt_semantic_seg[i, :h, :w]
            gt_semantic_seg = gt_semantic_new
        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_masks,gt_bboxes_ignore,gt_semantic_seg=gt_semantic_seg)
        return losses

    def simple_test(self, img, img_metas=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        bs = len(img_metas)
        assert bs == 1, f'Only batch_size 1 supported for testing, given {len(img_metas)}.'

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        return results_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results