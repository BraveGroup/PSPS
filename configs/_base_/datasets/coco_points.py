# dataset settings
dataset_type = 'COCOPanopticDataset'
data_root = 'data/coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
_img_scale_800 = [(480, 1333), (512, 1333), (544, 1333),
    (576, 1333), (608, 1333), (640, 1333),
    (672, 1333), (704, 1333), (736, 1333),
    (768, 1333), (800, 1333)]
_img_scale_600 = [
    (480, 1333), (512, 1333), (544, 1333),
    (576, 1333), (600, 1333)]
_img_scale = _img_scale_800

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPanopticAnnotations',
         with_bbox=True,
         with_mask=True,
         with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',
         img_scale=_img_scale,
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    #dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=_img_scale[-1],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_train2017.json',
        img_prefix=data_root + 'train2017/',
        seg_prefix=data_root + 'annotations/panoptic_train2017_1pnt_uniform/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric=['pq', 'iou'])
