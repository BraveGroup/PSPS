
_base_ = './coco_wsup.py'
lr_config = dict(policy='step', step=[8])
runner = dict(type='EpochBasedRunner', max_epochs=12)

custom_hooks = [dict(type='VisualizationHook2', priority='LOWEST')]
workflow = []


dataset_type = 'COCOPanopticDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
_size_div_ = 1
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='Resize', img_scale=(1333, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=_size_div_),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=_size_div_),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        #ann_file='./datasets/annotations/panoptic_train2017_detection_format.json',
        #img_prefix=data_root + 'train2017/',
        #ann_file=data_root + 'Panoptic/voc_panoptic_train_aug.json',
        ann_file=data_root + 'annotations/panoptic_train2017.json',
        img_prefix=data_root + 'train2017/',
        #seg_prefix=data_root + 'annotations/panoptic_train2017_1pnt_uniform_dynamic/',
        seg_prefix=data_root + 'annotations/panoptic_train2017_10pnt_uniform_dynamic/',
        #img_prefix=data_root + 'JPEGImages/',
        #seg_prefix=data_root + 'Panoptic/voc_panoptic_train_aug_1pnt_uniform/',
        #seg_prefix=data_root + 'Panoptic/voc_panoptic_train_aug/',
        #ann_file=data_root + 'Panoptic/voc_panoptic_val.json',
        #img_prefix=data_root + 'JPEGImages/',
        #seg_prefix=data_root + 'Panoptic/voc_panoptic_val/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
        #ann_file=data_root + 'Panoptic/voc_panoptic_val.json',
        #img_prefix=data_root + 'JPEGImages/',
        #seg_prefix=data_root + 'Panoptic/voc_panoptic_val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
        #ann_file=data_root + 'Panoptic/voc_panoptic_val.json',
        #img_prefix=data_root + 'JPEGImages/',
        #seg_prefix=data_root + 'Panoptic/voc_panoptic_val/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['pq', 'iou'])
