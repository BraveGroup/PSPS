_base_ = [
    '../_base_/datasets/voc_panoptic.py', '../_base_/default_runtime.py'
]

deformable_detr_head_ = dict(
    type='DeformableDETRHead',
    num_query=300,
    in_channels=2048,
    sync_cls_avg_factor=True,
    as_two_stage=False,
    with_box_refine=True,
    transformer=dict(
        type='DeformableDetrTransformer2',
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention', embed_dims=256),
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        decoder=dict(
            type='DeformableDetrTransformerDecoder',
            num_layers=6,
            return_intermediate=True,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.1,
                        proj_drop=0.1),
                    dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256)
                ],
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')))),
    positional_encoding=dict(
        type='SinePositionalEncoding',
        num_feats=128,
        normalize=True,
        offset=-0.5),
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0),
    loss_bbox=dict(type='L1Loss', loss_weight=5.0),
    loss_iou=dict(type='GIoULoss', loss_weight=2.0))

model = dict(
    type='Panformer',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='PanformerHead',
        num_thing_classes=20,
        num_stuff_classes=1,
        deformable_detr_head=deformable_detr_head_,
        thing_mask_head=dict(
            type='PanformerMaskDecoder',
            d_model=256,
            num_heads=8,
            num_layers=4),
        stuff_mask_head=dict(
            type='PanformerMaskDecoder',
            d_model=256,
            num_heads=8,
            num_layers=6,
            self_attn=True),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0, activate=False)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(
        max_per_img=100,
        mask_score_threshold=0.25,
        thing_overlap_threshold=0.4,
        stuff_overlap_threshold=0.2))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(320, 1333), (360, 1333),
                               (420, 1333), (480, 1333),
                               (512, 1333), (544, 1333), (576, 1333),
                               (600, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(320, 1333), (360, 1333),
                               (420, 1333), (480, 1333),
                               (512, 1333), (544, 1333), (576, 1333),
                               (600, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    #dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
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
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)
#train=dict(filter_empty_gt=False, pipeline=train_pipeline),
#val=dict(pipeline=test_pipeline),
#test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00014,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[18])
runner = dict(type='EpochBasedRunner', max_epochs=24)
#custom_hooks = [dict(type='CacheCleaner', priority='HIGHEST')]  # [dict(type='GradChecker',priority='HIGHEST')]
custom_hooks = [dict(type='VisualizationHook', priority='LOWEST', interval=50)]

custom_imports = dict(
    imports=["models", "datasets"],
    allow_failed_imports=False)

