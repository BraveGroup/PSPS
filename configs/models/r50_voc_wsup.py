_base_ = [
    '../_base_/datasets/voc_points.py',
    #'../_base_/datasets/voc_panoptic.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='PanopticSegmentor',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='WSupPanopticSegmentationHead',
        num_query=300,
        num_thing_classes=20,
        num_stuff_classes=1,
        point_expand_size=17,
        lambda_boundary=0.1,
        lambda_embedding=0.1,
        warmup_iter=700,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        thing_mask_decoder=dict(
            type='MaskTransformerDecoder',
            num_layers=4),
        stuff_mask_decoder=dict(
            type='MaskTransformerDecoder',
            num_layers=4, self_attn=True),
        semantic_mask_decoder=dict(
            type='MaskTransformerDecoder',
            num_layers=2, self_attn=False),
        transformer=dict(
            type='DeformableDetrTransformerForMask',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=4),
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
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=4),
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
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0, activate=False)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00014,
    #lr=0.0002,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=20)

#custom_hooks = [dict(type='VisualizationHook', priority='LOWEST', interval=50)]
custom_hooks = [dict(type='SimpleVisHook', priority='LOWEST', interval=50)]
custom_imports = dict(
    imports=["datasets", "models"],
    allow_failed_imports=False)

