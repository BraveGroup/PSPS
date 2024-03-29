
# previously named 'voc_wsup.py'

_base_ = [
    '../_base_/datasets/voc_points.py',
    '../_base_/default_runtime.py'
]
_dim_ = 256
_dim_half_ = _dim_//2
_feed_ratio_ = 4
_feed_dim_ = _feed_ratio_*_dim_
_num_levels_ = 4

# model
model = dict(
    type='PanSeg',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
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
        out_channels=_dim_,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=_num_levels_),
    bbox_head=dict(
        type='WsupPanformerHead',
        lambda_diff_prob=1,
        lambda_diff_bond=0.1,
        lambda_diff_feat=0.1,
        lambda_color_prior=3,
        expand_size=17,
        num_query=300,
        num_classes=21,  # 80+53
        num_things_classes=20,
        num_stuff_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        transformer=dict(
            type='Deformable_Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=_num_levels_,
                         ),
                    feedforward_channels=_feed_dim_,
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
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=_num_levels_,
                        )
                    ],
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_dim_half_,
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
        #loss_mask=dict(type='DiceLoss2', loss_weight=2.0),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0, activate=False),
        thing_transformer_head=dict(type='MaskHead',d_model=_dim_,nhead=8,num_decoder_layers=4),
        stuff_transformer_head=dict(type='MaskHead',d_model=_dim_,nhead=8,num_decoder_layers=6,self_attn=True),
        ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ),
        assigner_with_mask=dict(
            type='HungarianAssigner_multi_info',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            mask_cost=dict(type='DiceCost', weight=2.0),
            ),
        sampler =dict(type='PseudoSampler'),    
        sampler_with_mask =dict(type='PseudoSampler_segformer'),    
        ),
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

custom_hooks = [dict(type='VisualizationHook', priority='LOWEST', interval=50)]
custom_imports = dict(
    imports=["datasets", "easymd", "models"],
    allow_failed_imports=False)

