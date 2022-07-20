_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/voc_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20
        ),
        mask_head=dict(
            num_classes=20
        ),
    )
)

# lr policy
lr_config = dict(step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)

evaluation = dict(interval=1)

# runtime
checkpoint_config = dict(interval=5)
#workflow = [('train', 5), ('val', 1)]
