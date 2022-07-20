
_base_ = './voc_base.py'
lr_config = dict(policy='step', step=[20])
runner = dict(type='EpochBasedRunner', max_epochs=30)
