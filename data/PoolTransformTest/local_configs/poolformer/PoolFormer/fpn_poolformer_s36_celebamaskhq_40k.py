_base_ = [
    '../../_base_/models/fpn_r50.py',
    '../../_base_/datasets/celebaMaskHQ.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k.py'
]

# model settings
model = dict(
    type='EncoderDecoder',
    # pretrained='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar', # for old version of mmsegmentation
    backbone=dict(
        type='poolformer_s36_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='/data/yuhao/face_parsing/TopFormer/tools/work_dirs/fpn_poolformer_s36_lapa_40k/iter_28000.pth',
            checkpoint = 'https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar',
            ),
        ),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=16))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    samples_per_gpu=4,
    workers_per_gpu=0,
    )


# gpu_multiples = 1  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
# optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)

