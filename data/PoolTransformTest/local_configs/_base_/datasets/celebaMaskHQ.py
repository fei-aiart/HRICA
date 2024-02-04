# dataset settings
dataset_type = 'CelebaMaskHQDataset'
data_root = '/data/Datasets/CelebAMaskHQ_hry'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (473, 473)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/train/image',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/train/baimiao_nobg',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/train/canny',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/train/baimiao_nobg_simplify',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/train/ps_simplify',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/train/baimiao_nobg_noshape',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/train/baimiao_nobg_noappearance',
        
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/train/random_erasing',
        img_dir = '/data/Datasets/CelebAMaskHQ_hry/train/random_erasing_vec',
        ann_dir = '/data/Datasets/CelebAMaskHQ_hry/train/parsing',
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        data_root=data_root,
       
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/val/image',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/val/baimiao_nobg',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/val/canny',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/val/baimiao_nobg_simplify',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/val/ps_simplify',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/val/baimiao_nobg_noshape',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/val/baimiao_nobg_noappearance',

        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/val/random_erasing',
        img_dir = '/data/Datasets/CelebAMaskHQ_hry/val/random_erasing_vec',
        ann_dir = '/data/Datasets/CelebAMaskHQ_hry/val/parsing',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
      
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/test/image',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/test/baimiao_nobg',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/test/canny',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/test/baimiao_nobg_simplify',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/test/ps_simplify',
        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/test/baimiao_nobg_noshape',

        # img_dir = '/data/Datasets/CelebAMaskHQ_hry/test/random_erasing',
        img_dir = '/data/Datasets/CelebAMaskHQ_hry/test/random_erasing_vec',
        ann_dir = '/data/Datasets/CelebAMaskHQ_hry/test/parsing',
        pipeline=test_pipeline))
