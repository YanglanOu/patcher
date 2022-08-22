_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
import os
model = dict(
    backbone=dict(
        embed_dim=128,
        in_chans=3,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=2
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=2
    ))

# dataset settings
dataset_type = 'PolypDataset'
data_root = os.path.expanduser('~/data/polyp')
img_norm_cfg = dict(
    mean=[126.65899086500784, 76.80592546064584, 55.02459328356547], std=[81.59283363667217, 55.31233184266645, 44.052315257650505], to_rgb=False)
crop_size = (352, 352)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadStrokeAnnotations'),
    dict(type='Resize', img_scale=(352, 352), ratio_range=(0.7, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadStrokeAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(352, 352),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(352, 352),
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
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TrainDataset/img_dir',
        ann_dir='TrainDataset/ann_dir',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='ValDataset/img_dir',
        ann_dir='ValDataset/ann_dir',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TestDataset/Kvasir/img_dir',
        ann_dir='TestDataset/Kvasir/ann_dir',
        pipeline=test_pipeline))


# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
# data=dict(samples_per_gpu=2)

train_cfg = dict()
test_cfg = dict()