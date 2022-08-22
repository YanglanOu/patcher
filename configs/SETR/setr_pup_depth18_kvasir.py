_base_ = [
    '../SETR/setr_stroke_base.py',
    # '../_base_/datasets/stroke_norm_val_aug.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

import os

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(in_chans=3, img_size=256, align_corners=False,
                  pos_embed_interp=True, drop_rate=0.,  num_classes=2, random_init=True),
    decode_head=dict(img_size=256, align_corners=False, num_conv=4, upsampling_method='bilinear',
                     num_upsampe_layer=4),
    auxiliary_head=[dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=9,
        img_size=256,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=14,
        img_size=256,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=19,
        img_size=256,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=23,
        img_size=256,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ])


# dataset settings
dataset_type = 'KvasirDataset'
data_root = os.path.expanduser('~/data/Kvasir-SEG')
img_norm_cfg = dict(
    mean=[142.7585609055524, 83.26750583573453, 61.995762433978165], std=[80.46627778511683, 56.7390076138584, 48.607730326529776], to_rgb=False)
crop_size = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadStrokeAnnotations'),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 1.5)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadStrokeAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
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
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
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
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        pipeline=test_pipeline))


optimizer = dict(lr=0.01, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

crop_size = (256, 256)
test_cfg = dict(mode='whole', crop_size=crop_size, stride=(512, 512))
find_unused_parameters = True

train_cfg = dict()
test_cfg = dict()