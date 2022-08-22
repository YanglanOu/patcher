_base_ = [
    # '../../_base_/models/segformer.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]


depth = 18
in_index = 17
# in_index_0 = 3
# in_index_1 = 6
# in_index_2 = 9
# aux_depth = 12
embed_dim = 512
large_patch = 32
small_patch = 4
num_patch = (256/large_patch) ** 2
context_padding = 8
encoder_img_size = large_patch + context_padding*2

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    # pretrained='pretrained/mit_b0.pth',
    backbone=dict(
        type='PatchTransformer', img_size=256, patch_block_type='patchformer', 
        large_patch=[32,128,32,64,32,32,32], context_padding=[8,0,8,0,8,0,0],
        patch_sizes=[2,16,2,16,2,16,2], in_chans=2, embed_dims=[64,64,128,128,320,320,512], 
        num_heads=[1,1,2,2,5,5,8], mlp_ratios=[4,4,4,4,4,4,4],
        qkv_bias=True, depths=[6,6,6,6,6,6,6], sr_ratios=[2,1,2,1,1,1,1],
        drop_rate=0.0, drop_path_rate=0.1, alt=[False,True,False,True,False,True,False]),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0,2,4,6],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(768,768), stride=(768,768)))

# dataset settings
dataset_type = 'StrokeDataset'
data_root = '/data/stroke_trans/fold_0'
img_norm_cfg = dict(
    mean=[54.305244, 159.97537], std=[148.0489, 407.67776], to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadStrokeImageFromFile', to_float32=True),
    dict(type='LoadStrokeAnnotations'),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(1.0, 1.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadStrokeImageFromFile', to_float32=True),
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
    dict(type='LoadStrokeImageFromFile', to_float32=True),
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
    samples_per_gpu=4,
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


# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


train_cfg = dict()
test_cfg = dict()