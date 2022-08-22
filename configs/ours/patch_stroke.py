_base_ = [
    '../_base_/datasets/stroke.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='PatchEncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        model_name='vit_large_patch16_384',
        img_size=16,
        patch_size=1,
        in_chans=2,
        embed_dim=256,
        depth=6,
        num_heads=16,
        num_classes=2,
        mlp_ratio=1,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
        random_init=True,
    ),
    decode_head=dict(
        type='PatchVisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=5,
        img_size=16,
        embed_dim=256,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=0,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=[dict(
    #     type='PatchVisionTransformerUpHead',
    #     in_channels=1024,
    #     channels=512,
    #     in_index=9,
    #     img_size=16,
    #     embed_dim=256,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     num_conv=2,
    #     upsampling_method='bilinear',
    #     align_corners=False,
    #     conv3x3_conv1x1=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    #     dict(
    #     type='PatchVisionTransformerUpHead',
    #     in_channels=1024,
    #     channels=512,
    #     in_index=14,
    #     img_size=256,
    #     embed_dim=256,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     num_conv=2,
    #     upsampling_method='bilinear',
    #     align_corners=False,
    #     conv3x3_conv1x1=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    #     dict(
    #     type='PatchVisionTransformerUpHead',
    #     in_channels=1024,
    #     channels=512,
    #     in_index=19,
    #     img_size=16,
    #     embed_dim=256,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     num_conv=2,
    #     upsampling_method='bilinear',
    #     align_corners=False,
    #     conv3x3_conv1x1=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # ]
)

optimizer = dict(lr=0.01, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

crop_size = (256, 256)
train_cfg = dict(large_patch=16)
test_cfg = dict(mode='whole', crop_size=crop_size, stride=(512, 512), large_patch=16)
find_unused_parameters = True
data = dict(samples_per_gpu=1, workers_per_gpu=1)
