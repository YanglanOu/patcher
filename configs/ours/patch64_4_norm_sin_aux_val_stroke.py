_base_ = [
    '../_base_/datasets/stroke_norm_val.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
depth = 18
in_index = 17
in_index_0 = 5
in_index_1 = 10
in_index_2 = 15
aux_depth = 18
embed_dim = 512
large_patch = 64
small_patch = 4
num_patch = int((256/large_patch)**2)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='PatchEncoderDecoder',
    num_patch=num_patch,
    embed_dim=embed_dim,
    backbone=dict(
        type='PatchVisionTransformer',
        model_name='vit_large_patch16_384',
        img_size=large_patch,
        patch_size=small_patch,
        in_chans=2,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=16,
        num_classes=2,
        mlp_ratio=1,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
        random_init=True,
        pos_embed_type='sin_cos',
        aux=True
    ),
    aux_backbone=dict(
        type='VisionTransformer',
        model_name='vit_large_patch16_384',
        img_size=256,
        patch_size=large_patch,
        in_chans=2,
        embed_dim=embed_dim,
        depth=aux_depth,
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
        in_index=in_index,
        img_size=large_patch,
        embed_dim=embed_dim,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[dict(
        type='PatchVisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=in_index_0,
        img_size=large_patch,
        embed_dim=embed_dim,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        conv3x3_conv1x1=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='PatchVisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=in_index_1,
        img_size=large_patch,
        embed_dim=embed_dim,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        conv3x3_conv1x1=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='PatchVisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=in_index_2,
        img_size=large_patch,
        embed_dim=embed_dim,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        conv3x3_conv1x1=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ]
)

optimizer = dict(lr=0.01, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

crop_size = (256, 256)
train_cfg = dict(large_patch=large_patch)
test_cfg = dict(mode='whole', crop_size=crop_size, stride=(512, 512), large_patch=large_patch)
find_unused_parameters = True
data = dict(samples_per_gpu=4, workers_per_gpu=1)
