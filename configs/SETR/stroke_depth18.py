_base_ = [
    '../_base_/models/setr_naive_pup.py',
    '../_base_/datasets/stroke.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(in_chans=2, img_size=256, align_corners=False,
                  pos_embed_interp=True, drop_rate=0., num_classes=2, random_init=True),
    decode_head=dict(img_size=256, align_corners=False, num_conv=2,
                     upsampling_method='bilinear', conv3x3_conv1x1=False, num_classes=2),
    auxiliary_head=[dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=5,
        img_size=256,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        conv3x3_conv1x1=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=10,
        img_size=256,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        conv3x3_conv1x1=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=15,
        img_size=256,
        embed_dim=1024,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        conv3x3_conv1x1=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ])

optimizer = dict(lr=0.01, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

crop_size = (256, 256)
test_cfg = dict(mode='whole', crop_size=crop_size, stride=(512, 512))
find_unused_parameters = True
data = dict(samples_per_gpu=6, workers_per_gpu=1)
