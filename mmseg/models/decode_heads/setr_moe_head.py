# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import math

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), final_act=True, activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.final_act = final_act
        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for affine in self.affine_layers:
            x = affine(x)
            if affine != self.affine_layers[-1] or self.final_act:
                x = self.activation(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


@HEADS.register_module()
class SETRMOEHead(BaseDecodeHead):
    """
    """
    def __init__(self, feature_strides, prescale_mlp_dims=None, prescale_mlp_final_act=True,
                 afterscale_mlp_dims=[512, 256], afterscale_mlp_final_act=True, moe_mlp_dims=[512, 256], moe_conv_dims=None, activation='relu', use_linear_fuse=True, **kwargs):
        super(SETRMOEHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.prescale_mlp_dims = prescale_mlp_dims
        self.afterscale_mlp_dims = afterscale_mlp_dims
        self.use_linear_fuse = use_linear_fuse

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        cur_dim = sum(self.in_channels)
        if prescale_mlp_dims is not None:
            self.prescale_mlp = nn.ModuleList()
            for in_channel in self.in_channels:
                mlp = MLP(in_channel, prescale_mlp_dims, prescale_mlp_final_act, activation)
                self.prescale_mlp.append(mlp)

        cur_dim = len(self.in_channels) * prescale_mlp_dims[-1]

        if moe_conv_dims is not None:
            self.moe_conv = nn.ModuleList()
            conv_dims = moe_conv_dims + [len(self.in_channels)]
            for conv_dim in conv_dims:
                conv_layer = ConvModule(
                    in_channels=cur_dim,
                    out_channels=conv_dim,
                    kernel_size=3, stride=1, padding=1,
                    norm_cfg=dict(type='SyncBN', requires_grad=True)
                )
                cur_dim = conv_dim
                self.moe_conv.append(conv_layer)
        else:
            self.moe_conv = None


        if moe_mlp_dims is not None:
            self.moe_mlp = MLP(cur_dim, moe_mlp_dims + [len(self.in_channels)], False, activation)
        else:
            self.moe_mlp = None

        if afterscale_mlp_dims is not None:
            self.afterscale_mlp = MLP(prescale_mlp_dims[-1], afterscale_mlp_dims, afterscale_mlp_final_act, activation)
        cur_dim = afterscale_mlp_dims[-1]
        
        if use_linear_fuse:
            self.linear_fuse = ConvModule(
                in_channels=cur_dim,
                out_channels=embedding_dim,
                kernel_size=1,
                norm_cfg=dict(type='SyncBN', requires_grad=True)
            )
            cur_dim = embedding_dim

        self.linear_pred = nn.Conv2d(cur_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        x_scaled = []
        for i, x_i in enumerate(x):
            if x_i.dim() == 3:
                if x_i.shape[1] % 48 != 0:
                    x_i= x_i[:, 1:]
                # x_i = self.norm(x_i)

            if x_i.dim() == 3:
                n, hw, c = x_i.shape
                h = w = int(math.sqrt(hw))
                x_i = x_i.transpose(1, 2).reshape(n, c, h, w) 

            if i == 0:
                largest_size = x_i.shape[-2:]


            if self.prescale_mlp_dims is not None:
                x_i = self.prescale_mlp[i](x_i)

            if x_i.shape[-2:] != largest_size:
                x_i_scaled = resize(x_i, size=largest_size, mode='bilinear', align_corners=False)
            else:
                x_i_scaled = x_i

            x_scaled.append(x_i_scaled)

        x_stacked = torch.stack(x_scaled, dim=1) 
        x = torch.cat(x_scaled, dim=1)

        if self.moe_conv is not None:
            for conv_layer in self.moe_conv:
                x = conv_layer(x)

        if self.moe_mlp is not None:
            x = self.moe_mlp(x)

        moe_weights = torch.softmax(x, dim=1)
        x = (x_stacked * moe_weights.unsqueeze(2)).sum(1)

        if self.afterscale_mlp_dims is not None:
            x = self.afterscale_mlp(x)

        if self.use_linear_fuse:
            x = self.linear_fuse(x)

        x = self.dropout(x)
        x = self.linear_pred(x)

        return x
