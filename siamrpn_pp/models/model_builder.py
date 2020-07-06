from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

from siamrpn_model.siamrpn_r50 import config as cfg
from siamrpn_pp.models.backbone.resnet_atrous import resnet50
from siamrpn_pp.models.neck.neck import AdjustAllLayer
from siamrpn_pp.models.head.rpn import MultiRPN

EPSILON = 1e-10


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        # build backbone
        self.backbone = resnet50(used_layers=cfg.BACKBONE.LAYERS)
        # build adjust layer
        self.neck = AdjustAllLayer(cfg.BACKBONE.CHANNELS, cfg.ADJUST.ADJUST_CHANNEL)
        # build rpn head
        channels = cfg.ADJUST.ADJUST_CHANNEL
        if len(channels) == 1:
            channels = channels[0]

        if cfg.RPN.WEIGHTED:
            self.rpn_head = MultiRPN(cfg.ANCHOR.ANCHOR_NUM, channels, True)
        else:
            self.rpn_head = MultiRPN(cfg.ANCHOR.ANCHOR_NUM, channels)

    # spatial attention
    def spatial_attention(self, tensor, spatial_type='sum'):
        spatial = None
        if spatial_type is 'mean':
            spatial = tensor.mean(dim=1, keepdim=True)
        elif spatial_type is 'sum':
            spatial = tensor.sum(dim=1, keepdim=True)
        return spatial

    # channel attention
    def channel_attention(self, tensor):
        # average global pooling
        AAP = nn.AdaptiveAvgPool2d((1))
        channel = AAP(tensor)

        return channel

    def fusion_spatial(self, f_crops):
        type = 'sum'
        num = 3
        zf_crop = []
        for i in range(num):
            shape = f_crops[0][i].size()
            # l1-norm spatial attention
            spatial1 = self.spatial_attention(f_crops[0][i], spatial_type=type)
            spatial2 = self.spatial_attention(f_crops[1][i], spatial_type=type)
            # get weight map, soft-max
            spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
            spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
            # fusion
            spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
            spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
            tensor_f_s = spatial_w1 * f_crops[0][i] + spatial_w2 * f_crops[1][i]

            # channel attention
            channel1 = self.channel_attention(f_crops[0][i])
            channel2 = self.channel_attention(f_crops[1][i])
            channel_w1 = torch.exp(channel1) / (torch.exp(channel1) + torch.exp(channel2) + EPSILON)
            channel_w2 = torch.exp(channel2) / (torch.exp(channel1) + torch.exp(channel2) + EPSILON)
            # fusion
            channel_w1 = channel_w1.repeat(1, 1, shape[2], shape[3])
            channel_w2 = channel_w2.repeat(1, 1, shape[2], shape[3])
            tensor_f_c = channel_w1 * f_crops[0][i] + channel_w2 * f_crops[1][i]

            tensor_f = 0.5*tensor_f_s + 0.5*tensor_f_c
            zf_crop.append(tensor_f)
        return zf_crop

    def template(self, z_crop):
        zf = []
        num = len(z_crop)
        for i in range(num):
            zf.append(self.backbone(z_crop[i]))
        if num is not 1:
            # fusion spatial based on l1-norm
            zf = self.fusion_spatial(zf)
        else:
            zf = zf[0]
        zf = self.neck(zf)
        self.zf = zf

    def track(self, x_crop):
        xf = []
        num = len(x_crop)
        for i in range(num):
            xf.append(self.backbone(x_crop[i]))
        if num is not 1:
            # fusion spatial based on l1-norm
            xf = self.fusion_spatial(xf)
        else:
            xf = xf[0]
        xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
               }

