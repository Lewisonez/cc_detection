# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from mmcv.ops.merge_cells import GlobalPoolingCell, SumCell

from ..builder import NECKS
from .fpn import FPN


@NECKS.register_module()
class GAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `GAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(GAFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.GAFPN_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            GAFPN_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.GAFPN_convs.append(GAFPN_conv)
        self.nasfpn_stages = nn.ModuleList()
        stage0 = GlobalPoolingCell(with_out_conv=False)
        stage1 = GlobalPoolingCell(with_out_conv=False)
        stage2 = GlobalPoolingCell(with_out_conv=False)
        self.nasfpn_stages.extend([stage0, stage1,stage2])


    @auto_fp16()
    def forward(self, inputs):#input:[1, 64, 112, 112],[1, 128, 56, 56],[1, 320, 28, 28],[1, 512, 14, 14],
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [#lateral_conv是1x1的卷积,处理后存在laterals中：[1, 256, 112, 112],[1, 256, 56, 56],[1, 256, 28, 28],[1, 256, 14, 14]
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        lateral_temp_0 = self.downsample_convs[0](laterals[0])
        lateral_temp_1 = self.downsample_convs[1](laterals[1])
        lateral_temp_2 = self.downsample_convs[2](laterals[2])

        inter_outs = []##三处inter_outs改为outs
        inter_outs.append(self.fpn_convs[-1](laterals[-1]))
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]#获得h和w，作为上采样插值目标
            # fix runtime error of "+=" inplace operation in PyTorch 1.10
            laterals[i - 1] = laterals[i - 1] + F.interpolate(#使用插值上采样，然后相加
                inter_outs[0], size=prev_shape, mode='nearest')#结果存在新的laterals中，shape不变
            inter_outs.insert(0,self.fpn_convs[i - 1](laterals[i - 1]))

        # # build top-down path
        # used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     prev_shape = laterals[i - 1].shape[2:]#获得h和w，作为上采样插值目标
        #     # fix runtime error of "+=" inplace operation in PyTorch 1.10
        #     laterals[i - 1] = laterals[i - 1] + F.interpolate(#使用插值上采样，然后相加
        #         laterals[i], size=prev_shape, mode='nearest')#结果存在新的laterals中，shape不变

        # # build outputs
        # # part 1: from original levels
        # inter_outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]#使用3X3的卷积处理，shape不变

        # ###part 2: add bottom-up path
        # for i in range(0, used_backbone_levels - 1):
        #     inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])#通过3x3卷积进行下采样，后和上一层直接相加

        outs = []
        outs.append(inter_outs[0])#[1, 256, 112, 112]
        outs.extend([#上边三层用3x3卷积处理
            self.GAFPN_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])#总的各层shape不变，outs：[1, 64, 112, 112],[1, 128, 56, 56],[1, 320, 28, 28],[1, 512, 14, 14]

        outs[1] = self.nasfpn_stages[0](lateral_temp_0,outs[1],out_size=outs[1].shape[-2:])
        outs[2] = self.nasfpn_stages[1](lateral_temp_1,outs[2],out_size=outs[2].shape[-2:])
        outs[3] = self.nasfpn_stages[2](lateral_temp_2,outs[3],out_size=outs[3].shape[-2:])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))#增加了一层输出，maxpool
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
