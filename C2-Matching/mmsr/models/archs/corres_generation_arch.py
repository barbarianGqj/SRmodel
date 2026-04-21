import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmsr.models.archs.arch_util import tensor_shift
from mmsr.models.archs.ref_map_util import feature_match_index
from mmsr.models.archs.vgg_arch import VGGFeatureExtractor

logger = logging.getLogger('base')


class CorrespondenceGenerationArch(nn.Module):

    def __init__(self,
                 patch_size=3,
                 stride=1,
                 vgg_layer_list=['relu3_1', 'relu2_1', 'relu1_1'],
                 vgg_type='vgg19'):
        super(CorrespondenceGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

        self.vgg_layer_list = vgg_layer_list
        self.vgg = VGGFeatureExtractor(
            layer_name_list=vgg_layer_list, vgg_type=vgg_type)

    def index_to_flow(self, max_idx):
        """
        :param max_idx: 特征匹配后，每个点得到的参考图中的最优匹配patch的一维索引坐标存储在max_idx矩阵中
        :return: 返回偏移量矩阵，代表矩阵对应的点到达最优参考点需要的偏移量
        这里面h,w均比原特征图的尺寸小2，pad后补回到原来的尺寸
        """
        device = max_idx.device
        # max_idx to flow
        h, w = max_idx.size()
        flow_w = max_idx % w # flow_w[h][w]，最优匹配点的横坐标
        flow_h = max_idx // w # 纵坐标

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device),
            torch.arange(0, w).to(device))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device) # [1,h,w,2] dim = 3为坐标[x, y]
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h), dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid  # 核心：偏移流=匹配点坐标 - 原始坐标（表示“需要移动多少到匹配位置”）
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2)) # [1, h+2, w+2, 2] 6个元素构成三对填充，从后向前填充。对dim=3开头填充0行，结尾补0行；对dim =2 开头填充0行，结尾补2行；dim=1开头0，结尾2行

        return flow

    def forward(self, dense_features, img_ref_hr):
        """
        :param dense_features: (dict) class ContrasExtractorSep use ContrasExtractorLayer(img) to generate {'dense_features1': dense_features1, 'dense_features2': dense_features2}
        :param img_ref_hr: image_{reference, highResolution}  [batch_size, C, H, W]
        :return:
        """
        batch_offset_relu3 = [] # element is [9, h+2, w+2, 2], [batch_size, 9, h+2, w+2, 2]
        batch_offset_relu2 = []
        batch_offset_relu1 = []
        for ind in range(img_ref_hr.size(0)): # ind range from 0 to batch_size
            feat_in = dense_features['dense_features1'][ind]
            feat_ref = dense_features['dense_features2'][ind]
            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w) # feat_in:[c, h, w]  feat_in.reshape(c, -1):[c, h * w] .view():[c, h, w]
            feat_ref = F.normalize(feat_ref.reshape(c, -1), dim=0).view(c, h, w)

            _max_idx, _max_val = feature_match_index(
                feat_in,
                feat_ref,
                patch_size=self.patch_size,
                input_stride=self.stride,
                ref_stride=self.stride,
                is_norm=True,
                norm_input=True)

            # offset map for relu3_1
            offset_relu3 = self.index_to_flow(_max_idx) # [1, h+2, w+2, 2]
            # shift offset relu3
            # 目的：增强偏移流的鲁棒性，覆盖邻域匹配可能  maybe
            shifted_offset_relu3 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu3, (i, j)) # 向右下平移0，1，2个单位，舍弃的刚好是之前pad的0，新增的也是0 [1,h+2,w+2,2]
                    shifted_offset_relu3.append(flow_shift)
            shifted_offset_relu3 = torch.cat(shifted_offset_relu3, dim=0) #[9, h+2, w+2, 2]
            batch_offset_relu3.append(shifted_offset_relu3)

            # offset map for relu2_1
            offset_relu2 = torch.repeat_interleave(offset_relu3, 2, 1) # relu2_1特征图分辨率是relu3_1的2倍 → 偏移流需要缩放+插值
            offset_relu2 = torch.repeat_interleave(offset_relu2, 2, 2)
            offset_relu2 *= 2 # 偏移值也要放大两倍，因为图偏大了两倍
            # shift offset relu2
            shifted_offset_relu2 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu2, (i * 2, j * 2))
                    shifted_offset_relu2.append(flow_shift)
            shifted_offset_relu2 = torch.cat(shifted_offset_relu2, dim=0)
            batch_offset_relu2.append(shifted_offset_relu2)

            # offset map for relu1_1
            offset_relu1 = torch.repeat_interleave(offset_relu3, 4, 1)
            offset_relu1 = torch.repeat_interleave(offset_relu1, 4, 2)
            offset_relu1 *= 4
            # shift offset relu1
            shifted_offset_relu1 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu1, (i * 4, j * 4))
                    shifted_offset_relu1.append(flow_shift)
            shifted_offset_relu1 = torch.cat(shifted_offset_relu1, dim=0)
            batch_offset_relu1.append(shifted_offset_relu1)

        # size: [b, 9, h, w, 2], the order of the last dim: [x, y]
        batch_offset_relu3 = torch.stack(batch_offset_relu3, dim=0)
        batch_offset_relu2 = torch.stack(batch_offset_relu2, dim=0)
        batch_offset_relu1 = torch.stack(batch_offset_relu1, dim=0)

        pre_offset = {}
        pre_offset['relu1_1'] = batch_offset_relu1
        pre_offset['relu2_1'] = batch_offset_relu2
        pre_offset['relu3_1'] = batch_offset_relu3

        img_ref_feat = self.vgg(img_ref_hr) # 再次调用vgg提取参考高清图像的指定relu层的特征
        return pre_offset, img_ref_feat
