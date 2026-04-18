from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg


class ContrasExtractorLayer(nn.Module):

    def __init__(self):
        super(ContrasExtractorLayer, self).__init__()

        vgg16_layers = [ # 利用vgg预训练的卷积层来提取图像的特征
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
            'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        """
        conv3_1作为中层，能够捕捉提取复杂的纹理、形状和局部结构，而高层捕捉复杂的语义，底层捕捉简单的线条、边缘、颜色斑点等低级纹理特征
        选择 conv3_1 的目的： 通常是为了在**语义（Semantics）和细节（Details）**之间取得平衡。
        """
        conv3_1_idx = vgg16_layers.index('conv3_1') # 拿到列表中conv3_1的索引
        features = getattr(vgg,
                           'vgg16')(pretrained=True).features[:conv3_1_idx + 1] # 加载features模块（卷积层），保留从输入层到 conv3_1 的所有层。
        modified_net = OrderedDict()
        for k, v in zip(vgg16_layers, features):
            modified_net[k] = v # 通过有序字典将提取出的层与 vgg16_layers 中的字符串名称一一对应，从而可以通过具体的名称访问层

        self.model = nn.Sequential(modified_net)

        """
        VGG 是在 ImageNet 上训练的，输入图像必须先进行标准化才能得到正确的特征。
        当然，输入图像(RGB)首先要先约束到[0,1]，然后计算 img = (img - mean) / std
        通过 register_buffer，这两个张量会随模型一起移动到 GPU/CPU，但不会被视为模型参数进行更新。
        """
        # the mean is for image with range [0, 1]
        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # the std is for image with range [0, 1]
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, batch):
        batch = (batch - self.mean) / self.std
        output = self.model(batch)
        return output


class ContrasExtractorSep(nn.Module):

    def __init__(self):
        super(ContrasExtractorSep, self).__init__()

        self.feature_extraction_image1 = ContrasExtractorLayer()
        self.feature_extraction_image2 = ContrasExtractorLayer()

    def forward(self, image1, image2):
        dense_features1 = self.feature_extraction_image1(image1)
        dense_features2 = self.feature_extraction_image2(image2)

        return {
            'dense_features1': dense_features1,
            'dense_features2': dense_features2
        }
