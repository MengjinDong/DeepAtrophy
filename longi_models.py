#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:05:08 2019

@author: mdong
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 in_channel,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_date_diff_classes,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channel,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        last_size1 = int(math.ceil(sample_input_D / 8))
        last_size2 = int(math.ceil(sample_input_H / 8))
        last_size3 = int(math.ceil(sample_input_W / 8))
        self.avgpool = nn.AvgPool3d(
            (last_size1, last_size2, last_size3), stride=1)


        # self.fc_classification = nn.Linear(512 * block.expansion, num_classification_classes)

        self.fc_date_diff = nn.Linear(512 * block.expansion, num_date_diff_classes)


        # self.conv_seg = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         512 * block.expansion, # in_channel
        #         32, # out_channel
        #         2, # kernel_size
        #         stride=2
        #     ), # D_out = 2 * D_in; H_out = 2 * H_in; W_out = 2 * W_in;
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(
        #         32,
        #         32,
        #         kernel_size=3,
        #         stride=(1, 1, 1),
        #         padding=(1, 1, 1),
        #         bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(
        #         32,
        #         num_seg_classes,
        #         kernel_size=1,
        #         stride=(1, 1, 1),
        #         bias=False)
        # )



        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, image):

        # image = torch.cat((x1_seg, x2_jac), 1)
        image = self.conv1(image)
        image = self.bn1(image)
        image = self.relu(image)
        image = self.maxpool(image)
        image = self.layer1(image)
        image = self.layer2(image)
        image = self.layer3(image)
        image = self.layer4(image)
        image = self.avgpool(image)
        # image = self.conv_seg(image)

        image = image.view(image.size(0), -1)

        # out_weighted_jac = torch.flatten(out_weighted_jac, 1)
        # out_cls = self.fc_classification(image)
        out_t_order = self.fc_date_diff(image)

        return out_t_order

class ResNet_pair(nn.Module):
    def __init__(self, modelA, num_t_order_labels = 5):
        super(ResNet_pair, self).__init__()

        self.num_t_order_labels = num_t_order_labels
        self.modelA = modelA # modelA: segmentation network

    def forward(self, x_image):
        x1_image = x_image[:, 0:2, :, :, :]

        out_t_order_full1 = self.modelA(x1_image)

        # out_t_order1 = out_t_order_full1[:, 0:2]
        out_t_order1 = out_t_order_full1

        return out_t_order1

class ResNet_interval(nn.Module):
    def __init__(self, modelA, num_t_order_labels = 5, num_reg_labels= 4):
        super(ResNet_interval, self).__init__()

        self.num_t_order_labels = num_t_order_labels
        self.num_reg_labels = num_reg_labels
        self.modelA = modelA # modelA: segmentation network
        # self.fc_t_order = nn.Linear(num_t_order_labels, 2)
        self.fc = nn.Linear(2*num_t_order_labels, num_reg_labels)

    def forward(self, x_image):
        x1_image = x_image[:, 0:2, :, :, :]
        x2_image = x_image[:, 2:, :, :, :]
        # out_t_order1 = self.modelA(x1_image)
        # out_t_order2 = self.modelA(x2_image)
        #
        # out_range1 = torch.abs(out_t_order1[:, 0] - out_t_order1[:, 1])
        # out_range2 = torch.abs(out_t_order2[:, 0] - out_t_order2[:, 1]) * dd_ratio
        # out_range =  torch.cat((out_range1.unsqueeze_(-1), out_range2.unsqueeze_(-1)), 1)

        out_t_order_full1 = self.modelA(x1_image)
        out_t_order_full2 = self.modelA(x2_image)

        out_t_order1 = out_t_order_full1[:, 0:2]
        out_t_order2 = out_t_order_full2[:, 0:2]

        # out_t_order1 = self.fc_t_order(out_t_order_full1)
        # out_t_order2 = self.fc_t_order(out_t_order_full2)

        out_t_order_full = torch.cat((out_t_order_full1, out_t_order_full2), 1)
        out_range =  self.fc(out_t_order_full)

        return out_t_order1, out_t_order2, out_range

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


































    
