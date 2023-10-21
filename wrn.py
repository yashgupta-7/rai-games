'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
  " 3x3 convolution with padding "
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, usebn=True):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    if usebn:
      self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    if usebn:
      self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride
    self.usebn = usebn

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    if self.usebn:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.usebn:
      out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

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


class PreActBasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(PreActBasicBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.bn1(x)
    out = self.relu(out)

    if self.downsample is not None:
      residual = self.downsample(out)

    out = self.conv1(out)

    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)

    out += residual

    return out


class PreActBottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(PreActBottleneck, self).__init__()
    self.bn1 = nn.BatchNorm2d(inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.bn1(x)
    out = self.relu(out)

    if self.downsample is not None:
      residual = self.downsample(out)

    out = self.conv1(out)

    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)

    out = self.bn3(out)
    out = self.relu(out)
    out = self.conv3(out)

    out += residual

    return out


class WideResNet(nn.Module):

  def __init__(self, block, layers, width=1, num_classes=10, usebn=True):
    super(WideResNet, self).__init__()
    self.inplanes = 16
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.relu = nn.ReLU(inplace=True)
    self.layer1_out = None
    self.layer1_only = False

    self.layer1 = self._make_layer(block, 16 * width, layers[0], usebn=usebn)
    self.layer2 = self._make_layer(block, 32 * width, layers[1], stride=2, usebn=usebn)
    self.layer3 = self._make_layer(block, 64 * width, layers[2], stride=2, usebn=usebn)
    self.avgpool = nn.AvgPool2d(8, stride=1)
    self.feature = None
    self.fc = nn.Linear(64 * block.expansion * width, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, usebn=True):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion)
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, usebn=usebn))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, usebn=usebn))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    self.layer1_out = x
    if self.layer1_only:
      return x

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    self.feature = x.clone()
    x = self.fc(x)

    return x


def wrn_28(**kwargs):
  model = WideResNet(BasicBlock, [4, 4, 4], **kwargs)
  return model

def wrn_28_med(**kwargs):
  model = WideResNet(BasicBlock, [3, 3, 3], **kwargs)
  return model

def wrn_28_small(**kwargs):
  model = WideResNet(BasicBlock, [1, 1, 1], **kwargs)
  return model

def wrn_34(**kwargs):
  model = WideResNet(BasicBlock, [6, 6, 6], **kwargs)
  return model