import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable

from .operations import *


PRIMITIVES = [
    'none',
    'ag',
    'cross_att',
    'exc'
]


class Cell(nn.Module):
    """The searching form of cells.
    Adopt the standard DARTS scheme, where the output is the
    weighted sum of all options."""

    def __init__(self, in_channels, out_channels):
        super(Cell, self).__init__()
        mid_channels = in_channels // 4
        self.conv1a = Conv1x1(in_channels, mid_channels)
        self.conv1b = Conv1x1(in_channels, mid_channels)

        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        # Create interaction options.
        self._op2s = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](mid_channels, mid_channels)
            self._op2s.append(op)

        self.conv3a = LightConv3x3(mid_channels, mid_channels)
        self.conv3b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        # Create interaction options.
        self._op3s = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](mid_channels, mid_channels)
            self._op3s.append(op)

        self.conv4a = Conv1x1Linear(mid_channels, out_channels)
        self.conv4b = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x, weights):
        identity = x
        x1a = self.conv1a(x)
        x1b = self.conv1b(x)

        x2a = self.conv2a(x1a)
        x2b = self.conv2b(x1b)
        x2as, x2bs = [], []
        for op, w in zip(self._op2s, weights[0]):
            x2a_op, x2b_op = op((x2a, x2b))
            x2as.append(x2a_op * w)
            x2bs.append(x2b_op * w)
        x2a = sum(x2as)
        x2b = sum(x2bs)

        x3a = self.conv3a(x2a)
        x3b = self.conv3b(x2b)
        x3as, x3bs = [], []
        for op, w in zip(self._op3s, weights[1]):
            x3a_op, x3b_op = op((x3a, x3b))
            x3as.append(x3a_op * w)
            x3bs.append(x3b_op * w)
        x3a = sum(x3as)
        x3b = sum(x3bs)

        x4 = self.conv4a(x3a) + self.conv4b(x3b)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x4 + identity
        return F.relu(out)


class SearchCNN(nn.Module):
    """The searching form of MSINet.
    Simultaneously maintain the model parameters and architecture parameters."""

    def __init__(self, num_classes, channels, criterion):
        super(SearchCNN, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self._criterion = criterion

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.cells = nn.ModuleList()
        for i in range(3):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            self.cells += [
                Cell(in_channels, out_channels),
                Cell(out_channels, out_channels),
            ]
            if i != 2:
                self.cells += [
                    nn.Sequential(
                        Conv1x1(out_channels, out_channels),
                        nn.AvgPool2d(2, stride=2)
                    )
                ]

        out_planes = channels[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.BatchNorm1d(out_planes)
        self.head.bias.requires_grad_(False)
        self.classifier = nn.Linear(out_planes, num_classes, bias=False)

        self.reset_params()
        self.reset_alphas()

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        weights = F.softmax(self._arch_weights, dim=-1)
        for cell_idx, cell in enumerate(self.cells):
            if cell_idx % 3 == 2:
                x = cell(x)
            else:
                tmp_idx = cell_idx - cell_idx // 3
                x = cell(x, weights[tmp_idx * 2 : tmp_idx * 2 + 2])

        return x

    def forward(self, x):
        x = self.featuremaps(x)
        x = self.gap(x)
        x = x.view(x.shape[0], -1)

        if not self.training:
            return x

        bn_x = self.head(x)
        prob = self.classifier(bn_x)

        return x, prob

    def _loss(self, img, target):
        x, prob = self(img)
        return self._criterion(x, target)

    def reset_alphas(self):
        num_ops = len(PRIMITIVES)
        self._arch_weights = Variable(
            1e-3 * torch.randn((12, num_ops)).cuda(), requires_grad=True
        )
        self._arch_parameters = [self._arch_weights,]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        weights = self._arch_parameters[0]
        gene = []
        for i in range(12):
            w = weights[i]
            best = torch.argmax(w)
            gene.append(PRIMITIVES[best])

        return gene

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def build_search_model(criterion, num_classes=1000, pretrained=False):
    model = SearchCNN(
        num_classes,
        channels=[64, 256, 384, 512],
        criterion=criterion
    )
    return model

