import torch
from torch import nn
from torch.nn import functional as F


OPS = {
    'none': lambda C_in, C_out: Identity(),
    'exc': lambda C_in, C_out: Exchange(),
    'ag': lambda C_in, C_out: ChannelGate(C_in, C_out),
    'cross_att': lambda C_in, C_out: CrossAttention(C_in, C_out)
}


genotype_factory = {
    'msmt': (['ag', 'ag', 'exc', 'ag',
              'cross_att', 'ag', 'ag', 'none',
              'ag', 'cross_att', 'exc', 'cross_att'],
             'msinet_msmt.pth.tar'),
}


##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, groups=1, IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride,
            padding=0, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride,
            padding=1, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1,
            bias=False, groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Blocks for Multi-scale Feature Interaction between Two Branches
##########
class Identity(nn.Module):
    """Return the original features."""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Exchange(nn.Module):
    """Directly exchange the features of two branches."""
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x):
        x1, x2 = x
        return x2, x1


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self, in_channels, num_gates=None, return_gates=False,
        gate_activation='sigmoid', reduction=16, layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1,
            bias=True, padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction, num_gates, kernel_size=1,
            bias=True, padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, xs):
        out = []
        for x in xs:
            input = x
            x = self.global_avgpool(x)
            x = self.fc1(x)
            if self.norm1 is not None:
                x = self.norm1(x)
            x = self.relu(x)
            x = self.fc2(x)
            if self.gate_activation is not None:
                x = self.gate_activation(x)
            out.append(input * x)
        return out


class CrossAttention(nn.Module):
    """Exchange the key feature to calculate the correlation for two branches."""
    def __init__(self, in_channels, out_channels):
        super(CrossAttention, self).__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        xa, xb = x
        m_bs, C, height, width = xa.size()

        querya = xa.view(m_bs, C, -1)
        keya = xa.view(m_bs, C, -1).permute(0, 2, 1)

        queryb = xb.view(m_bs, C, -1)
        keyb = xb.view(m_bs, C, -1).permute(0, 2, 1)

        energya = torch.bmm(querya, keyb)
        energyb = torch.bmm(queryb, keya)

        def get_output(energy, xin):
            max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
            energy_new = max_energy_0 - energy
            attention = F.softmax(energy_new, dim=-1)
            proj_value = xin.view(m_bs, C, -1)

            out = torch.bmm(attention, proj_value)
            out = out.view(m_bs, C, height, width)

            gamma = self.gamma.to(out.device)
            out = gamma * out + xin
            return out

        return get_output(energya, xa), get_output(energyb, xb)
