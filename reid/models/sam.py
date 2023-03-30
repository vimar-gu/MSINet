import torch
import torch.nn as nn


class AlignModule(nn.Module):
    def __init__(self, input_height, input_width, in_planes):
        super(AlignModule, self).__init__()
        self.qaconv = QAConv(in_planes, input_height, input_width)
        self.pos_pam = PAM_Module(in_planes)

    def forward(self, x):
        self.qaconv.make_kernel(x)
        kernel_score = self.qaconv(x)
        pos_score = self.pos_pam(x)
        scores = torch.cat(
            (kernel_score.max(dim=3)[0],
             pos_score.unsqueeze(0).max(dim=3)[0]),
            dim=0
        )

        return scores


class QAConv(nn.Module):
    """Un-parametric correlation calculation"""
    def __init__(self, num_features, height, width):
        super(QAConv, self).__init__()
        self.num_features = num_features
        self.height = height
        self.width = width

    def make_kernel(self, features):
        self.kernel = features

    def forward(self, features):
        hw = self.height * self.width
        batch_size = features.shape[0]
        score = torch.einsum('g c h w, p c y x -> g p y x h w', features, self.kernel)
        score = score.view(batch_size, -1, hw, hw)

        return score


class PAM_Module(nn.Module):
    """Position Attention Module."""
    def __init__(self, in_planes):
        super(PAM_Module, self).__init__()
        self.in_planes = in_planes

        self.query_conv = nn.Conv2d(
            in_channels=in_planes, out_channels=in_planes // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_planes, out_channels=in_planes // 8, kernel_size=1
        )

    def forward(self, x):
        batch_size, C, height, width = x.shape
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)

        return energy
