import torch
import torch.nn as nn


class SpatialAlignLoss(nn.Module):
    """The spatial alignment loss for cross-domain Re-ID.
    """
    def __init__(self, mode='pos'):
        super(SpatialAlignLoss, self).__init__()
        self.mode = mode

    def forward(self, sam_logits, labels):
        unsup_corrs = sam_logits[:-1]
        pos_corrs = sam_logits[-1].unsqueeze(1)

        N = unsup_corrs.shape[0]
        M = unsup_corrs.shape[-1]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        # Calculate un-parametric self-attention activation.
        unsup_pos = unsup_corrs[is_pos].contiguous().view(N, -1, M)
        unsup_neg = unsup_corrs[is_neg].contiguous().view(N, -1, M)

        def batch_cosine_dist(x, y):
            bs = x.shape[0]
            bs1, bs2 = x.shape[1], y.shape[1]
            frac_up = torch.bmm(x, y.transpose(1, 2))
            frac_down1 = torch.sqrt(torch.sum(torch.pow(x, 2), dim=2)).view(bs, bs1, 1).repeat(1, 1, bs2)
            frac_down2 = torch.sqrt(torch.sum(torch.pow(y, 2), dim=2)).view(bs, 1, bs2).repeat(1, bs1, 1)

            return 1 - frac_up / (frac_down1 * frac_down2)

        all_losses = 0
        # Align the positive attention.
        if 'pos' in self.mode:
            pos_dist = batch_cosine_dist(pos_corrs, unsup_pos)
            all_losses += pos_dist.mean()
        # Align the negative attention.
        if 'neg' in self.mode:
            neg_dist = batch_cosine_dist(unsup_neg, unsup_neg)
            all_losses += neg_dist.mean()

        return all_losses
