import os
import torch
import os.path as osp

from ..utils.meters import AverageMeter
from ..utils.metrics import R1_mAP


def evaluate(args, model, test_loader, num_query):
    """Standard Re-ID evaluating engine."""
    print_freq = args.print_freq
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=True)
    evaluator.reset()

    model.eval()
    for n_iter, (img, pid, camid) in enumerate(test_loader):
        with torch.no_grad():
            img = img.cuda()
            feat = model(img)
            evaluator.update((feat, pid, camid))

        if n_iter % print_freq == 0:
            print('Evaluating: [{}/{}]'.format(n_iter, len(test_loader)))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    print('Validation Results')
    print('mAP: {:.1%}'.format(mAP))
    for r in [1, 5, 10]:
        print('CMC curve, Rank-{:<3}:{:.1%}'.format(r, cmc[r - 1]))

