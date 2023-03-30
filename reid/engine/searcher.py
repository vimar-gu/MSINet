import os
import math
import torch
import os.path as osp
from torch.nn import functional as F

from ..utils.meters import AverageMeter
from ..utils.metrics import R1_mAP


class Architect(object):
    """Architecture parameter maintenance and update"""
    def __init__(self, args, model):
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_lr, betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[150, 225, 350]
        )

    def step(self, input_valid, target_valid):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def step_scheduler(self):
        self.lr_scheduler.step()


def do_search(args, model, criterion, train_loader, valid_loader, test_loader,
              optimizer, lr_scheduler, num_query):
    """The engine for searching Re-ID architectures."""
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    print_freq = args.print_freq
    geno_interval = args.geno_interval
    eval_interval = args.eval_interval

    architect = Architect(args, model)

    for epoch in range(args.epochs):
        loss_meter.reset()
        acc_meter.reset()

        if (epoch + 1) % geno_interval == 0 or (epoch + 1) == args.epochs:
            genotype = model.genotype()
            print('genotype = {}'.format(genotype))
            print(F.softmax(model.arch_parameters()[0], dim=-1))

        model.train()
        for n_iter, (img, target, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.cuda()
            target = target.cuda()
            (img_s, target_s, _) = valid_loader.next()
            img_s = img_s.cuda()
            target_s = target_s.cuda()

            if n_iter > 0:
                # Update the architecture parameter.
                architect.step(img_s, target_s)

            feats, logits = model(img)
            loss = criterion(feats, target)

            # Update the model parameter.
            loss.backward()
            optimizer.step()

            acc = (logits.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if n_iter % print_freq == 0:
                print('Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Lr: {:.2e}'
                      .format(epoch, n_iter, len(train_loader), loss_meter.avg,
                              acc_meter.avg, lr_scheduler.get_last_lr()[0]))

        lr_scheduler.step()
        architect.step_scheduler()

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == args.epochs:
            torch.save(model.state_dict(), osp.join(args.logs_dir, 'model_{}.pth'.format(epoch)))
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
            print('Validation Results - Epoch[{}]'.format(epoch))
            print('mAP: {:.1%}'.format(mAP))
            for r in [1, 5, 10]:
                print('CMC curve, Rank-{:<3}:{:.1%}'.format(r, cmc[r - 1]))
            del evaluator
