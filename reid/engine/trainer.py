import os
import torch
import os.path as osp

from ..utils.meters import AverageMeter
from ..utils.metrics import R1_mAP


def do_train(args, model, criterion, train_loader, test_loader,
             optimizer, lr_scheduler, num_query):
    """Standard Re-ID training engine."""
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    print_freq = args.print_freq
    eval_interval = args.eval_interval

    for epoch in range(args.epochs):
        loss_meter.reset()
        acc_meter.reset()

        model.train()
        for n_iter, (img, pid, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.cuda()
            target = pid.cuda()

            feats, logits, f_feats, f_logits, sam_logits = model(img)
            loss = criterion(feats, logits, sam_logits, target, sam=True) + criterion(f_feats, f_logits, sam_logits, target)

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

