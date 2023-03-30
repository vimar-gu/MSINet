import torch
from .lr_scheduler import WarmupMultiStepLR


def build_optimizer(args, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if 'criterion' in key:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay, "momentum": args.momentum}]
    optimizer = torch.optim.SGD(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    return optimizer, lr_scheduler
