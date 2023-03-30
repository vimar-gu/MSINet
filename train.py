import os
import sys
import torch
import random
import argparse
import numpy as np
import os.path as osp
from torch.backends import cudnn

from reid.utils.logging import Logger
from reid.data import build_data
from reid.criterion import build_criterion
from reid.solver import build_optimizer
from reid.engine import do_train, evaluate
from reid.models.msinet import msinet_x1_0
from reid.utils.serialization import copy_state_dict


def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if ('classifier' not in name)) / 1e6


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    print('Running with:\n{}'.format(args))

    train_loader, test_loader, num_query, num_classes = build_data(args)
    model = msinet_x1_0(args, num_classes)
    print('Model Params: {}'.format(count_parameters(model)))
    model = model.cuda()

    if args.resume != '':
        copy_state_dict(torch.load(args.resume), model)

    if args.evaluate:
        evaluate(args, model, test_loader, num_query)
        if args.target_dataset != 'none':
            _, tar_test_loader, tar_num_query, _ = build_data(args, target=True)
            evaluate(args, model, tar_test_loader, tar_num_query)
        return

    criterion = build_criterion(args, num_classes)
    optimizer, lr_scheduler = build_optimizer(args, model)

    do_train(args, model, criterion, train_loader, test_loader,
             optimizer, lr_scheduler, num_query)

    if args.target_dataset != 'none':
        _, tar_test_loader, tar_num_query, _ = build_data(args, target=True)
        evaluate(args, model, tar_test_loader, tar_num_query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('-ds', '--source-dataset', type=str, default='market1501')
    parser.add_argument('-dt', '--target-dataset', type=str, default='none')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--num-instance', type=int, default=4)

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--reset-params', type=bool, default=False)
    parser.add_argument('--genotypes', type=str, default='msmt')

    # loss
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--sam-mode', type=str, default='none')
    parser.add_argument('--sam-ratio', type=float, default=2.0)

    # optimizer
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.065)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--milestones', nargs='+', type=int, default=[150, 225, 300])
    parser.add_argument('--warmup-step', type=int, default=10)

    # training configs
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=40)

    # misc
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--logs-dir', type=str, default='./logs')
    parser.add_argument('--pretrain-dir', type=str, default='./pretrained')

    args = parser.parse_args()
    main(args)
