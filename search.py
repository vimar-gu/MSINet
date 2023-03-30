import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
import os.path as osp
from torch.backends import cudnn

from reid.data import build_data
from reid.models.search_cnn import build_search_model
from reid.models.cm import ClusterMemory
from reid.criterion import build_criterion
from reid.solver import build_optimizer
from reid.engine.searcher import do_search
from reid.utils.logging import Logger


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    print('Running with:\n{}'.format(args))

    train_loader, valid_loader, test_loader, num_query,\
        num_train_classes, num_valid_classes, = build_data(args, search=True)
    train_memory = ClusterMemory(512, num_train_classes).cuda()
    valid_memory = ClusterMemory(512, num_valid_classes).cuda()
    model = build_search_model(valid_memory, num_train_classes)
    model = model.cuda()

    optimizer, lr_scheduler = build_optimizer(args, model)

    do_search(args, model, train_memory, train_loader, valid_loader,
              test_loader, optimizer, lr_scheduler, num_query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('-ds', '--source-dataset', type=str, default='msmt17')
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
    parser.add_argument('--loss', type=str, default='triplet_softmax')
    parser.add_argument('--triplet_margin', type=float, default=0.3)

    # optimizer
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--arch-lr', type=float, default=0.002)
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[150, 225, 300])
    parser.add_argument('--warmup-step', type=int, default=10)

    # training configs
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--geno-interval', type=int, default=5)
    parser.add_argument('--eval-interval', type=int, default=40)

    # misc
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--logs-dir', type=str, default='./logs')
    parser.add_argument('--pretrain-dir', type=str, default='./pretrained')

    args = parser.parse_args()
    main(args)

