import torchvision.transforms as T
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np

from .market1501 import Market1501
from .msmt17 import MSMT17
from .vehicleid import VehicleID
from .veri import VeRi
from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler


__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'vehicleid': VehicleID,
    'veri': VeRi,
}


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length
        else:
            return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


def separate_trainval(train):
    pid2data_dict = defaultdict(list)
    for data in train:
        pid2data_dict[data[1]].append(data)
    num_pids = len(pid2data_dict.keys())
    val_pids = list(np.arange(num_pids // 5 * 1, num_pids))
    val_pid2label = {pid: idx for idx, pid in enumerate(val_pids)}
    new_train = []
    new_valid = []
    for pid in pid2data_dict.keys():
        if pid < num_pids // 5 * 1:
            new_train += pid2data_dict[pid]
        elif pid >= num_pids // 5 * 3:
            for data in pid2data_dict[pid]:
                new_valid.append((data[0], val_pid2label[pid], data[2]))
        else:
            data = pid2data_dict[pid]
            data_len = len(data)
            for idx in range(data_len // 2):
                new_train.append((data[idx][0], pid, data[idx][2]))
            for idx in range(data_len // 2, data_len):
                new_valid.append((data[idx][0], val_pid2label[pid], data[idx][2]))

    return new_train, new_valid, num_pids // 5 * 3, num_pids - num_pids // 5 * 1


def build_data(args, target=False, search=False):
    if target:
        data_name = args.target_dataset
    else:
        data_name = args.source_dataset

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if args.target_dataset != 'none':
        train_transforms = T.Compose([
            T.Resize((args.height, args.width)),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((args.height, args.width)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3,
                         hue=0),
            T.ToTensor(),
            normalizer
        ])
    else:
        train_transforms = T.Compose([
            T.Resize((args.height, args.width)),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((args.height, args.width)),
            T.ToTensor(),
            normalizer,
            RandomErasing(probability=0.5, sh=0.4,
                          mean=(0.4914, 0.4822, 0.4465))
        ])
    test_transforms = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        normalizer
    ])

    dataset = __factory[data_name](args.data_dir)

    num_workers = args.workers
    num_classes = dataset.num_train_pids

    testset = ImageDataset(dataset.query + dataset.gallery, test_transforms)
    test_loader = DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=num_workers
    )

    if not search:
        trainset = ImageDataset(dataset.train, train_transforms)
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, num_workers=num_workers,
            sampler=RandomIdentitySampler(dataset.train, args.batch_size, args.num_instance),
            drop_last=True
        )

        return train_loader, test_loader, len(dataset.query), num_classes
    else:
        all_train_data = dataset.train
        new_train, new_valid, num_train_classes, num_valid_classes =\
            separate_trainval(all_train_data)

        trainset = ImageDataset(new_train, train_transforms)
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, num_workers=num_workers,
            sampler=RandomIdentitySampler(
                new_train, args.batch_size, args.num_instance
            ), drop_last=True
        )
        validset = ImageDataset(new_valid, train_transforms)
        valid_loader = DataLoader(
            validset, batch_size=args.batch_size, num_workers=num_workers,
            sampler=RandomIdentitySampler(
                new_valid, args.batch_size, args.num_instance
            ), drop_last=True
        )
        valid_loader = IterLoader(valid_loader)
        return train_loader, valid_loader, test_loader, len(dataset.query),\
            num_train_classes, num_valid_classes

