from __future__ import print_function, absolute_import
import os.path as osp
import tarfile

import glob
import re
import urllib
import zipfile


style='MSMT17_V1'
def _pluck_msmt(list_file, subdir, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ret = []
    pids_ = []
    cams_ = []
    for line in lines:
        line = line.strip()
        fname = line.split(' ')[0]
        pid, _, cam = map(int, pattern.search(osp.basename(fname)).groups())
        if pid not in pids_:
            pids_.append(pid)
        if cam not in cams_:
            cams_.append(cam)

        img_path=osp.join(subdir,fname)
        ret.append((osp.join(subdir,fname), pid, cam))

    return ret, pids_, cams_


class Dataset_MSMT(object):
    def __init__(self, root):
        self.root = root
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, style)

    def load(self, verbose=True):
        exdir = osp.join(self.root, style)
        nametrain= osp.join(exdir, 'train')
        nametest = osp.join(exdir, 'test')
        self.train, train_pids, train_cams = _pluck_msmt(osp.join(exdir, 'list_train.txt'), nametrain)
        self.val, val_pids, val_cams = _pluck_msmt(osp.join(exdir, 'list_val.txt'), nametrain)
        self.train = self.train + self.val
        self.query, query_pids, query_cams = _pluck_msmt(osp.join(exdir, 'list_query.txt'), nametest)
        self.gallery, gallery_pids, gallery_cams = _pluck_msmt(osp.join(exdir, 'list_gallery.txt'), nametest)
        self.num_train_pids = len(list(set(train_pids).union(set(val_pids))))
        self.num_train_cams = len(list(set(train_cams).union(set(val_cams))))

        if verbose:
            print(self.__class__.__name__, "v1~~~ dataset loaded")
            print("  ---------------------------------------")
            print("  subset   | # ids | # images | # cams")
            print("  ---------------------------------------")
            print("  train    | {:5d} | {:8d} | {:5d}"
                  .format(self.num_train_pids, len(self.train), self.num_train_cams))
            print("  query    | {:5d} | {:8d} | {:5d}"
                  .format(len(query_pids), len(self.query), len(query_cams)))
            print("  gallery  | {:5d} | {:8d} | {:5d}"
                  .format(len(gallery_pids), len(self.gallery), len(gallery_cams)))
            print("  ---------------------------------------")


class MSMT17(Dataset_MSMT):

    def __init__(self, data_dir, split_id=0, download=False):
        super(MSMT17, self).__init__(data_dir)

        if download:
            self.download()

        self.load()

    def download(self):

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root)
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, style)
        if osp.isdir(fpath):
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually to {}".format(fpath))
