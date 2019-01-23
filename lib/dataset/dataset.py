from __future__ import print_function
import os.path as osp
from PIL import Image
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset as TorchDataset
from .transform import transform
from ..utils.file import load_pickle
class Dataset(TorchDataset):
    """Args:
        samples: None or a list of dicts; samples[i] has key 'im_path' and optional 'label', 'cam'.
    """

    im_root = None
    split_spec = None

    def __init__(self, cfg, samples=None):
        self.cfg = cfg
        self.root = osp.join(cfg.ROOT, cfg.NAME)#data/market1501
        if samples is None:
            self.samples = self.load_split()
        else:
            self.samples = samples
            cfg.split = 'None'
        print(self.summary)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cfg = self.cfg
        # Deepcopy to inherit all meta items
        #pdb.set_trace()
        sample = deepcopy(self.samples[index])
        im_path = sample['im_path']
        sample['im'] = self.get_im(im_path)
        transform(sample, cfg)
        return sample

    def save_split(self, spec, save_path):
        raise NotImplementedError

    def load_split(self):
        cfg = self.cfg
        save_path = osp.join(self.root, cfg.SPLIT + '.pkl')#data/market1501/train.pkl
        if not osp.exists(save_path):
            self.save_split(self.split_spec[cfg.SPLIT], save_path)
        samples = load_pickle(save_path)
        return samples

    def get_im(self, im_path):
        return Image.open(osp.join(self.root, im_path)).convert("RGB")

    # Use property (instead of setting it in self.__init__) in case self.samples is changed after initialization.
    @property
    def num_samples(self):
        return len(self.samples)

    @property
    def num_ids(self):
        return len(set([s['label'] for s in self.samples])) if 'label' in self.samples[0] else -1

    @property
    def num_cams(self):
        return len(set([s['cam'] for s in self.samples])) if 'cam' in self.samples[0] else -1

    @property
    def summary(self):
        summary = ['=' * 25]
        summary += [self.__class__.__name__]
        summary += ['=' * 25]
        summary += ['   split: {}'.format(self.cfg.SPLIT)]
        summary += ['# images: {}'.format(self.num_samples)]
        summary += ['   # ids: {}'.format(self.num_ids)]
        summary += ['  # cams: {}'.format(self.num_cams)]
        summary = '\n'.join(summary) + '\n'
        return summary
