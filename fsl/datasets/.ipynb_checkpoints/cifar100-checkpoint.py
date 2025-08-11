import os
import math
import random
import copy
from collections import defaultdict

import torchvision.transforms as transforms
from .utils import Datum, DatasetBase, build_data_loader

def listdir_nohidden(path):
    p = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            p.append(f)
    return p


template = ['a photo of a {}.']


class CIFAR100(DatasetBase):

    dataset_dir = 'cifar100'

    def __init__(self, root, shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'cifar100png')

        self.template = template
        train = self.read_data(os.path.join(self.image_dir, "train"))
        test = self.read_data(os.path.join(self.image_dir, "test"))
        val = copy.deepcopy(test)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, image_dir, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        data = []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]
            data.extend(_collate(images, label, category))

        return data