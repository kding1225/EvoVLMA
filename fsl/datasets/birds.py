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


template = ['a photo of a {}, a type of bird.']


class Birds(DatasetBase):

    dataset_dir = 'UCSDBirds'

    def __init__(self, root, shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'CUB_200_2011/images')
        self.list_file = os.path.join(self.dataset_dir, 'CUB_200_2011/images.txt')
        self.split_file = os.path.join(self.dataset_dir, 'CUB_200_2011/train_test_split.txt')
        self.class_file = os.path.join(self.dataset_dir, 'CUB_200_2011/classes.txt')

        with open(self.list_file, "r") as f:
            image_list = [x.strip() for x in f]

        with open(self.split_file, "r") as f:
            split_flag = [int(x.strip().split()[1]) for x in f]
        assert len(image_list) == len(split_flag)

        with open(self.class_file, "r") as f:
            class_names = [x.strip() for x in f]
        class_names = [x.split('.')[1] for x in class_names]
        self.class_names = class_names

        train_list = [x for x,y in zip(image_list, split_flag) if y]
        test_list = [x for x,y in zip(image_list, split_flag) if not y]
        
        self.template = template
        train = self.read_data(train_list)
        test = self.read_data(test_list)
        val = copy.deepcopy(test)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, file_list):
        
        def _collate(ims, y, c):
            items = []
            for im in ims:
                  # is already 0-based
                items.append(item)
            return items

        data = []
        for i, path in enumerate(file_list):
            img_path = os.path.join(self.image_dir, path.split()[1])
            category = path.split()[1].split("/")[0].split('.')[1]
            y = self.class_names.index(category)
            item = Datum(impath=img_path, label=y, classname=category)
            data.append(item)

        return data