import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import random

import logging
logger = logging.getLogger("my_logger")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class GetData():
    def __init__(self, train_feats_path, train_labels_path, test_feats_path, test_labels_path, 
                 clip_weights_path, shots, inst_per_shots=2, sample_class=False, ntest_per_class=None, instance_cache_path=""):

        logger.info("load data from:")
        logger.info(f" * {train_feats_path}")
        logger.info(f" * {train_labels_path}")
        logger.info(f" * {test_feats_path}")
        logger.info(f" * {test_labels_path}")
        logger.info(f" * {clip_weights_path}")
        
        self.train_feats = [torch.load(x, map_location=torch.device('cuda')) for x in train_feats_path] # m*d
        self.train_labels = [torch.load(x, map_location=torch.device('cuda')) for x in train_labels_path] # m
        self.test_feats = [torch.load(x, map_location=torch.device('cuda')) for x in test_feats_path] # n*d
        self.test_labels = [torch.load(x, map_location=torch.device('cuda')) for x in test_labels_path] # n
        self.clip_weights = [torch.load(x, map_location=torch.device('cuda')) for x in clip_weights_path] # d*c

        logger.info("train_feats: "+", ".join([str(x.shape) for x in self.train_feats]))
        logger.info("train_labels: "+", ".join([str(x.shape) for x in self.train_labels]))
        logger.info("test_feats: "+", ".join([str(x.shape) for x in self.test_feats]))
        logger.info("test_labels: "+", ".join([str(x.shape) for x in self.test_labels]))
        logger.info("clip_weights: "+", ".join([str(x.shape) for x in self.clip_weights]))
        
        self.shots = shots
        self.inst_per_shots = inst_per_shots
        self.sample_class = sample_class
        self.num_classes = [torch.max(x).item() + 1 for x in self.test_labels]
        self.num_tasks = len(self.num_classes)
        self.ntest_per_class = ntest_per_class
        self.instance_cache_path = instance_cache_path

        os.makedirs(instance_cache_path, exist_ok=True)
        
    def subset_samples(self, shots, taskid):
        if self.sample_class:
            num_classes = torch.randint(int(0.1*self.num_classes[taskid]), self.num_classes[taskid], (1,)).item()
        else:
            num_classes = self.num_classes[taskid]
        cls_indices = torch.randperm(self.num_classes[taskid])[:num_classes]
        train_indices = []
        test_indices = []
        train_labels = []
        test_labels = []
        for j, i in enumerate(cls_indices):
            mask = self.train_labels[taskid] == i
            idx = torch.nonzero(mask)[:, 0]
            indices = torch.randperm(len(idx))[:shots]
            train_indices.append(idx[indices])
            train_labels.extend([j]*shots)

            mask = self.test_labels[taskid] == i
            idx = torch.nonzero(mask)[:, 0]
            if self.ntest_per_class is not None and self.ntest_per_class>0:
                idx = idx[:self.ntest_per_class]
            test_indices.append(idx)
            test_labels.extend([j]*len(idx))
        
        train_indices = torch.cat(train_indices).to(self.train_feats[0].device)
        test_indices = torch.cat(test_indices).to(self.train_feats[0].device)
        train_labels = torch.tensor(train_labels).to(self.train_feats[0].device)
        test_labels = torch.tensor(test_labels).to(self.train_feats[0].device)
        
        return train_indices, test_indices, cls_indices, train_labels, test_labels
        
    def generate_instances(self, taskid, seed=123):
        set_seed(seed)
        instance_data = []
        for shot in self.shots:
            for _ in range(self.inst_per_shots):
                train_idx, test_idx, cls_idx, train_label, test_label = self.subset_samples(shot, taskid)
                instance_data.append((train_idx, test_idx, cls_idx, train_label, test_label))
        return instance_data

    def make_fs_task(self, taskid, train_idx, test_idx, cls_idx, train_labels, test_labels):
        # out:
        # train_feats: m*d
        # test_feats: n*d
        # train_labels: m*c
        # clip_weights: d*c
        
        train_feats = self.train_feats[taskid][train_idx].clone()
        test_feats = self.test_feats[taskid][test_idx].clone()
        clip_weights = self.clip_weights[taskid][:, cls_idx].clone()

        return train_feats.half(), train_labels, test_feats.half(), test_labels, clip_weights.half()

    def generate_instances_mt(self):
        return [self.generate_instances(i) for i in range(self.num_tasks)]

    def generate_and_save_instances(self):
        
        instance_indices = self.generate_instances_mt()

        train_feats = None
        paths = []
        num = 0
        for taskid, task_data in enumerate(instance_indices):
            for inst_idx, (train_idx, test_idx, cls_idx, train_label, test_label) in enumerate(task_data):
                train_feats, train_labels, test_feats, test_labels, clip_weights = \
                    self.make_fs_task(taskid, train_idx, test_idx, cls_idx, train_label, test_label)

                data = {
                    "train_feats": train_feats.view(clip_weights.size(1), -1, train_feats.shape[-1]),
                    "train_labels": train_labels.reshape(clip_weights.size(1), -1),
                    "test_feats": test_feats,
                    "test_labels": test_labels,
                    "clip_weights": clip_weights.t(),
                    "num": num,
                    "taskid": taskid,
                    "num_classes": len(cls_idx)
                }
                if os.path.exists(os.path.join(self.instance_cache_path, f"{num}.pth")):
                    print(os.path.join(self.instance_cache_path, f"{num}.pth")+" exists")
                else:
                    torch.save(data, os.path.join(self.instance_cache_path, f"{num}.pth"))
                paths.append(os.path.join(self.instance_cache_path, f"{num}.pth"))
                num += 1

        return paths, train_feats.shape[-1]