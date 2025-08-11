import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from torch.autograd import Variable
import numpy as np

from sklearn.covariance import LedoitWolf, OAS, GraphicalLassoCV, GraphicalLasso

_tokenizer = _Tokenizer()
train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='backbone')
    parser.add_argument('--save_dir', type=str, default="RN50_caches")
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join(args.save_dir, cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    
    # Load cfg for conditional prompt.
    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
        
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots']) 
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    torch.save(clip_weights, "{}/clip_weights.pt".format(cfg['cache_dir']))

    # extract train features with augmentation
    load_train_features_aug(cfg, clip_model, train_loader_cache, 3)

    # extract val features
    pre_load_features(cfg, "val", clip_model, val_loader)
    
    
if __name__ == '__main__':
    main()
    