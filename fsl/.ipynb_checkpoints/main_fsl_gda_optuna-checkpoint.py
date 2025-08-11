import os
import random
import argparse
import yaml
from tqdm import tqdm
import sys
import types
import warnings
import time

import torch
import json
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from datasets.imagenet import ImageNet

import functools
import optunahub, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--save_dir', type=str, default="RN50_caches")
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ub', type=float, default=100, help='upper bound')
    parser.add_argument('--ntrials', type=int, default=500)
    parser.add_argument('--dtype', type=str, default="float")
    args = parser.parse_args()

    return args

def fit_gda_model(train_feats, train_labels, num_classes):
    
    # compute per-class mean features
    mus = torch.cat([train_feats[train_labels == i].mean(dim=0, keepdim=True) for i in range(num_classes)])

    # use KS Estimator to estimate inverted covariance matrix
    center_vecs = torch.cat([train_feats[train_labels == i] - mus[i].unsqueeze(0) for i in range(num_classes)])
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda())   
    cov_inv = cov_inv.type(train_feats.dtype)

    ps = torch.ones(num_classes).cuda() * 1. / num_classes
    W = mus @ cov_inv
    b = ps.log() - (W*mus).sum(dim=1) / 2
    
    return W.t(), b

def objective(trial, cfg, R_fW, features, labels, W, b):
    alpha = trial.suggest_float("alpha", 1e-9, cfg["ub"], log=True)
    with torch.no_grad():
        logits = R_fW + alpha * (features @ W + b)
    acc = cls_acc(logits, labels)
    return acc

def GDA(cfg, train_feats, train_labels, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print(test_labels.shape)
    print(test_features.shape)
    feat_dim, cate_num = clip_weights.shape
    
    # Zero-shot CLIP
    R_fW = 100. * test_features @ clip_weights
    acc = cls_acc(R_fW, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))
    
    alpha = 0.1
    W, b = fit_gda_model(train_feats, train_labels, cate_num)
    logits = R_fW + alpha * (test_features @ W + b)
    acc = cls_acc(logits, test_labels)
    print("**** Before search, test accuracy: {:.2f}. ****\n".format(acc))

    obj = functools.partial(objective, cfg=cfg, 
                            R_fW=100. * val_features @ clip_weights, features=val_features, 
                            labels=val_labels, W=W, b=b
                        )
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=cfg["ntrials"])
    print(study.best_trial.value, study.best_trial.params)
    params = study.best_trial.params
    best_alpha = params["alpha"]

    time1 = time.time()
    R_fW = 100. * test_features @ clip_weights
    logits = R_fW + best_alpha * (test_features @ W + b)
    time2 = time.time()
    
    print("used time: "+str(time2-time1)+"s, test shape: ", test_features.shape)
    
    acc = cls_acc(logits, test_labels)
    print("**** GDA's test accuracy: {:.2f}. ****\n".format(acc))


def main():

    # Load config file
    args = get_arguments()
    assert os.path.exists(args.config)
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join(args.save_dir, cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['shots'] = args.shots
    cfg['ntrials'] = args.ntrials
    cfg['ub'] = args.ub
    cfg['dtype'] = args.dtype
    
    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Preparing dataset.")
    if cfg['dataset'] == "imagenet":
        print("Preparing ImageNet dataset.")
        dataset = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
        val_loader = test_loader

        train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)
    else:
        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load train features
    print("\nLoading visual features and labels from train set.")
    train_features, train_labels = build_cache_model_nomean(cfg, clip_model, train_loader, args.seed)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader, args.seed)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader, args.seed)

    if args.dtype == "half":
        GDA(cfg, train_features.half(), train_labels, val_features.half(), val_labels, 
            test_features.half(), test_labels, clip_weights.half())
    else:
        GDA(cfg, train_features.float(), train_labels, val_features.float(), val_labels, 
            test_features.float(), test_labels, clip_weights.float())

if __name__ == '__main__':
    main()
