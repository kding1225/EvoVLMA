import os
import random
import argparse
import yaml
from tqdm import tqdm
import sys
import types
import warnings
import time

import math
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

def cal_criterion(clip_weights, cache_keys, w0, w1, topk, training_free=True):
    # clip_weights: c*d
    # cache_keys: c*k*d
    
    cate_num, feat_dim = clip_weights.shape
    text_feat = clip_weights.unsqueeze(1)
    cache_feat = cache_keys
    
    print('Calculating criterion...')
    feats = torch.cat([text_feat, cache_feat], dim=1)
    samp_num = feats.shape[1]
    sim_sum = torch.zeros((feat_dim)).cuda()
    count = 0
    for i in range(cate_num):
        for j in range(cate_num):
            if i != j:
                sim_sum += (feats[i].unsqueeze(1) * feats[j].unsqueeze(0)).mean(dim=0).mean(dim=0)
                count += samp_num**2
    
    sim = sim_sum / count
    criterion = (-1) * w0 * sim + w1 * torch.var(clip_weights, dim=0)
    # _, indices = torch.topk(criterion, k=topk)
    indices = torch.argsort(criterion, descending=True)
    
    return indices

def objective(trial, cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, indices):

    alpha = trial.suggest_float("alpha", 1e-9, cfg["ub"], log=True)
    beta = trial.suggest_float("beta", 1e-9, cfg["ub"], log=True)
    gamma = trial.suggest_float("gamma", 1e-9, cfg["ub"], log=True)
    if indices is not None:
        topk = trial.suggest_float("topk", 0.001, 1, log=False)
        topk = math.ceil(topk*len(indices))
        indices = indices[:topk]
    
    new_clip_weights = clip_weights[indices, :]
    new_cache_keys = cache_keys[:, indices]
    new_val_features = val_features[:, indices]
    
    new_clip_weights = new_clip_weights / new_clip_weights.norm(dim=0, keepdim=True)
    new_cache_keys = new_cache_keys /  new_cache_keys.norm(dim=-1, keepdim=True)
    new_val_features = new_val_features /  new_val_features.norm(dim=-1, keepdim=True)

    key_logits = new_cache_keys @ new_clip_weights
    key_logits = key_logits.softmax(1)
    cache_div = torch.sum(cache_values * torch.log2((cache_values + 1e-6) / (key_logits + 1e-6)), dim=1)[:, None]
    R_FW = (cache_div * gamma).exp()
    soft_cache_values = cache_values * R_FW
    
    R_fF = new_val_features @ new_cache_keys.t()
    R_fW = 100. * val_features @ clip_weights
    with torch.no_grad():
        soft_cache_values = cache_values * (cache_div * gamma).exp()                    
        cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ soft_cache_values
        ape_logits = R_fW + cache_logits * alpha
    acc = cls_acc(ape_logits, val_labels)

    return acc

def APE(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print(test_labels.shape)
    print(test_features.shape)
    feat_dim, cate_num = clip_weights.shape
    indices = cal_criterion(clip_weights.t(), cache_keys.reshape(cate_num, cfg['shots'], feat_dim), w0=cfg['w_training_free'][0], w1=cfg['w_training_free'][1], topk=cfg['training_free_feat_num'])

    obj = functools.partial(objective, cfg=cfg, cache_keys=cache_keys, 
                            cache_values=cache_values, val_features=val_features,
                            val_labels=val_labels, clip_weights=clip_weights,
                            indices=indices
                        )
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=cfg["ntrials"])
    print(study.best_trial.value, study.best_trial.params)
    params = study.best_trial.params
    topk = math.ceil(params["topk"]*len(indices))
    indices = indices[:topk]

    new_clip_weights = clip_weights[indices, :]
    new_cache_keys = cache_keys[:, indices]
    new_test_features = test_features[:, indices]
    new_val_features = val_features[:, indices]
    new_clip_weights = new_clip_weights / new_clip_weights.norm(dim=0, keepdim=True)
    new_cache_keys = new_cache_keys /  new_cache_keys.norm(dim=-1, keepdim=True)
    new_test_features = new_test_features /  new_test_features.norm(dim=-1, keepdim=True)
    new_val_features = new_val_features /  new_val_features.norm(dim=-1, keepdim=True)
    
    key_logits = new_cache_keys @ new_clip_weights
    key_logits = key_logits.softmax(1)
    cache_div = torch.sum(cache_values * torch.log2((cache_values + 1e-6) / (key_logits + 1e-6)), dim=1)[:, None]

    time1 = time.time()
    
    R_fW = 100. * test_features @ clip_weights
    R_fF = new_test_features @ new_cache_keys.t()
    soft_cache_values = cache_values * (cache_div * params["gamma"]).exp()
    cache_logits = ((-1) * (params["beta"] - params["beta"] * R_fF)).exp() @ soft_cache_values
    
    ape_logits = R_fW + cache_logits * params["alpha"]

    time2 = time.time()

    print("used time: "+str(time2-time1)+"s, test shape: ", test_features.shape)
    
    acc = cls_acc(ape_logits, test_labels)
    print("**** APE's test accuracy: {:.2f}. ****\n".format(acc))


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
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

        train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)
        train_loader_F = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=True)
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

        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
        train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache, args.seed)
    cache_values = cache_values.t()

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader, args.seed)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader, args.seed)
    
    cache_keys = cache_keys.t()
    cache_values = cache_values.t()

    if args.dtype == "half":
        APE(cfg, cache_keys.half(), cache_values.half(), val_features.half(), val_labels, test_features.half(), test_labels, clip_weights.half())
    else:
        APE(cfg, cache_keys.float(), cache_values.float(), val_features.float(), val_labels, test_features.float(), test_labels, clip_weights.float())

if __name__ == '__main__':
    main()
