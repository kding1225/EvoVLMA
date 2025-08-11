import os
import random
import argparse
import yaml
from tqdm import tqdm
import sys
import math
import types
import warnings
import re

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
    parser.add_argument('--method', type=str, default="tip")
    parser.add_argument('--save_dir', type=str, default="RN50_caches")
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--feat_func', type=str, default="")
    parser.add_argument('--logit_func', type=str, default="")
    parser.add_argument('--dtype', type=str, default="float")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ub', type=float, default=100, help='upper bound')
    parser.add_argument('--ntrials', type=int, default=500)
    args = parser.parse_args()

    return args

def get_pop_size(file):
    with open(file, "r") as f:
        json_dict = json.load(f)
    return len(json_dict)

def get_feat_func(cfg, idx):
    print("use searched feature selection function: "+cfg["feat_func"])
    with open(cfg["feat_func"], "r") as f:
        json_dict = json.load(f)
    code_string = json_dict[idx]["code"]
    # import pdb; pdb.set_trace()
    # print(code_string)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Create a new module object
        module = types.ModuleType("module")

        # Execute the code string in the new module's namespace
        exec(code_string, module.__dict__)

        # Add the module to sys.modules so it can be imported
        sys.modules[module.__name__] = module

        feat_func = module.feat_selection
    return feat_func

def convert_code(code):
    
    my_pinv = "my_pinv = lambda x: raise_(Exception('foobar')) if torch.all(x==0) else torch.linalg.pinv(x.float()).half()\n"
    my_inv = "my_inv = lambda x: raise_(Exception('foobar')) if torch.all(x==0) else torch.inverse(x.float()).half()\n"
    my_inv2 = "my_inv2 = lambda x: raise_(Exception('foobar')) if torch.all(x==0) else torch.linalg.inv(x.float()).half()\n"
    my_svd = "my_svd = lambda x: torch.svd(x.float()).half()\n"
    my_eig = "my_eig = lambda x: torch.linalg.eig(x.float()).half()\n"
    my_eigvals = "my_eigvals = lambda x: torch.linalg.eigvals(x.float()).half()\n"
    my_zeros = "my_zeros = lambda x: torch.zeros(x, dtype=torch.half)\n"
    my_ones = "my_ones = lambda x: torch.ones(x, dtype=torch.half)\n"
    my_zeros_like = "my_zeros_like = lambda x: torch.zeros_like(x, dtype=torch.half)\n"
    my_ones_like = "my_ones_like = lambda x: torch.ones_like(x, dtype=torch.half)\n"
    my_eye = "my_eye = lambda x: torch.eye(x, dtype=torch.half)\n"
    my_onehot = "my_onehot = lambda x: F.one_hot(x).half()\n"
    #my_exp = "my_exp = lambda x: torch.exp(x).half()\n"
    my_cov = "my_cov = lambda x: torch.cov(x.float()).half()\n"
    
    code = code.replace("torch.linalg.pinv", "my_pinv")
    code = code.replace("torch.inverse", "my_inv")
    code = code.replace("torch.linalg.inv", "my_inv2")
    code = code.replace("torch.svd", "my_svd")
    code = code.replace("torch.linalg.eig", "my_eig")
    code = code.replace("torch.linalg.eigvals", "my_eigvals")
    code = code.replace("torch.zeros", "my_zeros")
    code = code.replace("torch.ones", "my_ones")
    code = code.replace("torch.eye", "my_eye")
    code = code.replace("F.one_hot", "my_onehot")
    #code = code.replace("torch.exp", "my_exp")
    #code = code.replace("exp()", "exp().half()")
    code = code.replace("torch.cov", "my_cov")
    code = code.replace(".cov()", ".float().cov()")
    
    code = code.split("def ")[0] + my_pinv+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_inv+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_inv2+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_svd+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_eig+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_eigvals+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_zeros+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_ones+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_zeros_like+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_ones_like+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_eye+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_onehot+"def "+"def ".join(code.split("def ")[1:])
    #code = code.split("def ")[0] + my_exp+"def "+"def ".join(code.split("def ")[1:])
    code = code.split("def ")[0] + my_cov+"def "+"def ".join(code.split("def ")[1:])

    raise_str = """
def raise_(ex):
    raise ex
"""
    code = code + raise_str

    return code

def get_logit_func(cfg, idx):
    # print("use searched logit function: "+cfg["logit_func"])
    with open(cfg["logit_func"], "r") as f:
        json_dict = json.load(f)
    print(cfg["logit_func"])
    print("len json_dict:", len(json_dict))
    code_string = json_dict[idx]["code"]

    if cfg["dtype"] == "half":
        code_string = convert_code(code_string)
        # print(code_string)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Create a new module object
        module = types.ModuleType("module")

        # Execute the code string in the new module's namespace
        exec(code_string, module.__dict__)

        # Add the module to sys.modules so it can be imported
        sys.modules[module.__name__] = module

        if "compute_logits_with_fs" in code_string:
            logit_func = module.compute_logits_with_fs
        else:
            logit_func = module.compute_logits
        
    return logit_func

def batched_logit_func(cfg, logit_func, cache_keys, cache_values, features, 
                       clip_weights, indices, alpha, beta, gamma):

    # logit_func = compute_logits

    if cfg["dataset"] not in ["imagenet", "sun397"]:
        if indices is None:
            return logit_func(cache_keys, cache_values, features, clip_weights, alpha, beta, gamma)
        else:
            return logit_func(cache_keys, cache_values, features, clip_weights, indices, alpha, beta, gamma)

    num = features.shape[0]
    bs = 4000
    num_batches = math.ceil(num/bs)
    all_logits = []
    for i in range(num_batches):
        if i%10 == 0:
            print(f"batch {i}/{num_batches}")
        st = i*bs
        ed = min((i+1)*bs, num)
        try:
            if indices is None:
                logits = logit_func(cache_keys, cache_values, features[st:ed], clip_weights, alpha, beta, gamma)
            else:
                logits = logit_func(cache_keys, cache_values, features[st:ed], clip_weights, indices, alpha, beta, gamma)
        except Exception as e:
            print("logit_func error: "+str(e))
            return
        all_logits.append(logits)
    return torch.cat(all_logits)

def objective(trial, cfg, cache_keys, cache_values, features, labels, clip_weights, indices, fs_pop_size, logit_pop_size):

    # pop_size = 10
    alpha0 = trial.suggest_float("alpha0", 1e-9, cfg["ub"], log=True)
    alpha1 = trial.suggest_float("alpha1", 1e-9, cfg["ub"], log=True)
    alpha2 = trial.suggest_float("alpha2", 1e-9, cfg["ub"], log=True)
    if indices is not None:
        idx1 = trial.suggest_categorical('idx1', list(range(fs_pop_size)))
        topk = trial.suggest_float("topk", 0.001, 1, log=False)
        topk = math.ceil(topk*len(indices[idx1]))
        indices = indices[idx1][:topk]

    idx2 = trial.suggest_categorical('idx2', list(range(logit_pop_size)))
    logit_func = get_logit_func(cfg, idx2)
    logits = batched_logit_func(cfg, logit_func, cache_keys, cache_values, features, 
                   clip_weights, indices, alpha0, alpha1, alpha2)
    if logits is None:
        return 0.0
    acc = cls_acc(logits, labels)

    return acc

def train_free(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print(test_labels.shape)
    print(test_features.shape)
    feat_dim, cate_num = clip_weights.shape
    logit_pop_size = get_pop_size(cfg["logit_func"])
    # logit_pop_size = 1

    fs_pop_size = None
    if cfg["feat_func"].strip() != "none":
        indices = []
        fs_pop_size = get_pop_size(cfg["feat_func"])
        print(cfg["feat_func"].strip())
        fs_pop_size = 5
        for i in range(fs_pop_size):
            print(i)
            idx = get_feat_func(cfg, i)(clip_weights.t(), cache_keys.reshape(cate_num, -1, feat_dim), 
                                     w0=0.5, w1=0.5, 
                                     topk=feat_dim)
            indices.append(idx)
    else:
        indices = None
        fs_pop_size = 1
        # indices = torch.arange(feat_dim, device=clip_weights.device)
    print("fs_pop_size:", fs_pop_size)
    print("logit_pop_size", logit_pop_size)
    
    obj = functools.partial(objective, cfg=cfg, 
                            cache_keys=cache_keys.view(cate_num, -1, feat_dim), 
                            cache_values=cache_values.view(cate_num, -1), 
                            features=val_features, labels=val_labels, 
                            clip_weights=clip_weights.t(), indices=indices,
                            fs_pop_size=fs_pop_size, logit_pop_size=logit_pop_size
                        )
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=cfg["ntrials"])
    print(study.best_trial.value, study.best_trial.params)
    params = study.best_trial.params
    best_alpha, best_beta, best_gamma = params["alpha0"], params["alpha1"], params["alpha2"]
    if indices is not None:
        idx1 = params["idx1"]
        topk = math.ceil(params["topk"]*len(indices[idx1]))
        indices = indices[idx1][:topk]

    idx2 = params["idx2"]
    logit_func = get_logit_func(cfg, idx2)
    logits = batched_logit_func(
        cfg, logit_func,
        cache_keys.view(cate_num, -1, feat_dim),
        cache_values.view(cate_num, -1),
        test_features, clip_weights.t(), indices,
        best_alpha, best_beta, best_gamma
    )
    
    acc = cls_acc(logits, test_labels)
    print("**** {}'s test accuracy: {:.2f}. ****\n".format(cfg['method'].upper(), acc))


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join(args.save_dir, cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['method'] = args.method
    cfg['shots'] = args.shots
    cfg['feat_func'] = args.feat_func
    cfg['logit_func'] = args.logit_func
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
    if cfg['method'] in ['tip', 'ape']:
        cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache, args.seed)
        cache_keys = cache_keys.t()
        cache_values = cache_values.argmax(dim=1)
    elif cfg['method'] == 'gda':
        d, c = clip_weights.shape
        cache_keys, cache_values = build_cache_model_nomean(cfg, clip_model, train_loader_cache, args.seed)
        indices = torch.cat([torch.nonzero(cache_values==i)[:, 0] for i in range(c)])
        cache_keys = cache_keys[indices]
        cache_values = cache_values[indices]
    else:
        raise
    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader, args.seed)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader, args.seed)

    if cfg["dtype"] == "half":
        cache_keys = cache_keys.half()
        val_features = val_features.half()
        test_features = test_features.half()
        clip_weights = clip_weights.half()
    else:
        cache_keys = cache_keys.float()
        val_features = val_features.float()
        test_features = test_features.float()
        clip_weights = clip_weights.float()
    
    train_free(cfg, cache_keys, cache_values, val_features, val_labels, 
               test_features, test_labels, clip_weights)

if __name__ == '__main__':
    main()
