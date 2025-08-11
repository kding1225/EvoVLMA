from tqdm import tqdm

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import json
import clip
import time

import logging
from logging import FileHandler

def set_logger(log_file):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
     
    file_handler = FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)d %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def clip_classifier_cupl(classnames, prompt_path, clip_model, template):
    f = open(prompt_path)
    prompts = json.load(f)
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')

            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = template_texts + cupl_texts

            texts_token = clip.tokenize(texts, truncate=True).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def build_cache_model(cfg, clip_model, train_loader_cache, seed):
    
    cache_keys_path = cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + f"shots_seed{seed}.pt"
    cache_values_path = cfg['cache_dir'] + '/values_' + str(cfg['shots']) + f"shots_seed{seed}.pt"
    
    if os.path.exists(cache_keys_path) and os.path.exists(cache_values_path):
        cache_keys = torch.load(cache_keys_path)
        cache_values = torch.load(cache_values_path)
    else:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cache_keys_path)
        torch.save(cache_values, cache_values_path)

    return cache_keys, cache_values

def build_cache_model_nomean(cfg, clip_model, loader, seed):

    feat_path = cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + f"shots_seed{seed}_nomean.pt"
    label_path = cfg['cache_dir'] + '/values_' + str(cfg['shots']) + f"shots_seed{seed}_nomean.pt"
    
    if os.path.exists(feat_path) and os.path.exists(label_path):
        features = torch.load(feat_path)
        labels = torch.load(label_path)
    else:
        features, labels = [], []

        with torch.no_grad():
            for _ in range(cfg['augment_epoch']):
                for i, (images, target) in enumerate(tqdm(loader)):
                    images, target = images.cuda(), target.cuda()
                    image_features = clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features)
                    labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, feat_path)
        torch.save(labels, label_path)
    
    return features, labels


def load_train_features_aug(cfg, clip_model, loader, aug=1):

    split = "train"
    feat_path = cfg['cache_dir'] + "/" + split + f"_f_aug{aug}.pt"
    label_path = cfg['cache_dir'] + "/" + split + f"_l_aug{aug}.pt"
    
    if os.path.exists(feat_path) and os.path.exists(label_path):
        features = torch.load(feat_path)
        labels = torch.load(label_path)
    else:
        features, labels = [], []

        with torch.no_grad():
            for idx in range(aug):
                for i, (images, target) in enumerate(tqdm(loader)):
                    images, target = images.cuda(), target.cuda()
                    image_features = clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features)
                    labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, feat_path)
        torch.save(labels, label_path)
    
    return features, labels


def pre_load_features(cfg, split, clip_model, loader, seed=None):

    if seed is None:
        feat_path = cfg['cache_dir'] + "/" + split + f"_f.pt"
        label_path = cfg['cache_dir'] + "/" + split + f"_l.pt"
    else:
        feat_path = cfg['cache_dir'] + "/" + split + f"_f_seed{seed}.pt"
        label_path = cfg['cache_dir'] + "/" + split + f"_l_seed{seed}.pt"
    
    if os.path.exists(feat_path) and os.path.exists(label_path):
        features = torch.load(feat_path)
        labels = torch.load(label_path)
    else:
        features, labels = [], []
        times = []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                st = time.time()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)
                ed = time.time()
                times.append(ed-st)

        features, labels = torch.cat(features), torch.cat(labels)
        times = np.array(times)
        print(times[10:-1].mean())

        torch.save(features, feat_path)
        torch.save(labels, label_path)
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha
