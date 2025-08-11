from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import functools
import traceback
import sys
import types
import warnings
import math
import time
import numpy as np
import inspect
import gc
import concurrent.futures

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from torch.cuda.memory import set_per_process_memory_fraction
set_per_process_memory_fraction(0.5)

app = Flask(__name__)
pool_size = 4
dtype = torch.half

def convert_dtype(data, dtype):

    if dtype == torch.half:
        data["train_feats"] = data["train_feats"].half()
        data["test_feats"] = data["test_feats"].half()
        data["clip_weights"] = data["clip_weights"].half()
    else:
        data["train_feats"] = data["train_feats"].float()
        data["test_feats"] = data["test_feats"].float()
        data["clip_weights"] = data["clip_weights"].float()
    return data

def collect_mem():
    gc.collect()
    torch.cuda.empty_cache()

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc, pred

def convert_code(code):

    my_pinv = "my_pinv = lambda x: torch.linalg.pinv(x.float()).half()\n"
    my_inv = "my_inv = lambda x: torch.inverse(x.float()).half()\n"
    my_inv2 = "my_inv2 = lambda x: torch.linalg.inv(x.float()).half()\n"
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
    #my_cov = "my_cov = lambda x: torch.cov(x.float()).half()\n"
    
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
    #code = code.replace("torch.cov", "my_cov")
    #code = code.replace(".cov()", ".float().cov()")

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
    #code = code.split("def ")[0] + my_cov+"def "+"def ".join(code.split("def ")[1:])

    return code

def get_feat_selection_module(code_string, dtype):

    if dtype == torch.half:
        code_string = convert_code(code_string)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Create a new module object
        module = types.ModuleType("module")

        # Execute the code string in the new module's namespace
        exec(code_string, module.__dict__)

        # Add the module to sys.modules so it can be imported
        sys.modules[module.__name__] = module

        feat_selection = module.feat_selection
    return feat_selection

def get_logit_compute_module(code_string, dtype):

    if dtype == torch.half:
        code_string = convert_code(code_string)
    
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


def compute_acc_batched(compute_logits, train_feats, train_labels, test_feats, test_labels, clip_weights, 
                       indices, alphas):

    # args = inspect.getfullargspec(compute_logits)
    
    num = test_feats.shape[0]
    bs = 5000
    num_batches = math.ceil(num/bs)
    all_logits = []
    alpha0, alpha1, alpha2 = alphas
    for i in range(num_batches):
        st = i*bs
        ed = min((i+1)*bs, num)
        if indices is None:
            logits = compute_logits(
                train_feats, 
                train_labels, 
                test_feats[st:ed], clip_weights, 
                alpha0, alpha1, alpha2
            )
        else:
            logits = compute_logits(
                train_feats, 
                train_labels, 
                test_feats[st:ed], clip_weights, indices, 
                alpha0, alpha1, alpha2
            )
        all_logits.append(logits)
    return cls_acc(torch.cat(all_logits), test_labels)

def compute_searched_acc(compute_logits, indices, cache_keys, cache_values, val_features, val_labels, 
                         clip_weights, params):

    results = [None for _ in params]
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        future_to_index = {executor.submit(compute_acc_batched, compute_logits, cache_keys, 
                             cache_values, val_features, val_labels, clip_weights, indices, params[i]): i for i in range(len(params))}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    # task = functools.partial(compute_acc_batched, compute_logits, cache_keys, 
    #                          cache_values, val_features, val_labels, clip_weights, indices)
    #results = [task(x) for x in params]
    
    idx = np.argmax([r[0] for r in results])
    acc = results[idx][0]
    pred = results[idx][1]
    return acc, pred


def compute_searched_acc_optuna(compute_logits, indices, cache_keys, cache_values, val_features, val_labels, 
                         clip_weights, params, ub=20.0, ntrials=10, n_jobs=1):

    def objective(trial):
        alpha0 = trial.suggest_float("alpha0", 1e-9, ub)
        alpha1 = trial.suggest_float("alpha1", 1e-9, ub)
        alpha2 = trial.suggest_float("alpha2", 1e-9, ub)
        acc, pred = compute_acc_batched(compute_logits, cache_keys, 
                             cache_values, val_features, val_labels, clip_weights, indices, (alpha0, alpha1, alpha2))
        return acc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=ntrials, n_jobs=n_jobs)
    acc = study.best_trial.value
    print(study.best_trial.value, study.best_trial.params)

    alpha0 = study.best_trial.params["alpha0"]
    alpha1 = study.best_trial.params["alpha1"]
    alpha2 = study.best_trial.params["alpha2"]
    acc, pred = compute_acc_batched(compute_logits, cache_keys, 
                             cache_values, val_features, val_labels, clip_weights, indices, (alpha0, alpha1, alpha2))
    
    return acc, pred

@app.route('/feat_select', methods=['POST'])
def feat_select():
    # import pdb; pdb.set_trace()
    logstr = "============= feat_select =================\n"
    time1 = time.time()
    code = request.json['code']
    data_path = request.json['data_path']
    w0 = request.json['w0']
    w1 = request.json['w1']
    topk = request.json['topk']
    data = torch.load(data_path)
    data = convert_dtype(data, dtype)

    try:
        feat_selection = get_feat_selection_module(code, dtype)
        indices = feat_selection(data["clip_weights"], data["train_feats"], w0, w1, topk)
        indices = indices.tolist()
        indices = ",".join(list(map(str, indices)))
        status = 1
        info = "ok"
    except:
        indices = ""
        status = 0
        info = traceback.format_exc()
        if "CUDA error: device-side assert triggered" in info:
            collect_mem()
            exit

    time2 = time.time()
    
    ret = {'output': indices, "status": status, "info": info, "time": time2-time1}
    logstr += "train_feats: {}, train_labels: {}, test_feats: {}, test_labels: {}, clip_weights:  {}".format(data["train_feats"].shape, data["train_labels"].shape, data["test_feats"].shape, data["test_labels"].shape, data["clip_weights"].shape) + "\n"
    logstr += str(ret) + "\n\n"
    print(logstr)

    return jsonify(ret)


@app.route('/eval', methods=['POST'])
def eval():
    # import pdb; pdb.set_trace()
    time1 = time.time()
    logstr = "=============== eval ==================="
    indices = request.json['indices']
    params = request.json['params']
    code = request.json['code']
    data_path = request.json['data_path']
    data = torch.load(data_path)
    data = convert_dtype(data, dtype)

    print("params:", params)

    if indices is None:
        logstr += "indices is None\n"
    else:
        logstr += "indices is not None\n"
    
    try:
        compute_logits = get_logit_compute_module(code, dtype)
        
        # acc, pred = compute_acc_batched(
        #     compute_logits, 
        #     data["train_feats"], 
        #     data["train_labels"], 
        #     data["test_feats"], 
        #     data["test_labels"], 
        #     data["clip_weights"], 
        #     indices, 
        #     params
        # )

        if len(params) == 0:
            print("use optuna")
            acc, pred = compute_searched_acc_optuna(
                compute_logits,
                indices, 
                data["train_feats"], 
                data["train_labels"], 
                data["test_feats"], 
                data["test_labels"], 
                data["clip_weights"], 
                params
            )
        else:
            acc, pred = compute_searched_acc(
                compute_logits,
                indices, 
                data["train_feats"], 
                data["train_labels"], 
                data["test_feats"], 
                data["test_labels"], 
                data["clip_weights"], 
                params
            )
        
        pred = pred.reshape(-1).cpu().tolist()
        pred = ",".join(list(map(str, pred)))
        status = 1
        info = "ok"
    except:
        acc = -1
        pred = ""
        status = 0
        info = traceback.format_exc()
        if "CUDA error: device-side assert triggered" in info:
            collect_mem()
            exit

    time2 = time.time()
    ret = {'acc':acc, 'pred': pred, "status": status, "info": info, "time": time2-time1}

    logstr += "train_feats: {}, train_labels: {}, test_feats: {}, test_labels: {}, clip_weights:  {}".format(data["train_feats"].shape, data["train_labels"].shape, data["test_feats"].shape, data["test_labels"].shape, data["clip_weights"].shape) + "\n"
    logstr += str(ret) + "\n\n"
    print(logstr)

    return jsonify(ret)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
