import os
import sys
import json
from eoh import eoh
from eoh.utils.getParas import Paras
from utils import set_logger

def save_params(paras):
    attributes = vars(paras)
    json_string = json.dumps(attributes)
    os.makedirs(paras.exp_output_path, exist_ok=True)
    with open(os.path.join(paras.exp_output_path, "paras.json"), 'w') as file:
        file.write(json_string)

def makedirs(path):
    assert not os.path.exists(path), path
    os.makedirs(path)

def get_dataset(data_name):
    if data_name == "5d":
        return "cifar100,fashionmnist,objectnet,ucmerced,birds"
    elif data_name == "4d":
        return "cifar100,fashionmnist,objectnet,ucmerced"
    elif data_name == "3d":
        return "cifar100,fashionmnist,objectnet"
    elif data_name == "2d":
        return "cifar100,fashionmnist"
    else:
        assert data_name in ['cifar100','fashionmnist','objectnet','ucmerced','birds'], data_name
        return data_name

method = sys.argv[1]
net = sys.argv[2]
srcdata0 = sys.argv[3]
srcdata = get_dataset(sys.argv[3])
ec_pop_size = 10
ec_n_pop = 10
evomethod = "ael"
inst = int(sys.argv[4])
sample_class = int(sys.argv[5])
exp_n_proc = 0
ntest_per_class = int(sys.argv[8])

logit_exp_output_path = f"{net}_evol/{srcdata0}_inst{inst}_nt{ntest_per_class}/{method}_logit+fs-s1/{evomethod}_popsize{ec_pop_size}_npop{ec_n_pop}_sc{sample_class}"
fs_exp_output_path = f"{net}_evol/{srcdata0}_inst{inst}_nt{ntest_per_class}/{method}_logit+fs-s2/{evomethod}_popsize{ec_pop_size}_npop{ec_n_pop}_sc{sample_class}"

makedirs(logit_exp_output_path)
makedirs(fs_exp_output_path)

# *************** optimization for logit computation ***************

set_logger(f"{logit_exp_output_path}/log.txt")
# Parameter initilization #
paras = Paras()
# Set parameters #
problem_params = {
    "train_feats_path": [f"{net}_caches/{d}/train_f_aug3.pt" for d in srcdata.split(',')],
    "train_labels_path": [f"{net}_caches/{d}/train_l_aug3.pt" for d in srcdata.split(',')],
    "test_feats_path": [f"{net}_caches/{d}/val_f.pt" for d in srcdata.split(',')],
    "test_labels_path": [f"{net}_caches/{d}/val_l.pt" for d in srcdata.split(',')],
    "clip_weights_path": [f"{net}_caches/{d}/clip_weights.pt" for d in srcdata.split(',')],
    "shots": [4,8,16] if method=="gda" else [1,2,4,8,16],
    "inst_per_shots": inst,
    "feat_sel_path": "seeds/feat_selection.json",
    "init_alpha": [0.1, 1.0, 10.0],
    "init_w0": [0.5], 
    "init_percent": [0.7],
    "sample_class": sample_class,
    "ntest_per_class": ntest_per_class,
    "instance_cache_path": f"{net}_evol/{srcdata0}_inst{inst}_nt{ntest_per_class}",
    "service": [f"http://localhost:600{i}" for i in range(8)],
    "pool_size": 5,
    "timeout": 300,
}
paras.set_paras(method = evomethod,    # ['ael','eoh']
                problem = "logit_func_opt", #['tsp_construct','bp_online']
                llm_api_endpoint = "api.deepseek.com", # set your LLM endpoint
                llm_api_key = os.environ.get("APIKEY"),   # set your key
                llm_model = "deepseek-coder",
                ec_pop_size = ec_pop_size, # number of samples in each population
                ec_n_pop = ec_n_pop,  # number of populations
                exp_n_proc = exp_n_proc,  # multi-core parallel
                exp_debug_mode = False,
                problem_params = problem_params,
                exp_use_seed = True,
                exp_seed_path = f"seeds/logit_func_opt-{method}.json",
                exp_output_path = logit_exp_output_path,
                log_file = f"{logit_exp_output_path}/log.txt",
            )
save_params(paras)
# initilization
evolution = eoh.EVOL(paras)
# run
evolution.run()

# *************** optimization for feat selection ***************

set_logger(f"{fs_exp_output_path}/log.txt")
# Parameter initilization #
paras = Paras()
# Set parameters #
problem_params = {
    "train_feats_path": [f"{net}_caches/{d}/train_f_aug3.pt" for d in srcdata.split(',')],
    "train_labels_path": [f"{net}_caches/{d}/train_l_aug3.pt" for d in srcdata.split(',')],
    "test_feats_path": [f"{net}_caches/{d}/val_f.pt" for d in srcdata.split(',')],
    "test_labels_path": [f"{net}_caches/{d}/val_l.pt" for d in srcdata.split(',')],
    "clip_weights_path": [f"{net}_caches/{d}/clip_weights.pt" for d in srcdata.split(',')],
    "shots": [4,8,16] if method=="gda" else [1,2,4,8,16],
    "inst_per_shots": inst,
    "logit_func_path": f"{logit_exp_output_path}/results/pops_best/population_generation_{ec_n_pop}.json",
    "init_alpha": [0.1, 1.0, 10.0],
    "init_w0": [0.5], 
    "init_percent": [0.7],
    "sample_class": sample_class,
    "ntest_per_class": ntest_per_class,
    "instance_cache_path": f"{net}_evol/{srcdata0}_inst{inst}_nt{ntest_per_class}",
    "service": [f"http://localhost:600{i}" for i in range(8)],
    "pool_size": 5,
    "timeout": 300,
}
paras.set_paras(method = evomethod,    # ['ael','eoh']
                problem = "feat_selection", #['tsp_construct','bp_online']
                llm_api_endpoint = "api.deepseek.com", # set your LLM endpoint
                llm_api_key = os.environ.get("APIKEY"),   # set your key
                llm_model = "deepseek-coder",
                ec_pop_size = ec_pop_size, # number of samples in each population
                ec_n_pop = ec_n_pop,  # number of populations
                exp_n_proc = exp_n_proc,  # multi-core parallel
                exp_debug_mode = False,
                problem_params = problem_params,
                exp_use_seed = True,
                exp_seed_path = f"seeds/feat_selection.json",
                exp_output_path = fs_exp_output_path,
            )
save_params(paras)
# initilization
evolution = eoh.EVOL(paras)
# run
evolution.run()