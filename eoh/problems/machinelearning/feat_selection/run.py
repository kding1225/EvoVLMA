import numpy as np
import pickle
import sys
import types
import warnings
import torch
import os
import time
import math
import random
import json
import itertools
import torch.nn.functional as F
import functools
import concurrent.futures

import requests
from .prompts import GetPrompts
from .get_instance import GetData

import logging
logger = logging.getLogger("my_logger")

class FeatSel():
    def __init__(self, problem_params) -> None:
        self.ndelay = 1
        self.running_time = 10
        self.prompts = GetPrompts()
        self.instance_paths, self.feat_dim = GetData(
            train_feats_path=problem_params["train_feats_path"],
            train_labels_path=problem_params["train_labels_path"],
            test_feats_path=problem_params["test_feats_path"],
            test_labels_path=problem_params["test_labels_path"],
            clip_weights_path=problem_params["clip_weights_path"],
            shots=problem_params["shots"],
            inst_per_shots=problem_params["inst_per_shots"],
            sample_class=problem_params["sample_class"],
            ntest_per_class=problem_params["ntest_per_class"],
            instance_cache_path=problem_params["instance_cache_path"]
        ).generate_and_save_instances()
        self.num_tasks = len(self.instance_paths)
        self.timeout = problem_params["timeout"]

        self.logit_func_path = problem_params["logit_func_path"]
        self.alpha0_list = problem_params["init_alpha"]
        self.alpha1_list = problem_params["init_alpha"]
        self.alpha2_list = problem_params["init_alpha"]
        self.params = list(itertools.product(self.alpha0_list, self.alpha1_list, self.alpha2_list))
        self.pool_size = problem_params["pool_size"]
        
        self.w0_list = problem_params['init_w0']
        self.percent_list = problem_params['init_percent']
        self.service = problem_params['service']

        self.code_cache = {}

    def request_random(self, route, headers, payload, max_retries = 1):
        services = self.service
        if isinstance(self.service, str):
            services = [self.service]
        service = random.choice(services)

        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.post(service+route, headers=headers, 
                                         data=json.dumps(payload), timeout=self.timeout)
                logger.info(route+" RESPONSE CODE: "+str(response.status_code)+" "+response.text)
                return response
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    attempt += 1
                else:
                    logger.error("Maximum retries exceeded")
                    return
    
    def call_feat_selection(self, code, data_path, w0, w1, topk):
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "code": code,
            "data_path": data_path,
            "w0": w0,
            "w1": w1,
            "topk": topk
        }
        response = self.request_random("/feat_select", headers, payload)

        try:
            if response.json()["status"]:
                result = response.json()["output"]
                result = list(map(int, result.split(',')))
                return result
            else:
                return None
        except:
            return None

    def call_eval(self, indices, data_path, max_retries=1):
        
        with open(self.logit_func_path, "r") as f:
            code = json.load(f)[0]["code"]
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "indices": indices,
            "params": self.params,
            "code": code,
            "data_path": data_path,
        }

        response = self.request_random("/eval", headers, payload, max_retries=max_retries)
        
        try:
            result = response.json()
            if result["status"]:
                acc = result["acc"]
                pred = result["pred"]
                pred = torch.tensor(list(map(int, pred.split(','))))
                return acc, pred
            else:
                return None, None
        except:
            return None, None
        
    # @timeout(TIMEOUT)
    def compute_error(self, code):
        # import pdb; pdb.set_trace()

        def run_inst(inst_path, w0, w1, topk):
            indices = self.call_feat_selection(code, inst_path, w0, w1, topk)
            acc, pred = self.call_eval(indices, inst_path)
            return acc, pred
        
        accs = []
        preds = []
        num = 0
        for w0 in self.w0_list:
            for percent in self.percent_list:

                w1 = 1 - w0
                topk = int(percent*self.feat_dim)

                acc, pred = run_inst(self.instance_paths[0], w0, w1, topk)
                if acc is None:
                    return None, None

                results = [None for _ in self.instance_paths]
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_index = {executor.submit(run_inst, self.instance_paths[i], w0, w1, topk): i for i in range(len(self.instance_paths))}
                    for future in concurrent.futures.as_completed(future_to_index):
                        index = future_to_index[future]
                        results[index] = future.result()

                print([x[0] for x in results])
                for inst_idx, (acc, pred) in enumerate(results):
                    if acc is None:
                        return None, None
                    num += 1
                    accs.append(acc)
                    preds.append(pred)
        
        # import pdb; pdb.set_trace()
        return 100.0-sum(accs)/len(accs), torch.cat(preds).cpu() # cls error

    def evaluate(self, code_string):

        logger.info("EVALUATED CODE:")
        logger.info(code_string)

        if code_string in self.code_cache:
            fitness = self.code_cache[code_string]
            logger.info("cache hit")
        else:
            fitness = self.compute_error(code_string)
            self.code_cache[code_string] = fitness
        
        return fitness
