import numpy as np
import time
import sys, os
from .ael_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import traceback

import torch
import logging
logger = logging.getLogger("my_logger")

import ast

class NestedLoopChecker(ast.NodeVisitor):  
    def __init__(self):
        self.max_depth = 0  
        self.current_depth = 0
    
    def visit_For(self, node):  
        self.current_depth += 1  
        if self.current_depth > self.max_depth:  
            self.max_depth = self.current_depth  
        self.generic_visit(node)  
        self.current_depth -= 1  
  
    def visit_While(self, node):  
        self.current_depth += 1  
        if self.current_depth > self.max_depth:  
            self.max_depth = self.current_depth  
        self.generic_visit(node)  
        self.current_depth -= 1  
  
    def get_nested_loops_depth(self):  
        return self.max_depth
  
def check_for_nested_loops(code):  
    try:
        tree = ast.parse(code)  
        checker = NestedLoopChecker()  
        checker.visit(tree)  
        return checker.get_nested_loops_depth()  
    except SyntaxError:  
        print("The provided code has syntax errors.")  
        return -1

class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select,n_p,timeout,use_numba, **kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        # -----------------------------------------------------------

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode,prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba
        
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self):
        n_create = 2
        population = []
        for i in range(n_create):
            _,pop = self.get_algorithm([],'i1',{})
            for p in pop:
                population.append(p)
        return population
    
    def population_generation_seed(self,seeds,n_p):

        population = []
        try:
            fitness = [self.interface_eval.evaluate(seed['code']) for seed in seeds]
        except Exception as e:
            logger.info("EVAL SEED ERROR:")
            logger.info(traceback.format_exc())
            exit

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i][0])
                seed_alg['objective'] = np.round(obj, 5)
                seed_alg['preds'] = fitness[i][1]
                population.append(seed_alg)

            except Exception as e:
                logger.info("EVAL SEED ERROR:")
                logger.info(traceback.format_exc())
                exit()

        logger.info("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population

    def _get_alg(self, pop, operator, history):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        
        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.i1()            
        elif operator == "crossover":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.crossover(parents)
        elif operator == "mutation":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.mutation(parents[0]) 
        else:
            logger.info(f"Evolution operator [{operator}] has not been implemented ! \n")
            raise

        return parents, offspring

    def get_offspring(self, pop, operator, history):

        offspring0 = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }

        code, offspring = None, None
        try:
            p, offspring = self._get_alg(pop, operator, history)
            
            if self.use_numba:
                
                # Regular expression pattern to match function definitions
                pattern = r"def\s+(\w+)\s*\(.*\):"

                # Search for function definitions in the code
                match = re.search(pattern, offspring['code'])

                function_name = match.group(1)

                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            n_retry = 1
            while self.check_duplicate(pop, offspring['code']):
                
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                    
                p, offspring = self._get_alg(pop, operator, history)

                if self.use_numba:
                    # Regular expression pattern to match function definitions
                    pattern = r"def\s+(\w+)\s*\(.*\):"

                    # Search for function definitions in the code
                    match = re.search(pattern, offspring['code'])

                    function_name = match.group(1)

                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']
                 
                if n_retry > 1:
                    break

        except Exception as e:
            logger.info("GET ALGORITHM ERROR:")
            logger.info(traceback.format_exc())
            
            offspring = offspring0
            p = None
            return p, offspring

        if code is None:
            return None, offspring0

        depth = check_for_nested_loops(code)
        if depth >= 3:
            logger.info("TOO DEEP NESTED LOOPS: ", depth)
            return None, offspring0
        
        try:
            # # self.code2file(offspring['code'])
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     future = executor.submit(self.interface_eval.evaluate, code)
            #     fitness, preds = future.result(timeout=self.timeout)
            #     offspring['objective'] = np.round(fitness, 5)
            #     offspring['preds'] = preds
            #     future.cancel()
            fitness, preds = self.interface_eval.evaluate(code)
            offspring['objective'] = np.round(fitness, 5)
            offspring['preds'] = preds
        except Exception as e:
            logger.info("EVAL ALGORITHM ERROR:")
            err_info = traceback.format_exc()
            logger.info(err_info)
            offspring = offspring0
            p = None
            return p, offspring

        # Round the objective values
        return p, offspring
    
    def get_algorithm(self, pop, operator, history):

        results = []
        if self.n_p >= 1:
            try:
                results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator, history) for _ in range(self.pop_size))
            except Exception as e:
                logger.info(f"JOBLIB ERROR: {e}")
            time.sleep(2)
        else:
            results = [self.get_offspring(pop, operator, history) for _ in range(self.pop_size)]

        out_p = []
        out_off = []
        for p, off in results:
            out_p.append(p)
            out_off.append(off)
        return out_p, out_off
    
