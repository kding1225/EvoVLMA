import numpy as np
import json
import random
import time

import logging
logger = logging.getLogger("my_logger")

def save_pop(filename, population):

    population = [{k:v for k,v in pop.items() if k!="preds"} for pop in population]
    indices = np.argsort([pop["objective"] for pop in population])
    
    with open(filename, 'w') as f:
        json.dump([population[i] for i in indices], f, indent=5)

from .ael_interface_EC import InterfaceEC
# main class for AEL
class AEL:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            logger.info("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator
        logger.info("- AEL parameters loaded -")

        self.history = {}

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)

    def update_history(self, pop):
        for p in pop:
            if p["algorithm"] not in self.history:
                self.history[p["algorithm"]] = p["objective"]

    # run ael 
    def run(self):

        logger.info("- Evolution Start -")

        time_start = time.time()

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_prob, use_local_llm=self.use_local_llm, url=self.url, select=self.select,n_p=self.exp_n_proc,timeout=self.timeout,use_numba=self.use_numba
                                   )

        # initialization
        logger.info("INIT START")
        tim = time.time()
        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data, self.exp_n_proc)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            save_pop(filename, population)
            n_start = 0
        else:
            if self.load_pop:  # load population from files
                logger.info("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                logger.info("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                logger.info("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)
                
                logger.info(f"Pop initial: ")
                logger.info("|".join([" Obj: "+str(off['objective']) for off in population])+"\n")
                logger.info("initial population has been created!")
                # Save population to a file
                filename = self.output_path + "/results/pops/population_generation_0.json"
                save_pop(filename, population)
                n_start = 0
        logger.info("INIT END, elapsed time {}, #ind={}".format(time.time()-tim, len(population)))
        
        # main loop
        n_op = len(self.operators)
        self.update_history(population)

        for pop in range(n_start, self.n_pop):  
            #print(f" [{na + 1} / {self.pop_size}] ", end="|")    
            logger.info(f"{pop}-th ITER START")
            tim = time.time()
            for i in range(n_op):
                op = self.operators[i]
                logger.info(f" OP: {op}, [{i + 1} / {n_op}] ") 
                op_w = self.operator_weights[i]
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op, self.history)
                self.add2pop(population, offsprings)  # Check duplication, and add the new offspring
                self.update_history(offsprings)
                logger.info("|".join([" Obj: "+str(off['objective']) for off in offsprings]))
                # populatin management
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                logger.info("\n")
            logger.info(f"{pop}-th ITER END, elapsed time {time.time()-tim}, #ind={len(population)}")

            # Save population to a file
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            save_pop(filename, population)

            # Save the best one to a file
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            save_pop(filename, population)

            logger.info(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            logger.info("Pop Objs: "+" ".join([str(population[i]['objective']) for i in range(len(population))]))
            logger.info("\n")

            logger.info(f"Time Cost: {time.time()-time_start}")

