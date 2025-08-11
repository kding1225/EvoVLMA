import os, sys
import re
import numpy as np
from collections import defaultdict


def parse_result(file, method):
    with open(file, "r") as f:
       text = f.read()
    
    pattern = r"{}'s test accuracy:\s*(-?\d+\.\d+)".format(method) 
    match = re.search(pattern, text)  
    if match:  
        number = match.group(1)  
        return float(number)
    else:
        print(file)
        raise
    

datasets = ["imagenet", "caltech101", "oxford_pets", "stanford_cars", "oxford_flowers", "food101", "fgvc", "sun397", "dtd", "eurosat", "ucf101"]

f = sys.argv[1]
method = sys.argv[2]
shots = sys.argv[3]
path = "RN50_caches/{}/{}.txt"
shots = [1,2,4,8,16] if shots=="all" else [int(shots)]

all_ret = []
per_ret = defaultdict(list)
for shot in shots:
    test_accs1 = []
    test_accs2 = []
    val_accs1 = []
    val_accs2 = []
    for data in datasets:
       test_acc1 = [parse_result(path.format(data, f).replace("shots1", "shots"+str(shot)).replace("seed1", "seed"+str(seed)), method)
                   for seed in [1,2,3]
                   ]
       test_acc1 = np.mean(test_acc1)
       test_accs1.append(test_acc1)
       print("{}, acc={:.2f}".format(data, test_acc1))
       per_ret[data].append(test_acc1)
    
    print(test_accs1)
    print("TEST: {:.2f}".format(np.mean(test_accs1)))
    all_ret.append(np.mean(test_accs1))

print("".join(["{:.2f} ".format(x) for x in all_ret]))

print(per_ret)

