import os, signal
import subprocess
import requests
import json
import time
import concurrent.futures

def run_par(func, params, max_workers):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, *param) for param in params]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

def run_cmd(command):
    result = subprocess.run(command, shell=True, check=True, 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = result.stdout
    return stdout

def get_unicorn_work_pid():
    cmd = "ps -eo pid,ppid,cmd | awk '/[g]unicorn/ && $2 != 1 {print $1}'"
    ret = run_cmd(cmd)
    pids = ret.strip().split("\n")
    return pids

def get_pid_port_pairs(pids):
    cmd = "lsof -iTCP -sTCP:LISTEN -P | grep gunicorn"
    ret = run_cmd(cmd)
    pairs = []
    for line in ret.strip().split("\n"):
        line = line.split()
        pid = line[1]
        port = line[-2].replace("*:", "")
        if pid in pids:
            pairs.append([int(pid), int(port)])

    return pairs

def kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except:
        print(f"kill {pid} failed")

def post_feat(pid, port):

    with open("seeds/feat_selection_for_server_test.json", "r") as f:
        code = json.load(f)[0]["code"]
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "code": code,
        "data_path": "server_test_data.pth",
        "w0": 0.5,
        "w1": 0.5,
        "topk": 200
    }
    assert os.path.exists(payload["data_path"]), payload["data_path"]
    try:
        response = requests.post(f"http://localhost:{port}/feat_select", headers=headers, 
                                         data=json.dumps(payload))
    except:
        print(f"call error, kill {pid}")
        kill(pid)
        return
    try:
        if response.status_code == 500:
            print(f"code=500, kill {pid}")
            kill(pid)
    except:
        pass

def post_eval(pid, port):
    with open("seeds/logit_func_opt-clip.json", "r") as f:
        code = json.load(f)[0]["code"]
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "code": code,
        "data_path": "server_test_data.pth",
        "indices": [1,2,3,4,5],
        "params": [(1.0, 1.0, 1.0)],
    }
    assert os.path.exists(payload["data_path"]), payload["data_path"]

    try:
        response = requests.post(f"http://localhost:{port}/eval", headers=headers, 
                                         data=json.dumps(payload))
    except:
        print(f"call error, kill {pid}")
        kill(pid)
        return

    try:
        if response.status_code == 500:
            print(f"code=500, kill {pid}")
            kill(pid)
    except:
        pass


if __name__ == "__main__":
    
    while True:

        pids = get_unicorn_work_pid()
        pairs = get_pid_port_pairs(pids)
        print(pairs)
        
        run_par(post_feat, pairs, max_workers=5)
        run_par(post_eval, pairs, max_workers=5)
        time.sleep(2)
        # print(1)
    
