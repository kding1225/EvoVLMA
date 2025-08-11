#!/bin/bash

# examples: 
# stop servers:
# bash server.sh stop
#
# stop servers:
# bash server.sh start <gpu_ids> <processes_per_gpu> [base_port]
# for example:
# bash server.sh start 0,1 4 6000

if [ "$1" == "stop" ]; then
    echo "Stopping all gunicorn servers..."
    pkill -9 gunicorn
    echo "Done."
elif [ "$1" == "start" ]; then
    echo "Starting gunicorn servers..."
    rm -rf logs/*
    
    IFS=',' read -ra GPU_IDS <<< "$2"
    PROCESSES_PER_GPU=$3
    BASE_PORT=${4:-6000}
    TIMEOUT=300
    
    # Check required parameters
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Usage: $0 start <gpu_ids> <processes_per_gpu> [base_port]"
        echo "Example: $0 4,5 4 6000 2"
        exit 1
    fi
    
    # Calculate total processes
    TOTAL_PROCESSES=$(( ${#GPU_IDS[@]} * PROCESSES_PER_GPU ))
    
    echo "Starting $TOTAL_PROCESSES processes..."
    echo "${#GPU_IDS[@]} GPUs, $PROCESSES_PER_GPU processes per GPU"
    echo "Base port: $BASE_PORT"
    
    # Start all processes
    for (( i=0; i<${#GPU_IDS[@]}; i++ )); do
        GPU_ID=${GPU_IDS[$i]}
        for (( j=0; j<$PROCESSES_PER_GPU; j++ )); do
            PORT=$(( BASE_PORT + i * PROCESSES_PER_GPU + j ))
            echo "Starting process on GPU $GPU_ID, port $PORT"
            CUDA_VISIBLE_DEVICES=$GPU_ID gunicorn -w 1 -t $TIMEOUT \
                --access-logfile "logs/gu_access_${GPU_ID}_${j}.log" \
                --error-logfile "logs/gu_error_${GPU_ID}_${j}.log" \
                --bind 0.0.0.0:$PORT \
                model_server:app &
        done
    done
    
    echo "All processes started."
    echo "Servers started."
else
    echo "Usage: $0 [start|stop] ..."
    exit 1
fi