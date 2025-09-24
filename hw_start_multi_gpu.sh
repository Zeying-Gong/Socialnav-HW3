#!/bin/bash

# ==================== Parameters Check ====================
gpus="$@"
if [ -z "$gpus" ]; then
    echo "Usage: $0 <gpu_id1> <gpu_id2> ..."
    echo "Example: $0 0 1 (both GPUs will run in BOTH mode)"
    exit 1
fi

# ==================== Configuration ====================
PROJECT_DIR="/home/zeyingg/competition/SocialNav/Socialnav-HW" # for homework
MAIN_SCRIPT="${PROJECT_DIR}/remote_challenge_evaluation/main.py"
LOG_DIR="${PROJECT_DIR}/logs"
PID_FILE="${PROJECT_DIR}/worker_pids.txt"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# ==================== Kill Old Processes ====================
echo "Killing old evaluation workers..."
if [ -f "$PID_FILE" ]; then
    for pid in $(cat "$PID_FILE"); do
        echo "Attempting to kill old worker with PID: $pid"
        # Use `kill -0` to check if process is running before killing
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        else
            echo "PID $pid not found or already dead."
        fi
    done
    rm -f "$PID_FILE"
    sleep 2
fi

# Fallback pkill in case PID file was corrupted or not cleaned up
pkill -9 -f "^python -u ${MAIN_SCRIPT}$" 2>/dev/null
sleep 2

# ==================== Change to Project Directory ====================
cd "$PROJECT_DIR" || { echo "Error: Project directory '$PROJECT_DIR' not found!"; exit 1; }

# ==================== Load Environment Variables ====================
if [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    set -o allexport
    source .env
    set +o allexport
else
    echo "Error: .env file not found at '$PROJECT_DIR'!"
    exit 1
fi

# ==================== Start Workers ====================
echo "Starting evaluation workers on GPUs: $gpus (ALL running in BOTH mode)"
> "$PID_FILE"  # Clear the old PID file

for gpu_id in $gpus; do
    echo "Starting worker for GPU $gpu_id in BOTH mode..."
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="${LOG_DIR}/gpu_${gpu_id}_${timestamp}.log"
    
    # Set CUDA_VISIBLE_DEVICES to bind the worker to a specific GPU
    # Set WORKER_ROLE_OVERRIDE to BOTH for all workers
    CUDA_VISIBLE_DEVICES=$gpu_id \
    WORKER_ROLE_OVERRIDE=BOTH \
    python -u "$MAIN_SCRIPT" > "$log_file" 2>&1 &
    
    pid=$!
    echo "Worker for GPU $gpu_id started with PID: $pid in BOTH mode (log: $log_file)"
    echo "$pid" >> "$PID_FILE"
done

echo "All workers started in BOTH mode. PIDs saved to $PID_FILE"
echo "Both GPUs can now process both minival and general phases."