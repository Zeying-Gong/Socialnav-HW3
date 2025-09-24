#!/bin/bash
PID_FILE="/home/zeyingg/competition/SocialNav/Socialnav-HW/worker_pids.txt"
if [ -f "$PID_FILE" ]; then
    echo "Stopping workers..."
    xargs kill < "$PID_FILE" 2>/dev/null
    rm -f "$PID_FILE"
    echo "All workers stopped."
else
    echo "No PID file found, nothing to stop."
fi
