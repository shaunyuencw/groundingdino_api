#!/bin/bash

# Ports used by your Gunicorn servers
ports=(8001 8002 8003 8004)

for port in "${ports[@]}"
do
    echo "Shutting down server on port $port..."
    # Find the process listening on each port and kill it
    pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        kill -9 $pid
        echo "Server on port $port shut down."
    else
        echo "No server found running on port $port."
    fi
done

echo "All specified servers have been shut down."
