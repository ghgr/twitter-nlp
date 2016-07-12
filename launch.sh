#!/bin/bash 
NPROC=$(nproc)
echo "Launching $NPROC processes..."
for ((i=1;i<=NPROC;i++)); do
    python main.py &
done

