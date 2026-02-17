#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
set -euo pipefail

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_NODE_IP=$(srun -N1 -n1 -w "$HEAD_NODE" bash -lc 'hostname -I | awk "{print \$1}"')
RAY_PORT=6379
RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

echo "Launching head node: RAY_ADDRESS=$RAY_ADDRESS"
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
  ray start --head --node-ip-address="$HEAD_NODE_IP" --port="$RAY_PORT" --block &
sleep 5

NUM_WORKERS=$((SLURM_JOB_NUM_NODES - 1))
echo "Launching ${NUM_WORKERS} worker node(s)"
if [[ "$NUM_WORKERS" -gt 0 ]]; then
  srun -n "$NUM_WORKERS" --nodes="$NUM_WORKERS" --ntasks-per-node=1 --exclude "$HEAD_NODE" \
    ray start --address "$RAY_ADDRESS" --block &
fi

# Keep job alive (or replace with running your workload on the head)
wait
