#!/usr/bin/env bash
set -euo pipefail

# Get SLURM variables
NODELIST="${SLURM_STEP_NODELIST:-${SLURM_JOB_NODELIST:-}}"
NUM_NODES="${SLURM_STEP_NUM_NODES:-${SLURM_JOB_NUM_NODES:-}}"

if [[ -z "$NODELIST" || -z "$NUM_NODES" ]]; then
  echo "ERROR: Not running under Slurm (missing SLURM_* vars)."
  exit 1
fi

# Get head node
NODES=$(scontrol show hostnames "$NODELIST")
NODES_ARRAY=($NODES)
RAY_IP="${NODES_ARRAY[0]}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_ADDRESS="${RAY_IP}:${RAY_PORT}"

# Ensure command exists (Ray >= 2.49 per docs)
if ! ray symmetric-run --help >/dev/null 2>&1; then
  echo "ERROR: 'ray symmetric-run' not available. Check Ray installation (needs Ray 2.49+)."
  exit 1
fi

# Launch cluster!
echo "Ray head node: $RAY_ADDRESS"

exec ray symmetric-run \
  --address "$RAY_ADDRESS" \
  --min-nodes "$NUM_NODES" \
  -- \
  "$@"

