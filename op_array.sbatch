#!/bin/bash
#SBATCH --job-name=mama_mia_ddp
#SBATCH --nodes=1
#SBATCH --nodelist=dgxanode03
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --output=ddp_%j.out
#SBATCH --error=ddp_%j.err

set -e

echo "========== JOB START =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================"

# Use exactly six GPUs, exclude GPU 6
export CUDA_VISIBLE_DEVICES=1,2,3,5,7
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Deterministic unique rendezvous
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 2000))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# Pip safety
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

# NCCL stability
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# Prevent CPU oversubscription
export OMP_NUM_THREADS=4

srun --mpi=pmi2 \
     --container-image /home/hpc_apps/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts /home/neeraj:/hostd \
     --container-workdir /hostd \
     bash -c "
        set -e

        echo 'Inside container'
        echo 'Visible GPUs:' \$CUDA_VISIBLE_DEVICES
        echo 'MASTER_ADDR:' \$MASTER_ADDR
        echo 'MASTER_PORT:' \$MASTER_PORT

        nvidia-smi

        # Install only user-space scientific stack, never torch or torchaudio
        pip install --user \
            monai \
            nibabel \
            scikit-image \
            scikit-learn \
            tqdm \
            matplotlib \
            scipy \
            pillow

        echo 'Launching torchrun DDP'

        torchrun \
            --nproc_per_node=5 \
            --rdzv_backend=c10d \
            --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
            /hostd/mama_mia_dataset/ganeev_ddp.py
     "

echo "End time: $(date)"
echo "=========== JOB END ==========="