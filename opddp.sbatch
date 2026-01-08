#!/bin/bash
#SBATCH --job-name=mama_mia_ddp
#SBATCH --nodes=1
#SBATCH --nodelist=dgxanode03
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=defq
#SBATCH --time=1500:00:00
#SBATCH --output=ddp_%j.out
#SBATCH --error=ddp_%j.err

set -euo pipefail

echo "========== JOB START =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (NUM_GPUS=${NUM_GPUS})"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 10000))
export RDZV_ID="mama_mia_${SLURM_JOB_ID}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

export PYTHONUNBUFFERED=1
ulimit -n 65536 || true

srun --mpi=none \
     --container-image /home/hpc_apps/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts /home/neeraj:/hostd \
     --container-workdir /hostd \
     bash -c "
        set -euo pipefail

        echo 'Inside container'
        echo 'Visible GPUs:' \$CUDA_VISIBLE_DEVICES
        nvidia-smi || true

        pip install --user --upgrade --no-input \
            monai nibabel scikit-image scikit-learn tqdm matplotlib scipy pillow

        echo 'Launching torchrun with ${NUM_GPUS} processes'
        torchrun \
            --nproc_per_node=${NUM_GPUS} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
            --rdzv_id=${RDZV_ID} \
            /hostd/mama_mia_dataset/5phase.py
     "

echo "End time: $(date)"
echo "=========== JOB END =========="
