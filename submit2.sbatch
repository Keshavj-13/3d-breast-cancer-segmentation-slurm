#!/bin/bash
#SBATCH --job-name neerajjob
#SBATCH --nodes 1
#SBATCH --nodelist dgxanode03
#SBATCH --ntasks-per-node 1
#SBATCH --time 100:00:00
#SBATCH --partition defq
#SBATCH --output output_job_%j.out
#SBATCH --error error_job_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

export CUDA_VISIBLE_DEVICES=3

# Launch a container shell that installs the missing libraries, then runs your script
srun --mpi pmi2 \
     --container-image /home/hpc_apps/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts /home/neeraj:/hostd \
     --container-workdir /hostd \
     bash -c "
        pip install --user monai nibabel scikit-image scikit-learn tqdm matplotlib scipy pillow &&
        python3 /hostd/mama_mia_dataset/ganeev.py
     "

echo "Job finished on $(date)"