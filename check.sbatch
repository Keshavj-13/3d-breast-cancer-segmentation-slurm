#!/bin/bash
#SBATCH --job-name=gpu_sanity
#SBATCH --output=gpu_sanity_%j.log
#SBATCH --error=gpu_sanity_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -e

echo "========== SLURM INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo

echo "========== NVIDIA SMI =========="
nvidia-smi
echo

echo "========== ENVIRONMENT =========="
which python
python --version
echo

echo "========== PYTORCH CHECK =========="
python << 'EOF'
import torch
import os
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("SM count:", props.multi_processor_count)
    print("Total memory GB:", props.total_memory / 1e9)
else:
    print("WARNING: CUDA NOT AVAILABLE")

print("OMP threads:", os.environ.get("OMP_NUM_THREADS"))
EOF

echo
echo "========== DATALOADER STRESS TEST =========="
python << 'EOF'
import torch, time

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(1024, 1024, device=device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    y = x @ x
torch.cuda.synchronize()
t1 = time.time()

print("Matmul time seconds:", t1 - t0)
EOF

echo
echo "========== TRAIN LOOP MICRO BENCH =========="
python << 'EOF'
import torch, time

device = "cuda"
model = torch.nn.Linear(4096, 4096).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(256, 4096, device=device)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(200):
    opt.zero_grad()
    y = model(x).sum()
    y.backward()
    opt.step()
torch.cuda.synchronize()
t1 = time.time()

print("Mini train loop seconds:", t1 - t0)
EOF

echo "========== DONE =========="
