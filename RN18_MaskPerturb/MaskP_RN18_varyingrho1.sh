#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o RN18_maskp_varyrho_01sparsity.out

source /home/sliu/miniconda3/etc/profile.d/conda.sh
source activate TJ

cd ..
pwd

for rho in 0.1 1 5 10 15
do
python main.py --config configs/smallscale/resnet18/resnet18-kn-unsigned.yaml --multigpu 0  --data dataset --prune-rate 0.1 --rho $rho --print-freq 400
done
