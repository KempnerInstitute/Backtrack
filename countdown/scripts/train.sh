#!/bin/sh

#SBATCH -c 48 # Number of cores requested
#SBATCH -t 1-16:00 # Runtime in minutes
#SBATCH -p kempner_h100 # Partition to submit to
#SBATCH --mem=250G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -o ../slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e ../slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=cd-17sft-hint

module purge
module load Mambaforge
module load cuda cudnn
mamba activate reason

set -x 

cd ../src

# accelerate launch --config_file ../configs/accelerate.yaml train.py --config ../configs/oft-mix-4-cd.conf 

accelerate launch --config_file ../configs/accelerate.yaml \
  train.py \
    --config ../configs/sft-hint-4-cd-17M-qwen.conf


    # --ckpt /n/netscratch/dam_lab/Lab/sqin/outputs/oft/checkpoint-9500 \
    # --resume 
