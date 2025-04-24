#!/bin/sh

#SBATCH -c 48 # Number of cores requested
#SBATCH -t 3-00:00 # Runtime in minutes
#SBATCH -p kempner_h100 # Partition to submit to
#SBATCH --mem=250G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this file
#SBATCH --account=kempner_barak_lab
#SBATCH --job-name=grpo_17cd


module purge
module load Mambaforge
module load cuda cudnn
module load gcc/12.2.0-fasrc01 
mamba activate verl

cd /n/home05/sqin/self-correct/verl/my_scripts/countdown
pwd

bash grpo_countdown.sh