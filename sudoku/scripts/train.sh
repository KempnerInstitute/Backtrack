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
#SBATCH --exclude=holygpu8a19604,holygpu8a19303,holygpu8a17402,holygpu8a17402
#SBATCH --job-name=sdk-38sft-easy
#SBATCH --exclude=holygpu8a11404,holygpu8a17504,holygpu8a13201

module purge
module load Mambaforge
module load cuda cudnn
mamba activate reason

cd ../

accelerate launch \
  --config_file configs/accelerate.yaml \
  train.py \
  --config configs/38M_sft_strat_easy.conf 


  # --resume \
  # --ckpt /n/netscratch/dam_lab/Lab/sqin/reason/sudoku/outputs/17M-sft-strat-fixed/checkpoint-590000

