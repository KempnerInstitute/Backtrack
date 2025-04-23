#!/bin/sh

#SBATCH -c 16 # Number of cores requested
#SBATCH -t 0-10:00 # Runtime in minutes
#SBATCH -p kempner # Partition to submit to
#SBATCH --mem=250G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o ../slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e ../slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_barak_lab
#SBATCH --job-name=e-17easy

module purge
module load Mambaforge
module load cuda cudnn
mamba activate reason

cd ../

# SFT
# date 
# echo "38M-SFT-strat-easy pass@1 ckpt=latest"
# python eval.py \
#     --ckpt /n/netscratch/dam_lab/Lab/sqin/reason/sudoku/outputs/38M-sft-strat-easy \
#     --num 200 \
#     --offset 0 \
#     --data easy_sudoku_strategy_data_sample.json \
#     --temperature 0.0 \
#     --batch_size 1 \
#     --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/sudoku/strategy_data_easy \
#     --gens 1 \
#     --ctx 4096  
# date 


# OFT
# k range: 1 4 16 32 64 128
ks=(16)
for k in ${ks[@]}; do
    date 
    echo "17M-OFT-strat-easy pass@${k} temperature=0.7 ckpt=latest"
    python eval.py \
        --ckpt /n/netscratch/dam_lab/Lab/sqin/reason/sudoku/outputs/17M-oft-strat-easy/checkpoint-1312500 \
        --num 200 \
        --offset 0 \
        --data easy_sudoku_strategy_data_sample.json  \
        --temperature 0.7 \
        --batch_size 1 \
        --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/sudoku/strategy_data_easy \
        --gens ${k} \
        --ctx 512 
done
