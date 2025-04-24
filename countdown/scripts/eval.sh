#!/bin/sh

#SBATCH -c 16 # Number of cores requested
#SBATCH -t 0-06:00 # Runtime in minutes
#SBATCH -p kempner_h100 # Partition to submit to
#SBATCH --mem=64G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o ../slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e ../slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=cd-e-17oft

module purge
module load Mambaforge
module load cuda cudnn
mamba activate reason

set -x

cd ../src

# Currrently, for Qwen models please use bacth_size 1 because I think the padding is not fixed
# date
# echo "3M-SFT pass@1 temperature=0.0 ckpt=latest"
# python eval_neo.py \
#     --ckpt /n/netscratch/dam_lab/Lab/sqin/reason/sos/sft-countdown-dfs5e5-3M-qwen/checkpoint-545760 \
#     -n 200 \
#     -o 0 \
#     -d val1_b4_t100_n500000_dfs.json  \
#     --temperature 0.0 \
#     --batch_size 1 \
#     --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs \
#     --gens 1 \
#     --ctx 4096 
# date


# ks=(64)
# for k in ${ks[@]}; do
#     date
#     echo "17M-OFT pass@${k} temperature=0.7 ckpt=best"
#     python eval_neo.py \
#         --ckpt /n/netscratch/dam_lab/Lab/sqin/reason/sos/oft-countdown-cd3e5-17M-qwen/checkpoint-118760 \
#         --num 200 \
#         --offset 0 \
#         --data val1_b4_t100_n500000_dfs.json \
#         --temperature 0.7 \
#         --batch_size 1 \
#         --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs \
#         --gens ${k} \
#         --ctx 512 \
#         --seed 8
# done

perform temperature sampling on SFT model
ks=(64)
for k in ${ks[@]}; do
    date
    echo "17M-SFT-GRPO pass@${k} temperature=0.7 ckpt=best"
    python eval_neo.py \
        --ckpt /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-sft-dfs-17M-qwen/hf_model/global_step_350 \
        --num 200 \
        --offset 200 \
        --data val1_b4_t100_n500000_dfs.json \
        --temperature 0.7 \
        --batch_size 1 \
        --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs \
        --gens ${k} \
        --ctx 4096 
done

# # eval GRPO'ed model - SFT
# python eval_neo.py \
#     --ckpt /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-sft-dfs-3M-qwen/hf_model/global_step_1830 \
#     -n 200 \
#     -o 0 \
#     -d val_target1_b4_t100_n500000_dfs.json \
#     --temperature 0.0 \
#     --batch_size 1 \
#     --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs \
#     --gens 1 \
#     --ctx 4096 
# date


# # eval GRPO'ed model - OFT
# ks=(2 4 8 16)
# for k in ${ks[@]}; do
#     date
#     python eval_neo.py \
#         --ckpt /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-oft-dfs-17M-qwen/hf_model/global_step_3100 \
#         --num 200 \
#         --offset 0 \
#         --data val1_b4_t100_n500000_dfs.json \
#         --temperature 0.7 \
#         --batch_size 1 \
#         --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs \
#         --gens ${k} \
#         --ctx 512 
# done
