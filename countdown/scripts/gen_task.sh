#!/bin/sh

#SBATCH -c 24 # Number of cores requested
#SBATCH -t 0-03:00 # Runtime in minutes
#SBATCH -p kempner # Partition to submit to
#SBATCH --mem=64G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o ../slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e ../slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_dam_lab

module purge
module load Mambaforge
module load cuda cudnn
mamba activate reason

cd ../src

set -x 
# python countdown_generate.py --seed 4 --data_dir data/b4_3_random/ --min_range 4 --start_range 4 --num_samples 500000

# # generate BFS data with different beam width heuristics 
# python countdown_generate.py --seed 4 --data_dir data/b4_3_bfs/ --min_range 4 --start_range 4 --num_samples 20000 --search bfs

# # DFS: currently only ablate on the type of heuristic used while fixing the pruning threshold to target
# python countdown_generate.py --seed 4 --data_dir data/b4_3_dfs/ --min_range 4 --start_range 4 --num_samples 20000 --search dfs

# generate DFS training data
# python countdown_generate.py --seed 4 --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs/ --min_range 4 --start_range 4 --num_samples 500000 --search dfs

# generate DFS-hint training data
# python countdown_generate.py --seed 4 --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs_hint --min_range 4 --start_range 4 --num_samples 500000 --search dfs_hint

# generate deep-countdown training data
python countdown_deep_generate.py --seed 4 --offset 4 --data_dir /n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs_deep --min_range 8 --start_range 8 --num_samples 100000 --search dfs


