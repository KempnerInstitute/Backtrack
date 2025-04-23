#!/bin/sh

#SBATCH -c 4 # Number of cores requested
#SBATCH -t 0-01:00 # Runtime in minutes
#SBATCH -p kempner # Partition to submit to
#SBATCH --mem=60G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o ../slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e ../slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_barak_lab
#SBATCH --exclude=holygpu8a19604,holygpu8a19303,holygpu8a17402,holygpu8a17402

module purge
module load Mambaforge
module load cuda cudnn
mamba activate reason

cd ../

echo "generating data" $start_idx $end_idx
python data.py $start_idx $end_idx
