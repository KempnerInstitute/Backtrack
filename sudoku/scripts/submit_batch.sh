#!/bin/sh

# data generation script
# Loop through start_idx from 0 to 3,000,000 with increments of 100,000
for ((start_idx=2000000; start_idx<3000000; start_idx+=100000)); do 
    # Calculate end_idx
    end_idx=$((start_idx + 100000))
    
    # Export variables so they are available in the SLURM job
    export start_idx
    export end_idx
    sbatch gen_data.sh $start_idx $end_idx
    sleep 1
done


# offsets=(0 100)
# for offset in ${offsets[@]}; do
#     export offset
#     sbatch eval.sh $offset
#     sleep 1
# done

