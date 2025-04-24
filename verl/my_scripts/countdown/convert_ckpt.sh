# convert GRPO'ed SFT model
python3 scripts/merge_state_dicts.py \
        /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-sft-dfs-17M-qwen \
        --dest_dir /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-sft-dfs-17M-qwen/ \
        --specify_step 350

python3 scripts/save_hf.py\
    /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-sft-dfs-17M-qwen/state_dict_0.pt \
    --model_name /n/netscratch/dam_lab/Lab/sqin/reason/sos/sft-countdown-hint5e5-17M-qwen/checkpoint-336920 \
    --save_path /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-sft-dfs-17M-qwen//hf_model/global_step_350


# convert GRPO'ed OFT model
python3 scripts/merge_state_dicts.py \
        /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-oft-dfs-17M-qwen \
        --dest_dir /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-oft-dfs-17M-qwen \
        --specify_step 3100

python3 scripts/save_hf.py\
    /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-oft-dfs-17M-qwen/state_dict_0.pt \
    --model_name /n/netscratch/dam_lab/Lab/sqin/reason/sos/oft-countdown-hint3e5-17M-qwen/checkpoint-178140 \
    --save_path /n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-oft-dfs-17M-qwen/hf_model/global_step_3100
