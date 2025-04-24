set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo \
    /n/home05/sqin/self-correct/verl/my_scripts/sudoku/grpo_sudoku.yaml