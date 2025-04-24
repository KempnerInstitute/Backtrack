import sys, os
import datasets
import argparse
import copy
from datasets import load_dataset, DatasetDict
sys.path.append('/n/home05/sqin/self-correct/sudoku')
from tokenization import data_preprocessing, load_tokenizer


tokenizer, mapping, reverse_mapping = load_tokenizer()

def format_board(board):
    text = tokenizer.bos_token_w_space \
        + board + '\n'\
        + tokenizer.sol_start_token  + '\n'
               
    processed_text = data_preprocessing([text], mapping)
    return processed_text

def map_func(datum, idx):
    datum_=copy.deepcopy(datum)
    datum_["data_source"]=data_source
    datum_['prompt'] = [{
        "role": "user",
        "content": format_board(datum_["board"])
        }]
    datum_["reward_model"]={
        "style":"rule",
        "ground_truth":datum["strategy_success"]
    }
    datum_["extra_info"]={
        'split': split,
        'index': idx
    }
    # remove solution, search_path from datum_
    del datum_["strategy_full_trace_solution"]
    del datum_["strategy_shortcut_solution"]

    return datum_

def push_data_to_hf():
    # Load dataset splits
    data_dir = "/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/strategy_data_fixed"
    train_file = os.path.join(data_dir, "sudoku_strategy_data_0_100000.json")
    val_file = os.path.join(data_dir, "sudoku_strategy_data_sample.json")
    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "test": val_file}
    )

    # only take the first 200 val examples
    dataset['train'] = dataset['train'].select(list(range(2000)))
    dataset['test'] = dataset['test'].select(list(range(200)))

    # Print dataset structure
    print(dataset)

    # Upload dataset to Hugging Face Hub
    repo_id = "sunnytqin/sudoku"
    dataset.push_to_hub(repo_id)

    print(f"✅ Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/n/home05/sqin/self-correct/verl/my_scripts/sudoku')
    args = parser.parse_args()

    push_data_to_hf()

    data_source = 'sunnytqin/sudoku'
    dataset = datasets.load_dataset(data_source)

    splits=["train", "test"]

    for split in splits:
        ds=dataset[split]
        ds=ds.map(function=map_func, with_indices=True, load_from_cache_file=False)
        local_dir=args.local_dir
        ds.to_parquet(os.path.join(local_dir,f"grpo_{split}.parquet"))
        ds.to_json(os.path.join(local_dir,f"grpo_{split}.json"))