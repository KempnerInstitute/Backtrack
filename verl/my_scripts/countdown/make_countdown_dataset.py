import os
import datasets
import argparse
import copy
from datasets import load_dataset, DatasetDict


# change HF defaul cache dir
os.environ["HF_HOME"] = "/n/netscratch/dam_lab/Lab/sqin/"

def map_func(datum, idx):
    datum_=copy.deepcopy(datum)
    datum_["data_source"]=data_source
    # datum_["prompt"]=datum['search_path'].split('\n')[0]
    datum_['prompt'] = [{
        "role": "user",
        "content": "START "+datum['search_path'].split('\n')[0]
        }]
    datum_["reward_model"]={
        "style":"rule",
        "ground_truth":datum["solution"]
    }
    datum_["extra_info"]={
        'split': split,
        'index': idx
    }
    # remove solution, search_path from datum_
    del datum_["solution"]
    del datum_["optimal_path"]
    del datum_["heuristic"]
    del datum_["search_path"]
    del datum_["rating"]

    return datum_

def push_data_to_hf():
    # Load dataset splits
    data_dir = "/n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs"
    train_file = os.path.join(data_dir, "train1_b4_t100_n500000_dfs.json")
    val_file = os.path.join(data_dir, "val1_b4_t100_n500000_dfs.json")
    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "test": val_file}
    )

    # only take the first 200 val examples
    dataset['test'] = dataset['test'].select(list(range(200)))

    # Print dataset structure
    print(dataset)

    # Upload dataset to Hugging Face Hub
    repo_id = "sunnytqin/countdown"
    dataset.push_to_hub(repo_id)

    print(f"✅ Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs')
    args = parser.parse_args()

    # push_data_to_hf()

    data_source = 'sunnytqin/countdown'
    dataset = datasets.load_dataset(data_source,
                                    cache_dir="/n/netscratch/dam_lab/Lab/sqin/",)

    splits=["train", "test"]

    for split in splits:
        ds=dataset[split]
        ds=ds.map(function=map_func, with_indices=True, load_from_cache_file=False)
        local_dir=args.local_dir
        ds.to_parquet(os.path.join(local_dir,f"grpo_{split}.parquet"))
        # ds.to_json(os.path.join(local_dir,f"grpo_{split}.json"))