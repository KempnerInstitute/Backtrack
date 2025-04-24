"""
Script for  generating data for the countdown task.
"""
import json
import argparse
import random
import tiktoken
import os

import tqdm

from countdown_deep import CountDownDeep
from countdown_utils import *
from countdown_deep_dfs import dfs_deep

parser = argparse.ArgumentParser()

# data args
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--data_dir", type=str, default=None, help="Directory to store data")

# countdown specific
parser.add_argument("--start_range", type=int, default=3, help="Max Range of starting numbers [M, N]")
parser.add_argument("--min_range", type=int, default=3, help="Min Range of starting numbers [M, N]")
parser.add_argument("--max_target", type=int, default=100, help="Maximum target number")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of data samples to generate")

# search args
parser.add_argument("--search", type=str, default="random", help="Search type")

# split for growth mode on or off 
parser.add_argument("--grow", action="store_true", help="grow mode on or off, only a new train set is created")
parser.add_argument("--offset", type=int, default=1, help="offset for random seed")


if __name__ == "__main__":
    args = parser.parse_args()
    # set random seed
    random.seed(args.seed)
    target_nums = [i for i in range(10, args.max_target+1)]

    # save 10% of target numbers for validation
    random.shuffle(target_nums)
    val_target_nums = target_nums[:len(target_nums)//10]
    print(val_target_nums)
    train_nums = target_nums[len(target_nums)//10:]

    if args.grow:
        splits = ["grow"]
        target_list = [train_nums]
        # to avoid reusing the same samples from the train set, change the seed
        random.seed(args.seed + args.offset)
    else:
        splits = ["train", "val_target", "val"]
        target_list = [train_nums, val_target_nums, train_nums]
        random.seed(args.seed + args.offset)

    average_token_length = {7: [], 8: [], 9: []}
    average_reward = {7: [], 8: [], 9: []}
    average_zeros = {7: [], 8: [], 9: []}
    total_samples = {7: 0, 8: 0, 9: 0}

    data_samples = {}
    
    for split, target_nums in zip(splits, target_list):

        data_samples[split] = []
        if split == "train" or split=="grow":
            num_samples = args.num_samples
        else:
            num_samples = 1000

        zero_count = 0
        for t in tqdm.tqdm(range(num_samples)):
            start_size = random.randint(args.min_range, args.start_range)
            cd = CountDownDeep(args.max_target, start_size)
            max_nodes = None
            if start_size == 7:
                max_rating = 0
                # naive calculation of first 4 max nodes: 4c2 x 4 x 3c2 x 4 x 4 = 1152
                max_rating += 1152
                # naive calculation of rest 3 max nodes: 3c2 x 4 x 4 = 48
                max_rating += 3*4*4
            elif start_size == 8:
                # two 4 by 4 games 
                max_rating = 1152 * 2
            target = random.choice(target_nums)
            nums, solution, solution_tuple = cd.generate(target)
            no_backtrack_trace = cd.convert_to_path(target, nums, solution_tuple)
            if split == "val":
                while solution in [s["solution"] for s in data_samples["train"]]:
                    target = random.choice(target_nums)
                    nums, solution = cd.generate(target)
            if args.search == "astar":
                # astar not adapted to new format
                raise NotImplementedError
            elif args.search == "dfs":
                search = dfs_deep
                heuristic = random.choice([sum_heuristic, mult_heuristic])
                search_path = dfs_deep(target, nums, heuristic=heuristic, threshold=target)

            elif args.search == 'dfs_hint':
                raise NotImplementedError
            elif args.search == "bfs":
                raise NotImplementedError
            elif args.search == "random":
                raise NotImplementedError
            else:
                raise ValueError(f"Search type {args.search} not supported")
            if "Goal Reached" in search_path:
                rating = 1. - simple_rating(search_path) / max_rating
                rating = max(0., rating)
            else:
                rating = 0.
            if rating == 0.:
                zero_count += 1

            search_type = search.__name__
            if search_type == "bfs":
                search_type += f"_{beam_size}"

            data_samples[split].append({
                "nums": nums,
                "target": target,
                "solution": solution,
                "search_path": search_path,
                "rating": rating,
                "search_type": search_type,
                "optimal_path": no_backtrack_trace,
                "heuristic": heuristic.__name__
            })
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(search_path)
            if rating == 0:
                average_zeros[start_size].append(1)
            average_token_length[start_size].append(len(tokens))
            average_reward[start_size].append(rating)
            total_samples[start_size] += 1

        print(f"Zero count: {zero_count}")
        print(f"Total samples: {total_samples}")
        print(f"average zeros: start size 7: {sum(average_zeros[7])}, start size 8: {sum(average_zeros[8])}, start size 9: {sum(average_zeros[9])}")
        print(f"average token length: start size 7: {(sum(average_token_length[7]) / total_samples[7]) if total_samples[7] else None if total_samples[7] else None}, start size 4: {(sum(average_token_length[8]) / total_samples[8]) if total_samples[8] else None}, start size 9: {(sum(average_token_length[9]) / total_samples[9]) if total_samples[9] else None}")
        print(f"average reward: start size 7: {(sum(average_reward[7]) / total_samples[7]) if total_samples[7] else None}, start size 8: {(sum(average_reward[8]) / total_samples[8]) if total_samples[8] else None}, start size 9: {(sum(average_reward[9]) / total_samples[9]) if total_samples[9] else None}")

        os.makedirs(args.data_dir, exist_ok=True)
        with open(f"{args.data_dir}/{split}{args.offset}_b{args.start_range}_t{args.max_target}_n{args.num_samples}_{args.search}.json", "w") as f:
            json.dump(data_samples[split], f, indent=4)
