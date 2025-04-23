import os
import json
import argparse
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import Qwen2ForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

from tokenization import data_postprocessing, data_preprocessing, generate_mapping
from solver import check_sudoku_solution, convert_full_trace_shortcut_solution_fuzzy, convert_full_trace_shortcut_solution

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--ckpt", type=str, help="path to checkpoint")
parser.add_argument("-n", "--num",type=int, default=10)
parser.add_argument("-o", "--offset",type=int, default=0)
parser.add_argument("--data_dir", type=str, default="data/")
parser.add_argument("-d", "--data",type=str, default="val_b3_t100_n100000_random.json")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--ctx", type=int, default=4096)
parser.add_argument("--gens", type=int, default=1)

# set torch floating point precision
torch.set_default_dtype(torch.bfloat16)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def eval_ll(model ,tokenizer, data, gen_len=4096, temperature=0.0, n=1):
    output_texts = []
    for b in tqdm(range(len(data))):
        batch = data[b]
        inputs = tokenizer(batch, return_tensors="pt", padding=False).to("cuda")
        inputs = inputs['input_ids']

        if n == 1:
            if temperature == 0.0:
                outputs = model.generate(input_ids=inputs, 
                                         pad_token_id=106, # need to confirm from original data where I specified PAD token
                                         attention_mask=torch.ones_like(inputs), 
                                         max_length=gen_len, 
                                         num_beams=1, 
                                         do_sample=False
                                         )
                
            else:
                outputs = model.generate(input_ids=inputs, 
                                         pad_token_id=106,
                                         attention_mask=torch.ones_like(inputs),  
                                         max_length=gen_len, 
                                         num_beams=1, 
                                         do_sample=True, 
                                         temperature=temperature
                                         )
            
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            output_texts.extend(output_text)
        
        if n > 1:
            assert temperature > 0.0
            # replicate inputs n times
            inputs = inputs.repeat(n, 1)
            outputs = model.generate(input_ids=inputs, 
                                     pad_token_id=106,
                                     attention_mask=torch.ones_like(inputs),  
                                     max_length=gen_len, 
                                     num_beams=1, 
                                     do_sample=True, 
                                     temperature=temperature
                                     )
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            output_texts.append(output_text)

    return output_texts


def convert_sudoku_str_to_arr(sudoku_str):
    sudoku_array = (sudoku_str.replace("START ", "")
                  .replace("END ", "")
                  .replace("SOL_S", "")
                  .replace("SOL_E", "")
                  .replace("solving", "")
                  .strip()).split("\n")

    return sudoku_array

def fill_in_cells(sudoku_array, solution_array):
    for step in solution_array:
        if "guess" in step or "revert" in step:
            raise NotImplementedError("Backtracked solution is not implemented")
        elif "=" in step:
            match = re.match(r"\((\d+), (\d+)\) = (\d+)", step)
            row, col, val = map(int, match.groups()) 
            sudoku_array[row, col] = val
        else:
            raise NotImplementedError("Backtracked solution is not implemented")

    return sudoku_array

def check_per_cell_acc(pred, target):
    pred_array = np.zeros((9, 9), dtype=int)
    pred_array = fill_in_cells(pred_array, pred)

    target_array = np.zeros((9, 9), dtype=int)
    target_array = fill_in_cells(target_array, target)
    
    # compare the two arrays
    return np.mean(pred_array == target_array)


def get_sol_accuracy(post_processed_pred, val_data):
    acc_results = []
    for i in range(len(post_processed_pred)):
        sol = post_processed_pred[i]

        # extract the string that starts with START and ends with sovling
        board = sol[sol.find("START"):sol.find("solving")]
        board = convert_sudoku_str_to_arr(board)

        # extract the string that starts with solving and ends with END
        solution = sol[sol.find("solving"):sol.find("END")]
        solution = convert_sudoku_str_to_arr(solution)
        solution = convert_full_trace_shortcut_solution_fuzzy(solution) # only needed for full-trace but shorcut doesn't care if processed
        # solution = convert_full_trace_shortcut_solution(solution)

        # check if the solution is correct
        # is_valid = check_sudoku_solution(board, solution)
        # print(is_valid)

        # compare to the target solution - this also functions as "check solution correctness"
        try:
            gt_solution = val_data["shortcut_solution"][i]
        except:
            gt_solution = val_data["strategy_shortcut_solution"][i]
        gt_solution = convert_sudoku_str_to_arr(gt_solution)
        # confirm that the solution is correct
        is_valid = check_sudoku_solution(board, gt_solution)
        assert is_valid

        per_cell_acc = check_per_cell_acc(solution, gt_solution)
        acc_results.append(per_cell_acc)
        
    return acc_results

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    if "checkpoint" not in args.ckpt:
        ckpt_files = os.listdir(args.ckpt)
        ckpt_files = [f for f in ckpt_files if "checkpoint" in f]
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("checkpoint-")[1]))
        args.ckpt = os.path.join(args.ckpt, ckpt_files[-1])
    print(f"Loading model from {args.ckpt}")

    model = Qwen2ForCausalLM.from_pretrained(args.ckpt, 
                                             torch_dtype=torch.bfloat16, 
                                             device_map="auto")
    model.eval()
    print(model)
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

    tokenizer.sol_start_token = "SOL_S "
    tokenizer.sol_end_token = "SOL_E "
    tokenizer.bos_token_w_space = "START "
    tokenizer.eos_token_w_space = "END "

    # print tokenizer vocab size
    data_file = os.path.join(args.data_dir, args.data)

    val_data = load_dataset("json", data_files=data_file)['train']

    # get only a subset of the data
    val_data = val_data.select(list(range(args.offset, args.offset + args.num)))

    # generate mapping 
    len_vocab = 23 # a hard coded number from tokenization.py
    mapping, reverse_mapping = generate_mapping(len_vocab)

    # Only need the board 
    text = [
            tokenizer.bos_token_w_space 
            + val_data["board"][e] + '\n'
            + tokenizer.sol_start_token  + '\n'
            for e in range(len(val_data))
            ]
    board_data = data_preprocessing(text, mapping)

    model_prediction =  eval_ll(model, 
                                tokenizer, 
                                board_data, 
                                gen_len=args.ctx, 
                                temperature=args.temperature, 
                                n=args.gens)

    if args.gens == 1:
        post_processed_pred = data_postprocessing(model_prediction, reverse_mapping)
        acc_results = get_sol_accuracy(post_processed_pred, val_data)
    else:
        # check each genereation 
        acc_results_gen = []
        post_processed_pred_gen = []
        for gen_i in range(args.gens):
            model_prediction_gen = [gen[gen_i] for gen in model_prediction]
        
            post_processed_pred = data_postprocessing(model_prediction_gen, reverse_mapping)
            acc_results = get_sol_accuracy(post_processed_pred, val_data)
            acc_results_gen.append(list(acc_results))
            post_processed_pred_gen.append(list(post_processed_pred))

        # then take the best effort
        acc_results = np.max(np.array(acc_results_gen), axis=0) # acc_results_gen shape (gens, test_size)
    
    
    print(f"{args.ckpt} Accuracy pass@{args.gens}: {np.mean(acc_results)}")

    # save results
    ckpt_num = args.ckpt.split("checkpoint-")[1]
    ckpt_num = f"ckpt{ckpt_num}"
    ckpt_dir = os.path.dirname(args.ckpt)
    results_file = os.path.join(ckpt_dir, f"results_{args.data.replace('/','_')}_{args.num}_{args.offset}_{ckpt_num}_@{args.gens}")

    # save results
    if args.gens == 1: # pass@1 so save only one result
        with open(results_file, "w") as f:
            json.dump({
                "ratings": list(acc_results), 
                "model_output": post_processed_pred,
                }, f, indent=4)
    else:
        with open(results_file, "w") as f:
            json.dump({
                "ratings": list(acc_results), 
                "ratings_gen": acc_results_gen,
                "model_output_gen": post_processed_pred_gen, # shape (gens, test_size)
                }, f, indent=4)
