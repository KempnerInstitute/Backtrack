import os
import json
import random
import argparse

import tqdm

import numpy as np
import torch
from transformers import pipeline, PreTrainedTokenizerFast, AutoModelForCausalLM, GPTNeoForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset

from countdown_utils import *
from countdown_bfs import bfs
from countdown_dfs import dfs

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

def find_answer_len(output_text):
    goal_lines = re.finditer(r"\d+,\d+ equal: Goal Reached", output_text)
    goal_lines = list(goal_lines)
    if not goal_lines:
        print("WARN: No goal reached")
        print(output_text)
        return 0
    else:
        return goal_lines[0].end()


def eval_ll(model, tokenizer, data, batch_size=128, context_len=4096, temperature=0.0, n=1):
    """
    Evaluate the model on the data using a sliding window so that the context length is not exceeded
    """
    output_texts_concat = []
    # answer_len_all = []
    all_ratings_concat = []
    all_outputs_concat = []
    for b in tqdm.trange(0, len(data), batch_size):
        batch = data[b:min(b+batch_size, len(data))]
        output_texts = ["" for _ in range(len(batch))]
        inputs = tokenizer(batch, padding=False, truncation=False, return_tensors="pt", return_attention_mask=True).to("cuda")
        inputs = inputs['input_ids']
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):

        if n == 1:
            if temperature == 0.0:
                outputs = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_length=context_len, num_beams=1, do_sample=False)
            else:
                outputs = model.generate(input_ids=inputs, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs), max_length=context_len, num_beams=1, do_sample=True, temperature=temperature)
            # split output vector into first N tokens and the rest
            output_tokens = outputs
            output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
            tokenizer.padding_side = "left"
            output_texts = [ot + ot_now for ot, ot_now in zip(output_texts, output_text)]

            # print token lens of tokenized outputs
            # print([len(tokenizer(ot)['input_ids']) for ot in output_texts])
            
            # for i in range(len(output_texts)): # estimate average lengths 
            #     answer_len = find_answer_len(output_texts[i])
            #     tokenized_output = tokenizer(output_texts[i][:answer_len])
            #     answer_len_all.append(len(tokenized_output['input_ids']))
            output_texts_concat += output_texts
            
        elif n > 1:
            assert temperature > 0.0, "Temperature must be greater than 0 for sampling"
            
            # old: parallel generation using a for loop
            # all_outputs = []
            # all_ratings = []
            # for i in range(n):
            #     outputs = model.generate(input_ids=inputs, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs), max_length=context_len, do_sample=True, temperature=temperature)
            #     output_tokens = outputs
            #     output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
            #     tokenizer.padding_side = "left"
            #     # get rating for each output
            #     if "qwen" in args.ckpt:
            #         ratings = [metric_fn_select(parse_trajectory_qwen(ot), mode="sft")[0] for ot in output_text]
            #     else: # for GPTNeo
            #         ratings = [metric_fn_select(ot.split(tokenizer.bos_token)[1], mode="sft")[0] for ot in output_text]
                
            #     all_ratings.append(ratings)
            #     all_outputs.append(output_text)

            # instead, we can batch it
            inputs = inputs.repeat(n, 1)
            outputs = model.generate(input_ids=inputs, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs), max_length=context_len, do_sample=True, temperature=temperature)
            output_tokens = outputs
            output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)     
            all_ratings = [[metric_fn_select(parse_trajectory_qwen(ot), mode="sft")[0]] for ot in output_text]
            all_outputs = [[o] for o in output_text]  
            # only keep the output with the highest rating for each input
            all_ratings = np.array(all_ratings)

            print(f"average rating across attemps", np.mean(all_ratings)) 
            # all ratings is n x batch_size
            max_ratings = np.argmax(all_ratings, axis=0)
            max_rating_vals = np.max(all_ratings, axis=0)
            print(f"average of max ratings", np.mean(max_rating_vals))
            # max ratings is batch_size, output_texts is n x batch_size
            output_texts = [all_outputs[max_r][i] for i, max_r in enumerate(max_ratings)]
            output_texts_concat += output_texts
            # all ratings
            all_ratings_concat.append(list(all_ratings.flatten()))
            all_outputs_concat.append(all_outputs)
        else:
            raise ValueError(f"Invalid n: {n}")
    
        # answer_len_all = np.array(answer_len_all)
        # # remove zero-elements
        # print("total answer len:", len(answer_len_all))
        # print("no answer len: ", np.sum(answer_len_all == 0))
        # answer_len_all = answer_len_all[answer_len_all != 0]
        # print("averge answer len: ", np.mean(answer_len_all))
        # print("  max answer len: ", np.max(answer_len_all))
        # print("  min answer len: ", np.min(answer_len_all))
        # print("  std answer len: ", np.std(answer_len_all))

    return output_texts_concat, (all_ratings_concat, all_outputs_concat) 

args = parser.parse_args()
# set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(args)
if 'qwen' in args.ckpt:
    model = AutoModelForCausalLM.from_pretrained(args.ckpt,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 )
    
    model.eval()
    # an ugly way to load the tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="countdown_tokenizer.json")
    tokenizer.pad_token = "PAD"
    tokenizer.bos_token = " START "
    tokenizer.eos_token = " END "

else:
    model = GPTNeoForCausalLM.from_pretrained(args.ckpt, 
                                            torch_dtype=torch.bfloat16,
                                            device_map="auto",
                                            )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, padding_side='left')
    # tokenizer.pad_token = tokenizer.eos_token

data_file = os.path.join(args.data_dir, args.data)

with open(data_file, "r") as json_file:
    data = json.load(json_file)

# define metric_fn based on ckpt
if "deep" in args.ckpt:
    from countdown_deep_utils import metric_fn_deep as metric_fn_select
else:
    from countdown_utils import metric_fn as metric_fn_select

# rate true outputs
print("first true ratings...")
true_rating = []
for i in range(len(data[args.offset: args.offset + args.num])):
    search_path = data[i]['search_path']

    ''' control search_path by tokenized length '''
    tokenized_search_path = tokenizer(search_path, return_tensors="pt", padding=False)['input_ids']
    # if len(tokenized_search_path[0]) > 4096:
    #     print(len(tokenized_search_path[0]))
    tokenized_search_path = tokenized_search_path[0][0: args.ctx]
    search_path = tokenizer.decode(tokenized_search_path)
    # for qwen tokenizer, also process the search path
    if 'qwen' in args.ckpt:
        search_path = parse_trajectory_qwen(search_path)
   
    tr, reason = metric_fn_select(f"{search_path}", mode="sft")
    true_rating.append(tr)
print(f"Average true rating: {np.mean(true_rating)}")
print(f"True Accuracy: {np.mean([r > 0 for r in true_rating])}")

predictions = []
pred_ratings = []
pred_reasons = []
tokenizer.padding_side = "left"

# generation test prompt from scratch
# test_prompts = [tokenizer.bos_token + f"Current State: {sample['target']}:{sample['nums']}, Operations: []"  for sample in data[args.offset: args.offset + args.num]]

# a more generic approach: generate from solution trace
if "oft" in args.ckpt: 
    path = 'optimal_path'
elif "sft" in args.ckpt:
    path = 'search_path'
else:
    raise ValueError(f"Invalid ckpt: {args.ckpt}")
test_prompts = [tokenizer.bos_token + sample[path].split("\n")[0] for sample in data[args.offset: args.offset + args.num]]

len_nums = [len(sample['nums']) for sample in data[args.offset: args.offset + args.num]]
if "deep" in args.ckpt:
    data_l = [d for d, l in zip(test_prompts, len_nums) if l == 8]
else:
    data_l = [d for d, l in zip(test_prompts, len_nums) if l == 4]
print(f"Number of samples with 4 numbers: {len(data_l)} out of {len(test_prompts)}")
predictions, (all_ratings, all_outputs) = eval_ll(model, tokenizer, data_l, batch_size=args.batch_size, context_len=args.ctx, temperature=args.temperature, n=args.gens)

# rate outputs
for i in range(len(predictions)):
    # "mode" only matters when using "dt" - sft and oft are the same
    # for Qwen
    if 'qwen' in args.ckpt:
        processed_prediction = parse_trajectory_qwen(predictions[i]) # need to fix spacing issues 
        rating, reason = metric_fn_select(processed_prediction, mode="sft")
    else: # for GPTNeo 
        rating, reason = metric_fn_select(predictions[i].split(tokenizer.bos_token)[1], mode="sft")
    
    pred_ratings.append(rating)
    pred_reasons.append(reason)

# get max rating for each sample with its index
pred_ratings = np.array(pred_ratings)

# print results
print("Results Summary:")
print(f"Average rating: {np.mean(pred_ratings):.2f}")
print(f"Average true rating: {np.mean(true_rating):.2f}")
print(f"Accuracy: {np.mean([r > 0 for r in pred_ratings]):.2f}")
print(f"True Accuracy: {np.mean([r > 0 for r in true_rating]):.2f}")

ckpt_dir = os.path.dirname(args.ckpt)
# save results
results_file = os.path.join(ckpt_dir, f"results_{args.data.replace('/','_')}_{args.num}_{args.offset}_seed{args.seed}_@{args.gens}")
with open(results_file, "w") as f:
    json.dump({"trajectories": predictions, 
               "ratings": pred_ratings.tolist(), 
               "reasons": pred_reasons,
               "true_ratings": true_rating,
               "all_ratings": all_ratings,
               "all_outputs": all_outputs,
               }, f, indent=4)
