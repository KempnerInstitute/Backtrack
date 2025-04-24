
import sys, os
sys.path.append("/n/home05/sqin/self-correct/stream-of-search/src")
from countdown_utils import parse_trajectory_qwen, metric_fn

def compute_score(solution_str, ground_truth) -> float:
    processed_prediction = parse_trajectory_qwen(solution_str) # need to fix spacing issues 
    rating, reason = metric_fn(processed_prediction, mode="sft")
    correct = int(rating > 0)
    return correct