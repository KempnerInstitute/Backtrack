
import sys, os

def compute_score(solution_str, ground_truth) -> float:
    print("solution_str: ", solution_str)
    print("ground_truth: ", ground_truth)
    return len(solution_str) / len(ground_truth)