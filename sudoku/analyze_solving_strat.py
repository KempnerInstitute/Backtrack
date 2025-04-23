import numpy as np
import matplotlib.pyplot as plt
from analyze_results import load_results
import json 

def compare_dfs_sft_oft():
    # load data 
    dfs_data_path = "/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/strategy_data_fixed/sudoku_strategy_data_sample.json"
    with open(dfs_data_path, "r") as f:
        dfs_data = json.load(f)
    dfs_search_num = np.zeros(200)
    for i in range(200):
        dfs_search_num[i] = (
            dfs_data[i]["strategy_full_trace_solution"].count("\nrevert")
        )
    dfs_search_num = np.array(dfs_search_num)
    # clip the search number to 0-200
    # dfs_search_num = np.clip(dfs_search_num, 0, 200)

    # load 38M SFT result
    sft_strat_file = "/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/outputs/38M-sft-strat-fixed/results_sudoku_data_sample.json_200_0_ckpt797220_@1"
    sft_strat_result, sft_strat_model_output = load_results(sft_strat_file)
    
    sft_search_num = np.zeros(200)
    for i in range(200):
        # find number of "revert" in the model output
        if sft_strat_result[i] > 0.99:
            model_output_i = sft_strat_model_output[i].split("SOL_E")[0]
            sft_search_num[i] = (
                model_output_i.count("\nrevert")
            )
    sft_search_num = np.array(sft_search_num)
    print(sft_search_num)

    # load 38M OFT result
    gens = [1, 4, 16, 32, 64]
    oft_search_num = np.zeros(200)
    for g in gens:
        oft_strat_file = f"/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/outputs/38M-oft-strat/results_sudoku_data_sample.json_200_0_ckpt160000_@{g}"
        oft_strat_result, oft_strat_model_output = load_results(oft_strat_file)
        for i in range(200):
            if oft_search_num[i] == 0 and oft_strat_result[i] > 0.99:
                oft_search_num[i] = g    

    sort_idx = np.argsort(dfs_search_num)
    dfs_search_num = dfs_search_num[sort_idx]
    sft_search_num = sft_search_num[sort_idx]
    oft_search_num = oft_search_num[sort_idx]
    # normalize the search number to 0-1
    # plot the results
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("Set1")
    # only plot the non-zero values
    sft_search_num[sft_search_num == 0] = np.nan
    # oft_search_num[oft_search_num == 0] = np.nan
    plt.scatter(range(200), dfs_search_num, label='DFS Backtracking Trace', color=cmap(0))
    plt.scatter(range(200), sft_search_num, label='Backtracking Model', color=cmap(1))  
    plt.scatter(range(200), oft_search_num, label='No-Backtacking Model', color=cmap(2))
    plt.ylabel("Number of Searches (Normalized)", fontsize=15)
    plt.xlabel("Test Sample Index", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([-1, 40])
    plt.xlim([-5, 205])
    plt.legend()
    plt.savefig("sudoku_dfs_vs_oft_search.pdf")
    return


if __name__ == "__main__":
    compare_dfs_sft_oft()


