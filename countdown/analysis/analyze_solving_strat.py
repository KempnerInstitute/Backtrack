import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import sys
sys.path.append("/n/home05/sqin/self-correct/stream-of-search")
from src.countdown_deep_utils import metric_fn_deep
from analyze_results import load_results
import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix

def get_qwen_tokenizer():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="/n/home05/sqin/self-correct/stream-of-search/src/countdown_tokenizer.json")
    tokenizer.pad_token = "PAD"
    tokenizer.bos_token = " START "
    tokenizer.eos_token = " END "
    return tokenizer

def iou(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return intersection / union if union > 0 else 0.0

def analyze_bfs_data():
    tokenizer = get_qwen_tokenizer()
    data_path = "/n/home05/sqin/self-correct/stream-of-search/src/data/b4_3_bfs/train1_b4_t100_n20000_bfs.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    # for each data entry record heuristic used, search path, rating, beam_width
    heuristic = []
    solution_len = []
    acc = []
    beam_width = []
    for d in data:
        heuristic.append(d["heuristic"])
        solution = d["search_path"]
        tokenized_solution = tokenizer.encode(solution)
        solution_len.append(len(tokenized_solution))
        acc.append(d["rating"] > 0)
        beam_width.append(int(d["search_type"].split("_")[-1]))
    
    # for each beam width, plot average solution length versus accuracy, grouped by heuristic
    df = pd.DataFrame({"heuristic": heuristic, "solution_len": solution_len, "acc": acc, "beam_width": beam_width})
    plt.figure()
    for h in df["heuristic"].unique():
        for b in df["beam_width"].unique():
            df_hb = df[(df["heuristic"] == h) & (df["beam_width"] == b)]
            avg_sol_len = df_hb["solution_len"].mean()
            acc = df_hb["acc"].mean()
            if h == "sum_heuristic":
                plt.scatter(avg_sol_len, acc, label=f"{h}_{b}", color="red")
            else:
                plt.scatter(avg_sol_len, acc, label=f"{h}_{b}", color="blue")
    plt.xlabel("Average Solution Length")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("BFS Search Performance: Average Solution Length vs. Accuracy")
    plt.savefig("bfs_summary_stats.png")

    # for both heuristics, plot cumulative acc over solution length for each beam width
    plt.figure(figsize=(12, 6))
    for h in ["sum_heuristic", "mult_heuristic"]:
        plt.subplot(1, 2, 1 if h == "sum_heuristic" else 2)
        for b in [1, 2, 3, 4, 5]:
            df_b = df[(df["heuristic"] == h) & (df["beam_width"] == b)]
            df_b = df_b.sort_values("solution_len")
            df_b["cum_acc"] = df_b["acc"].cumsum() / len(df_b)
            plt.plot(df_b["solution_len"], df_b["cum_acc"], label=f"{h}_bw={b}")
        plt.xlabel("Solution Length")
        plt.ylabel("Cumulative Accuracy")   
        # also plot for all beam widths
        df_h = df[df["heuristic"] == h]
        df_h = df_h.sort_values("solution_len")
        df_h["cum_acc"] = df_h["acc"].cumsum() / len(df_h)
        plt.plot(df_h["solution_len"], df_h["cum_acc"], label=f"{h}_all", color="black")
        plt.legend()
    plt.suptitle("BFS Search Performance: Cumulative Accuracy over Solution Length")
    plt.savefig("bfs_len_acc.png")

    # save df
    df.to_csv("bfs_data.csv") 

    return 

def analyze_dfs_data():
    tokenizer = get_qwen_tokenizer()
    data_path = "/n/home05/sqin/self-correct/stream-of-search/src/data/b4_3_dfs/train1_b4_t100_n20000_dfs.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    # for each data entry record heuristic used, search path, rating, beam_width
    heuristic = []
    solution_len = []
    acc = []
    for d in data:
        heuristic.append(d["heuristic"])
        solution = d["search_path"]
        tokenized_solution = tokenizer.encode(solution)
        solution_len.append(len(tokenized_solution))
        acc.append(d["rating"] > 0)
    
    len_cutoff = [256, 512, 1024, 2048, 4096, 8192]
    for cutoff in len_cutoff:
        idx = np.where(np.array(solution_len) < cutoff)[0]
        # Number of samples solved within cutoff
        solved_idx = np.where(np.array(acc)[idx] == True)[0]
        print(f"Cutoff: {cutoff}, Number of Samples Solved: {len(solved_idx)}")
    
    # for both heuristics, plot cumulative acc over solution length
    df = pd.DataFrame({"heuristic": heuristic, "solution_len": solution_len, "acc": acc})
    plt.figure(figsize=(6, 6))
    for h in ["sum_heuristic", "mult_heuristic"]:
        df_h = df[df["heuristic"] == h]
        df_h = df_h.sort_values("solution_len")
        df_h["cum_acc"] = df_h["acc"].cumsum() / len(df_h)
        plt.plot(df_h["solution_len"], df_h["cum_acc"], label=f"{h}")
        plt.xlabel("Solution Length")
        plt.ylabel("Cumulative Accuracy")
        plt.xlim([0, 10_000]) 
        plt.legend()
    # plot an average for both heuristics
    df = df.sort_values("solution_len")
    df["cum_acc"] = df["acc"].cumsum() / len(df)
    plt.plot(df["solution_len"], df["cum_acc"], label="all", color="black")
    plt.legend()
    plt.title("DFS Search Performance: Cumulative Accuracy over Solution Length")
    plt.savefig("dfs_len_acc.png")  

    # save df
    df.to_csv("dfs_data.csv")
    return 

def analyze_dfs_hint_data():
    tokenizer = get_qwen_tokenizer()
    data_path = "/n/home05/sqin/self-correct/stream-of-search/src/data/b4_3_dfs_hint/train1_b4_t100_n20000_dfs_hint.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    # for each data entry record heuristic used, search path, rating, beam_width
    heuristic = []
    solution_len = []
    acc = []
    for d in data:
        heuristic.append(d["heuristic"])
        solution = d["search_path"]
        tokenized_solution = tokenizer.encode(solution)
        assert 42 not in tokenized_solution
        solution_len.append(len(tokenized_solution))
        acc.append(d["rating"] > 0)
    # for both heuristics, plot cumulative acc over solution length
    df = pd.DataFrame({"heuristic": heuristic, "solution_len": solution_len, "acc": acc})
    plt.figure(figsize=(6, 6))
    for h in ["sum_heuristic", "mult_heuristic"]:
        df_h = df[df["heuristic"] == h]
        df_h = df_h.sort_values("solution_len")
        df_h["cum_acc"] = df_h["acc"].cumsum() / len(df_h)
        plt.plot(df_h["solution_len"], df_h["cum_acc"], label=f"{h}")
        plt.xlabel("Solution Length")
        plt.ylabel("Cumulative Accuracy")
        plt.xlim([0, 10_000]) 
        plt.legend()
    # plot an average for both heuristics
    df = df.sort_values("solution_len")
    df["cum_acc"] = df["acc"].cumsum() / len(df)
    plt.plot(df["solution_len"], df["cum_acc"], label="all", color="black")
    plt.legend()
    plt.title("DFS-hint Search Performance: Cumulative Accuracy over Solution Length")
    plt.savefig("dfs_hint_len_acc.png")  

    # save df
    df.to_csv("dfs_hint_data.csv")
    return 

def analyze_deep_dfs_data():
    tokenizer = get_qwen_tokenizer()
    data_path = "/n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs_deep/val0_b8_t100_n100000_dfs.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    # for each data entry record heuristic used, search path, rating, beam_width
    heuristic = []
    solution_len = []
    acc = []
    for d in data:
        heuristic.append(d["heuristic"])
        solution = d["search_path"]
        tokenized_solution = tokenizer.encode(solution)
        solution_len.append(len(tokenized_solution))
        rating, reason = metric_fn_deep(solution)
        acc.append(rating > 0)
        # acc.append(d["rating"] > 0)
    # for both heuristics, plot cumulative acc over solution length
    df = pd.DataFrame({"heuristic": heuristic, "solution_len": solution_len, "acc": acc})
    plt.figure(figsize=(6, 6))
    for h in ["sum_heuristic", "mult_heuristic"]:
        df_h = df[df["heuristic"] == h]
        df_h = df_h.sort_values("solution_len")
        df_h["cum_acc"] = df_h["acc"].cumsum() / len(df_h)
        plt.plot(df_h["solution_len"], df_h["cum_acc"], label=f"{h}")
        plt.xlabel("Solution Length")
        plt.ylabel("Cumulative Accuracy")
        plt.xlim([0, 10_000]) 
        plt.legend()
    # plot an average for both heuristics
    df = df.sort_values("solution_len")
    df["cum_acc"] = df["acc"].cumsum() / len(df)
    plt.plot(df["solution_len"], df["cum_acc"], label="all", color="black")
    plt.legend()
    plt.title("DFS Deep Search Performance: Cumulative Accuracy over Solution Length")
    plt.savefig("dfs_deep_len_acc.png")  

    # save df
    df.to_csv("dfs_deep_data.csv")
    return 

def compare_bfs_dfs():
    dfs_data = pd.read_csv("dfs_data.csv")
    bfs_data = pd.read_csv("bfs_data.csv")   
    dfs_hint_data = pd.read_csv("dfs_hint_data.csv") 

    # sort data by solution length
    dfs_data = dfs_data.sort_values("solution_len")
    bfs_data = bfs_data.sort_values("solution_len")
    dfs_hint_data = dfs_hint_data.sort_values("solution_len")

    bfs_data['cum_acc'] = bfs_data['acc'].cumsum() / len(bfs_data)
    dfs_data['cum_acc'] = dfs_data['acc'].cumsum() / len(dfs_data)
    dfs_hint_data['cum_acc'] = dfs_hint_data['acc'].cumsum() / len(dfs_hint_data)

    # plot cumulative accuracy over solution length for each heuristic
    plt.figure(figsize=(8, 5))
    plt.plot(dfs_data["solution_len"], dfs_data["cum_acc"], label="DFS", color="red")
    plt.plot(bfs_data["solution_len"], bfs_data["cum_acc"], label="BFS", color="blue")
    plt.plot(dfs_hint_data["solution_len"], dfs_hint_data["cum_acc"], label="DFS-Hint", color="green")
    
    plt.xlim([200, 10_000])
    plt.xscale("log")
    plt.xticks([256, 512, 1024, 2048, 4096, 8192], ["256", "512", "1024", "2048", "4096", "8192"])
    plt.xlabel("Solution Length")
    plt.ylabel("Cumulative Accuracy")
    plt.legend()
    plt.title("DFS vs. BFS Search Performance: Cumulative Accuracy over Solution Length")
    plt.savefig("dfs_hint_vs_bfs_len_acc.png")

    return

def compare_dfs_and_bfs_hist():
    # Apply Seaborn style
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.5)

    # Load data
    bfs_data = pd.read_csv("bfs_data.csv")   
    dfs_data = pd.read_csv("dfs_data.csv")   
    # only take solution_len > 256
    # dfs_data = dfs_data[dfs_data["solution_len"] > 256]
    # only heuristic = sum_heuristic
    # dfs_data = dfs_data[dfs_data["heuristic"] == "mult_heuristic"]

    # Set up colors
    colors = {True: sns.color_palette("Set2")[0], False: sns.color_palette("Set2")[1]}

    # Parameters
    num_bins = 50
    beam_widths = [1, 2, 4]
    total_plots = len(beam_widths) + 1

    # Global bin edges (log-spaced)
    combined = pd.concat([bfs_data[['solution_len']], dfs_data[['solution_len']]])
    min_len = max(1, combined['solution_len'].min())
    max_len = combined['solution_len'].max()
    bin_edges = np.logspace(np.log10(min_len), np.log10(max_len), num_bins + 1)

    # Create subplots
    fig, axes = plt.subplots(total_plots, 1, figsize=(10, 2.2 * total_plots),
                            sharex=True, sharey=True, gridspec_kw={'hspace': 0.1})

    # Ensure iterable
    if total_plots == 1:
        axes = [axes]

    # BFS subplots
    for ax, bw in zip(axes[:-1], beam_widths):
        subset = bfs_data[bfs_data['beam_width'] == bw]

        true_counts, _ = np.histogram(subset[subset['acc'] == True]['solution_len'], bins=bin_edges)
        false_counts, _ = np.histogram(subset[subset['acc'] == False]['solution_len'], bins=bin_edges)

        ax.bar(bin_edges[:-1], false_counts, width=np.diff(bin_edges), align='edge',
            color=colors[False], label='Incorrect' if bw == beam_widths[0] else None)
        ax.bar(bin_edges[:-1], true_counts, width=np.diff(bin_edges), align='edge',
            bottom=false_counts, color=colors[True], label='Correct' if bw == beam_widths[0] else None)

        ax.text(0.98, 0.85, f' BFS Beam Width = {bw}', transform=ax.transAxes,
                ha='right', va='top', fontsize=18, color='black')
        ax.set_xscale('log')
        ax.set_yticks([1000, 2000])
        ax.set_yticklabels(['1K', '2K'])
        ax.tick_params(axis='both', labelsize=16)
        if bw == 1:
            ax.legend(loc='upper left', frameon=True, fontsize=14)
    # DFS plot
    ax = axes[-1]
    true_counts, _ = np.histogram(dfs_data[dfs_data['acc'] == True]['solution_len'], bins=bin_edges)
    false_counts, _ = np.histogram(dfs_data[dfs_data['acc'] == False]['solution_len'], bins=bin_edges)

    ax.bar(bin_edges[:-1], false_counts, width=np.diff(bin_edges), align='edge',
        color=colors[False], label='Incorrect')
    ax.bar(bin_edges[:-1], true_counts, width=np.diff(bin_edges), align='edge',
        bottom=false_counts, color=colors[True], label='Correct')

    ax.text(0.98, 0.85, 'DFS', transform=ax.transAxes,
            ha='right', va='top', fontsize=18, color='black')
    ax.set_xscale('log')
    ax.set_yticks([500, 2000])
    ax.set_yticklabels(['1K', '2K'])
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel('Search Length (log scale)', fontsize=20)

    # Shared y-label
    fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical', fontsize=20)

    # Final layout
    sns.despine()
    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Leave space for shared y-label
    plt.savefig("countdown_search_strategy_comparison.pdf", bbox_inches='tight')


    return    

def compare_dfs_oft_sft():
    tokenizer = get_qwen_tokenizer()

    # Load DFS data
    dfs_data_path = "/n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs/val1_b4_t100_n500000_dfs.json"
    with open(dfs_data_path, "r") as f:
        dfs_data = json.load(f)

    solution_len, acc, num_searches = [], [], []
    for d in dfs_data[:200]:
        tokenized_solution = tokenizer.encode(d["search_path"])
        solution_len.append(len(tokenized_solution))
        num_searches.append(d["search_path"].count("unequal: No Solution") + 1)
        acc.append(d["rating"] > 0)

    dfs_search_num = np.array(num_searches)
    dfs_acc = np.array(acc)
    # dfs_search_num = np.where(dfs_acc, dfs_search_num, 0).astype(float)

    # Load SFT results
    sft_result_path = "/n/netscratch/dam_lab/Lab/sqin/reason/sos/sft-countdown-dfs5e5-17M-qwen/results_val1_b4_t100_n500000_dfs.json_200_0_@1"
    sft_acc, sft_model_output = load_results(sft_result_path)
    sft_search_num = np.array([
        sft_model_output[i].split("END")[0].count("unequal : No Solution") + 1
        for i in range(200)
    ])
    sft_acc = np.array(sft_acc) > 0
    # for sft_search_num only record sft_acc > 0
    # sft_search_num = np.where(sft_acc, sft_search_num, 0).astype(float)

    # Load OFT results
    oft_result_path = "/n/netscratch/dam_lab/Lab/sqin/reason/sos/oft-countdown-cd3e5-17M-qwen"
    gens = np.concatenate([np.arange(1, 17), [22, 24, 26, 28, 30, 32, 34, 36, 38, 42]])
    oft_search_num = np.zeros(200, dtype=float)
    oft_acc = np.zeros(200)
    for g in gens:
        oft_file = f'{oft_result_path}/results_val1_b4_t100_n500000_dfs.json_200_0_@{g}'
        oft_acc_g, _ = load_results(oft_file)
        for i in range(200):
            if oft_acc_g[i] > 0 and oft_search_num[i] == 0:
                oft_search_num[i] = g
                oft_acc[i] = 1
    oft_acc = np.array(oft_acc) > 0
    # replace all 0 with max-searches
    oft_search_num[oft_search_num == 0] = 42

    # Sort everything by DFS search num
    sort_idx = np.argsort(dfs_search_num)
    dfs_search_num = dfs_search_num[sort_idx]
    oft_search_num = oft_search_num[sort_idx]
    sft_search_num = sft_search_num[sort_idx]
    dfs_acc = dfs_acc[sort_idx]
    sft_acc = sft_acc[sort_idx]
    oft_acc = oft_acc[sort_idx]
    # replace 0 search num with nan
    # dfs_search_num[dfs_search_num == 0] = np.nan
    # sft_search_num[sft_search_num == 0] = np.nan
    # oft_search_num[oft_search_num == 0] = np.nan


    dfs_smoothed = gaussian_filter1d(dfs_search_num, sigma=8)
    cmap = plt.get_cmap("Set1")

    # Use more distinctive colors
    colors = {
        "dfs": "#FF6F61",   # Coral
        "sft": "#6A5ACD",   # Slate Blue
        "oft": "#20B2AA",   # Light Sea Green
    }
    sns.set_theme(style="whitegrid", context="talk")
    rcParams.update({
        'font.size': 27,
        'axes.titlesize': 22,
        'axes.labelsize': 26,
        'xtick.labelsize': 26,
        'ytick.labelsize': 24,
        'legend.fontsize': 24
    })

    # Create 3 horizontal panels
    fig, axes = plt.subplots(1, 3, figsize=(32, 8), sharex=False)
    plt.subplots_adjust(wspace=0.2)  # Reduced space between plots

    # Plot 1: DFS vs SFT
    ax1 = axes[0]
    ax1.plot(dfs_smoothed, color=colors["dfs"], alpha=0.2, lw=18)
    # plot correct samples
    correct_dfs_idx = np.where(dfs_acc)[0]
    correct_sft_idx = np.where(sft_acc)[0]
    ax1.scatter(np.array(range(200))[correct_dfs_idx], dfs_search_num[correct_dfs_idx], label="DFS Backtracking Trace", color=colors["dfs"], s=30)
    ax1.scatter(np.array(range(200))[correct_sft_idx], sft_search_num[correct_sft_idx], label="Backtracking Model", color=colors["sft"], s=30)
    
    
    # incorrect_dfs_idx = np.where(~dfs_acc)[0]
    # incorrect_sft_idx = np.where(~sft_acc)[0]
    # ax1.scatter(np.array(range(200))[incorrect_dfs_idx], dfs_search_num[incorrect_dfs_idx], color=colors["dfs"], s=30, marker='x')
    # ax1.scatter(np.array(range(200))[incorrect_sft_idx], sft_search_num[incorrect_sft_idx], color=colors["sft"], s=30, marker='x')
    
    ax1.set_xlabel("Test Sample Index")
    ax1.set_ylabel("Number of Mistakes")
    ax1.set_xlim([-5, 205])
    ax1.set_ylim([-1, 45])
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.legend()

    # Plot 2: DFS vs OFT (twin y-axis)
    ax2 = axes[1]
    ax2.plot(dfs_smoothed, color=colors["dfs"], alpha=0.2, lw=18)
    correct_dfs_idx = np.where(dfs_acc)[0]
    incorrect_dfs_idx = np.where(~dfs_acc)[0]
    ax2.scatter(np.array(range(200))[correct_dfs_idx], dfs_search_num[correct_dfs_idx], label="DFS Backtracking Trace", color=colors["dfs"], s=30)
    ax2.scatter(np.array(range(200))[incorrect_dfs_idx], dfs_search_num[incorrect_dfs_idx], color=colors["dfs"], s=30, marker='x')

    # ax2b = ax2.twinx()
    correct_oft_idx = np.where(oft_acc)[0]
    # incorrect_oft_idx = np.where(~oft_acc)[0]
    ax2.scatter(np.array(range(200))[correct_oft_idx], oft_search_num[correct_oft_idx], label="Direct Solution Model", color=colors["oft"], s=40)  
    # ax2.scatter(np.array(range(200))[incorrect_oft_idx], oft_search_num[incorrect_oft_idx], color=colors["oft"], s=40, marker='x')
    # ax2b.scatter(range(200), oft_search_num, label="No-Backtracking Model", color=colors["oft"], s=40)
    ax2.set_xlabel("Test Sample Index")
    ax2.set_ylabel("Number of Mistakes")
    ax2.set_xlim([-5, 205])
    ax2.set_ylim([-1, 45])
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # ax2b.set_ylim([-1, 45])
    # ax2b.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # ax2b.tick_params(axis='y', labelcolor=colors["oft"])
    ax2.legend(loc="upper left")

    # Plot 3: Similarity
    ax3 = axes[2]
    bins = [1, 5, 10, 20, 30, 40, 50]
    dfs_sft_sim = []
    dfs_oft_sim = []
    # for b in bins:
    for i in range(len(bins) - 1):
        b_lw = bins[i]
        b_up = bins[i + 1]
        dfs_bin = ((dfs_search_num >= b_lw) & (dfs_search_num < b_up))
        sft_bin = ((sft_search_num >= b_lw) & (sft_search_num < b_up))
        dfs_sft_sim.append(iou(dfs_bin * dfs_acc, sft_bin * sft_acc))
        
        oft_bin = ((oft_search_num >= b_lw) & (oft_search_num < b_up))
        dfs_oft_sim.append(iou(dfs_bin * dfs_acc, oft_bin * oft_acc))

    
    ax3.plot(bins[:-1], dfs_sft_sim, label="Backtracking Model", marker='o', color=colors["sft"], lw=4)
    ax3.plot(bins[:-1], dfs_oft_sim, label="Direct Solution Model", marker='o', color=colors["oft"], lw=4)
    print("oft sim:", dfs_oft_sim)
    ax3.set_xlabel("Number of Mistakes")
    ax3.set_ylabel("Similarity (Jaccard Index)")
    ax3.set_ylim([-0.05, 0.5])
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3.legend()

    sns.despine()  # Removes top and right spines
    plt.tight_layout()
    plt.savefig("countdown_combined_dfs_vs_sft_oft.pdf", bbox_inches='tight')


    # direct compare OFT and SFT
    # Create 1 panel plot
    fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=False)

    # Plot 1: SFT vs OFT
    # Sort everything by DFS search num
    sort_idx = np.argsort(sft_search_num)
    oft_search_num = oft_search_num[sort_idx]
    sft_search_num = sft_search_num[sort_idx]
    dfs_acc = dfs_acc[sort_idx]
    sft_acc = sft_acc[sort_idx]
    oft_acc = oft_acc[sort_idx]

    ax1 = axes
    sft_smoothed = gaussian_filter1d(sft_search_num, sigma=8)
    ax1.plot(sft_smoothed, color=colors["sft"], alpha=0.2, lw=18)
    # plot correct samples
    correct_sft_idx = np.where(sft_acc)[0]
    correct_oft_idx = np.where(oft_acc)[0]
    ax1.scatter(np.array(range(200))[correct_sft_idx], sft_search_num[correct_sft_idx], label="Backtracking Model", color=colors["sft"], s=30)
    ax1.scatter(np.array(range(200))[correct_oft_idx], oft_search_num[correct_oft_idx], label="Direct Solution Model", color=colors["dfs"], s=30)
    # ax1.scatter(np.array(range(200)), sft_search_num, label="Backtracking Model", color=colors["sft"], s=30)
    # ax1.scatter(np.array(range(200)), oft_search_num, label="Direct Solution Model", color=colors["dfs"], s=30)
    
    
    ax1.set_xlabel("Test Sample Index")
    ax1.set_ylabel("Number of Mistakes")
    ax1.set_xlim([-5, 205])
    ax1.set_ylim([-1, 38])
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.legend()

    plt.savefig("countdown_sft_vs_oft_tweet.png", bbox_inches='tight')

    return

def compare_sft_and_sft_short():
    from sklearn.metrics import confusion_matrix
    sft_result_path = "/n/netscratch/dam_lab/Lab/sqin/reason/sos/sft-countdown-dfs5e5-17M-qwen/results_val1_b4_t100_n500000_dfs.json_200_0_@1"
    sft_acc, sft_model_output = load_results(sft_result_path)
    sft_acc = np.array(sft_acc) > 0
    print(len(sft_acc), np.sum(sft_acc))

    sft_short_result_path = "/n/netscratch/dam_lab/Lab/sqin/reason/sos/sft-countdown-hint5e5-17M-qwen/results_val1_b4_t100_n500000_dfs.json_200_0_@1"
    sft_short_acc, sft_short_model_output = load_results(sft_short_result_path)
    sft_short_acc = np.array(sft_short_acc) > 0
    print(len(sft_short_acc), np.sum(sft_short_acc))

    # Compute confusion matrix
    cm = confusion_matrix(sft_acc, sft_short_acc, labels=[1, 0])  # [True, False]

    # Plot
    plt.figure(figsize=(3, 3))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        square=True,
        xticklabels=['Correct', 'Incorrect'],
        yticklabels=['Correct', 'Incorrect'],
        annot_kws={"size": 12}
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("Think-Backtrack Model", fontsize=14)
    ax.set_ylabel("Backtrack Model", fontsize=14)

    plt.tight_layout()
    plt.savefig("countdown_sft_short_confusion_matrix.pdf", bbox_inches='tight')
    return
    
def compare_sft_and_grpo():
    tokenizer = get_qwen_tokenizer()

    # Load DFS data
    dfs_data_path = "/n/netscratch/dam_lab/Lab/sqin/reason/data/b4_3_dfs/val1_b4_t100_n500000_dfs.json"
    with open(dfs_data_path, "r") as f:
        dfs_data = json.load(f)

    solution_len, acc, num_searches = [], [], []
    for d in dfs_data[:200]:
        tokenized_solution = tokenizer.encode(d["search_path"])
        solution_len.append(len(tokenized_solution))
        num_searches.append(d["search_path"].count("unequal: No Solution") + 1)
        acc.append(d["rating"] > 0)

    dfs_search_num = np.array(num_searches)
    dfs_acc = np.array(acc)
    # dfs_search_num = np.where(dfs_acc, dfs_search_num, 0).astype(float)

    # Load SFT results
    sft_result_path = "/n/netscratch/dam_lab/Lab/sqin/reason/sos/sft-countdown-dfs5e5-17M-qwen/results_val1_b4_t100_n500000_dfs.json_200_0_@1"
    sft_acc, sft_model_output = load_results(sft_result_path)
    sft_search_num = np.array([
        sft_model_output[i].split("END")[0].count("unequal : No Solution") + 1
        for i in range(200)
    ])
    sft_acc = np.array(sft_acc) > 0
    # for sft_search_num only record sft_acc > 0
    # sft_search_num = np.where(sft_acc, sft_search_num, 0).astype(float)

    # Load GRPO results
    grpo_result_path = "/n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-sft-dfs-17M-qwen/hf_model/results_val1_b4_t100_n500000_dfs.json_200_0_@1"
    grpo_acc, grpo_model_output = load_results(grpo_result_path)
    grpo_search_num = np.array([
        grpo_model_output[i].split("END")[0].count("unequal : No Solution") + 1
        for i in range(200)
    ])
    grpo_acc = np.array(grpo_acc) > 0

    # Sort everything by DFS search num
    sort_idx = np.argsort(dfs_search_num)
    dfs_search_num = dfs_search_num[sort_idx]
    grpo_search_num = grpo_search_num[sort_idx]
    sft_search_num = sft_search_num[sort_idx]
    dfs_acc = dfs_acc[sort_idx]
    sft_acc = sft_acc[sort_idx]
    grpo_acc = grpo_acc[sort_idx]
    # replace 0 search num with nan
    # dfs_search_num[dfs_search_num == 0] = np.nan
    # sft_search_num[sft_search_num == 0] = np.nan
    # oft_search_num[oft_search_num == 0] = np.nan


    dfs_smoothed = gaussian_filter1d(dfs_search_num, sigma=8)
    cmap = plt.get_cmap("Set1")

    # Use more distinctive colors
    colors = {
        "dfs": "#FF6F61",   # Coral
        "sft": "#6A5ACD",   # Slate Blue
        "oft": "#20B2AA",   # Light Sea Green
        "grpo": "#A0522D",   # Sienna
    }
    sns.set_theme(style="whitegrid", context="talk")
    rcParams.update({
        'font.size': 27,
        'axes.titlesize': 22,
        'axes.labelsize': 26,
        'xtick.labelsize': 26,
        'ytick.labelsize': 24,
        'legend.fontsize': 24
    })

    # Create 3 horizontal panels
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=False)
    ax_combined = axes[0]
    ax3 = axes[1]
    plt.subplots_adjust(wspace=0.2)  # Reduced space between plots

    # Plot 1: DFS vs SFT
    ax1 = axes[0]
    ax1.plot(dfs_smoothed, color=colors["dfs"], alpha=0.2, lw=18)
    # plot correct samples
    correct_dfs_idx = np.where(dfs_acc)[0]
    correct_sft_idx = np.where(sft_acc)[0]
    ax1.scatter(np.array(range(200))[correct_dfs_idx], dfs_search_num[correct_dfs_idx], label="DFS Backtracking Trace", color=colors["dfs"], s=30)
    ax1.scatter(np.array(range(200))[correct_sft_idx], sft_search_num[correct_sft_idx], label="Backtracking Model (Before GRPO)", color=colors["sft"], s=30)
    
    
    # incorrect_dfs_idx = np.where(~dfs_acc)[0]
    # incorrect_sft_idx = np.where(~sft_acc)[0]
    # ax1.scatter(np.array(range(200))[incorrect_dfs_idx], dfs_search_num[incorrect_dfs_idx], color=colors["dfs"], s=30, marker='x')
    # ax1.scatter(np.array(range(200))[incorrect_sft_idx], sft_search_num[incorrect_sft_idx], color=colors["sft"], s=30, marker='x')
    
    ax1.set_xlabel("Test Sample Index")
    ax1.set_ylabel("Number of Mistakes")
    ax1.set_xlim([-5, 205])
    ax1.set_ylim([-1, 45])
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.legend(loc='upper left', fontsize=20)

    # Plot 2: DFS vs GRPO
    ax2 = axes[1]
    ax2.plot(dfs_smoothed, color=colors["dfs"], alpha=0.2, lw=18)
    # plot correct samples
    correct_dfs_idx = np.where(dfs_acc)[0]
    correct_grpo_idx = np.where(grpo_acc)[0]
    ax2.scatter(np.array(range(200))[correct_dfs_idx], dfs_search_num[correct_dfs_idx], label="DFS Backtracking Trace", color=colors["dfs"], s=30)
    ax2.scatter(np.array(range(200))[correct_grpo_idx], grpo_search_num[correct_grpo_idx], label="Backtracking Model (After GRPO)", color=colors["grpo"], s=30)
    
    # incorrect_dfs_idx = np.where(~dfs_acc)[0]
    # incorrect_grpo_idx = np.where(~grpo_acc)[0]
    # ax2.scatter(np.array(range(200))[incorrect_dfs_idx], dfs_search_num[incorrect_dfs_idx], color=colors["dfs"], s=30, marker='x')
    # ax2.scatter(np.array(range(200))[incorrect_grpo_idx], sft_search_num[incorrect_grpo_idx], color=colors["grpo"], s=30, marker='x')
    
    ax2.set_xlabel("Test Sample Index")
    ax2.set_ylabel("Number of Mistakes")
    ax2.set_xlim([-5, 205])
    ax2.set_ylim([-1, 45])
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.legend(loc='upper left', fontsize=20)

    sns.despine()  # Removes top and right spines
    plt.tight_layout()
    plt.savefig("countdown_combined_sft_vs_grpo.pdf", bbox_inches='tight')


    # Separately, I only want panel 3
    # Plot 3: Similarity
    fig, ax3 = plt.subplots(1, 1, figsize=(12, 8), sharex=False)
    bins = [1, 5, 10, 20, 30, 40, 50]
    dfs_sft_sim = []
    dfs_grpo_sim = []
    # for b in bins:
    for i in range(len(bins) - 1):
        b_lw = bins[i]
        b_up = bins[i + 1]
        dfs_bin = ((dfs_search_num >= b_lw) & (dfs_search_num < b_up))
        sft_bin = ((sft_search_num >= b_lw) & (sft_search_num < b_up))
        dfs_sft_sim.append(iou(dfs_bin * dfs_acc, sft_bin * sft_acc))

        grpo_bin = ((grpo_search_num >= b_lw) & (grpo_search_num < b_up))
        dfs_grpo_sim.append(iou(dfs_bin * dfs_acc, grpo_bin * grpo_acc))
    dfs_oft_sim = [0.32432432432432434, 0.05555555555555555, 0.02857142857142857, 0.0, 0.0, 0.0]
    
    ax3.plot(bins[:-1], dfs_sft_sim, label="Backtracking Model (Before GRPO)", marker='o', color=colors["sft"], lw=4)
    ax3.plot(bins[:-1], dfs_grpo_sim, label="After GRPO", marker='o', color=colors["grpo"], lw=4)
    ax3.plot(bins[:-1], dfs_oft_sim, label="Direct Solution Model", marker='o', color=colors["oft"], lw=4)
    ax3.set_xlabel("Number of Mistakes")
    ax3.set_ylabel("Similarity (Jaccard Index)")
    # ax3.set_ylim([-0.05, 0.6])
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3.legend()

    sns.despine()  # Removes top and right spines
    plt.tight_layout()
    plt.savefig("countdown_sft_vs_grpo_jaccard.pdf", bbox_inches='tight')


    # Addtionally, use SFT as a reference
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8), sharex=False)
    plt.subplots_adjust(wspace=0.2)  # Reduced space between plots

    sort_idx = np.argsort(sft_search_num)
    grpo_search_num = grpo_search_num[sort_idx]
    sft_search_num = sft_search_num[sort_idx]
    sft_acc = sft_acc[sort_idx]
    grpo_acc = grpo_acc[sort_idx]

    # Plot 1: SFT vs GRPO
    # ax1 = axes[0]
    # plot correct samples
    correct_dfs_idx = np.where(dfs_acc)[0]
    correct_grpo_idx = np.where(grpo_acc)[0]
    ax1.scatter(np.array(range(200))[correct_sft_idx], sft_search_num[correct_sft_idx], label="Backtracking Model", color=colors["sft"], s=30)
    ax1.scatter(np.array(range(200))[correct_grpo_idx], grpo_search_num[correct_grpo_idx], label="GRPO Model", color=colors["grpo"], s=30)
    
    
    # incorrect_dfs_idx = np.where(~dfs_acc)[0]
    # incorrect_grpo_idx = np.where(~grpo_acc)[0]
    # ax1.scatter(np.array(range(200))[incorrect_dfs_idx], dfs_search_num[incorrect_dfs_idx], color=colors["sft"], s=30, marker='x')
    # ax1.scatter(np.array(range(200))[incorrect_grpo_idx], grpo_search_num[incorrect_grpo_idx], color=colors["grpo"], s=30, marker='x')
    
    ax1.set_xlabel("Test Sample Index")
    ax1.set_ylabel("Number of Searches")
    ax1.set_xlim([-5, 205])
    ax1.set_ylim([-1, 45])
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.legend()

    sns.despine()  # Removes top and right spines
    plt.tight_layout()
    plt.savefig("countdown_combined_sft_vs_grpo_2.pdf", bbox_inches='tight')

    solution_num = np.load("/n/home05/sqin/self-correct/stream-of-search/analysis/val1_b4_t100_n500000_dfs.json_solution_num.npy")
    # print correctlation between accuracy and solution_num
    # bin the solution_num into 5 bins
    bins = np.array([0, 1, 2, 5, 40])
    # for each bin, compute the mean accuracy
    grpo_bin_acc = []
    dfs_bin_acc = []
    for i in range(len(bins) - 1):
        bin_mask = (solution_num >= bins[i]) & (solution_num < bins[i + 1])
        grpo_bin_acc.append(np.mean(grpo_acc[bin_mask]))
        dfs_bin_acc.append(np.mean(dfs_acc[bin_mask]))
    print("GRPO Bin Accuracy: ", grpo_bin_acc)
    print("DFS Bin Accuracy: ", dfs_bin_acc)

    return

def compare_oft_and_grpo():
    # Load OFT results
    oft_result_path = "/n/netscratch/dam_lab/Lab/sqin/reason/sos/oft-countdown-cd3e5-17M-qwen"
    gens = np.concatenate([np.arange(1, 17), [22, 24, 26, 28, 30, 32, 34, 36, 38, 42]])
    oft_search_num = np.zeros(200, dtype=float)
    oft_acc = np.zeros(200)
    for g in gens:
        oft_file = f'{oft_result_path}/results_val1_b4_t100_n500000_dfs.json_200_0_@{g}'
        oft_acc_g, _ = load_results(oft_file)
        for i in range(200):
            if oft_acc_g[i] > 0 and oft_search_num[i] == 0:
                oft_search_num[i] = g
                oft_acc[i] = 1
    oft_acc = np.array(oft_acc) > 0
    # replace search num zero with nans
    oft_search_num[oft_search_num == 0] = np.nan
    # Load GRPO results
    grpo_result_path = "/n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-oft-dfs-17M-qwen/hf_model/results_val1_b4_t100_n500000_dfs.json_200_0_@1"
    grpo_acc, grpo_model_output = load_results(grpo_result_path)
    grpo_acc = np.array(grpo_acc) > 0
    print("grpo acc: ", np.mean(grpo_acc))

    # find the set of questions OFT gets correct with less than 4 searches
    # oft_correct = (oft_acc) & (oft_search_num < 8)
    # create a confusion matrix between grpo_acc and oft_correct
    cm = confusion_matrix(oft_acc, grpo_acc, labels=[1, 0])
    # cm = confusion_matrix(model_a_acc, model_b_acc, labels=[1, 0])  # rows = A, cols = B
    labels = [[f"{v}" for v in row] for row in cm]

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, square=True,
                xticklabels=[f'grpo Correct', f'grpo Incorrect'],
                yticklabels=[f'oft Correct', f'oft Incorrect'])
    plt.xlabel("grpo")
    plt.ylabel("oft")
    plt.title('Model Accuracy Confusion Matrix')
    plt.tight_layout()
    plt.savefig("countdown_oft_vs_grpo_confusion.png", bbox_inches='tight')


     # for all the questions grpo gets correct, plot a histogram of number of searches
    correct_grpo_idx = np.where(grpo_acc)[0]
    # Set Seaborn whitegrid style and talk context for better aesthetics
    sns.set_theme(style="whitegrid", context="talk")

    # Optional: Custom color (e.g., dark green)
    custom_color = "#e24a33"  # light red

    plt.figure(figsize=(6, 5))
    plt.hist(oft_search_num[correct_grpo_idx], bins=20, alpha=0.8, label="No-backtracking model, \npre GRPO",
         color=custom_color, edgecolor='black')

    # Labels and legend
    plt.xlabel("Number of Mistakes", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.title("Problems solved post-GRPO pass@1")

    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))  # Integer y-axis ticks
    sns.despine(top=True, right=True)  # Remove spines
    plt.tight_layout()
    plt.savefig("countdown_oft_vs_grpo.pdf", bbox_inches='tight')

    return 

if __name__ == "__main__":
    # analyze_bfs_data()
    # analyze_dfs_data()
    # analyze_dfs_hint_data()
    # analyze_deep_dfs_data()
    # compare_bfs_dfs()
    # compare_dfs_hint_no_hint()
    compare_dfs_oft_sft()
    # compare_dfs_and_bfs_hist()
    # compare_sft_and_sft_short()
    # compare_sft_and_grpo()
    # compare_oft_and_grpo()