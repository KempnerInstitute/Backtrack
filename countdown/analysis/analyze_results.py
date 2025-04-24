import json
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from transformers import PreTrainedTokenizerFast
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
sys.path.append('../src/')
from countdown_utils import parse_trajectory_qwen, metric_fn, extract_final_answer

output_dir = '/n/netscratch/dam_lab/Lab/sqin/reason/sos/'


def plot_gradient_line(ax, x, y, colors, label=None, linestyle='solid'):
    from matplotlib.collections import LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors[:-1], linewidths=2, linestyle=linestyle, label=label)
    ax.add_collection(lc)

def load_results(result_file):
    if isinstance(result_file, list):
        pass
    else:
        result_file = [result_file]
    acc = []
    model_output = []
    for f in result_file:
        print(f'loading results from {f}')
        f = os.path.join(output_dir, f)
        with open(f, 'r') as f:
            data = json.load(f)
        acc += data['ratings']    
        model_output += data['trajectories']

    return acc, model_output

def load_all_outputs(result_file):
    if isinstance(result_file, list):
        pass
    else:
        result_file = [result_file]
    all_ratings = []
    all_outputs = []
    for f in result_file:
        f = os.path.join(output_dir, f)
        with open(f, 'r') as f:
            data = json.load(f)
        all_ratings += data['all_ratings']
        all_outputs += data['all_outputs']    

    return all_ratings, all_outputs

def get_tokenizer():
    # an ugly way to load the tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="../src/countdown_tokenizer.json")
    tokenizer.pad_token = "PAD"
    tokenizer.bos_token = " START "
    tokenizer.eos_token = " END "
    return tokenizer

def analyze_path_difficulty():
    validation_data_file = '/n/home05/sqin/self-correct/stream-of-search/src/data/val1_b4_t100_n500000_random.json'
    val_data = json.load(open(validation_data_file))
    path_ratings = np.array([d["rating"] for d in val_data])
    bins = np.concatenate([np.linspace(0, 500, 51), [1153]])
    pass_at = [1, 16, 32, 64, 128, 256]

    plt.figure(figsize=(5, 3))
    # convert rating to difficulty (measured by nodes explored)
    path_difficulty = (1. - path_ratings) * 1152.

    plt.hist(path_difficulty, bins=np.linspace(0, 1153, 51), align='right') 
    plt.ylabel('Frequency')
    plt.xlabel('Difficulty (Measured by Nodes Explored in Searched Path)')
    plt.title('Searched Path Ground Truth')
    plt.savefig('../results/histogram_of_val_ground_truth.png', bbox_inches='tight')


    # model results
    path_difficulty = np.digitize(path_difficulty, bins=bins)
    plt.figure(figsize=(5, 10))
    for i, p in enumerate(pass_at):
        model_result_file = f'/n/netscratch/dam_lab/Lab/sqin/outputs/sos/results_val1_b4_t100_n500000_random.json_1000_0_@{p}'
        perf_result = json.load(open(model_result_file))
        model_correct = np.array([i > 0.0 for i in perf_result["ratings"]])
        # compute the accuracy for each difficulty level
        accuracy = []
        for d in range(1, len(bins)):
            correct = np.sum(model_correct[path_difficulty == d])
            total = np.sum(path_difficulty == d)
            accuracy.append(correct / total)

        plt.subplot(len(pass_at), 1, i+1)
        # plot a stacked histogram of path difficulty for correct and incorrect problems
        # plt.hist([path_difficulty[~model_correct], path_difficulty[model_correct]], 
        #         bins=100, 
        #         stacked=True, 
        #         label=['Incorrect', 'Correct'], 
        #         color=['red', 'green'])
        # plot accuracy at each difficulty level
        plt.plot(bins[1:], accuracy, 'o')
        plt.ylabel('Acc')
        plt.title(f'Pass at {p}')
        if i < len(pass_at) - 1:
            plt.xticks([])
        

    plt.xlabel('Difficulty (Measured by Nodes Explored in Searched Path)')
    # plt.suptitle('Histogram of Model Correctly Solved Problem Difficulty')
    plt.savefig('../results/histogram_of_val_model_ratings.png', bbox_inches='tight')  

def load_sft_results(run_type, offsets, num):
    sft_files = []
    for o in offsets:
        # sft_files.append(f'{run_type}/results_val1_b4_t100_n500000_random.json_{num}_{o}_@1')
        sft_files.append(f'{run_type}/results_val1_b4_t100_n500000_dfs.json_{num}_{o}_@1')
        # sft_files.append(f'{run_type}/results_val_target1_b4_t100_n500000_dfs.json_{num}_{o}_@1')
        # sft_files.append(f'{run_type}/results_val1_b4_t100_n500000_dfs_hint.json_{num}_{o}_@1')


    sft_result, sft_model_output = load_results(sft_files)
    acc = np.sum([r > 0 for r in sft_result])
    print(f"{run_type} Pass@1 fully solved: {acc} out of {len(sft_result)}")
    print(f"    Average rating: {np.mean(sft_result):.2f} ", )
    
    sft_result = np.array(sft_result)
    
    # analyze model output length 
    tokenizer = get_tokenizer()
    output_lens = []
    for output in sft_model_output:
        # find the first instance of "END" token
        output = output.split("END")[0]
        tokenized_output = tokenizer.encode(output)
        output_lens.append(len(tokenized_output))
    output_lens = np.array(output_lens)

    len_cutoffs = [256, 512, 1024, 2048, 4096]
    for cutoff in len_cutoffs:
        cutoff_idx = np.argwhere(output_lens <= cutoff)
        sft_result_cutoff = sft_result[cutoff_idx]
        acc = np.sum([r > 0 for r in sft_result_cutoff]) 
        print(f"{run_type} Pass@1 fully solved with output length <= {cutoff}: {acc}")

    return sft_result

def load_oft_results(run_type, gens, offsets, num):
    oft_result = []
    for g in gens:  
        oft_files = []
        for o in offsets:
            oft_files.append(f'{run_type}/results_val1_b4_t100_n500000_dfs.json_{num}_{o}_@{g}')
            # oft_files.append(f'{run_type}/results_val1_b4_t100_n500000_dfs_hint.json_{num}_{o}_@{g}')
            # oft_files.append(f'{run_type}/results_val0_b8_t100_n100000_dfs.json_{num}_{o}_@{g}')

        oft_result_g, oft_model_output_g = load_results(oft_files)
        oft_result.append(oft_result_g)

        acc = np.sum([r > 0 for r in oft_result_g])
        print(f"{run_type} Pass@{g} fully solved: {acc} out of {len(oft_result_g)}")
        print(f"    Average accuracy: {np.mean(oft_result_g):.1f} ", )

        # all ratings
        oft_full_result_g, oft_full_model_output_g = load_all_outputs(oft_files)
        oft_full_result_g = np.array(oft_full_result_g)
        print(" all ratings:", oft_full_result_g.shape)
        oft_full_result_g = (oft_full_result_g > 0).astype(int)
        print(" all ratings:", oft_full_result_g.shape)
        print(f"    Average Solving Probability: {np.mean(oft_full_result_g, axis=1).shape} ", )
    
    oft_result = np.array(oft_result)

    # check length
    tokenizer = get_tokenizer()
    output_lens = []
    for output in oft_model_output_g:
        # find the first instance of "END" token
        output = output.split("END")[0]
        tokenized_output = tokenizer.encode(output)
        output_lens.append(len(tokenized_output))
    print("Average output length: ", np.mean(output_lens))
    
    return oft_result

def compute_auto_generation_flops(hidden_dim, kv_dim, inter_size, n_layer, gen_len, num_gen):    
    linear_flops = 2 * hidden_dim **2 + 2 * hidden_dim * kv_dim + 3 * hidden_dim * inter_size 
    quadratic_flops = hidden_dim
    per_gen_cost = n_layer * (linear_flops * gen_len + quadratic_flops * gen_len * (gen_len + 1)/2)
    total_cost = num_gen * per_gen_cost

    return total_cost

def compare_sft_and_oft():
    # SFT with mixed strategy
    # sft_acc = [
    #     [6.5, 12.5, 25, 33.5, 44.5], # 3M
    #     [12.5, 17.5, 32, 41, 51], # 17M
    #     [12, 17.5, 33, 41, 51.5], # 38M
    #     [12, 16.5, 35, 43, 54], # 144M
    #     [11.5, 15.5, 30.5, 41.5, 53] # 365M
    # ]

    # SFT with DFS only
    sft_acc = [
        [10.5, 13.5, 28.5, 33.5, 41.5],  # 3M
        [7.5, 11.5, 25.5, 35, 41.5],     # 17M
        [9.5, 14.5, 28, 35, 45],         # 38M
        [12.5, 16, 30, 35.5, 43],        # 144M
    ]

    oft_acc = [
        [8.5, 20.5, 30, 41.5, 52],     # 3M
        [17, 25, 37, 50, 61],     # 17M
        [11.5, 27, 33.5, 40, 63],   # 38M
        [21, 25.5, 37.5, 52, 68.5], # 144M
    ]

    oft_gens = [1, 2, 4, 8, 16]
    oft_base_gen_len = 212
    oft_gen_lens = [oft_base_gen_len * g for g in oft_gens]
    sft_gen_lens = [256, 512, 1024, 2048, 4096]
    sft_gen = 1

    model_sizes = [3, 17, 38, 144]
    model_hid_dim = [256, 512, 512, 1024]
    model_kv_dim = [64, 128, 128, 256]
    model_layers = [6, 8, 10, 12]
    model_inter_dim = [512, 1024, 2048, 3072]

    # === Seaborn style ===
    sns.set_theme(style='whitegrid', context='talk', rc={"grid.linewidth": 0.4})
    fig = plt.figure(figsize=(14, 6))  # Taller figure
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[1, 1])

    ax_sft = fig.add_subplot(gs[0, 0])      # Top-left: Backtracking
    ax_oft = fig.add_subplot(gs[1, 0], sharex=ax_sft)      # Bottom-left: No-Backtracking
    ax_diff = fig.add_subplot(gs[:, 1])     # Right: Full-height difference plot

    cmap = plt.get_cmap("magma_r")
    palette = [cmap(0.15 + 0.7 * i / (len(model_sizes) - 1)) for i in range(len(model_sizes))]

    # === Plot SFT curves ===
    for i in range(len(model_sizes)):
        color = palette[i]
        sft_flops = [compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=g, num_gen=sft_gen) for g in sft_gen_lens]
        ax_sft.plot(sft_flops, sft_acc[i], color=color, marker='o',
                    label=f'{model_sizes[i]}M', linewidth=2)

    ax_sft.set_xscale('log')
    # ax_sft.set_title('(A) Backtracking Models', fontsize=20)
    ax_sft.text(0.01, 0.95, '(A) Backtracking Models', transform=ax_sft.transAxes,
            fontsize=16, fontweight='bold', va='top')
    ax_sft.tick_params(axis='both', labelsize=14)
    ax_sft.set_ylim([0, 80])
    ax_sft.tick_params(axis='x', labelbottom=False)
    ax_sft.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # === Plot OFT curves ===
    for i in range(len(model_sizes)):
        color = palette[i]
        oft_flops = [compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=oft_base_gen_len, num_gen=g) for g in oft_gens]
        ax_oft.plot(oft_flops, oft_acc[i], color=color, linestyle='--',
                    marker='o', label=f'{model_sizes[i]}M', linewidth=2)

    ax_oft.set_xscale('log')
    ax_oft.set_xlabel('FLOPs', fontsize=18)
    # ax_oft.set_title('(B) No-Backtracking Models', fontsize=20)
    ax_oft.text(0.01, 0.95, '(B) Direct Solution Models', transform=ax_oft.transAxes,
            fontsize=16, fontweight='bold', va='top')
    ax_oft.tick_params(axis='both', labelsize=16)
    ax_oft.set_ylim([0, 80])
    # ax_oft.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_oft.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.text(0.0, 0.5, 'Unseen Problems Solved (%)',
         va='center', rotation='vertical', fontsize=18)
    

    # === Plot Difference Plot ===
    for i in range(len(model_sizes)):
        sft_vals = np.array(sft_acc[i])
        oft_vals = np.array(oft_acc[i])
        sft_flops = [compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=g, num_gen=sft_gen) for g in sft_gen_lens]

        diff = (oft_vals - sft_vals)

        ax_diff.plot(sft_flops, diff, color=palette[i], marker='o',
                     label=f'{model_sizes[i]}M', linewidth=2,
                     ls="-.")

    ax_diff.set_xscale('log')
    ax_diff.set_xlabel('FLOPs', fontsize=18)
    ax_diff.set_ylabel('Accuracy Gap \n Direct Solution − Backtrack (%)', fontsize=17)
    # ax_diff.set_title('(C) Performance Gap Between Models', fontsize=20)
    ax_diff.text(0.01, 0.95, '(C) Performance Gap Between Models', transform=ax_diff.transAxes,
             fontsize=16, fontweight='bold', va='top')
    ax_diff.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax_diff.tick_params(axis='both', labelsize=16)
    # ax_diff.xaxis.set_major_locator(MaxxNLocator(nbins=5))
    ax_diff.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_diff.legend(fontsize=14, framealpha=0.)
    sns.despine()
    plt.subplots_adjust(hspace=-0.05, wspace=0.3)
    plt.suptitle('CountDown', fontsize=24, fontweight='bold', y=0.9)
    plt.tight_layout()
    plt.savefig('countdown_flop_sft_oft.pdf', bbox_inches='tight')

    return

def compare_sft_and_oft_alt():
    # SFT with mixed strategy
    # sft_acc = [
    #     [6.5, 12.5, 25, 33.5, 44.5], # 3M
    #     [12.5, 17.5, 32, 41, 51], # 17M
    #     [12, 17.5, 33, 41, 51.5], # 38M
    #     [12, 16.5, 35, 43, 54], # 144M
    #     [11.5, 15.5, 30.5, 41.5, 53] # 365M
    # ]

    # SFT with DFS only
    sft_acc = [
        [10.5, 13.5, 28.5, 33.5, 41.5],  # 3M
        [7.5, 11.5, 25.5, 35, 41.5],     # 17M
        [9.5, 14.5, 28, 35, 45],         # 38M
        [12.5, 16, 30, 35.5, 43],        # 144M
    ]

    oft_acc = [
        [8.5, 20.5, 30, 41.5, 52],     # 3M
        [17, 25, 37, 50, 61],     # 17M
        [11.5, 27, 33.5, 40, 63],   # 38M
        [21, 25.5, 37.5, 52, 68.5], # 144M
    ]

    oft_gens = [1, 2, 4, 8, 16]
    oft_base_gen_len = 212
    oft_gen_lens = [oft_base_gen_len * g for g in oft_gens]
    sft_gen_lens = [256, 512, 1024, 2048, 4096]
    sft_gen = 1

    model_sizes = [3, 17, 38, 144]
    model_hid_dim = [256, 512, 512, 1024]
    model_kv_dim = [64, 128, 128, 256]
    model_layers = [6, 8, 10, 12]
    model_inter_dim = [512, 1024, 2048, 3072]

    # === Seaborn style ===
    sns.set_theme(style='whitegrid', context='talk', rc={"grid.linewidth": 0.4})
    fig = plt.figure(figsize=(14, 6))  # Taller figure
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[1, 1])

    ax_sft = fig.add_subplot(gs[0, 0])      # Top-left: Backtracking
    ax_oft = fig.add_subplot(gs[1, 0], sharex=ax_sft)      # Bottom-left: No-Backtracking
    ax_diff = fig.add_subplot(gs[:, 1])     # Right: Full-height difference plot

    cmap = plt.get_cmap("magma_r")
    palette = [cmap(0.15 + 0.7 * i / (len(model_sizes) - 1)) for i in range(len(model_sizes))]

    # === Plot SFT curves ===
    for i in range(len(model_sizes)):
        color = palette[i]
        sft_flops = [compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=g, num_gen=sft_gen) for g in sft_gen_lens]
        ax_sft.plot(sft_flops, sft_acc[i], color=color, marker='o',
                    label=f'{model_sizes[i]}M', linewidth=2)

    ax_sft.set_xscale('log')
    # ax_sft.set_title('(A) Backtracking Models', fontsize=20)
    ax_sft.text(0.01, 0.95, '(A) Backtracking Models', transform=ax_sft.transAxes,
            fontsize=16, fontweight='bold', va='top')
    ax_sft.tick_params(axis='both', labelsize=14)
    ax_sft.set_ylim([0, 80])
    ax_sft.tick_params(axis='x', labelbottom=False)
    ax_sft.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # === Plot OFT curves ===
    for i in range(len(model_sizes)):
        color = palette[i]
        oft_flops = [compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=oft_base_gen_len, num_gen=g) for g in oft_gens]
        ax_oft.plot(oft_flops, oft_acc[i], color=color, linestyle='--',
                    marker='o', label=f'{model_sizes[i]}M', linewidth=2)

    ax_oft.set_xscale('log')
    ax_oft.set_xlabel('FLOPs', fontsize=18)
    # ax_oft.set_title('(B) No-Backtracking Models', fontsize=20)
    ax_oft.text(0.01, 0.95, '(B) No-Backtracking Models', transform=ax_oft.transAxes,
            fontsize=16, fontweight='bold', va='top')
    ax_oft.tick_params(axis='both', labelsize=16)
    ax_oft.set_ylim([0, 80])
    # ax_oft.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_oft.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.text(0.0, 0.5, 'Unseen Problems Solved (%)',
         va='center', rotation='vertical', fontsize=18)
    

    # === Plot Last Point ===
    sft_final = [acc[-1] for acc in sft_acc]
    oft_final = [acc[-1] for acc in oft_acc]

    sft_flops_final = [
        compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=sft_gen_lens[-1], num_gen=sft_gen
        ) for i in range(len(model_sizes))
    ]

    oft_flops_final = [
        compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=oft_base_gen_len, num_gen=oft_gens[-1]
        ) for i in range(len(model_sizes))
    ]

    # Convert palette to np array
    sft_colors = np.array(palette)
    oft_colors = np.array(palette)

    # Plot gradient lines
    plot_gradient_line(ax_diff, sft_flops_final, sft_final, sft_colors, label='Backtracking')
    plot_gradient_line(ax_diff, oft_flops_final, oft_final, oft_colors, label='No-Backtracking', linestyle='--')

    # Plot per-model points using palette
    for i in range(len(model_sizes)):
        color = palette[i]
        ax_diff.plot(sft_flops_final[i], sft_final[i], marker='o', color=color,
                     linestyle='none', label=None)
        ax_diff.plot(oft_flops_final[i], oft_final[i], marker='o', color=color,
                     linestyle='none', label=None)

    ax_diff.set_xscale('log')
    ax_diff.set_xlabel('FLOPs', fontsize=16)
    ax_diff.set_ylim([30, 80])
    ax_diff.tick_params(axis='both', labelsize=14)
    ax_diff.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_diff.text(0.01, 0.95, '(F) Best Accuracy vs. FLOPs', transform=ax_diff.transAxes,
                 fontsize=16, fontweight='bold', va='top')

    ax_diff.legend(fontsize=12, loc='lower right')


    sns.despine()
    plt.subplots_adjust(hspace=-0.05, wspace=0.3)
    plt.suptitle('CountDown', fontsize=18, fontweight='bold', y=0.9)
    plt.tight_layout()
    plt.savefig('countdown_flop_sft_oft_alt.pdf', bbox_inches='tight')

    return

def compare_sft_dfs_and_sft_short_and_sft_mix():
    # === Sample Data ===
    sft_acc = [[7.5, 11.5, 25.5, 35, 41.5]]  # 17M
    oft_acc = [[17, 25, 37, 50, 61, 70]]     # 17M
    dfs_data_acc = [[12, 18, 30.5, 39.5, 58]]  # Upper bound
    sft_mix_acc = [[12.5, 17.5, 32, 41, 51]]   # Variation 2
    sft_hint_acc = [[17, 28.5, 45.5, 53.5, 59.5]]  # Variation 1

    oft_gens = [1, 2, 4, 8, 16, 32]
    oft_base_gen_len = 212
    oft_gen_lens = [oft_base_gen_len * g for g in oft_gens]
    sft_gen_lens = [256, 512, 1024, 2048, 4096]
    sft_gen = 1

    model_sizes = [17]
    model_hid_dim = [512]
    model_kv_dim = [128]
    model_layers = [8]
    model_inter_dim = [1024]

    # === Plot Setup ===
    sns.set_theme(style='whitegrid', context='talk')
    fig, ax = plt.subplots(figsize=(8, 7))
    blues = sns.color_palette("Blues", 4)[1:]  # Skip the lightest shade
    accent_colors = sns.color_palette("husl", 5)

    # Define a clean color palette
    palette = sns.color_palette("husl", 5)  # 5 distinct hues

    for i in range(len(sft_acc)):
        # SFT FLOPs
        sft_flops = [compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i],
            kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i],
            n_layer=model_layers[i],
            gen_len=g,
            num_gen=sft_gen,
        ) for g in sft_gen_lens]

        # OFT FLOPs
        oft_flops = [compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i],
            kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i],
            n_layer=model_layers[i],
            gen_len=oft_base_gen_len,
            num_gen=g,
        ) for g in oft_gens]

        # === Baselines ===
        ax.plot(sft_flops, sft_acc[i], color=blues[0], label='Backtrack', marker='o', markersize=10)
        ax.plot(sft_flops, dfs_data_acc[i], color="grey", linestyle='--', label='DFS Trace (Data Bound)', marker='o', markersize=10)
        ax.plot(oft_flops, oft_acc[i], color=palette[0], linestyle='--', label='Direct Soluion', marker='o', markersize=10)

        # === Variations ===
        ax.plot(sft_flops, sft_mix_acc[i], color=blues[2], label='Mix-Backtrack(Variation 1)', marker='*', markersize=14)
        ax.plot(sft_flops, sft_hint_acc[i], color=blues[1], label='Think-Backtrack(Variation 2)', marker='*', markersize=14)
        

    # === Aesthetic Settings ===
    
    ax.set_xscale('log')
    ax.set_xlabel('FLOPs', fontsize=18)
    ax.set_ylabel('Unseen Problems Solved (%)', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=14, loc='upper left')

    sns.despine()
    plt.tight_layout()
    plt.savefig('countdown_17M_flop_dfs_shorten_mix.pdf', bbox_inches='tight')

    return

def compare_sft_and_grpo():
    sns.set_theme(style='whitegrid', context='talk')

    # Accuracy
    sft_acc = [
        [10.5, 13.5, 28.5, 33.5, 41.5],  # 3M
        [7.5, 11.5, 25.5, 35, 41.5]      # 17M
    ]
    sft_grpo_acc = [
        [14.5, 20, 38.5, 58.5, 68],      # 3M
        [23.5, 31, 41, 52.5, 70.5]       # 17M
    ]
    oft_acc = [
        [8.5, 20.5, 30, 41.5, 52, 69.5], # 3M
        [17, 25, 37, 50, 61, 70]         # 17M
    ]
    oft_grpo_acc = [
        [33, 35, 37, 38.5, 41.5, 42.5], # 3M
        [42.5, 43.5, 44, 47, 47, 48], # 17M
    ]

    oft_gens = [1, 2, 4, 8, 16, 32]
    oft_base_gen_len = 212
    oft_gen_lens = [oft_base_gen_len * g for g in oft_gens]
    sft_gen_lens = [256, 512, 1024, 2048, 4096]
    sft_gen = 1

    model_sizes = [3, 17]
    model_hid_dim = [256, 512]
    model_kv_dim = [64, 128]
    model_layers = [6, 8]
    model_inter_dim = [512, 1024]

    for i in range(len(model_sizes)):
        # Compute FLOPs
        sft_flops = [compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=g, num_gen=sft_gen) for g in sft_gen_lens]
        oft_flops = [compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=oft_base_gen_len, num_gen=g) for g in oft_gens]

        fig, ax = plt.subplots(figsize=(6, 5))
        palette = sns.color_palette("Set1", 2)
        colors = {
            'sft': palette[0],
            'sft_grpo': palette[0],
            'oft': palette[1],
            'oft_grpo': palette[1]
        }

        # --- Plot SFT ---
        ax.plot(sft_flops, sft_acc[i], label='Backtracking (Before)', marker='o', color=colors['sft'], alpha=0.4)
        ax.plot(sft_flops, sft_grpo_acc[i], label='After GRPO', marker='o', color=colors['sft_grpo'], alpha=1.0)

        # Arrows for SFT
        for j, (x, y0, y1) in enumerate(zip(sft_flops, sft_acc[i], sft_grpo_acc[i])):
            if j % 2 == 0:
                dy = y1 - y0
                ax.annotate('', 
                            xy=(x, y0 + 0.9 * dy), 
                            xytext=(x, y0 + 0.1 * dy),
                            arrowprops=dict(arrowstyle='->', color=colors['sft_grpo'], lw=1.5, alpha=0.4))

        # --- Plot OFT ---
        ax.plot(oft_flops, oft_acc[i], label='No-Backtracking (Before)', marker='o', linestyle='--', color=colors['oft'], alpha=0.4)
        ax.plot(oft_flops, oft_grpo_acc[i], label='After GRPO', marker='o', linestyle='--', color=colors['oft_grpo'], alpha=1.0)

        
        # Arrows for OFT
        for j, (x, y0, y1) in enumerate(zip(oft_flops, oft_acc[i], oft_grpo_acc[i])):
            if j % 2 == 0 and not np.isnan(y1):
                dy = y1 - y0
                ax.annotate('', 
                            xy=(x, y0 + 0.9 * dy), 
                            xytext=(x, y0 + 0.1 * dy),
                            arrowprops=dict(arrowstyle='->', color=colors['oft_grpo'], lw=1.5, alpha=0.4))

        # Styling
        ax.set_xscale('log')
        ax.set_xlabel('FLOPs', fontsize=14)
        ax.set_ylabel('Unseen Problems Solved (%)', fontsize=14)
        # ax.set_title(f'GRPO Improvement — {model_sizes[i]}M Model', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=11)
        sns.despine()

        plt.tight_layout()
        plt.savefig(f'countdown_{model_sizes[i]}M_flop_sft_grpo.pdf', bbox_inches='tight')

    return

def compare_deep_and_default():
    sft_acc = [
        # [10.5, 13.5, 28.5, 33.5, 41.5],  # 3M
        [7.5, 11.5, 25.5, 35, 41.5],     # 17M
        [9.5, 14.5, 28, 35, 45],         # 38M
    ]
    sft_acc = np.array(sft_acc)

    oft_acc = [
        # [8.5, 20.5, 30, 41.5, 52],     # 3M
        [17, 25, 37, 50, 61],     # 17M
        [11.5, 27, 33.5, 40, 63],   # 38M
    ]
    oft_acc = np.array(oft_acc)

    sft_deep_acc = [
        # [0.5, 3.5, 10.5, 15, np.nan], # 3M
        [1, 7.5, 13, 18.5, np.nan], # 17M
        [0, 4.5, 10.5, 17.5, np.nan], # 38M

    ]
    sft_deep_acc = np.array(sft_deep_acc)

    oft_deep_acc = [
        # [1.5, 5, 8, 11, 18], # 3M
        [2.5, 6.5, 10, 10.5, 20], # 17M
        [4, 6.5, 12.5, 14, 21.5],  # 38M
    ]
    oft_deep_acc = np.array(oft_deep_acc)

    oft_gens = [1, 2, 4, 8, 16]
    oft_base_gen_len = 212
    oft_deep_base_gen_len = 400
    oft_gen_lens = np.array([oft_base_gen_len * g for g in oft_gens])
    sft_gen_lens = np.array([256, 512, 1024, 2048, 4096])
    sft_deep_gen_lens = np.array([512, 1024, 2048, 4096, 8192])
    sft_gen = 1

    model_sizes = [17, 38]
    model_hid_dim = [512, 512]
    model_kv_dim = [128, 128]
    model_layers = [8, 10]
    model_inter_dim = [1024, 2048]

     # Axes setup
    sns.set_theme(style='whitegrid', context='talk')
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    cmap = sns.color_palette("viridis", 2) 

    for i in range(2):
        ax = axes[i]
        color = cmap[i]

        # FLOPs
        flops = np.array([compute_auto_generation_flops(
            hidden_dim=model_hid_dim[i], kv_dim=model_kv_dim[i],
            inter_size=model_inter_dim[i], n_layer=model_layers[i],
            gen_len=g, num_gen=sft_gen) for g in sft_gen_lens])

        # Accuracy gaps
        default_diff = oft_acc[i] - sft_acc[i]
        deep_diff = oft_deep_acc[i] - sft_deep_acc[i]

        # Remove NaNs
        mask = ~np.isnan(deep_diff)
        flops = flops[mask]
        default_diff = default_diff[mask]
        deep_diff = deep_diff[mask]

        # Shared widths and smaller offset
        widths = flops * 0.12  # Slightly narrower bars
        offsets = flops * 0.06  # Smaller gap between bars

        # Plot bars: Default on left, Easy on right
        ax.bar(flops - offsets, default_diff, width=widths,
               label="Original", color=color)
        ax.bar(flops + offsets, deep_diff, width=widths,
               label="Stacked", color=color, alpha=0.5, hatch='//')
        
        # Style
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=16)
        # ax.set_ylim([0, 80])

        # Add legend with model size as title
        ax.legend(fontsize=16, title=f'CountDown {model_sizes[i]}M', title_fontsize=16)

    # Axis styling (outside loop)
    axes[1].set_xlabel('FLOPs', fontsize=18)
    fig.text(0.00, 0.5, 'Accuracy Gap\nDirect Solution − Backtrack (%)',
         va='center', ha='center', rotation='vertical', fontsize=18)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('countdown_flop_deep.pdf', bbox_inches='tight')


    return

def figure_1():
    sns.set_style("white")

    # === DATA ===
    countdown_sft_acc = [[7.5, 11.5, 25.5, 35, 41.5]]
    countdown_oft_acc = [[17, 25, 37, 50, 61]]
    sudoku_sft_strat_acc = [[18, 27.5, 64, 73.5, 77]]
    sudoku_oft_strat_acc = [[6.5, 16, 30.0, 39.5, 58.0]]

    oft_gens = [1, 2, 4, 8, 16]
    countdown_oft_base_gen_len = 212
    sudoku_oft_base_gen_len = 322
    sft_gen_lens = [256, 512, 1024, 2048, 4096]
    sft_gen = 1

    model_hid_dim = 512
    model_kv_dim = 128
    model_layers = 8
    model_inter_dim = 1024

    sudoku_layers = 10
    sudoku_inter = 2048

    countdown_sft_flops = [compute_auto_generation_flops(
        model_hid_dim, model_kv_dim, model_inter_dim, model_layers, g, sft_gen) for g in sft_gen_lens]
    countdown_oft_flops = [compute_auto_generation_flops(
        model_hid_dim, model_kv_dim, model_inter_dim, model_layers, countdown_oft_base_gen_len, g) for g in oft_gens]

    sudoku_sft_flops = [compute_auto_generation_flops(
        model_hid_dim, model_kv_dim, sudoku_inter, sudoku_layers, g, sft_gen) for g in sft_gen_lens]
    sudoku_oft_flops = [compute_auto_generation_flops(
        model_hid_dim, model_kv_dim, sudoku_inter, sudoku_layers, sudoku_oft_base_gen_len, g) for g in oft_gens]

    # GRPO (17M)
    sft_acc = [7.5, 11.5, 25.5, 35, 41.5]
    sft_grpo_acc = [23.5, 31, 41, 52.5, 70.5]
    oft_acc = [17, 25, 37, 50, 61, 70]
    oft_grpo_acc = [42.5, 43.5, 44, 47, 47, 48]
    oft_gens_grpo = [1, 2, 4, 8, 16, 32]
    oft_base_gen_len = 212
    oft_gen_lens_grpo = [oft_base_gen_len * g for g in oft_gens_grpo]
    sft_flops_grpo = [compute_auto_generation_flops(
        model_hid_dim, model_kv_dim, model_inter_dim, model_layers, g, sft_gen) for g in sft_gen_lens]
    oft_flops_grpo = [compute_auto_generation_flops(
        model_hid_dim, model_kv_dim, model_inter_dim, model_layers, oft_base_gen_len, g) for g in oft_gens_grpo]

    # === PLOT SETTINGS ===
    sudoku_color = "#1f78b4"
    sudoku_alt = "#33a0a4"
    countdown_color = "#e66100"
    countdown_alt = "#fdb863"

    grpo_sft_color = "#3c8dbc"   # NEW color 1 (blue)
    grpo_oft_color = "#e24a33"   # NEW color 2 (orange-red)

    fontsize = 26
    linewidth = 3
    markersize = 10

    fig, axes = plt.subplots(1, 3, figsize=(28, 6))
    plt.subplots_adjust(wspace=0.2)

    ax0_pos = axes[0].get_position()
    ax1_pos = axes[1].get_position()

    # === SHADED BACKGROUND BOX FOR (A) ===
    combined_x0 = ax0_pos.x0 - 0.04
    combined_y0 = ax0_pos.y0 - 0.12
    combined_width = ax1_pos.x1 - ax0_pos.x0 + 0.035 
    combined_height = ax0_pos.y1 - ax0_pos.y0 + 0.25

    shaded_box = patches.FancyBboxPatch(
        (combined_x0, combined_y0),
        combined_width,
        combined_height,
        boxstyle="round,pad=0.01",
        linewidth=1.5,
        edgecolor="gray",
        facecolor="#f5f5f5",  # grey fill
        transform=fig.transFigure,
        zorder=-1  # send to background
    )
    fig.patches.append(shaded_box)

    # === COUNTDOWN ===
    ax = axes[0]
    ax.plot(countdown_oft_flops, countdown_oft_acc[0], color=countdown_color,
            label='Direct Solution Model\n(Scaling w/ best-of-n)', marker='o', linewidth=linewidth, markersize=markersize)
    ax.plot(countdown_sft_flops, countdown_sft_acc[0], color=countdown_alt,
            linestyle='--', marker='o', label='Backtrack Model\n (Scaling w/ CoT)',
            linewidth=linewidth, markersize=markersize)
    ax.set_xscale('log')
    ax.set_title("Countdown", fontsize=fontsize + 6, pad=20, color=countdown_color, weight='bold')
    ax.set_xlabel("FLOPs", fontsize=fontsize)
    ax.set_ylabel("Problem Solved (%)", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize - 2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=fontsize - 6, loc='upper left', framealpha=0)

    # === SUDOKU ===
    ax = axes[1]
    ax.plot(sudoku_oft_flops, sudoku_oft_strat_acc[0], color=sudoku_color,
            label='Direct Solution\n(w/best-of-n)', marker='o', linewidth=linewidth, markersize=markersize)
    ax.plot(sudoku_sft_flops, sudoku_sft_strat_acc[0], color=sudoku_alt,
            linestyle='--', marker='o', label='Backtrack\n(w/ CoT)',
            linewidth=linewidth, markersize=markersize)
    ax.set_xscale('log')
    ax.set_title("Sudoku", fontsize=fontsize + 6, pad=20, color=sudoku_color, weight='bold')
    ax.set_xlabel("FLOPs", fontsize=fontsize)
    ax.set_ylabel("Problem Solved (%)", fontsize=fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='both', labelsize=fontsize - 2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=fontsize - 6, loc='upper left', framealpha=0)

     # === GRPO PANEL ===
    ax = axes[2]
    
    # SFT curves
    ax.plot(sft_flops_grpo, sft_acc, label='Backtrack (Before)', marker='o',
            color=grpo_sft_color, linestyle='-', alpha=0.4, linewidth=linewidth, markersize=markersize)
    ax.plot(sft_flops_grpo, sft_grpo_acc, label='After GRPO', marker='o',
            color=grpo_sft_color, linestyle='-', alpha=1.0, linewidth=linewidth, markersize=markersize)
    # add a grey horizontal line at = y=0.42
    ax.axhline(y=43, color='grey', linestyle='-', linewidth=4, alpha=0.4)
    
    # Annotate every other SFT point
    for i, (x, y0, y1) in enumerate(zip(sft_flops_grpo, sft_acc, sft_grpo_acc)):
        if i % 2 == 0:
            dy = y1 - y0
            ax.annotate('', xy=(x, y0 + 0.9 * dy), xytext=(x, y0 + 0.1 * dy),
                        arrowprops=dict(arrowstyle='-|>,head_width=0.5,head_length=1.0',
                                        color=grpo_sft_color, lw=2, alpha=0.))

    # OFT curves
    ax.plot(oft_flops_grpo, oft_acc, label='Direct Solution (Before)', marker='o',
            color=grpo_oft_color, linestyle='--', alpha=0.4, linewidth=linewidth, markersize=markersize)
    ax.plot(oft_flops_grpo, oft_grpo_acc, label='After GRPO', marker='o',
            color=grpo_oft_color, linestyle='--', alpha=1.0, linewidth=linewidth, markersize=markersize)
    
    # Annotate every other OFT point
    for i, (x, y0, y1) in enumerate(zip(oft_flops_grpo, oft_acc, oft_grpo_acc)):
        if i % 2 == 0:
            dy = y1 - y0
            ax.annotate('', xy=(x, y0 + 0.9 * dy), xytext=(x, y0 + 0.1 * dy),
                        arrowprops=dict(arrowstyle='-|>,head_width=0.5,head_length=1.0',
                                        color=grpo_oft_color, lw=2, alpha=0.4))

    # Final GRPO styling
    ax.set_xscale('log')
    ax.set_title("CountDown GRPO", fontsize=fontsize + 6, pad=20, color='#444', weight='bold')
    ax.set_xlabel("FLOPs", fontsize=fontsize)
    ax.set_ylabel("Problem Solved (%)", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize - 2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=fontsize - 8, loc='upper left', framealpha=0)

    # === PANEL LABELS ===
    fig.text(ax0_pos.x0 - 0.04, ax0_pos.y1 + 0.05, "(A)", fontsize=fontsize + 8, weight='bold')
    fig.text(ax1_pos.x1 + 0.04, ax1_pos.y1 + 0.05, "(B)", fontsize=fontsize + 8, weight='bold')

    # === SAVE ===
    plt.savefig("figure_1.pdf", bbox_inches="tight", dpi=300)
    return

def oft_grpo_training_dynamics():
    oft_result = []
    g = 64
    num=200
    o=0

    # --- Pre GRPO ---
    run_type = '/n/netscratch/dam_lab/Lab/sqin/reason/sos/oft-countdown-cd3e5-17M-qwen'
    oft_file = f'{run_type}/results_val1_b4_t100_n500000_dfs.json_{num}_{o}_@{g}'
    oft_full_result_g, _ = load_all_outputs(oft_file)
    oft_full_result_g = np.array(oft_full_result_g)
    oft_full_result_g = (oft_full_result_g > 0).astype(int)
    oft_solving_rate = np.mean(oft_full_result_g, axis=1)

    # --- Post GRPO ---
    run_type = '/n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-oft-dfs-17M-qwen/hf_model'
    grpo_file = f'{run_type}/results_val1_b4_t100_n500000_dfs.json_{num}_{o}_@{g}'
    grpo_full_result_g, _ = load_all_outputs(grpo_file)
    grpo_full_result_g = np.array(grpo_full_result_g)
    grpo_full_result_g = (grpo_full_result_g > 0).astype(int)
    grpo_solving_rate = np.mean(grpo_full_result_g, axis=1)

    # --- Sort by Pre-GRPO to order problems from easy to hard ---
    sort_idx = np.argsort(oft_solving_rate)[::-1]
    oft_sorted = oft_solving_rate[sort_idx]
    grpo_sorted = grpo_solving_rate[sort_idx]
    smoothed = sm.nonparametric.lowess(grpo_sorted, np.arange(num), frac=0.2)

    sns.set_theme(style="whitegrid", context="talk")

    num = len(oft_sorted)
    problem_index = np.arange(1, num + 1)

    # DataFrame for seaborn
    df = pd.DataFrame({
        "Problem Index": problem_index,
        "Pre GRPO": oft_sorted,
        "Post GRPO": grpo_sorted
    })

    # We only need 'Post GRPO' for vertical histogram; no x-axis marginal plot
    g = sns.jointplot(
        x="Problem Index",
        y="Post GRPO",
        data=df,
        kind="scatter",
        marginal_kws=dict(bins=30, fill=True, color='tab:red', alpha=0.3),
        color='tab:red',
        height=7,
        ratio=3,
        space=0.05
    )

    # Add Pre-GRPO (blue) and Post-GRPO (red) manually to main plot
    g.ax_joint.plot(problem_index, oft_sorted, linestyle='None', marker='o', color='tab:blue', label="Pre GRPO", markersize=4)
    g.ax_joint.plot(problem_index, grpo_sorted, linestyle='None', marker='o', color='tab:red', label="Post GRPO", markersize=4)

    # Smoothed LOWESS line over Post-GRPO
    smoothed = sm.nonparametric.lowess(grpo_sorted, problem_index, frac=0.2)
    g.ax_joint.plot(smoothed[:, 0], smoothed[:, 1], color='tab:red', linestyle='-', linewidth=4, alpha=0.3)

    # g.figure.subplots_adjust(wspace=0.01)
    # Clean up labels and legend
    g.ax_joint.set_xlim(0, num + 1)
    g.ax_joint.set_ylim(0, 1.05)
    g.ax_joint.set_xlabel("Problem Index", fontsize=14)
    g.ax_joint.set_ylabel("Solving Probability (Pass@64)", fontsize=14)
    g.ax_joint.tick_params(labelsize=12)
    g.ax_joint.legend(fontsize=12, loc="center right", title='Direct Solution Model', title_fontsize=12, framealpha=0.)

    # Remove x-axis marginal histogram
    g.ax_marg_x.remove()

    # set y limit for main plot
    g.ax_joint.set_ylim(-0.02, 1.02)

    # Finalize
    # g.figure.tight_layout()
    plt.savefig('countdown_oft_grpo_training_dynamics.pdf', bbox_inches='tight', dpi=300)

    return 

def oft_majority_vote():
    g = 64
    num=200
    o=0

    # --- Pre GRPO ---
    run_type = '/n/netscratch/dam_lab/Lab/sqin/reason/sos/oft-countdown-cd3e5-17M-qwen'
    oft_file = f'{run_type}/results_val1_b4_t100_n500000_dfs.json_{num}_{o}_seed8_@{g}'
    oft_full_result_g, oft_model_output = load_all_outputs(oft_file)
    oft_full_result_g = np.array(oft_full_result_g)
    oft_full_result_g = (oft_full_result_g > 0).astype(int)
    oft_solving_rate = np.mean(oft_full_result_g, axis=1)

    majority_vote_acc = []
    best_n_acc = []
    for i in range(200):
        if np.max(oft_full_result_g[i]) > 0: # examine when model gets correct at least once
            best_n_acc.append(1)
            # extract correct attempt
            correct_idx = np.argwhere(oft_full_result_g[i] > 0)[0, 0]
            attempt_dict = {}
            attempt_idx = {}
            for j in range(g): # examine each generation
                model_output_example = oft_model_output[i][j][0]
                processed_prediction = parse_trajectory_qwen(model_output_example) # need to fix spacing issues 
                success, operation_list = extract_final_answer(processed_prediction)
                if success:
                    operation_list = ",".join(operation_list)
                else:
                    operation_list = None
                    # print("WARNING: no operation list found", processed_prediction)
                if j == correct_idx:
                    correct_operation = operation_list
                if operation_list is not None:
                    if operation_list in attempt_dict:
                        attempt_dict[operation_list] += 1
                        attempt_idx[operation_list].append(j)
                    else:
                        attempt_dict[operation_list] = 1
                        attempt_idx[operation_list] = [j]
            
            # find the operation with the most attempts
            max_attempt = max(attempt_dict.values())
            max_attempt_list = [k for k, v in attempt_dict.items() if v == max_attempt]
            # randomly select one of the max attempts
            max_attempt_sample = np.random.choice(max_attempt_list)
            # get idx
            max_attempt_idx = np.random.choice(attempt_idx[max_attempt_sample]) 
            # check if the max attempt is correct
            maj_correct = int(oft_full_result_g[i][max_attempt_idx] > 0)
            majority_vote_acc.append(maj_correct)
            
        else:
            majority_vote_acc.append(0)
            best_n_acc.append(0)
    
    # Best-of-n accuracy
    majority_vote_rate = np.mean(majority_vote_acc)
    best_n_rate = np.mean(best_n_acc)
    print("Best-of-n accuracy", best_n_rate*100)
    print("Majority Voting acc", majority_vote_rate*100)

    # plotting

    # Apply Seaborn style
    sns.set_theme(style="whitegrid", context="talk")

    # Define n values and labels
    n_values = [1, 2, 4, 8, 16, 32, 64]
    x_labels = list(map(str, n_values))

    # Best-of-n accuracy
    best_of_n = [17, 25, 37, 50, 61, 66, 70]

    # Majority vote accuracy across 5 seeds (only Pass@4 to Pass@64)
    mv_matrix = np.array([
    [17.5, 18.0, 17.5, 15.0, 18.5],  # Pass@4
    [17.0, 22.0, 19.5, 16.5, 16.5],  # Pass@8
    [19.5, 18.0, 19.5, 17.5, 19.0],  # Pass@16
    [20.5, 19.0, 21.0, 19.0, 17.0],  # Pass@32
    [19.0, 18.5, 21.0, 19.5, 20.5],  # Pass@64
    ])

    # Compute mean and std
    mv_mean = mv_matrix.mean(axis=1)
    mv_std = mv_matrix.std(axis=1)
    mv_x = n_values[2:]  # [4, 8, 16, 32, 64]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, best_of_n, label="best-of-n", marker='o', color='tab:blue', linewidth=2, markersize=6)
    ax.errorbar(mv_x, mv_mean, yerr=mv_std, label="majority vote", marker='o',
                color='tab:red', linewidth=2, capsize=4, markersize=6)

    # Axes settings
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (log-scale)", fontsize=14)
    ax.set_ylabel("Accuracy %", fontsize=14)
    ax.set_title("Direct Solution Model Test Performance", fontsize=16)
    ax.set_xticks(n_values)
    ax.set_xticklabels(x_labels)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    # Save and show
    plt.tight_layout()
    plt.savefig("countdown_oft_majority_vote.pdf")
    
    return


if __name__ == '__main__':
    # analyze_path_difficulty()
    
    # load_sft_results(
    #     run_type='/n/netscratch/dam_lab/Lab/sqin/reason/sos/grpo-countdown-sft-dfs-3M-qwen/hf_model', 
    #     offsets=[0], 
    #     num=200
    # )

    # load_oft_results(
    #     run_type='/n/netscratch/dam_lab/Lab/sqin/reason/sos/oft-countdown-cd3e5-17M-qwen', 
    #     offsets=[0], 
    #     gens=[64],
    #     num=200
    # )

    # compare_sft_and_oft()
    # compare_sft_and_oft_alt()

    # compare_sft_and_sft_short()

    # compare_sft_and_grpo()

    # compare_sft_dfs_and_sft_short_and_sft_mix()

    # compare_deep_and_default()

    # figure_1()

    oft_grpo_training_dynamics()

    # oft_majority_vote()