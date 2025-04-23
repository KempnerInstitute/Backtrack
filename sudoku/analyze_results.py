import numpy as np
import matplotlib.pyplot as plt
import json
import os
from solver import check_sudoku_solution, convert_full_trace_shortcut_solution_fuzzy, convert_full_trace_shortcut_solution
from eval import convert_sudoku_str_to_arr
from tokenization import data_preprocessing, load_tokenizer
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
output_dir = '/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/outputs'

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
        try:
            model_output += data['model_output']
        except:
            model_output += data['model_output_gen']
    return acc, model_output

def load_sft_results(run_type, ckpt, offsets, num):
    sft_strat_files = []
    for o in offsets:
        # sft_strat_files.append(f'{run_type}/results_sudoku_data_sample.json_{num}_{o}_ckpt{ckpt}_@1')
        sft_strat_files.append(f'{run_type}/results_easy_sudoku_strategy_data_sample.json_{num}_{o}_ckpt{ckpt}_@1')

    sft_strat_result, sft_strat_model_output = load_results(sft_strat_files)
    print(f"{run_type} Pass@1 fully solved: {sum([1 for i in sft_strat_result if i > 0.99])} out of {len(sft_strat_result)}")
    print(f"    Average accuracy: {np.mean(sft_strat_result)*100:.1f} ", )
    
    sft_strat_result = np.array(sft_strat_result)
    print(sft_strat_result.shape)

    tokenizer, mapping, _ = load_tokenizer()
    output_lens  = []
    for output in sft_strat_model_output:
        # find the first instance of "END" token
        output = output.split("SOL_S")[1].split("END")[0]
        processed_text = data_preprocessing([output], mapping)
        tokenized_output = tokenizer(processed_text)['input_ids'][0]
        output_lens.append(len(tokenized_output))
    output_lens = np.array(output_lens)
    
    len_cutoffs = [256, 512, 1024, 2048, 4096]
    # len_cutoffs = [256, 512, 1024]
    for cutoff in len_cutoffs:
        cutoff_idx = np.argwhere(output_lens <= cutoff)
        sft_result_cutoff = sft_strat_result[cutoff_idx]
        acc = sum([1 for i in sft_result_cutoff if i > 0.99])
        print(f"{run_type} Pass@1 fully solved with output length <= {cutoff}: {acc}")

    return sft_strat_result

def backtrack_format_checking():

    # try a mode strict metric for backtrack
    solution_sample = sft_strat_model_output[0]
    # get board 
    board = solution_sample[solution_sample.find("START"):solution_sample.find("solving")]
    board = convert_sudoku_str_to_arr(board)
    print("Board: ", board)

    # convert solution into a list
    solution_sample = solution_sample[
        solution_sample.find("SOL_S"):
        solution_sample.find("SOL_E")].strip()

    solution_sample = solution_sample.split("\n")[1:] # remove "SOL_S" 

    # check if the solution is valid -- more strict
    converted_solution = convert_full_trace_shortcut_solution(solution_sample)
    is_valid = check_sudoku_solution(board, converted_solution)
    print("Is valid: ", is_valid)

    # check if the solution is valid -- less strict
    converted_solution = convert_full_trace_shortcut_solution_fuzzy(solution_sample)
    is_valid = check_sudoku_solution(board, converted_solution)
    print("Is valid: ", is_valid)

    return

def load_oft_results(run_type, ckpt, gens, offsets, num):
    oft_strat_result = []
    for g in gens:  
        oft_strat_files = []
        for o in offsets:
            # oft_strat_files.append(f'{run_type}/results_sudoku_data_sample.json_{num}_{o}_ckpt{ckpt}_@{g}')
            oft_strat_files.append(f'{run_type}/results_easy_sudoku_strategy_data_sample.json_{num}_{o}_ckpt{ckpt}_@{g}')

        oft_strat_result_g, oft_strat_model_output_g = load_results(oft_strat_files)
        oft_strat_result.append(oft_strat_result_g)

        print(f"{run_type} Pass@{g} fully solved: {sum([1 for i in oft_strat_result_g if i > 0.99])} out of {len(oft_strat_result_g)}")
        print(f"    Average accuracy: {np.mean(oft_strat_result_g)*100:.1f} ", )

    oft_strat_result = np.array(oft_strat_result)
    print(oft_strat_result.shape)

    # check length
    tokenizer, mapping, _ = load_tokenizer()
    output_lens  = []
    for output in oft_strat_model_output_g:
        # find the first instance of "END" token - take one generatation from each board
        output = output[0].split("SOL_S")[1].split("END")[0]
        processed_text = data_preprocessing([output], mapping)
        tokenized_output = tokenizer(processed_text)['input_ids'][0]
        output_lens.append(len(tokenized_output))
    output_lens = np.array(output_lens)
    print("Average output length: ", np.mean(output_lens), np.std(output_lens))

    return oft_strat_result

def plot_results():
    # plot histograms of results for full results
    plt.figure(figsize=(10, 5))
    # plt.hist(sft_full_result, bins=20, alpha=0.5, label='No Elimination Strategy (Naive)')
    plt.hist(sft_strat_result, bins=20, alpha=0.5, label='W/ Elimination Strategy', color='orange')
    plt.xlabel('Per cell accuracy')
    plt.xlim([0, 1])
    plt.ylabel('Counts')
    plt.legend()
    plt.title('Histogram of per cell accuracy for Backtrack Pass@1')
    plt.savefig('sudoku_backtrack_results.png', bbox_inches='tight')

    gens = [1, 4, 16, 32, 64]
    plt.figure(figsize=(12, 5))
    # plot histograms of results for oft results
    for i in range(len(gens)):
        plt.subplot(2, 3, i+1)
        try:
            plt.hist(oft_result[i], bins=20, alpha=0.5, label='No Elimination Strategy (Naive)')
        except:
            pass
        plt.hist(oft_strat_result[i], bins=20, alpha=0.5, label='W/ Elimination Strategy')
        plt.xlabel('Per cell accuracy')
        plt.xlim([0.2, 1.02])
        plt.ylabel('Counts')
        plt.title(f'Pass@{gens[i]}')
        
        if i == 0:
            plt.legend()
        if i == len(gens)-1:
            plt.xlabel('Per cell accuracy')
            
    plt.suptitle('Histogram of per cell accuracy for No-Backtrack Pass@k')
    plt.tight_layout()
    plt.savefig('sudoku_nobacktrack_results.png', bbox_inches='tight')

def compute_auto_generation_flops(hidden_dim, kv_dim, inter_size, n_layer, gen_len, num_gen):    
    linear_flops = 2 * hidden_dim **2 + 2 * hidden_dim * kv_dim + 3 * hidden_dim * inter_size 
    quadratic_flops = hidden_dim
    per_gen_cost = n_layer * (linear_flops * gen_len + quadratic_flops * gen_len * (gen_len + 1)/2)
    total_cost = num_gen * per_gen_cost

    return total_cost

def compare_sft_and_oft():
    # this is the fixed version
    sft_acc = [
        [0, 0.5, 0.5, 0.5, 0.5], # 3M
        [10, 20.5, 45, 51, 53], # 17M
        [9, 40.5, 66, 74.5, 77],  # 38M
        [10.5, 47.5, 72.5, 85, 89], # 144M
    ]

    oft_acc = [
        [0, 0, 0, 0.5, 1], # 3M
        [13, 25, 29.5, 44.5, 58], # 17M
        [6.5, 16, 30.0, 39.5, 58.0],  # 38M
        [15.5, 29, 40, 62, 74], # 144M
    ]

    oft_gens = [1, 2, 4, 8, 16]
    oft_base_gen_len = 332
    oft_gen_lens = [oft_base_gen_len * g for g in oft_gens]
    sft_gen_lens = [400, 512, 1024, 2048, 4096]
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

    cmap = plt.get_cmap("plasma_r")
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
    ax_sft.text(0.01, 0.95, '(D) Backtracking Models', transform=ax_sft.transAxes,
            fontsize=16, fontweight='bold', va='top')
    ax_sft.tick_params(axis='both', labelsize=14)
    ax_sft.set_ylim([0, 92])
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
    ax_oft.text(0.01, 0.95, '(E) Direct Solution Models', transform=ax_oft.transAxes,
            fontsize=16, fontweight='bold', va='top')
    ax_oft.tick_params(axis='both', labelsize=16)
    ax_oft.set_ylim([0, 92])
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
    ax_diff.text(0.01, 0.95, '(F) Performance Gap Between Models', transform=ax_diff.transAxes,
             fontsize=16, fontweight='bold', va='top')
    ax_diff.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax_diff.tick_params(axis='both', labelsize=16)
    # ax_diff.xaxis.set_major_locator(MaxxNLocator(nbins=5))
    ax_diff.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_diff.legend(fontsize=14, loc='center left', framealpha=0.0)
    ax_diff.set_ylim([None, 12])
    sns.despine()
    plt.subplots_adjust(hspace=-0.05, wspace=0.3)
    plt.suptitle('Sudoku', fontsize=24, fontweight='bold', y=0.9)
    plt.tight_layout()
    plt.savefig('sudoku_flop_sft_oft.pdf', bbox_inches='tight')

def compare_sft_and_oft_alt():
    # this is the fixed version
    sft_acc = [
        [0, 0.5, 0.5, 0.5, 0.5], # 3M
        [10, 20.5, 45, 51, 53], # 17M
        [9, 40.5, 66, 74.5, 77],  # 38M
        [10.5, 47.5, 72.5, 85, 89], # 144M
    ]

    oft_acc = [
        [0, 0, 0, 0.5, 1], # 3M
        [13, 25, 29.5, 44.5, 58], # 17M
        [6.5, 16, 30.0, 39.5, 58.0],  # 38M
        [15.5, 29, 40, 62, 74], # 144M
    ]

    oft_gens = [1, 2, 4, 8, 16]
    oft_base_gen_len = 332
    oft_gen_lens = [oft_base_gen_len * g for g in oft_gens]
    sft_gen_lens = [400, 512, 1024, 2048, 4096]
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

    cmap = plt.get_cmap("plasma_r")
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
    ax_sft.text(0.01, 0.95, '(D) Backtracking Models', transform=ax_sft.transAxes,
            fontsize=16, fontweight='bold', va='top')
    ax_sft.tick_params(axis='both', labelsize=14)
    ax_sft.set_ylim([-2, 92])
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
    ax_oft.text(0.01, 0.95, '(E) No-Backtracking Models', transform=ax_oft.transAxes,
            fontsize=16, fontweight='bold', va='top')
    ax_oft.tick_params(axis='both', labelsize=16)
    ax_oft.set_ylim([-2, 92])
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
    ax_diff.set_ylim([-1, 92])
    ax_diff.tick_params(axis='both', labelsize=14)
    ax_diff.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_diff.text(0.01, 0.95, '(F) Best Accuracy vs. FLOPs', transform=ax_diff.transAxes,
                 fontsize=16, fontweight='bold', va='top')

    ax_diff.legend(fontsize=12, loc='lower right')

    sns.despine()
    plt.subplots_adjust(hspace=-0.05, wspace=0.3)
    plt.suptitle('Sudoku', fontsize=18, fontweight='bold', y=0.9)
    plt.tight_layout()
    plt.savefig('sudoku_flop_sft_oft_alt.pdf', bbox_inches='tight')
    
    return

def compare_backtrack_steps():
    result_dir = '/n/netscratch/dam_lab/Lab/sqin/reason/sudoku/outputs'
    oft_result ='17M-oft-strat/results_sudoku_data_sample.json_200_0_ckpt160000_@64'
    sft_result = '17M-sft-strat/results_sudoku_data_sample.json_200_0_ckpt790000_@1'

    # read both json file
    with open(os.path.join(result_dir, oft_result), 'r') as f:
        data = json.load(f)
    print("oft:", data.keys())
    oft_acc = data['ratings']
    oft_model_output = data['model_output_gen']
    oft_acc_gen = np.array(data['ratings_gen']) 

    print(oft_acc_gen.shape) # 64, 200

    with open(os.path.join(result_dir, sft_result), 'r') as f:
        data = json.load(f)
    sft_acc = data['ratings']
    sft_model_output = data['model_output']

    sft_backtrack = []
    oft_pass = []
    for i in range(200):
        sft_trace = sft_model_output[i]
        # search number of backtrack steps
        backtrack_steps = sft_trace.count("revert")
        sft_backtrack.append(backtrack_steps)

        # get pass at k for oft
        oft_acc = oft_acc_gen[:, i]
        oft_pass.append(sum([1 for i in oft_acc if i > 0.99]))

    # compure the correlation
    sft_backtrack = np.array(sft_backtrack)
    oft_pass = np.array(oft_pass)
    corr = np.corrcoef(sft_backtrack, oft_pass)
    print("Correlation: ", corr)
    plt.figure(figsize=(8, 5))
    # heat map
    plt.hexbin(sft_backtrack, oft_pass, gridsize=50, cmap="plasma", bins="log")  # "log" for better visibility
    plt.colorbar(label="Density")
    plt.xlabel('Backtrack Steps')
    plt.ylabel('Pass@64 Success')
    plt.title(f'Sudoku 17M Backtrack Steps vs. Pass@64 Success \n corr={corr[0, 1]:.2f}')
    plt.savefig('backtrack_vs_pass.png', bbox_inches='tight')



    return

def compare_easy_and_default():
    sns.set_theme(style='whitegrid', context='talk')

    # Accuracy arrays
    sft_acc = np.array([
        [10, 20.5, 45, 51, 53],           # 17M
        [9, 40.5, 66, 74.5, 77],          # 38M
    ])
    oft_acc = np.array([
        [13, 25, 29.5, 44.5, 58],         # 17M
        [6.5, 16, 30.0, 39.5, 58.0],      # 38M
    ])
    sft_easy_acc = np.array([
        [43, 73.5, 81, 81.5, 81.5],   # 17M
        [42, 76, 83.5, 83.5, 83.5], # 38M
    ])
    oft_easy_acc = np.array([
        [45.5, 56.5, 70.5, 82, 89], # 17M
        [44, 52.5, 63.5, 71, 78],   # 38M
    ])

    oft_gens = [1, 2, 4, 8, 16]
    oft_base_gen_len = 200
    sft_gen_lens = [256, 512, 1024, 2048, 4096]
    sft_gen = 1

    model_sizes = [17, 38]
    model_hid_dim = [512, 512]
    model_kv_dim = [128, 128]
    model_layers = [8, 10]
    model_inter_dim = [1024, 2048]

    # Axes setup
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
        easy_diff = oft_easy_acc[i] - sft_easy_acc[i]

        # Remove NaNs
        mask = ~np.isnan(easy_diff)
        flops = flops[mask]
        default_diff = default_diff[mask]
        easy_diff = easy_diff[mask]

        # Shared widths and smaller offset
        widths = flops * 0.12  # Slightly narrower bars
        offsets = flops * 0.06  # Smaller gap between bars

        # Plot bars: Default on left, Easy on right
        ax.bar(flops - offsets, default_diff, width=widths,
               label="Original", color=color)
        ax.bar(flops + offsets, easy_diff, width=widths,
               label="Easy", color=color, alpha=0.5, hatch='//')
        
        # Style
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=16)
        # ax.set_ylim([0, 80])

        # Add legend with model size as title
        ax.legend(fontsize=14, title=f'Sudoku {model_sizes[i]}M', title_fontsize=16)

    # Axis styling (outside loop)
    axes[1].set_xlabel('FLOPs', fontsize=18)
    fig.text(0.00, 0.5, 'Accuracy Gap\nDirect Solution − Backtrack (%)',
         va='center', ha='center', rotation='vertical', fontsize=18)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('sudoku_flop_easy.pdf', bbox_inches='tight')

    return


if __name__ == '__main__':
    # load SFT results
    # load_sft_results(
    #     run_type='17M-sft-strat-easy', 
    #     ckpt=1600000, 
    #     offsets=[0], 
    #     num=200
    # )

    # load OFT results
    # load_oft_results(
    #     run_type='17M-oft-strat-easy',
    #     ckpt=1312500,
    #     gens=[1, 2, 4, 8, 16],
    #     offsets=[0],
    #     num=200
    # )

    # plot results
    compare_sft_and_oft()
    # compare_sft_and_oft_alt()

    # backtracking strategy analysis
    # compare_backtrack_steps()

    # compare_easy_and_default()