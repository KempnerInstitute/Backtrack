import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from analyze_results import compute_auto_generation_flops

def create_grpo_transition_gif():
    sns.set_theme(style="whitegrid", context="talk")

    # Accuracy values
    sft_acc = np.array([7.5, 11.5, 25.5, 35, 41.5])
    sft_grpo_acc = np.array([23.5, 31, 41, 52.5, 70.5])
    oft_acc = np.array([17, 25, 37, 50, 61, 70])
    oft_grpo_acc = np.array([42.5, 43.5, 44, 47, 47, 48])

    sft_gen_lens = [256, 512, 1024, 2048, 4096]
    oft_gens = [1, 2, 4, 8, 16, 32]
    oft_base_gen_len = 212

    # Compute FLOPs
    sft_flops = [compute_auto_generation_flops(512, 128, 1024, 8, g, 1) for g in sft_gen_lens]
    oft_flops = [compute_auto_generation_flops(512, 128, 1024, 8, oft_base_gen_len, g) for g in oft_gens]

    # Animation params
    n_transition_frames = 30
    n_hold_frames = 10
    alpha_seq = np.concatenate([
        np.linspace(0, 1, n_transition_frames),
        np.ones(n_hold_frames)  # hold at final post-GRPO values
    ])
    n_frames = len(alpha_seq)

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 6))
    grpo_sft_color = "#3c8dbc"
    grpo_oft_color = "#e24a33"

    def update(frame):
        ax.clear()
        alpha = alpha_seq[frame]

        # Interpolated values for animation
        sft_interp = (1 - alpha) * sft_acc + alpha * sft_grpo_acc
        oft_interp = (1 - alpha) * oft_acc + alpha * oft_grpo_acc

        # Static pre-GRPO curves
        # Animated post-GRPO curves (fainter)
        ax.plot(sft_flops, sft_acc, marker='o', color=grpo_sft_color,
                linewidth=3, label='Backtracing Model (Before)')
        ax.plot(sft_flops, sft_interp, marker='o', color=grpo_sft_color,
                linewidth=3, linestyle='-', alpha=0.6, label='After GRPO')
        ax.plot(oft_flops, oft_acc, marker='o', color=grpo_oft_color,
                linewidth=3, linestyle='--', label='Direct Solution Model (Before)')
        ax.plot(oft_flops, oft_interp, marker='o', color=grpo_oft_color,
                linewidth=3, linestyle='--', alpha=0.6, label='After GRPO')

        ax.set_xscale('log')
        ax.set_ylim(0, 75)
        ax.set_xlabel("FLOPs", fontsize=14)
        ax.set_ylabel("Unseen Problem Solved (%)", fontsize=14)
        ax.set_title("Finetuning Models with GRPO ", fontsize=16)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, repeat=False)
    gif_path = "grpo_animation.gif"
    anim.save(gif_path, writer='pillow', fps=10)
    plt.close()

    return gif_path
if __name__ == "__main__":

    gif_path = create_grpo_transition_gif()
