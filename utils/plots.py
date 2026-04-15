import numpy as np
import matplotlib.pyplot as plt

def moving_average(values, window=10):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")

def plot_training(rewards, losses, wait_parts=None, count_parts=None, save_path="training_results.png"):
    """
    Plots training rewards, losses, and optional component breakdowns.
    Expects step-wise data to show granularity per action selection.
    """
    n_rows = 3 if wait_parts is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4.5 * n_rows), facecolor="#1e1e2e")
    
    # Apply standard theme
    for ax in axes:
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#313244")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cba6f7")

    # Reward panel (Step-wise)
    steps = np.arange(1, len(rewards) + 1)
    # Scatter with low alpha to see density
    axes[0].scatter(steps, rewards, color="#94e2d5", alpha=0.15, s=6)
    
    if len(rewards) >= 100:
        window = 100
        sm = moving_average(rewards, window=window)
        axes[0].plot(steps[window-1:], sm, color="#94e2d5", linewidth=2.5,
                     label=f"Moving avg ({window} steps)")
    
    axes[0].set_xlabel("Decision Point (Action Selection)")
    axes[0].set_ylabel("Step Reward")
    axes[0].set_title("Training Reward (per Step)")
    axes[0].legend(facecolor="#313244", labelcolor="#cdd6f4",
                   fontsize=9, loc="lower right")
    axes[0].grid(True, color="#313244", alpha=0.6)

    # Loss panel (Step-wise)
    # Note: agents might start updating late, so we match based on the last N steps if needed,
    # but here we just plot the sequence of recorded losses.
    loss_steps = np.arange(1, len(losses) + 1)
    axes[1].plot(loss_steps, losses, color="#f38ba8", alpha=0.4, linewidth=1)
    
    if len(losses) >= 100:
        window = 100
        sm_loss = moving_average(losses, window=window)
        axes[1].plot(loss_steps[window-1:], sm_loss,
                     color="#f38ba8", linewidth=2.5, label=f"Moving avg ({window} steps)")
                     
    axes[1].set_xlabel("Update Step")
    axes[1].set_ylabel("Loss (Huber)")
    axes[1].set_title("Training Loss")
    axes[1].legend(facecolor="#313244", labelcolor="#cdd6f4")
    axes[1].grid(True, color="#313244", alpha=0.6)

    # Component panel (if provided)
    if wait_parts is not None and count_parts is not None:
        axes[2].plot(steps, wait_parts, color="#fab387", alpha=0.5, label="Wait Component")
        axes[2].plot(steps, count_parts, color="#89b4fa", alpha=0.5, label="Count Component")
        
        if len(wait_parts) >= 100:
            axes[2].plot(steps[99:], moving_average(wait_parts, 100), color="#fab387", linewidth=2)
            axes[2].plot(steps[99:], moving_average(count_parts, 100), color="#89b4fa", linewidth=2)
            
        axes[2].set_xlabel("Decision Point (Action Selection)")
        axes[2].set_ylabel("Component Reward")
        axes[2].set_title("Reward Breakdown (Harmonic Components)")
        axes[2].legend(facecolor="#313244", labelcolor="#cdd6f4")
        axes[2].grid(True, color="#313244", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [plot saved] {save_path}")
    plt.close(fig)

def plot_comparison(rl_rewards, static_rewards, save_path="evaluation_results.png"):
    """
    Plots a comparison between RL agent and static controller.
    """
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#1e1e2e")
    ax.set_facecolor("#2a2a3e")

    rl_mean, static_mean = np.mean(rl_rewards), np.mean(static_rewards)
    rl_std, static_std = np.std(rl_rewards), np.std(static_rewards)

    labels = ["RL\nAgent", "Static\nCtrl"]
    means = [rl_mean, static_mean]
    stds = [rl_std, static_std]
    colors = ["#89b4fa", "#f38ba8"]

    bars = ax.bar(labels, means, color=colors, width=0.45,
                  yerr=stds, capsize=8,
                  error_kw={"ecolor": "#cdd6f4", "lw": 2})
    
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{m:.0f}", ha="center", va="bottom",
                color="#cdd6f4", fontsize=10, fontweight="bold")

    improvement = ((rl_mean - static_mean) / abs(static_mean) * 100
                   if static_mean != 0 else 0)
    ax.set_title(f"Performance Comparison ({improvement:+.1f}%)",
                 color="#cba6f7", fontsize=12)
    ax.set_ylabel("Total Reward", color="#cdd6f4")
    ax.tick_params(colors="#cdd6f4")
    for sp in ax.spines.values():
        sp.set_color("#313244")
    ax.grid(axis="y", color="#313244", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [plot saved] {save_path}")
    plt.close(fig)
