import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_traffic_phases(csv_path):
    df = pd.read_csv(csv_path)
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    phases = df['phase'].unique()
    phases.sort()
    
    # We want to show 1 for the lane group in green, 0 otherwise
    # Since there are 4 groups, let's offset them slightly for visibility if needed,
    # or use subplots. Subplots are cleaner.
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    group_names = [
        "Group 0: N-S Straight + Right",
        "Group 1: E-W Straight + Right",
        "Group 2: N-S Left Turn",
        "Group 3: E-W Left Turn"
    ]
    
    group_colors = ['red', 'blue' ,'green', 'orange']

    for i in range(4):
        # 1 if phase == i, else 0
        active = (df['phase'] == i).astype(int)
        axes[i].step(df['sim_time'], active, where='post', color=group_colors[i], linewidth=2)
        axes[i].set_ylabel("Status (0/1)")
        axes[i].set_title(group_names[i])
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].grid(True, alpha=0.3)
    
    plt.xlabel("Simulation Time (s)")
    plt.tight_layout()
    
    # Save the plot
    output_path = "traffic_phases_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Find the most recent eval log
    log_files = glob.glob("logs/eval_steps_*.csv")
    if not log_files:
        print("No evaluation logs found in logs/")
    else:
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"Plotting phases from: {latest_log}")
        plot_traffic_phases(latest_log)
