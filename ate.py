import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ###################### Configuration Settings ######################
# Toggle visibility for different Visual Odometry (VO) methods
SHOW_ALIKED = False    # Display results for ALIKED-based VO
SHOW_SUPERPOINT = True  # Display results for SuperPoint-based VO

# Column names for timestamp and position data in CSV files
time_col = '#timestamp [ns]'
pos_cols = [' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']

# Directory paths for input data
gt_dir = "gt"
aliked_dir = "vo_pred_aliked_1024"
superpoint_dir = "vo_pred_super_512"

# Visualization settings for trajectory plots
line_styles = {
    'gt': {'color': 'red', 'lw': 2, 'label': 'Ground Truth'},
    'aliked': {'color': 'green', 'lw': 1, 'ls': '-', 'label': 'ALIKED'},
    'superpoint': {'color': 'blue', 'lw': 1, 'ls': '--', 'label': 'SuperPoint'}
}
# ####################################################################

def load_and_process_data(sequence_num):
    """
    Loads and processes ground truth and VO trajectory data for a given sequence.
    It reads CSV files, converts timestamps to a common numeric type, sorts by time,
    and aligns VO data to ground truth timestamps using 'merge_asof'.

    Args:
        sequence_num (int): The numerical identifier of the dataset sequence (e.g., 1 for MH01).

    Returns:
        dict: A dictionary containing pandas DataFrames for 'gt', 'aliked', and 'superpoint'.
              Returns None if ground truth data cannot be loaded.
    """
    results = {'gt': None, 'aliked': None, 'superpoint': None}

    # Load Ground Truth data
    gt_file = os.path.join(gt_dir, f"mh{sequence_num}_gt.csv")
    try:
        results['gt'] = pd.read_csv(gt_file)
        results['gt'][time_col] = pd.to_numeric(results['gt'][time_col]).astype(np.int64)
        results['gt'].sort_values(time_col, inplace=True)
    except Exception as e:
        print(f"Error reading Ground Truth for MH{sequence_num}: {e}")
        return None

    # Load ALIKED VO data if enabled
    if SHOW_ALIKED:
        aliked_file = os.path.join(aliked_dir, f"mh{sequence_num}_ns.csv")
        try:
            df = pd.read_csv(aliked_file)
            df[time_col] = pd.to_numeric(df[time_col]).astype(np.int64)
            df.sort_values(time_col, inplace=True)
            # Align ALIKED data with ground truth timestamps
            results['aliked'] = pd.merge_asof(df, results['gt'], on=time_col, suffixes=('_pred', '_gt'))
        except Exception as e:
            print(f"Error reading ALIKED data for MH{sequence_num}: {e}")
            results['aliked'] = None # Continue processing other sequences if this one fails

    # Load SuperPoint VO data if enabled
    if SHOW_SUPERPOINT:
        superpoint_file = os.path.join(superpoint_dir, f"mh{sequence_num}_ns.csv")
        try:
            df = pd.read_csv(superpoint_file)
            df[time_col] = pd.to_numeric(df[time_col]).astype(np.int64)
            df.sort_values(time_col, inplace=True)
            # Align SuperPoint data with ground truth timestamps
            results['superpoint'] = pd.merge_asof(df, results['gt'], on=time_col, suffixes=('_pred', '_gt'))
        except Exception as e:
            print(f"Error reading SuperPoint data for MH{sequence_num}: {e}")
            results['superpoint'] = None # Continue processing other sequences if this one fails

    return results

def calculate_metrics(aligned_df, method_name):
    """
    Calculates Absolute Trajectory Error (ATE) and per-axis RMSE from aligned data.
    ATE is computed as the RMSE of the 3D position errors.

    Args:
        aligned_df (pd.DataFrame): DataFrame with '_pred' and '_gt' columns for x, y, z positions.
        method_name (str): Name of the VO method for reporting.

    Returns:
        str: A formatted string containing ATE and RMSE metrics.
             Returns a message if input data is None.
    """
    if aligned_df is None:
        return f"{method_name} data not available"

    rmse_x = np.sqrt(np.mean((aligned_df[f"{pos_cols[0]}_pred"] - aligned_df[f"{pos_cols[0]}_gt"])**2))
    rmse_y = np.sqrt(np.mean((aligned_df[f"{pos_cols[1]}_pred"] - aligned_df[f"{pos_cols[1]}_gt"])**2))
    rmse_z = np.sqrt(np.mean((aligned_df[f"{pos_cols[2]}_pred"] - aligned_df[f"{pos_cols[2]}_gt"])**2))
    # ATE is the root of the sum of squared per-axis RMSEs, equivalent to 3D RMSE.
    ate = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)

    return f"{method_name}\nATE: {ate:.2f}m\nRMSE X: {rmse_x:.2f}m\nY: {rmse_y:.2f}m\nZ: {rmse_z:.2f}m"

# Visualization setup
nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8), squeeze=False) # Create a 2x3 grid of subplots
axes = axes.flatten()
num_sequences_processed = 0

for seq in range(1, 6):
    ax = axes[seq-1]
    data = load_and_process_data(seq) # Load data for the current sequence

    title_seq_num = f"{seq:02d}" # Format sequence number for titles (e.g., 01, 02)

    if data is None or data['gt'] is None:
        ax.set_title(f"MH{title_seq_num} - Data Unavailable")
        continue

    # Plot Ground Truth trajectory
    ax.plot(data['gt'][pos_cols[0]], data['gt'][pos_cols[1]], **line_styles['gt'])

    # Initialize text for displaying metrics on the plot
    metrics_text = ""

    # Plot ALIKED trajectory and calculate/add its metrics if enabled and available
    if SHOW_ALIKED and data['aliked'] is not None:
        ax.plot(data['aliked'][f"{pos_cols[0]}_pred"],
                data['aliked'][f"{pos_cols[1]}_pred"],
                **line_styles['aliked'])
        metrics_text += calculate_metrics(data['aliked'], "ALIKED") + "\n\n"

    # Plot SuperPoint trajectory and calculate/add its metrics if enabled and available
    if SHOW_SUPERPOINT and data['superpoint'] is not None:
        ax.plot(data['superpoint'][f"{pos_cols[0]}_pred"],
                data['superpoint'][f"{pos_cols[1]}_pred"],
                **line_styles['superpoint'])
        metrics_text += calculate_metrics(data['superpoint'], "SuperPoint")

    # Configure plot aesthetics
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("X (m)")
    if (seq - 1) % ncols == 0: # Add Y-axis label only to the first plot in each row
        ax.set_ylabel("Y (m)")
    ax.set_title(f"MH{title_seq_num}")
    num_sequences_processed += 1

    # Display metrics text on the plot
    ax.text(0.05, 0.95, metrics_text.strip(), transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

plt.suptitle("Trajectory Comparison" + 
             (" - ALIKED" if SHOW_ALIKED else "") +
             (" - SuperPoint" if SHOW_SUPERPOINT else ""), y=0.98) # Adjust super title position

# Hide any unused subplots if the number of sequences is less than nrows*ncols
for i in range(num_sequences_processed, nrows * ncols):
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
output_filename = "comparison"
output_filename += "_aliked" if SHOW_ALIKED else ""
output_filename += "_superpoint" if SHOW_SUPERPOINT else ""
plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
plt.show()