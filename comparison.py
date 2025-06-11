import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

def read_position_data(dataset_path, dataset_name):
    """
    Reads ground truth position data from a CSV file.
    The CSV file is expected to contain timestamped 3D position (x, y, z).
    This data serves as the reference for evaluating estimated trajectories.
    """
    imu_csv = dataset_path / 'imu_interp_gt' / f"{dataset_name}_imu_with_interpolated_groundtruth.csv"
    if not imu_csv.exists():
        # The file path is constructed based on a predefined directory structure.
        raise FileNotFoundError(f"Ground truth data file not found: {imu_csv}")

    df = pd.read_csv(imu_csv)

    required_cols = ["#timestamp [ns]", " p_RS_R_x [m]", " p_RS_R_y [m]", " p_RS_R_z [m]"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column not found in ground truth data: {col}")

    t = df["#timestamp [ns]"].values.astype(np.float64)
    x = df[" p_RS_R_x [m]"].values
    y = df[" p_RS_R_y [m]"].values
    z = df[" p_RS_R_z [m]"].values
    return t, x, y, z

def read_vo_data(dataset_path, dataset_name):
    """
    Reads Visual Odometry (VO) position data from a CSV file.
    The CSV file is expected to contain timestamped 3D position (x, y, z)
    as estimated by a VO algorithm.
    """
    try:
        # Extracts numerical index from dataset name (e.g., "MH01" -> 1).
        index = int(dataset_name[2:])
    except Exception as e:
        raise ValueError(f"Dataset name format is not as expected: {dataset_name}") from e

    vo_csv = dataset_path / 'VO'/ f"mh{index}_ns.csv"
    if not vo_csv.exists():
        # The file path is constructed based on a predefined directory structure.
        raise FileNotFoundError(f"VO data file not found: {vo_csv}")

    df = pd.read_csv(vo_csv)

    required_cols = ["#timestamp [ns]", " p_RS_R_x [m]", " p_RS_R_y [m]", " p_RS_R_z [m]"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column not found in VO data: {col}")

    t = df["#timestamp [ns]"].values.astype(np.float64)
    x = df[" p_RS_R_x [m]"].values
    y = df[" p_RS_R_y [m]"].values
    z = df[" p_RS_R_z [m]"].values
    return t, x, y, z

def read_adaptive_data(dataset_path, dataset_name):
    """
    Reads position data from an adaptive algorithm (e.g., adaptive VIO/filter output).
    The CSV file is expected to contain timestamped 3D position (x, y, z)
    as estimated by the adaptive method.
    """
    try:
        # Extracts numerical index from dataset name (e.g., "MH01" -> 1).
        index = int(dataset_name[2:])
    except Exception as e:
        raise ValueError(f"Dataset name format is not as expected: {dataset_name}") from e

    adaptive_csv = dataset_path / f"outputs/adaptive_mh{index:02d}.csv"
    if not adaptive_csv.exists():
        # The file path is constructed based on a predefined directory structure.
        raise FileNotFoundError(f"Adaptive algorithm data file not found: {adaptive_csv}")

    df = pd.read_csv(adaptive_csv)

    required_cols = ["#timestamp [ns]", " p_RS_R_x [m]", " p_RS_R_y [m]", " p_RS_R_z [m]"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column not found in adaptive data: {col}")

    t = df["#timestamp [ns]"].values.astype(np.float64)
    x = df[" p_RS_R_x [m]"].values
    y = df[" p_RS_R_y [m]"].values
    z = df[" p_RS_R_z [m]"].values
    return t, x, y, z

def align_trajectories(t_gt, x_gt, y_gt, z_gt, t_pred, x_pred, y_pred, z_pred):
    """
    Aligns predicted trajectory data with ground truth data based on timestamps.
    This involves:
    1. Finding the overlapping time segment between ground truth and prediction.
    2. Interpolating the predicted trajectory data to match the ground truth timestamps
       within this overlapping segment.
    This alignment is crucial for consistent error metric calculation (e.g., ATE).
    """
    t_start = max(t_gt[0], t_pred[0])
    t_end   = min(t_gt[-1], t_pred[-1])

    valid = (t_gt >= t_start) & (t_gt <= t_end)
    t_gt_valid = t_gt[valid]
    x_gt_valid = x_gt[valid]
    y_gt_valid = y_gt[valid]
    z_gt_valid = z_gt[valid]
    
    x_pred_interp = np.interp(t_gt_valid, t_pred, x_pred)
    y_pred_interp = np.interp(t_gt_valid, t_pred, y_pred)
    z_pred_interp = np.interp(t_gt_valid, t_pred, z_pred)
    
    return t_gt_valid, x_gt_valid, y_gt_valid, z_gt_valid, x_pred_interp, y_pred_interp, z_pred_interp

def compute_ate(x_gt, y_gt, z_gt, x_pred, y_pred, z_pred):
    """
    Computes the Absolute Trajectory Error (ATE) as Root Mean Squared Error (RMSE)
    between aligned ground truth and predicted 3D positions.
    ATE is a standard metric for evaluating the global consistency of a trajectory.
    """
    squared_errors = (x_gt - x_pred)**2 + (y_gt - y_pred)**2 + (z_gt - z_pred)**2
    ate = np.sqrt(np.mean(squared_errors))
    return ate

def plot_2d_trajectories(datasets, base_path=Path("."), margin=0.1):
    """
    Generates and displays 2D (X-Y plane) plots of trajectories for multiple datasets.
    Each subplot shows:
    - Ground Truth (GT) trajectory.
    - Visual Odometry (VO) trajectory.
    - Adaptive algorithm's trajectory (if available).
    - Calculated ATE values for VO and the adaptive method.
    This visualization aids in qualitative and quantitative comparison of different
    odometry/SLAM approaches.
    """
    num_datasets = len(datasets)
    if num_datasets == 0:
        print("No datasets selected for plotting.")
        return

    # Determine a reasonable grid layout based on the number of datasets
    if num_datasets <= 3:
        ncols = num_datasets
        nrows = 1
    elif num_datasets == 4:
        ncols = 2
        nrows = 2
    elif num_datasets == 5: # Arrange as 2 rows, 3 columns, last one potentially empty
        ncols = 3
        nrows = 2
    elif num_datasets == 6:
        ncols = 3
        nrows = 2
    else:  # For more than 6 datasets, use 3 columns and calculate rows
        ncols = 3
        nrows = (num_datasets + ncols - 1) // ncols

    fig, all_axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = all_axes.flatten()

    for idx, ds_name in enumerate(datasets):
        if idx >= len(axes_flat): # Should not happen with correct nrows/ncols logic
            print(f"Warning: Trying to plot dataset {ds_name} but not enough subplots. Skipping.")
            continue
        ax = axes_flat[idx]

        try:
            # Read ground truth data.
            t_gt, x_gt, y_gt, z_gt = read_position_data(base_path, ds_name)
            # Read Visual Odometry (VO) data.
            t_vo, x_vo, y_vo, z_vo = read_vo_data(base_path, ds_name)

            # Attempt to read data from the adaptive algorithm.
            try:
                t_ad, x_ad, y_ad, z_ad = read_adaptive_data(base_path, ds_name)
                adaptive_available = True
            except FileNotFoundError as fnf:
                adaptive_available = False
                print(f"Adaptive data not found for {ds_name}: {fnf}")

            # Align VO data with ground truth and compute ATE.
            t_valid_vo, x_gt_vo, y_gt_vo, z_gt_vo, x_vo_interp, y_vo_interp, z_vo_interp = align_trajectories(
                t_gt, x_gt, y_gt, z_gt, t_vo, x_vo, y_vo, z_vo)
            ate_vo = compute_ate(x_gt_vo, y_gt_vo, z_gt_vo, x_vo_interp, y_vo_interp, z_vo_interp)

            if adaptive_available:
                # Align adaptive data with ground truth and compute ATE.
                t_valid_ad, x_gt_ad, y_gt_ad, z_gt_ad, x_ad_interp, y_ad_interp, z_ad_interp = align_trajectories(
                    t_gt, x_gt, y_gt, z_gt, t_ad, x_ad, y_ad, z_ad)
                ate_ad = compute_ate(x_gt_ad, y_gt_ad, z_gt_ad, x_ad_interp, y_ad_interp, z_ad_interp)
            else:
                ate_ad = None

            # Plot trajectories: GT (red), VO (blue), Adaptive (green, dashed if available).
            ax.plot(x_gt, y_gt, label="GT", color='red', linewidth=2)
            ax.plot(x_vo, y_vo, label="VO", color='blue', linewidth=1)
            if adaptive_available:
                ax.plot(x_ad, y_ad, label="Adaptive", color='green', linewidth=1, linestyle='--')
            else:
                ax.text(0.05, 0.85, "Adaptive data N/A", transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

            # Display ATE values on the plot.
            if ate_ad is not None:
                ax.text(0.05, 0.95, f"ATE VO: {ate_vo:.3f} m\nATE Adaptive: {ate_ad:.3f} m",
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
            else:
                ax.text(0.05, 0.95, f"ATE VO: {ate_vo:.3f} m", transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

            ax.set_title(ds_name)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.legend(loc='best')
            ax.grid(True)

            # Adjust axis limits to encompass all plotted trajectories with a margin.
            all_x = np.concatenate([x_gt, x_vo] + ([x_ad] if adaptive_available else []))
            all_y = np.concatenate([y_gt, y_vo] + ([y_ad] if adaptive_available else []))
            x_range = all_x.max() - all_x.min()
            y_range = all_y.max() - all_y.min()

            # Handle cases where range is zero (e.g., single point data or all data is identical)
            if x_range == 0: x_range = 1.0 # Default range if no variation
            if y_range == 0: y_range = 1.0 # Default range if no variation

            # Determine the larger dimension for setting equal aspect ratio limits
            max_dim_range = max(x_range, y_range) / 2.0
            expanded_range = max_dim_range * (1 + margin) # Add margin

            mid_x = (all_x.max() + all_x.min()) * 0.5
            mid_y = (all_y.max() + all_y.min()) * 0.5
            ax.set_xlim(mid_x - expanded_range, mid_x + expanded_range)
            ax.set_ylim(mid_y - expanded_range, mid_y + expanded_range)
            ax.set_aspect('equal', adjustable='box')

        except Exception as e:
            print(f"Error processing position data for {ds_name}: {e}")
            ax.text(0.5, 0.5, "Data could not be loaded/processed", transform=ax.transAxes,
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(ds_name) # Still set title for problematic plots
            ax.axis('off') # Turn off axis for problematic plots

    # Hide any unused subplots if the grid is larger than num_datasets
    for i in range(num_datasets, nrows * ncols):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Define the order of datasets for plotting.
    # Top row: MH05, MH04, MH03; Bottom row: MH02, MH01.
    mh_datasets = [
        "MH05",
        "MH04"
    ]
    base_path = Path(".")
    plot_2d_trajectories(mh_datasets, base_path, margin=0.1)