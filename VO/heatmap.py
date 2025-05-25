import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Path to the input CSV file containing VO/SLAM metrics.
file_path = r"vo_pred_aliked/mh5_ns.csv"

# Columns to be excluded from the primary analysis and heatmap.
# These typically include raw pose data, timestamps, and direct error metrics
# if the focus is on the correlation of internal system parameters with ATE.
exclude_columns = [
    "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]", # Position
    "q_RS_x []", "q_RS_y []", "q_RS_z []", "q_RS_w []", # Orientation (Quaternion)
    "#timestamp [ns]",                             # Timestamp
    "x_error_m", "y_error_m", "z_error_m",         # Per-axis error components
    "graph_chi2_error", "gba_chi2_error", "lba_chi2_error", # Chi2 errors from optimization
    "num_inliers", "num_matched_map_points",        # Feature matching and mapping statistics
    "last_num_static_stereo_map_points","last_num_fused_points" # Specific map point counts
]
# --- End Configuration ---

# Load the dataset from the specified CSV file.
df = pd.read_csv(file_path)

# Preprocessing: Clean column names by removing leading/trailing whitespace.
df.columns = df.columns.str.strip()

# Extract timestamps if available, converting them to datetime objects for time series plotting.
if "#timestamp [ns]" in df.columns:
    timestamps = pd.to_datetime(df["#timestamp [ns]"], unit="ns")
else:
    timestamps = None
# Data Cleaning and Preparation:
# Drop specified irrelevant columns. `errors='ignore'` prevents errors if a column is not found.
filtered_df = df.drop(columns=[col for col in exclude_columns if col in df.columns], errors='ignore')
# Handle missing values using forward fill followed by backward fill.
# This is a common strategy for time series data to propagate last known good values.
filtered_df = filtered_df.ffill().bfill()

# Normalization: Scale numeric features to the [0, 1] range.
# Min-max normalization is applied to ensure all features contribute equally
# to correlation analysis, regardless of their original scale.
for col in filtered_df.columns:
    if pd.api.types.is_numeric_dtype(filtered_df[col]):
        col_min = filtered_df[col].min()
        col_max = filtered_df[col].max()
        if col_max != col_min:
            filtered_df[col] = (filtered_df[col] - col_min) / (col_max - col_min)
        else: # Handle cases where all values in a column are the same.
            filtered_df[col] = 0.0

# Feature Engineering: Calculate ATE-related metrics if 'ate' column exists.
# 'ate_delta' represents the change in ATE, scaled by a factor (30) for visualization.
# 'ate_delta_abs' is the absolute change, useful for identifying magnitude of ATE fluctuations.
if "ate" in df.columns:
    original_ate = df["ate"].ffill().bfill()
    ate_delta = original_ate.diff() * 30
    ate_delta_abs = ate_delta.abs()
    filtered_df["ate_delta"] = ate_delta
    filtered_df["ate_delta_abs"] = ate_delta_abs
else:
    print("Warning: 'ate' column not found in DataFrame. 'ate_delta' and 'ate_delta_abs' could not be computed.")

# Prepare DataFrame for Correlation Analysis:
# Exclude the newly created 'ate_delta' and 'ate_delta_abs' from the primary correlation matrix
# if the goal is to see correlations of other parameters with the original 'ate'.
columns_to_drop_for_corr = []
if "ate_delta" in filtered_df.columns: columns_to_drop_for_corr.append("ate_delta")
if "ate_delta_abs" in filtered_df.columns: columns_to_drop_for_corr.append("ate_delta_abs")
corr_df = filtered_df.drop(columns=columns_to_drop_for_corr, errors='ignore')
# Select only numeric columns for correlation calculation.
corr_df_numeric = corr_df.select_dtypes(include=['number'])
corr_matrix = corr_df_numeric.corr()

# --- Visualization 1: Correlation Matrix Heatmap ---
# This heatmap visualizes the pairwise Pearson correlation coefficients between all numeric variables.
fig1, ax1 = plt.subplots(figsize=(12, 10))
fig1.subplots_adjust(top=0.963, bottom=0.23, left=0.191, right=0.988, hspace=0.2, wspace=0.2)
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5, ax=ax1)
ax1.set_title("Correlation Matrix of All Variables")
annotation = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="white", alpha=0.9),
                          arrowprops=dict(arrowstyle="->"))
annotation.set_visible(False)
def update_annotation(event):
    if event.inaxes == ax1:
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            col = int(round(x))
            row = int(round(y))
            if 0 <= row < len(corr_matrix) and 0 <= col < len(corr_matrix.columns):
                value = corr_matrix.iloc[row, col]
                text = f"{value:.2f}" if pd.api.types.is_number(value) and not pd.isna(value) else "N/A"
                annotation.xy = (col + 0.5, row + 0.5)
                annotation.set_text(text)
                annotation.set_visible(True)
            else:
                annotation.set_visible(False)
        else:
            annotation.set_visible(False)
    else:
        annotation.set_visible(False)
    fig1.canvas.draw_idle()

fig1.canvas.mpl_connect("motion_notify_event", update_annotation)
plt.show()

# --- Visualization 2: Bar Plot of Correlations with ATE ---
# This plot shows the correlation of each variable with the 'ate' (Absolute Trajectory Error).
# It helps identify parameters that are most strongly (positively or negatively) correlated with ATE.
ate_corr = pd.Series(dtype='float64') # Initialize an empty Series
if "ate" in corr_df_numeric.columns:
    ate_corr = corr_matrix["ate"].drop("ate", errors='ignore').sort_values(ascending=False)
    if not ate_corr.empty:
        plt.figure(figsize=(10, 6))
        ate_corr.plot(kind="bar")
        plt.title("Correlation of Variables with ATE")
        plt.xlabel("Variables")
        plt.ylabel("Correlation Coefficient with ATE")
        plt.tight_layout()
        plt.show()
    else:
        print("No other columns found to correlate with 'ate' (ate_corr is empty).")
else:
    print("Warning: 'ate' column not found in numeric columns for correlation. Plot 2 cannot be generated.")

# --- Visualization 3: Time Series Analysis of Key Metrics ---
# This section plots the time series of:
# 1. The top N metrics most correlated with ATE.
# 2. The 'ate_delta' (change in ATE) if available.
# This helps in observing the temporal behavior of these metrics and their relationship with ATE changes.
if timestamps is not None:
    cols_to_plot = []
    if not ate_corr.empty:
        # Select the top 2 metrics with the highest absolute correlation with ATE.
        top_abs_corr_metrics = ate_corr.abs().sort_values(ascending=False).head(2).index.tolist()
        cols_to_plot.extend(top_abs_corr_metrics)

    # Proceed if there are metrics to plot or if 'ate_delta' is available.
    if len(cols_to_plot) > 0 or "ate_delta" in filtered_df.columns:
        num_metrics_to_plot = len(cols_to_plot)
        num_subplots = num_metrics_to_plot
        if "ate_delta" in filtered_df.columns:
            num_subplots += 1

        if num_subplots > 0:
            fig3, axes_array = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots), sharex=True)

            # Ensure current_axes_list is always a list, even for a single subplot.
            if num_subplots == 1:
                current_axes_list = [axes_array]
            else:
                current_axes_list = axes_array.flatten().tolist()

            current_ax_idx = 0
            # Plot selected metrics correlated with ATE.
            for col_name in cols_to_plot:
                if col_name in filtered_df.columns and current_ax_idx < len(current_axes_list):
                    ax_to_plot_on = current_axes_list[current_ax_idx]
                    ax_to_plot_on.plot(timestamps, filtered_df[col_name], label=f"{col_name} (Normalized)")
                    ax_to_plot_on.set_title(f"Time Series of {col_name}")
                    ax_to_plot_on.legend()
                    ax_to_plot_on.set_ylabel("Normalized Value")
                    current_ax_idx += 1
                else:
                    if col_name not in filtered_df.columns:
                        print(f"Warning: Column '{col_name}' not found in filtered_df for time series plot.")
                    if current_ax_idx >= len(current_axes_list):
                         print(f"Warning: Not enough subplots available for '{col_name}'.")

            # Plot ATE Delta if available.
            if "ate_delta" in filtered_df.columns and current_ax_idx < len(current_axes_list):
                ax_to_plot_on = current_axes_list[current_ax_idx]
                ax_to_plot_on.plot(timestamps, filtered_df["ate_delta"], label="ATE Delta x30", color='green')
                ax_to_plot_on.set_title("ATE Delta Zaman Serisi (Orijinal Ölçek)")
                ax_to_plot_on.legend()
                ax_to_plot_on.set_ylabel("ATE Delta Değeri")
                current_ax_idx +=1
            elif "ate_delta" in filtered_df.columns and current_ax_idx >= len(current_axes_list):
                print(f"Warning: Not enough subplots available for ATE Delta.")

            if num_subplots > 0 and current_axes_list:
                current_axes_list[-1].set_xlabel("Timestamp")

            vlines = []
            # Add interactive vertical lines for scrubbing through the time series.
            if not timestamps.empty and current_axes_list:
                initial_x_val = timestamps.iloc[0]
                vlines = [ax.axvline(x=initial_x_val, color='r', linestyle='--', alpha=0.7)
                          for ax in current_axes_list if ax is not None]
            elif timestamps.empty:
                print("Warning: Timestamp data is empty, vertical line cannot be added.")

            def on_move(event):
                # Update vertical line position on mouse move within any subplot.
                should_update = False
                if event.inaxes is not None and event.xdata is not None:
                    for ax_check in current_axes_list:
                        if event.inaxes == ax_check:
                            should_update = True
                            break

                if should_update:
                    for vline_obj in vlines:
                        if vline_obj:
                            vline_obj.set_xdata([event.xdata, event.xdata])
                    if fig3.canvas:
                        fig3.canvas.draw_idle()

            fig3.canvas.mpl_connect('motion_notify_event', on_move)
            plt.tight_layout(pad=1.0)
            plt.show()
        else: # Case where num_subplots is 0 (no metrics to plot and no ate_delta).
            print("No metrics selected for time series plot (Plot 3).")

    elif "ate_delta" in filtered_df.columns: # Case where only ATE Delta is available for plotting.
        fig3, ax_delta = plt.subplots(1, 1, figsize=(12, 4), sharex=True)
        ax_delta.plot(timestamps, filtered_df["ate_delta"], label="ATE Delta x30", color='green')
        ax_delta.set_title("ATE Delta Time Series (Original Scale)")
        ax_delta.legend()
        ax_delta.set_xlabel("Timestamp")
        ax_delta.set_ylabel("ATE Delta Değeri")
        # Note: Interactive vline and on_move could be added here too for consistency.
        plt.tight_layout()
        plt.show()
    else:
        print("No ATE-correlated columns found and ATE Delta data is unavailable. Plot 3 (metrics) cannot be generated.")
else:
    print("Timestamp data is not available. Plot 3 (time series) cannot be generated.")