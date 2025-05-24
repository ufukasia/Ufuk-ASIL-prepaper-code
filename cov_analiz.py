#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_and_merge(sequence, output_dir="outputs"):
    """
    Loads and merges covariance-related metrics (sigma_v and sigma_p)
    for a given sequence from specified CSV files.
    The merging is performed based on the 'Timestamp [ns]' column using an inner join.

    Args:
        sequence (str): The sequence identifier (e.g., "MH05").
        output_dir (str): Directory containing the CSV files.

    Returns:
        pd.DataFrame or None: A merged DataFrame containing sigma_v and sigma_p metrics
                              if both files are found and successfully merged, otherwise None.
    """
    # Construct file paths for sigma_v (delta metrics) and sigma_p (static metrics)
    csv_v = os.path.join(output_dir, f"adaptive_sigma_v_{sequence.lower()}.csv")
    csv_p = os.path.join(output_dir, f"adaptive_sigma_p_{sequence.lower()}.csv")

    if not os.path.exists(csv_v):
        print(f"Sigma_v file not found: {csv_v}")
        return None
    if not os.path.exists(csv_p):
        print(f"Sigma_p file not found: {csv_p}")
        return None

    df_v = pd.read_csv(csv_v, index_col=False)
    df_p = pd.read_csv(csv_p, index_col=False)

    # Clean column names by stripping leading/trailing whitespace
    df_v.columns = df_v.columns.str.strip()
    df_p.columns = df_p.columns.str.strip()

    # Define required columns for sigma_v and sigma_p data
    # These typically represent different aspects of system uncertainty or performance.
    # sigma_v (delta metrics) might relate to changes or dynamics.
    cols_v = ["Timestamp [ns]", "norm_delta_intensity", "norm_delta_pose_chi2", "norm_delta_culled_keyframes"]
    # sigma_p (static metrics) might relate to more stable or accumulated uncertainties.
    cols_p = ["Timestamp [ns]", "scaled_norm_static_inv_entropy", "scaled_norm_static_pose_chi2_error", "scaled_norm_static_culled_keyframes"]

    # Validate the presence of required columns
    for col in cols_v:
        if col not in df_v.columns:
            print(f"Required column '{col}' not found in {csv_v}")
            return None
    for col in cols_p:
        if col not in df_p.columns:
            print(f"Required column '{col}' not found in {csv_p}")
            return None

    # Standardize timestamp format to int64 after stripping potential whitespace and converting
    df_v["Timestamp [ns]"] = df_v["Timestamp [ns]"].astype(str).str.strip().astype(float).astype(np.int64)
    df_p["Timestamp [ns]"] = df_p["Timestamp [ns]"].astype(str).str.strip().astype(float).astype(np.int64)

    # Select only the required columns
    df_v = df_v[cols_v]
    df_p = df_p[cols_p]

    # Diagnostic print for checking common timestamps (can be removed in final version)
    # print("Timestamps in sigma_v data (head):", df_v["Timestamp [ns]"].head())
    # print("Timestamps in sigma_p data (head):", df_p["Timestamp [ns]"].head())

    ortak = set(df_v["Timestamp [ns]"]).intersection(set(df_p["Timestamp [ns]"]))
    print(f"Number of common timestamps for {sequence}: {len(ortak)}")

    # Merge DataFrames based on common timestamps
    df_merged = pd.merge(df_v, df_p, on="Timestamp [ns]", how="inner")
    if df_merged.empty:
        print(f"No common timestamps found for merging in {sequence}. Check data alignment.")
        return None
    return df_merged

def main():
    """
    Main function to process specified sequences, load/merge their covariance metrics,
    and generate plots visualizing these metrics over time.
    Static metrics (sigma_p) are plotted on the left Y-axis, and
    Delta metrics (sigma_v) are plotted on the right Y-axis (inverted).
    This allows for comparative analysis of different uncertainty indicators.
    """
    sequences = ["MH05", "MH01"]
    output_dir = "outputs"
    n_seq = len(sequences)

    # Create subplots, one for each sequence
    fig, axs = plt.subplots(n_seq, 1, figsize=(12, 3 * n_seq), sharex=False)
    if n_seq == 1:
        axs = [axs]

    handles_delta = []
    labels_delta = []
    handles_static = []
    labels_static = []

    for i, (ax, seq) in enumerate(zip(axs, sequences)):
        df = load_and_merge(seq, output_dir)
        if df is None:
            ax.set_title(f"{seq} data not found")
            continue

        time_ns = df["Timestamp [ns]"].values

        # Create a twin Y-axis for delta metrics (sigma_v)
        ax_delta = ax.twinx()

        is_first_seq = (i == 0)
        # Define labels for legend; only add for the first sequence to avoid duplicates
        label_static_inv_entropy = "Static Inverse Entropy" if is_first_seq else None
        label_static_pose_chi2 = r"Static Pose $\chi^2$ Error" if is_first_seq else None
        label_static_culled = "Static Culled Keyframes" if is_first_seq else None
        label_delta_intensity = r"$\Delta$ Intensity" if is_first_seq else None
        label_delta_pose_chi2 = r"$\Delta$ Pose $\chi^2$" if is_first_seq else None
        label_delta_culled = r"$\Delta$ Culled Keyframes" if is_first_seq else None

        # Plot Static metrics (sigma_p) on the left Y-axis (solid lines)
        ax.plot(time_ns, df["scaled_norm_static_inv_entropy"], label=label_static_inv_entropy, color="blue", linestyle="-", alpha=0.7)
        ax.plot(time_ns, df["scaled_norm_static_pose_chi2_error"], label=label_static_pose_chi2, color="red", linestyle="-", alpha=0.7)
        ax.plot(time_ns, df["scaled_norm_static_culled_keyframes"], label=label_static_culled, color="green", linestyle="-", alpha=0.7)

        # Plot Delta metrics (sigma_v) on the right Y-axis (dotted lines)
        ax_delta.plot(time_ns, df["norm_delta_intensity"], label=label_delta_intensity, color="blue", linestyle=":")
        ax_delta.plot(time_ns, df["norm_delta_pose_chi2"], label=label_delta_pose_chi2, color="red", linestyle=":")
        ax_delta.plot(time_ns, df["norm_delta_culled_keyframes"], label=label_delta_culled, color="green", linestyle=":")

        # Add a horizontal line at y=0 for reference on the delta axis
        ax_delta.axhline(0, color="black", linewidth=0.5)

        # Set titles and labels for axes
        ax.set_title(seq)
        ax.set_ylabel("Static Metrics (Sigma_p)")
        ax_delta.set_ylabel("Delta Metrics (Sigma_v)")
        ax.grid(True)

        # Set Y-axis limits. Static metrics are typically normalized [0,1].
        # Delta metrics are plotted on an inverted Y-axis [1,0] for specific visualization purposes.
        ax.set_ylim(0, 1)
        ax_delta.set_ylim(1, 0)
        ax.set_xlim(time_ns.min(), time_ns.max())

        # Collect legend handles and labels from the first sequence's plot
        if is_first_seq:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_delta.get_legend_handles_labels()
            handles_static.extend(h1)
            labels_static.extend(l1)
            handles_delta.extend(h2)
            labels_delta.extend(l2)

    axs[-1].set_xlabel("Timestamp [ns]")
    # A super title can be added if desired, e.g.:
    # fig.suptitle("Per-Sequence Analysis: Delta (right Y-axis) and Static (left Y-axis) Covariance Metrics", fontsize=14)

    handles = handles_static + handles_delta
    labels = labels_static + labels_delta
    fig.legend(handles, labels, loc='upper right', fontsize="small")

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to accommodate the legend
    plt.show()

if __name__ == "__main__":
    main()
