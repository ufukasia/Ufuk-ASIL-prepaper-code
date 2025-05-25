import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import sys

# --- Configuration and Setup ---
# Set the default Plotly renderer to open plots in a browser window.
pio.renderers.default = "browser"

# Parameter Selection:
# The script analyzes the impact of a specific parameter on RMSE.
# This parameter can be provided as a command-line argument.
# If no argument is given, a default parameter is used.
if len(sys.argv) > 1:
    selected_param = sys.argv[1]
else:
    selected_param = "Culled_Norm_Min"  # Default parameter for analysis

# Validate the selected parameter against a list of known valid parameters.
valid_params = ["Entropy_Norm_Min", "Pose_Chi2_Norm_Min", "Culled_Norm_Min"]
if selected_param not in valid_params:
    print(f"Error: '{selected_param}' is not a valid parameter.")
    print(f"Valid parameters: {', '.join(valid_params)}")
    selected_param = "Entropy_Norm_Min"  # Fallback to a default if input is invalid
    print(f"Default parameter '{selected_param}' will be used.")

print(f"Selected parameter for analysis: {selected_param}")

# Load data from the results CSV file.
file_path = "results.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Results CSV file not found at '{file_path}'. Ensure 'main.py' has been run to generate it.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

# Ensure the selected parameter column exists in the DataFrame.
if selected_param not in df.columns:
     print(f"Error: Selected parameter '{selected_param}' not found in {file_path}. "
           "This might indicate that 'main.py' was run without this parameter, "
           "or the parameter name in 'results.csv' is different.")
     sys.exit(1)

# Data Processing: RMSE_Position
# The 'RMSE_Position' column in the CSV might store a list of RMSE values as a string.
# This section processes this column to calculate a single mean RMSE value per row.
# This involves parsing string representations of lists of RMSE values,
# converting them to numeric arrays, and calculating their mean.
try:
    # Convert stringified list "[val1 val2 ...]" to a mean of numeric values.
    df["RMSE_Position"] = df["RMSE_Position"].apply(
        lambda x: np.mean(np.fromstring(str(x).strip("[]"), sep=" ")) if isinstance(x, str) and x.strip("[]") else np.nan
    )
    # Remove rows where RMSE_Position could not be computed (resulting in NaN).
    df.dropna(subset=["RMSE_Position"], inplace=True)
    if df.empty:
        print("Error: No valid RMSE_Position data found after processing.")
        sys.exit(1)
except Exception as e:
    print(f"Error processing RMSE_Position column: {e}")
    sys.exit(1)

# Get a unique list of all sequences present in the data.
all_sequences = df["Sequence"].unique()

# Define a color cycle for plotting. Data from the same sequence (both Adaptive=True and False)
# will share the same base color for visual consistency.
colors = [
    "blue", "red", "green", "orange", "purple", "brown",
    "pink", "gray", "cyan", "magenta"
]
# --- End Configuration and Setup ---

# --- Plot 1: Mean RMSE vs. Selected Parameter for All Sequences ---
# This plot visualizes how the mean RMSE_Position changes with the `selected_param`.
# - For 'Adaptive = False' runs, a horizontal line represents the mean RMSE (as the parameter is not varied).
# - For 'Adaptive = True' runs, points show the mean RMSE for each value of `selected_param`.
fig = go.Figure()

for i, sequence in enumerate(all_sequences):
    df_seq = df[df["Sequence"] == sequence].copy()

    # Seçilen parametre kolonunu sayısal yap, hataları NaN yap
    df_seq[selected_param] = pd.to_numeric(df_seq[selected_param], errors='coerce')
    # Drop rows where the selected parameter or RMSE is NaN after conversion.
    df_seq.dropna(subset=[selected_param], inplace=True)
    if df_seq.empty:
        # print(f"Warning: No valid data for sequence '{sequence}' and parameter '{selected_param}' after cleaning. Skipping this sequence for Plot 1.")
        continue

    # Assign a color for the current sequence.
    chosen_color = colors[i % len(colors)]

    # Plot data for 'Adaptive = False'
    df_false = df_seq[df_seq["Adaptive"] == False]
    if not df_false.empty:
        # Calculate the single mean RMSE for Adaptive=False runs.
        rmse_val_false = df_false["RMSE_Position"].mean()

        # Determine the x-axis range for the horizontal line from Adaptive=True data for this sequence.
        seq_param_vals_true = df_seq[df_seq["Adaptive"] == True][selected_param].unique()
        if len(seq_param_vals_true) > 0:
            min_val = min(seq_param_vals_true)
            max_val = max(seq_param_vals_true)
            if min_val == max_val:
                x_line = [min_val, min_val + 0.0001]
            else:
                x_line = np.linspace(min_val, max_val, 50)
            y_line = [rmse_val_false] * len(x_line)

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name=f"{sequence} (False)",
                line=dict(color=chosen_color),
                hovertemplate=f'{selected_param}: %{{x}}<br>RMSE: %{{y:.4f}}<extra></extra>'
            ))

    # ---- Adaptif Olan (True) ----
    df_true = df_seq[df_seq["Adaptive"] == True]
    if not df_true.empty:
        # For Adaptive=True, group by the selected parameter and calculate mean RMSE for each parameter value.
        param_rmse_mean_true = df_true.groupby(selected_param)["RMSE_Position"].mean().reset_index()

        # Plot as markers.
        fig.add_trace(go.Scatter(
            x=param_rmse_mean_true[selected_param],
            y=param_rmse_mean_true["RMSE_Position"],
            mode='markers',
            name=f"{sequence} (True)",
            marker=dict(color=chosen_color),
            hovertemplate=f'{selected_param}: %{{x}}<br>RMSE: %{{y:.4f}}<extra></extra>'
        ))

# Annotation for Overall Minimum Average RMSE (for Adaptive=True data):
# This identifies the value of `selected_param` that yields the lowest average RMSE
# *across all sequences where that parameter value was tested with Adaptive=True*.
df_all_true = df[df["Adaptive"] == True].copy()
if not df_all_true.empty:
    # Ensure selected_param is numeric and drop NaNs.
    df_all_true[selected_param] = pd.to_numeric(df_all_true[selected_param], errors='coerce')
    df_all_true.dropna(subset=[selected_param, "RMSE_Position"], inplace=True)

    # Identify sequences that have Adaptive=True data.
    sequences_adaptive = df_all_true["Sequence"].unique()
    total_seq_count = len(sequences_adaptive)

    if total_seq_count > 0:
        # Group by the selected parameter. For each parameter value, calculate:
        # 1. The mean RMSE across all sequences that used this parameter value.
        # 2. The number of unique sequences that used this parameter value.
        overall_stats = df_all_true.groupby(selected_param).agg(
            mean_rmse=("RMSE_Position", "mean"),
            seq_count=("Sequence", lambda x: x.nunique())
        ).reset_index()

        # Filter for parameter values that were present in ALL Adaptive=True sequences.
        # This ensures a fair comparison for finding the "overall best" parameter value.
        complete_params = overall_stats[overall_stats["seq_count"] == total_seq_count]

        if not complete_params.empty:
            # Among these common parameter values, find the one with the minimum average RMSE.
            min_rmse_row = complete_params.loc[complete_params["mean_rmse"].idxmin()]
            min_param_val = min_rmse_row[selected_param]
            min_rmse_val = min_rmse_row["mean_rmse"]

            # Add an annotation to the plot marking this overall minimum.
            fig.add_annotation(
                x=min_param_val,
                y=min_rmse_val,
                text=f"Min Avg RMSE ({min_rmse_val:.4f})<br>at {selected_param}={min_param_val}",
                showarrow=True,
                arrowhead=1,
                ax=0,  # Horizontal offset of the arrow tail
                ay=-40 # Vertical offset of the arrow tail (points downwards)
            )
        else:
            print(f"Uyarı: Tüm sekanslar için ortak olan bir '{selected_param}' değeri bulunamadı. Minimum ortalama RMSE anotasyonu eklenemedi.")
    else:
         print("Uyarı: Adaptive=True için geçerli veri bulunamadı.")


fig.update_layout(
    title=f"Mean RMSE_Position versus {selected_param} for All Sequences",
    xaxis_title=f"{selected_param} Value",
    yaxis_title="Mean RMSE_Position",
    legend_title="Sequences (Adaptive state)",
    hovermode="closest"
)

fig.show()

# --- Plot 2: Subplots for Sequence 'MH05' - RMSE vs. All Valid Parameters ---
# This plot focuses on a single sequence (e.g., 'MH05') and shows how RMSE_Position
# varies with *each* of the `valid_params` in separate subplots.
# This allows for a detailed view of parameter sensitivity for a specific dataset.
sequence_name = "MH05"
df_filtered = df[df["Sequence"] == sequence_name].copy()

if not df_filtered.empty:
    print(f"\nGenerating subplots for sequence: {sequence_name}")

    # Parameters to be plotted in subplots.
    params_for_subplots = ["Entropy_Norm_Min", "Pose_Chi2_Norm_Min", "Culled_Norm_Min"]
    # Filter this list to include only parameters actually present in the DataFrame for this sequence.
    valid_subplot_params = [p for p in params_for_subplots if p in df_filtered.columns]

    if not valid_subplot_params:
        print(f"Error: None of the parameters {params_for_subplots} found in the data for sequence {sequence_name} to generate subplots.")
    else:
        # Create subplots, one for each valid parameter.
        num_subplots = len(valid_subplot_params)
        fig_subplots = make_subplots(rows=num_subplots, cols=1,
                                     subplot_titles=[f"Mean RMSE_Position versus {param} Value" for param in valid_subplot_params])

        # Iterate through each valid parameter and create its subplot.
        for i, param in enumerate(valid_subplot_params):
            row = i + 1
            col = 1

            # Ensure the parameter column is numeric and handle NaNs.
            df_filtered[param] = pd.to_numeric(df_filtered[param], errors='coerce')
            df_filtered_param = df_filtered.dropna(subset=[param, "RMSE_Position"])

            if not df_filtered_param.empty:
                # Group by the current parameter and calculate mean RMSE.
                param_rmse_mean = df_filtered_param.groupby(param)["RMSE_Position"].mean().reset_index()
                fig_subplots.add_trace(
                    go.Scatter(
                        x=param_rmse_mean[param],
                        y=param_rmse_mean["RMSE_Position"],
                        mode='lines+markers',
                        name=param,
                        hovertemplate=f'{param}: %{{x}}<br>RMSE: %{{y:.4f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
            else:
                print(f"Warning: No valid data for parameter '{param}' in sequence '{sequence_name}' for subplot generation.")

            fig_subplots.update_xaxes(title_text=f"{param} Value", row=row, col=col)
            fig_subplots.update_yaxes(title_text="Mean RMSE_Position", row=row, col=col)

        fig_subplots.update_layout(
            height=300 * num_subplots, # Adjust height based on the number of subplots.
            width=1200,
            title_text=f"Selected Parameters for Sequence: {sequence_name}",
            showlegend=False
        )

        fig_subplots.show()
else:
    print(f"Error: Sequence '{sequence_name}' not found in the dataset. Cannot generate subplots.")
