import os
import pandas as pd
import csv
import numpy as np

# Define the minimum and maximum bounds for the adaptive visual position covariance (sigma_p).
MIN_SIGMA_P = 1e-0   # Minimum value for sigma_p (e.g., 1.0)
MAX_SIGMA_P = 1e2   # Maximum value for sigma_p (e.g., 100.0)

# Define the typical range (max - min) for each metric used in sigma_p calculation.
# These are used to dynamically set the normalization maximum based on a configurable minimum.
# These values are typically derived from empirical analysis of the datasets.
# Example source values (illustrative):
# Metric                  | Min      | Max      | Diff (Max - Min)
# static_entropy          | 0.116945 | 0.981526 | 0.864581 (approx. ENTROPY_DIFF)
# last_num_culled_keyframes | -1.0     | 3.0      | 4.0 (CULLED_DIFF considers positive range)
# pose_opt_chi2_error     | 0.191120 | 3.515726 | 3.324606 (approx. POSE_CHI2_DIFF)
ENTROPY_DIFF = 0.865
POSE_CHI2_DIFF = 3.325
CULLED_DIFF = 3.0 # This typically represents the positive range of culled keyframes.

def casef(x, s):
    """
    Clipped Adaptive Saturation Exponential Function (CASEF)
    Negatif girdilerde 0, 0–1 aralığında üssel olarak ölçeklenir,
    1 üzerindeki girdilerde doygunluk (saturasyon) ile 1 değerini alır.
    """
    x_clip = np.clip(x, 0.0, 1.0)

    if np.isclose(s, 0.0):
        return x_clip # Linear behavior for s near 0

    # Define safe exponent argument limits to prevent overflow/underflow
    MAX_SAFE_EXP_ARG = 708
    MIN_SAFE_EXP_ARG = -708

    if s > MAX_SAFE_EXP_ARG:
        return 1.0 if np.isclose(x_clip, 1.0) else 0.0

    if s < MIN_SAFE_EXP_ARG:
        return 0.0 if np.isclose(x_clip, 0.0) else 1.0

    exp_s = np.exp(s)
    numerator = np.exp(s * x_clip) - 1.0
    denominator = exp_s - 1.0

    if np.isclose(denominator, 0.0):
        return x_clip # Fallback to linear behavior

    return numerator / denominator

def min_max_normalize(value, min_val, max_val):
    """
    Normalizes a given value to the [0, 1] range based on provided min and max values.
    If min_val and max_val are equal, returns 0 to avoid division by zero.
    """
    if max_val != min_val:
        return (value - min_val) / (max_val - min_val)
    else:
        return 0

def compute_adaptive_sigma_p(config, csv_file, sequence):
    """
    Computes an adaptive visual position measurement noise covariance (sigma_p)
    based on inverse static entropy, pose optimization chi-squared error,
    and the number of culled keyframes from visual odometry data.
    The computed sigma_p values are mapped to timestamps.
    Normalization ranges are dynamically adjusted based on configuration parameters.
    """

    # Determine normalization ranges: min is from config, max is min + predefined_difference.
    entropy_min = config.get('entropy_norm_min', 0.0) 
    entropy_max = entropy_min + ENTROPY_DIFF

    pose_chi2_min = config.get('pose_chi2_norm_min', 0.5)
    pose_chi2_max = pose_chi2_min + POSE_CHI2_DIFF

    culled_min = config.get('culled_norm_min', 0.0)
    culled_max = culled_min + CULLED_DIFF

    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(csv_file):
        print(f"Visual data CSV file not found: {csv_file}")
        return {}

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading visual data CSV: {e}")
        return {}

    # Check for required columns for sigma_p calculation
    required_cols = ["static_entropy", "pose_opt_chi2_error"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Required column '{col}' not found in {csv_file}.")
            return {}

    # If 'last_num_culled_keyframes' is missing, assume it's 0.
    if 'last_num_culled_keyframes' not in df.columns:
        df['last_num_culled_keyframes'] = 0.0

    # Standardize timestamp column to 'timestamp_ns'
    if "#timestamp [ns]" in df.columns:
        df['timestamp_ns'] = pd.to_numeric(df["#timestamp [ns]"], errors='coerce')
    elif "timestamp" in df.columns:
        df['timestamp_ns'] = (pd.to_numeric(df["timestamp"], errors='coerce') * 1e9).astype('int64')
    else:
        print(f"Timestamp information not found in {csv_file}.")
        return {}

    sigma_p_map = {}
    output_rows = []

    for idx, row in df.iterrows():
        t_ns = int(row['timestamp_ns']) # Ensure timestamp is integer

        # Raw metric values from the visual data
        inv_entropy_val = 1.0 - row['static_entropy']  # Inverse entropy: higher value indicates less texture/more uniformity
        pose_val        = row['pose_opt_chi2_error']
        culled_val      = row['last_num_culled_keyframes']

        # Normalize raw metric values using the dynamically determined min/max ranges
        inv_entropy_norm = min_max_normalize(inv_entropy_val, entropy_min, entropy_max)
        pose_norm        = min_max_normalize(pose_val, pose_chi2_min, pose_chi2_max)
        culled_norm      = min_max_normalize(culled_val, culled_min, culled_max)

        # Ölçeklendirme (config'den gelen yeni parametrelerle)
        scaled_inv_entropy = config['beta_p'] * inv_entropy_norm
        scaled_pose        = config['epsilon_p'] * pose_norm
        scaled_culled      = config['zeta_p'] * culled_norm

        # Combine the scaled, normalized metrics by taking the maximum value.
        # This implies that the metric indicating the poorest quality (highest scaled value) dictates the combined value.
        combined_value = max(scaled_inv_entropy, scaled_pose, scaled_culled)

        # Apply the CASEF activation function and subsequent thresholding to derive theta_p.
        # theta_p acts as a scaling factor for sigma_p.
        theta_p_raw = casef(combined_value, config['s'])
        if theta_p_raw < config['w_thr']:
            theta_p = 0.0
        elif theta_p_raw > config['d_thr']:
            theta_p = 1.0
        else:
            ratio = (theta_p_raw - config['w_thr']) / (config['d_thr'] - config['w_thr'])
            # Linear interpolation between w_thr and d_thr
            theta_p = ratio

        sigma_p = MIN_SIGMA_P + theta_p * (MAX_SIGMA_P - MIN_SIGMA_P)
        sigma_p_map[t_ns] = sigma_p

        output_rows.append([
            # Data for output CSV, useful for debugging and analysis
            idx + 1,
            t_ns,
            inv_entropy_val,     
            pose_val,            
            culled_val,          
            inv_entropy_norm,    
            pose_norm,           
            culled_norm,         
            scaled_inv_entropy,  
            scaled_pose,         
            scaled_culled,       
            combined_value,
            theta_p_raw,
            theta_p,
            sigma_p
        ])

    # Save the detailed computation steps for adaptive_sigma_p to a CSV file.
    out_file = f"outputs/adaptive_sigma_p_{sequence.lower()}.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Satir_No",
            "Timestamp [ns]",
            "static_inv_entropy",
            "static_pose_chi2_error",
            "static_culled_keyframes",
            "norm_static_inv_entropy",          
            "norm_static_pose_chi2_error",                   
            "norm_static_culled_keyframes",
            "scaled_norm_static_inv_entropy",
            "scaled_norm_static_pose_chi2_error",
            "scaled_norm_static_culled_keyframes",
            "combined_value",
            "theta_p_raw",
            "theta_p",
            "sigma_p"
        ])
        writer.writerows(output_rows)

    return sigma_p_map
