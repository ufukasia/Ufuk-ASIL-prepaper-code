import os
import pandas as pd
import csv
import numpy as np
 
def casef(x, s):
    """
    Clipped Adaptive Saturation Exponential Function (CASEF).
    This function provides a tunable non-linear mapping.
    It returns 0 for negative inputs, scales exponentially within the [0, 1] range,
    and saturates at 1 for inputs greater than or equal to 1.
    The parameter 's' controls the steepness of the exponential curve.
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

    # Fallback to linear behavior if the denominator is near zero (s was extremely close to 0).
    if np.isclose(denominator, 0.0):
        return x_clip

    return numerator / denominator

# Define the minimum and maximum bounds for the adaptive visual velocity covariance (sigma_v).
MIN_COV = 1e-2 # Minimum value for sigma_v
MAX_COV = 1e-0 # Maximum value for sigma_v

def min_max_normalize(value, min_val, max_val):
    """
    Normalizes a given value to the [0, 1] range based on provided min and max values.
    If min_val and max_val are equal, returns 0 to avoid division by zero.
    """
    if max_val != min_val:
        return (value - min_val) / (max_val - min_val)
    else:
        return 0

# Global normalization bounds for the metrics used in sigma_v calculation.
# These values are typically derived from empirical analysis of the datasets.
INTENSITY_MIN = 0.044404
INTENSITY_MAX = 1.000000
POSE_CHI2_MIN = 0.191120
POSE_CHI2_MAX = 3.515726
CULLED_MIN = 0.0
CULLED_MAX = 3.0

def compute_adaptive_sigma_v(config, visual_file, sequence):
    """
    Computes an adaptive visual velocity measurement noise covariance (sigma_v)
    based on changes in image intensity, pose optimization chi-squared error,
    and the number of culled keyframes from visual odometry data.
    The computed sigma_v values are mapped to timestamps.
    """
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(visual_file):
        print(f"Visual data CSV file not found: {visual_file}")
        return {}
    try:
        df = pd.read_csv(visual_file)
    except Exception as e:
        print(f"Error reading visual data CSV: {e}")
        return {}

    # Beklenen sütunlar kontrolü
    required_cols = ["static_intensity", "pose_opt_chi2_error", "last_num_culled_keyframes"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"The following required columns were not found in {visual_file}: {', '.join(missing_cols)}")
        return {}

    # Standardize timestamp column to 'timestamp_ns'
    if "#timestamp [ns]" in df.columns:
        df['timestamp_ns'] = pd.to_numeric(df["#timestamp [ns]"], errors='coerce')
    elif "timestamp" in df.columns:
        df['timestamp_ns'] = (pd.to_numeric(df["timestamp"], errors='coerce') * 1e9).astype('int64')
    else:
        print(f"Timestamp information not found in {visual_file}.")
        return {}

    # Ensure required metric columns are numeric and fill NaNs with 0
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate weighted absolute difference in static intensity
    diff_intensity = df['static_intensity'].diff().fillna(0)
    alpha_v = config.get('alpha_v', 1.0)
    df['weighted_delta_intensity'] = alpha_v * np.abs(diff_intensity)

    # Calculate weighted absolute difference in pose optimization chi-squared error
    diff_pose_chi2 = df['pose_opt_chi2_error'].diff().fillna(0)
    epsilon_v = config.get('epsilon_v', 1.0) # epsilon_L/H yerine epsilon_v kullan
    df['weighted_delta_pose_chi2'] = epsilon_v * np.abs(diff_pose_chi2) # Yöne bakılmaksızın mutlak farkı ağırlıklandır

    # Culled keyframes (Zeta H/L ayrı kalıyor)
    diff_culled_kf = df['last_num_culled_keyframes'].diff().fillna(0)
    zeta_L = config.get('zeta_L', 1.0)
    zeta_H = config.get('zeta_H', 1.0)
    df['weighted_delta_culled_kf'] = 0.0
    df.loc[diff_culled_kf > 0, 'weighted_delta_culled_kf'] = zeta_H * diff_culled_kf[diff_culled_kf > 0]
    df.loc[diff_culled_kf < 0, 'weighted_delta_culled_kf'] = zeta_L * np.abs(diff_culled_kf[diff_culled_kf < 0])

    sigma_v_map = {}
    output_rows = []

    for idx, row in df.iterrows():
        t_ns = int(row['timestamp_ns'])

        # Retrieve the pre-calculated weighted delta values for the current row
        delta_intensity_val = row['weighted_delta_intensity']
        delta_pose_chi2_val = row['weighted_delta_pose_chi2']
        delta_culled_kf_val = row['weighted_delta_culled_kf']

        # Normalize the weighted delta values using predefined global min/max bounds
        intensity_norm = min_max_normalize(delta_intensity_val, INTENSITY_MIN, INTENSITY_MAX)
        pose_chi2_norm = min_max_normalize(delta_pose_chi2_val, POSE_CHI2_MIN, POSE_CHI2_MAX)
        culled_norm    = min_max_normalize(delta_culled_kf_val, CULLED_MIN, CULLED_MAX)

        # Combine the normalized metrics by taking the maximum value.
        # This implies that the most significant change among the metrics dictates the combined value.
        combined_value = max(intensity_norm, pose_chi2_norm, culled_norm)

        # Apply the CASEF activation function and subsequent thresholding to derive theta_v.
        # theta_v acts as a scaling factor for sigma_v.
        theta_v_raw = casef(combined_value, config['s'])
        if theta_v_raw < config['w_thr']:
            theta_v = 0.0
        elif theta_v_raw > config['d_thr']:
            theta_v = 1.0
        else:
            # Linear interpolation between w_thr and d_thr
            ratio = (theta_v_raw - config['w_thr']) / (config['d_thr'] - config['w_thr'])
            theta_v = ratio

        sigma_v = MIN_COV + theta_v * (MAX_COV - MIN_COV)
        sigma_v_map[t_ns] = sigma_v

        output_rows.append([
            # Data for output CSV, useful for debugging and analysis
            idx+1,
            t_ns,
            delta_intensity_val, 
            delta_pose_chi2_val, 
            delta_culled_kf_val, 
            intensity_norm,
            pose_chi2_norm,
            culled_norm,
            intensity_norm, 
            pose_chi2_norm, 
            culled_norm, 
            combined_value,
            theta_v_raw,
            theta_v,
            sigma_v
        ])

    # Save the detailed computation steps for adaptive_sigma_v to a CSV file.
    out_file = f"outputs/adaptive_sigma_v_{sequence.lower()}.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Satir_No",
            "Timestamp [ns]",
            "weighted_delta_intensity",
            "weighted_delta_pose_chi2",
            "weighted_delta_culled_keyframes", 
            "norm_delta_intensity",
            "norm_delta_pose_chi2",
            "norm_delta_culled_keyframes",
            "combined_value",
            "theta_v_raw",
            "theta_v",
            "sigma_v"
        ])
        writer.writerows(output_rows)

    return sigma_v_map
