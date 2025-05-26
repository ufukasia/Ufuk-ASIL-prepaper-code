SAVE_RESULTS_CSV = True
SAVE_RESULTS_CSV_NAME = "results.csv"

import numpy as np
import pandas as pd
import time
import cv2
import os
import csv
import sys
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm 
import concurrent.futures  
import copy
from collections import deque 

from cov_sigma_v import compute_adaptive_sigma_v  
from cov_sigma_p import compute_adaptive_sigma_p  


import argparse
parser = argparse.ArgumentParser()

# Weights for sigma_p
parser.add_argument("--beta_p", type=float, default=1, help="Weight for inv_entropy (for sigma_p)")
parser.add_argument("--epsilon_p", type=float, default=1, help="Weight for pose_chi2 (for sigma_p)")
parser.add_argument("--zeta_p", type=float, default=1, help="Weight for culled_keyframes (for sigma_p)")

# Minimum values for sigma_p normalization terms.
parser.add_argument("--entropy_norm_min", type=float, default=0, help="Minimum value for Entropy normalization (sigma_p)")
parser.add_argument("--pose_chi2_norm_min", type=float, default=1, help="Minimum value for Pose Chi2 normalization (sigma_p)")
parser.add_argument("--culled_norm_min", type=float, default=0.5, help="Minimum value for Culled Keyframes normalization (sigma_p)")

# Weights for sigma_v
parser.add_argument("--alpha_v", type=float, default=6.2, help="Weight for Intensity difference (sigma_v)")
parser.add_argument("--epsilon_v", type=float, default=2, help="Weight for Pose_chi2 difference (sigma_v)")
parser.add_argument("--zeta_H", type=float, default=1, help="Weight for increasing culled_keyframes (sigma_v)")
parser.add_argument("--zeta_L", type=float, default=1, help="Weight for decreasing culled_keyframes (sigma_v)")

parser.add_argument("--w_thr", type=float, default=0.25, help="w_thr parameter for image confidence")
parser.add_argument("--d_thr", type=float, default=0.99, help="d_thr parameter for image confidence")
parser.add_argument("--s", type=float, default=3.0, help="s parameter for CASEF activation function")
parser.add_argument("--adaptive", action="store_true", default=False, help="Whether to use adaptive covariance")


parser.add_argument("--zupt_acc_thr", type=float, default=0.1, help="Accelerometer std threshold for ZUPT [m/s^2]")
parser.add_argument("--zupt_gyro_thr", type=float, default=0.1, help="Gyroscope std threshold for ZUPT [rad/s]")
parser.add_argument("--zupt_win", type=int, default=60, help="Window size for ZUPT (number of samples)")

args = parser.parse_args()

# =========================================================
# Activation Functions
# =========================================================
def casef(x, s):
    """
    Clipped Adaptive Saturation Exponential Function (CASEF).
    Outputs 0 for negative inputs, scales exponentially in the 0-1 range,
    and saturates to 1 for inputs above 1.
    """
    x_clip = np.clip(x, 0.0, 1.0)

    if np.isclose(s, 0.0):
        return x_clip # Linear behavior for s near 0
    MAX_SAFE_EXP_ARG = 100 
    MIN_SAFE_EXP_ARG = -100

    # Handle large positive s
    if s > MAX_SAFE_EXP_ARG: # exp(s) will be inf
        return 1.0 if np.isclose(x_clip, 1.0) else 0.0

    if s < MIN_SAFE_EXP_ARG: # exp(s) will be 0.0
        return 0.0 if np.isclose(x_clip, 0.0) else 1.0

    exp_s = np.exp(s)
    numerator = np.exp(s * x_clip) - 1.0
    denominator = exp_s - 1.0

    # If denominator is zero (s was extremely close to 0, exp_s is 1.0)
    if np.isclose(denominator, 0.0):
        return x_clip # Fallback to linear behavior

    return numerator / denominator

activation_func = lambda x: casef(x, args.s)

config = {
    'alpha_v': args.alpha_v, 
    'beta_p': args.beta_p,      
    'epsilon_v': args.epsilon_v, 
    'epsilon_p': args.epsilon_p, 
    'zeta_H': args.zeta_H, 
    'zeta_L': args.zeta_L, 
    'zeta_p': args.zeta_p, 
    'w_thr': args.w_thr,
    'd_thr': args.d_thr,
    's': args.s,

    'entropy_norm_min': args.entropy_norm_min,
    'pose_chi2_norm_min': args.pose_chi2_norm_min,
    'culled_norm_min': args.culled_norm_min,
}

# =========================================================
# Helper Functions (Quaternion Operations, etc.)
# =========================================================
def skew(vector):
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def convert_quaternion_order(q_wxyz):
    q_wxyz = np.array(q_wxyz)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def delta_quaternion(delta_theta):
    return np.concatenate([[1.0], 0.5 * delta_theta])


def quaternion_to_euler(q):
    q_xyzw = [q[1], q[2], q[3], q[0]]
    r = R.from_quat(q_xyzw)
    return r.as_euler('xyz', degrees=True)

def ensure_quaternion_continuity(q, q_prev):
    if np.dot(q, q_prev) < 0:
        return -q
    return q

def align_quaternions(estimated_quaternions, true_quaternions):
    min_length = min(len(estimated_quaternions), len(true_quaternions))
    aligned_quaternions = np.copy(estimated_quaternions[:min_length])
    for i in range(min_length):
        if np.dot(estimated_quaternions[i], true_quaternions[i]) < 0:
            aligned_quaternions[i] = -aligned_quaternions[i]
    return aligned_quaternions

def ensure_angle_continuity(angles, threshold=180):
    return np.unwrap(angles, axis=0, period=2 * threshold)

def angle_difference(angle1, angle2):
    diff = angle1 - angle2
    return (diff + 180) % 360 - 180

def calculate_angle_rmse(predictions, targets):
    diff = np.array([angle_difference(p, t) for p, t in zip(predictions, targets)])
    return np.sqrt((diff ** 2).mean(axis=0))

# =========================================================
# ESKF Class (with Bias Estimation) - Original Base Class
# =========================================================
class ErrorStateKalmanFilterVIO:
    def __init__(self, initial_quaternion, initial_velocity, initial_position,
                 initial_accel_bias=np.zeros(3), initial_gyro_bias=np.zeros(3),
                 gravity_vector=np.array([0, 0, -9.81]),
                 # Gauss-Markov & Noise Parameters (EuRoC MAV Values)
                 tau_a=300.0, tau_g=1000.0,         # Correlation times (s)
                 sigma_wa=2e-5, sigma_wg=2e-5, 
                 sigma_g=2e-4, sigma_a=2e-1):

        # Nominal State
        self.q = initial_quaternion / np.linalg.norm(initial_quaternion) # Ensure initial norm
        self.v = initial_velocity
        self.p = initial_position
        self.b_a = initial_accel_bias
        self.b_g = initial_gyro_bias
        self.g = gravity_vector
        # Error State Covariance (15x15)
        # [delta_theta, delta_v, delta_p, delta_b_a, delta_b_g]
        P_init_ori = 1e-6 * np.eye(3)
        P_init_vel = 1e-3 * np.eye(3)
        P_init_pos = 1e-3 * np.eye(3)
        P_init_ba = 1e-5 * np.eye(3) 
        P_init_bg = 1e-7 * np.eye(3)
        
        self.P = np.block([
            [P_init_ori, np.zeros((3,12))],
            [np.zeros((3,3)), P_init_vel, np.zeros((3,9))],
            [np.zeros((3,6)), P_init_pos, np.zeros((3,6))],
            [np.zeros((3,9)), P_init_ba, np.zeros((3,3))],
            [np.zeros((3,12)), P_init_bg]
        ])

        # Noise densities (measurement + bias)
        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        self.sigma_wa = sigma_wa # Accel random walk
        self.sigma_wg = sigma_wg # Gyro random walk
        self.tau_a = tau_a
        self.tau_g = tau_g
        # Continuous-time process covariance (15x15)
        Qc = np.zeros((15,15))
        # Measurement noises:
        Qc[0:3,  0:3]   = (self.sigma_g**2) * np.eye(3)  # gyro white noise
        Qc[3:6,  3:6]   = (self.sigma_a**2) * np.eye(3)  # accel white noise
        # Bias diffusion (Gauss–Markov random walk):
        Qc[9:12, 9:12]  = (self.sigma_wa**2) * np.eye(3) # accel bias random walk
        Qc[12:15,12:15] = (self.sigma_wg**2) * np.eye(3) # gyro bias random walk
        self.Q_c = Qc
        # Measurement Noise Covariances
        self.R_zupt = np.diag([1e-3, 1e-3, 1e-3]) # For ZUPT (this value can be reviewed)
        fixed_sigma_p = 1e0 
        fixed_sigma_v = 1e-2
        self.R_vis_6d = np.diag([fixed_sigma_p**2, fixed_sigma_p**2, fixed_sigma_p**2,
                                 fixed_sigma_v**2, fixed_sigma_v**2, fixed_sigma_v**2]) 

    def _compute_van_loan_matrices(self, R_nav_from_body, accel_corrected, gyro_corrected, dt):
        """ Computes discrete Phi and Qd matrices using the Van Loan method. """
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # Continuous-time error-state dynamics matrix A (15x15)
        A = np.zeros((15, 15))
        A[0:3, 0:3] = -skew(gyro_corrected)    # d(delta_theta)/dt
        A[0:3, 12:15] = -I3                    # Dependency of d(delta_theta)/dt on delta_b_g
        A[3:6, 0:3] = -R_nav_from_body @ skew(accel_corrected) # Dependency of d(delta_v)/dt on delta_theta
        A[3:6, 9:12] = -R_nav_from_body        # Dependency of d(delta_v)/dt on delta_b_a
        A[6:9, 3:6] = I3                       # Dependency of d(delta_p)/dt on delta_v
        A[9:12, 9:12] = -1/self.tau_a * I3     # d(delta_b_a)/dt (Gauss-Markov)
        A[12:15, 12:15] = -1/self.tau_g * I3   # d(delta_b_g)/dt (Gauss-Markov)


        # Matrix construction for Van Loan (30x30)
        M = np.zeros((30, 30))
        M[0:15, 0:15] = A * dt
        M[0:15, 15:30] = self.Q_c * dt
        M[15:30, 15:30] = -A.T * dt # This term is -A.T * dt as per Van Loan method.

        # Compute matrix exponential
        M_exp = expm(M)

        # Extract discrete Phi and Qd (Correct Van Loan extraction)
        Phi = M_exp[0:15, 0:15] # expm(A*dt)
        Qd = M_exp[0:15, 15:30] @ Phi.T 
        # Ensure Qd is symmetric
        Qd = 0.5 * (Qd + Qd.T)

        return Phi, Qd

    def predict(self, gyro_raw, accel_raw, dt):
        if dt <= 0:
            return
        q = self.q; v = self.v; p = self.p; b_a = self.b_a; b_g = self.b_g

        # Correct for biases
        accel_corrected = accel_raw - b_a
        gyro_corrected = gyro_raw - b_g

        # --- Nominal State Update ---
        # Quaternion update (with gyro_corrected)
        dq_dt = 0.5 * np.array([
            -q[1]*gyro_corrected[0] - q[2]*gyro_corrected[1] - q[3]*gyro_corrected[2], # w_dot
             q[0]*gyro_corrected[0] + q[2]*gyro_corrected[2] - q[3]*gyro_corrected[1],
             q[0]*gyro_corrected[1] - q[1]*gyro_corrected[2] + q[3]*gyro_corrected[0],
             q[0]*gyro_corrected[2] + q[1]*gyro_corrected[1] - q[2]*gyro_corrected[0]
        ])
        q_new = q + dq_dt * dt
        q_new /= np.linalg.norm(q_new) # Normalization

        # Rotation matrix (from updated q_new)
        q_xyzw = convert_quaternion_order(q_new)
        R_nav_from_body = R.from_quat(q_xyzw).as_matrix()

        # Velocity update (with accel_corrected and current R)
        a_nav = R_nav_from_body @ accel_corrected + self.g
        v_new = v + a_nav * dt
        p_new = p + v * dt + 0.5 * a_nav * dt**2 

        # Bias state remains constant (only error state is updated)
        b_a_new = b_a
        b_g_new = b_g

        # --- Error State Covariance Update (Van Loan) ---
        Phi, Qd = self._compute_van_loan_matrices(R_nav_from_body, accel_corrected, gyro_corrected, dt)

        # Update covariance: P = Phi * P * Phi^T + Qd
        P_new = Phi @ self.P @ Phi.T + Qd
        # Ensure P remains symmetric
        self.P = 0.5 * (P_new + P_new.T)

        # Update nominal state
        self.q = q_new
        self.v = v_new
        self.p = p_new
        self.b_a = b_a_new
        self.b_g = b_g_new

    def _update_common(self, delta_x):
        """ Common state update logic. """
        
        dq = delta_quaternion(delta_x[0:3])
        self.q = quaternion_multiply(self.q, dq)
        self.q /= np.linalg.norm(self.q) 

        self.v += delta_x[3:6]
        self.p += delta_x[6:9]
        self.b_a += delta_x[9:12]
        self.b_g += delta_x[12:15]

    def zero_velocity_update(self):
        q_xyzw = convert_quaternion_order(self.q)
        R_nav_from_body = R.from_quat(q_xyzw).as_matrix()
        R_body_from_nav = R_nav_from_body.T

        # Measurement: Velocity in navigation frame expressed in body frame
        v_body_predicted = R_body_from_nav @ self.v
        y = -v_body_predicted # Measurement residual

        H = np.zeros((3, 15))
        # Jacobian of measurement residual (0 - R_body_from_nav @ v) w.r.t. delta_theta
        H[:, 0:3] = -skew(R_body_from_nav @ self.v) 
        # Jacobian of R_body_from_nav @ v w.r.t delta_v
        H[:, 3:6] = R_body_from_nav            

        # Kalman update steps
        S = H @ self.P @ H.T + self.R_zupt
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: ZUPT S matrix is not invertible!")
            return # Update cannot be performed
        K = self.P @ H.T @ S_inv # Kalman gain (15x3)

        delta_x = K @ y # Error state correction (15x1)

        # Update state and covariance (Joseph Form)


        # Velocity (self.v) will be reset when ZUPT is detected.
        self.v = np.zeros(3)
        # Other states are updated based on the correction:
        self.p += delta_x[6:9]  # Position is updated via delta_x[6:9]; note H for ZUPT has no direct position terms.
        self.b_a += delta_x[9:12]
        self.b_g += delta_x[12:15]
        
        I15 = np.eye(15)
        P_new = (I15 - K @ H) @ self.P @ (I15 - K @ H).T + K @ self.R_zupt @ K.T # Joseph form
        self.P = 0.5 * (P_new + P_new.T) # Ensure symmetry
    def vision_posvel_update(self, p_meas, v_meas, R_vision):
        """ Updates with visual position and velocity measurement. R_vision is provided externally. """
        y = np.concatenate([p_meas - self.p, v_meas - self.v]) # Measurement residual (6x1)
        H = np.zeros((6, 15))
        H[0:3, 6:9] = np.eye(3)  # H_p: Effect of position error on measurement
        H[3:6, 3:6] = np.eye(3)  # H_v: Effect of velocity error on measurement

        # Kalman update steps
        S = H @ self.P @ H.T + R_vision # Updated R_vision is used
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Vision S matrix is not invertible!")
            return # Update cannot be performed
        K = self.P @ H.T @ S_inv # Kalman kazancı (15x6)

        delta_x = K @ y # Error state correction (15x1)

        # Update state and covariance (Joseph Form)
        self._update_common(delta_x)
        I15 = np.eye(15)
        P_new = (I15 - K @ H) @ self.P @ (I15 - K @ H).T + K @ R_vision @ K.T # Joseph form
        self.P = 0.5 * (P_new + P_new.T) # Ensure symmetry

    def gravity_update(self, accel_raw): # Takes accel_raw only now
        """ Updates accelerometer bias using gravity during static conditions. """
        # Calculate R_nav_from_body from current state quaternion
        q_xyzw = convert_quaternion_order(self.q)
        R_nav_from_body = R.from_quat(q_xyzw).as_matrix()
        R_body_from_nav = R_nav_from_body.T

        # Expected gravity in body frame
        g_body = R_body_from_nav @ (-self.g) 
        # Measurement residual
        y = (accel_raw - self.b_a) - g_body
        
        # Jacobian H
        H = np.zeros((3,15))
        # Jacobian of (accel_raw - b_a) - (R_body_from_nav @ (-g)) w.r.t delta_theta
        H[:, 0:3] = skew(R_body_from_nav @ self.g) # Correct Jacobian for gravity update
        # Jacobian w.r.t delta_b_a
        H[:, 9:12] = -np.eye(3)
        
        # Measurement noise
        R_acc = (self.sigma_a**2) * np.eye(3)

        # Kalman update
        S  = H @ self.P @ H.T + R_acc
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Gravity Update S matrix is not invertible!")
            return # Update cannot be performed
        K  = self.P @ H.T @ S_inv
        dx = K @ y
        
        self.v += dx[3:6]     # Velocity error dx[3:6] is updated; H for gravity has no direct velocity terms.
        self.p += dx[6:9]     # Position error dx[6:9] is updated; H for gravity has no direct position terms.
        self.b_a += dx[9:12]
        self.b_g += dx[12:15] # Gyro bias error dx[12:15] is updated; H for gravity has no direct gyro bias terms.

        I15 = np.eye(15)
        P_new = (I15 - K @ H) @ self.P @ (I15 - K @ H).T + K @ R_acc @ K.T # Joseph form
        self.P = 0.5 * (P_new + P_new.T) # Ensure symmetry

# =========================================================
# UKF-ESKF Class (Additive UKF + SUT)
# =========================================================
class ErrorStateKalmanFilterVIO_UKF(ErrorStateKalmanFilterVIO):
    def __init__(self, initial_quaternion, initial_velocity, initial_position,
                 initial_accel_bias=np.zeros(3), initial_gyro_bias=np.zeros(3),
                 gravity_vector=np.array([0, 0, -9.81]),
                 # Noise parameters (same as base class)
                 tau_a=300.0, tau_g=1000.0,
                 sigma_wa=2e-5, sigma_wg=2e-5,
                 sigma_g=2e-4, sigma_a=2e-1):

        # Call the base class constructor
        super().__init__(initial_quaternion, initial_velocity, initial_position,
                         initial_accel_bias, initial_gyro_bias, gravity_vector,
                         tau_a, tau_g, sigma_wa, sigma_wg, sigma_g, sigma_a)

        # UKF/SUT Parameters
        self.n = 15                # Error-state dimension [delta_theta, delta_v, delta_p, delta_b_a, delta_b_g]
        self.α = 0.5               # SUT scaling parameter alpha
        self.β = 2.0               # SUT scaling parameter beta (optimal for Gaussian)
        self.κ = 3 - self.n        # SUT scaling parameter kappa (Merwe's recommendation: 3-n)
        self.λ = self.α**2 * (self.n + self.κ) - self.n # Composite scaling parameter lambda
        
        # Sigma point weights (Merwe's Scaled Unscented Transform)
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.λ))) # Weights for mean
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.λ))) # Weights for covariance
        self.Wm[0] = self.λ / (self.n + self.λ)                         # Weight for mean (center point)
        self.Wc[0] = self.Wm[0] + (1 - self.α**2 + self.β)             # Weight for covariance (center point)

    def _sigma_points(self, x_mean, P):
        """ Generates sigma points for the UKF based on mean and covariance. """
        num_states = P.shape[0] # Should be self.n
        if num_states != self.n:
             print(f"Warning: P matrix size {P.shape} does not match state size {self.n} in _sigma_points")
             # Potentially handle error or adjust self.n; proceeding with P's dimension.
             # This might indicate an issue elsewhere if P size changes unexpectedly.

        # Ensure P is positive semi-definite before Cholesky
        P = 0.5 * (P + P.T) # Ensure symmetry
        # Add small diagonal jitter for numerical stability
        P_stable = P + np.eye(num_states) * 1e-9 

        try:
            # Use Cholesky decomposition.
            S = np.linalg.cholesky((num_states + self.λ) * P_stable)
        except np.linalg.LinAlgError:
            print(f"Error: Cholesky decomposition failed in _sigma_points even after jitter.")
            # Fallback: e.g., use SVD or return mean. Returning mean for now.
            chi = np.zeros((2 * num_states + 1, num_states))
            chi[0] = x_mean
            return chi # Degenerate case

        chi = np.zeros((2 * num_states + 1, num_states))
        chi[0] = x_mean                   # Center sigma point
        for i in range(num_states):
            chi[i + 1]        = x_mean + S[:, i] # Positive direction sigma points 
            chi[num_states + 1 + i] = x_mean - S[:, i] # Negative direction sigma points
        return chi

    # --- Helper functions using explicit nominal state ---
    def _inject_explicit(self, q_nom, v_nom, p_nom, ba_nom, bg_nom, delta_x):
        """ Injects the error state delta_x into the provided nominal state. """
        delta_theta = delta_x[0:3]
        delta_v     = delta_x[3:6]
        delta_p     = delta_x[6:9]
        delta_ba    = delta_x[9:12]
        delta_bg    = delta_x[12:15]

        dq = delta_quaternion(delta_theta)
        q_inj = quaternion_multiply(q_nom, dq)
        q_inj /= np.linalg.norm(q_inj)

        v_inj = v_nom + delta_v
        p_inj = p_nom + delta_p
        ba_inj = ba_nom + delta_ba
        bg_inj = bg_nom + delta_bg

        return q_inj, v_inj, p_inj, ba_inj, bg_inj

    def _retract_explicit(self, q_nom, v_nom, p_nom, ba_nom, bg_nom, q_inj, v_inj, p_inj, ba_inj, bg_inj):
        """ Retracts the injected state back to an error state relative to the provided nominal state. """
        # Ensure nominal quaternion is normalized
        q_nom_norm = q_nom / np.linalg.norm(q_nom)
        q_inj_norm = q_inj / np.linalg.norm(q_inj)

        delta_q = quaternion_multiply(q_inj_norm, quaternion_conjugate(q_nom_norm))
        delta_q /= np.linalg.norm(delta_q) # Ensure unit quaternion difference

        # Convert quaternion difference to angle error vector (SO(3) logarithm)
        if delta_q[0] < 0: # Ensure positive scalar part for acos
             delta_q = -delta_q
        
        # Clamp angle slightly below 1.0 to avoid acos domain errors
        cos_half_angle = np.clip(delta_q[0], -1.0, 1.0 - 1e-12) 
        half_angle = np.arccos(cos_half_angle)
        angle = 2 * half_angle
        
        sin_half_angle = np.sqrt(1.0 - cos_half_angle**2)

        if sin_half_angle < 1e-9: # Check for near-zero rotation
            delta_theta = np.zeros(3)
        else:
            axis = delta_q[1:] / sin_half_angle
            delta_theta = angle * axis

        # Retract other errors (Direct subtraction)
        delta_v = v_inj - v_nom
        delta_p = p_inj - p_nom
        delta_ba = ba_inj - ba_nom
        delta_bg = bg_inj - bg_nom

        delta_x = np.concatenate([delta_theta, delta_v, delta_p, delta_ba, delta_bg])
        return delta_x

    def _propagate_nominal(self, q, v, p, b_a, b_g, gyro_raw, accel_raw, dt):
        if dt <= 0:
            # Return current state if dt is invalid
            return q, v, p, b_a, b_g 

        # Correct biases
        accel_corrected = accel_raw - b_a
        gyro_corrected = gyro_raw - b_g

        # Quaternion update (e.g., Euler integration)
        # Euler integration:
        dq_dt = 0.5 * np.array([
            -q[1]*gyro_corrected[0] - q[2]*gyro_corrected[1] - q[3]*gyro_corrected[2],
             q[0]*gyro_corrected[0] + q[2]*gyro_corrected[2] - q[3]*gyro_corrected[1],
             q[0]*gyro_corrected[1] - q[1]*gyro_corrected[2] + q[3]*gyro_corrected[0],
             q[0]*gyro_corrected[2] + q[1]*gyro_corrected[1] - q[2]*gyro_corrected[0]
        ])
        q_new = q + dq_dt * dt
        q_new /= np.linalg.norm(q_new) # Normalize

        # Rotation matrix from the new quaternion
        q_xyzw = convert_quaternion_order(q_new)
        R_nav_from_body = R.from_quat(q_xyzw).as_matrix()

        # Velocity and Position update (using trapezoidal integration for better accuracy)
        a_nav_start = R.from_quat(convert_quaternion_order(q)).as_matrix() @ accel_corrected + self.g
        a_nav_end = R_nav_from_body @ accel_corrected + self.g
        v_new = v + 0.5 * (a_nav_start + a_nav_end) * dt
        p_new = p + v * dt + 0.25 * (a_nav_start + a_nav_end) * dt**2 # More accurate position update using trapezoidal rule for acceleration

        # Biases remain constant during propagation in this model
        b_a_new = b_a
        b_g_new = b_g

        return q_new, v_new, p_new, b_a_new, b_g_new

    def predict(self, gyro_raw, accel_raw, dt):
        """ Predicts the state using UKF for the error state. """
        if dt <= 0:
            return

        # --- (a) Nominal State Propagation ---
        # Store current nominal state before propagation
        q_nom_old, v_nom_old, p_nom_old = self.q.copy(), self.v.copy(), self.p.copy()
        ba_nom_old, bg_nom_old = self.b_a.copy(), self.b_g.copy()

        # Propagate the current nominal state using the internal helper
        q_new, v_new, p_new, ba_new, bg_new = self._propagate_nominal(
            q_nom_old, v_nom_old, p_nom_old, ba_nom_old, bg_nom_old,
            gyro_raw, accel_raw, dt
        )
        # Update nominal state immediately
        self.q, self.v, self.p = q_new, v_new, p_new
        self.b_a, self.b_g = ba_new, bg_new # Biases are constant in _propagate_nominal

        # --- (b) UKF Error State Covariance Propagation ---
        # Calculate components for Qd based on the nominal state before propagation
        accel_corrected_old = accel_raw - ba_nom_old
        gyro_corrected_old = gyro_raw - bg_nom_old
        q_xyzw_old = convert_quaternion_order(q_nom_old)
        R_nav_from_body_old = R.from_quat(q_xyzw_old).as_matrix()

        # Get discrete process noise Qd using Van Loan based on the pre-propagation state
        _, Qd = self._compute_van_loan_matrices(R_nav_from_body_old, accel_corrected_old, gyro_corrected_old, dt)
        # Note: Qd from Van Loan method is used as the discrete process noise covariance.

        # 1. Generate sigma points for the error state (mean = 0)
        # Use current error covariance P
        chi = self._sigma_points(np.zeros(self.n), self.P)
        chi_pred = np.zeros_like(chi)

        # 2. Propagate sigma points through the non-linear error dynamics
        for i, δx in enumerate(chi):
            # Inject error sigma point into the pre-propagation nominal state
            q_inj, v_inj, p_inj, ba_inj, bg_inj = self._inject_explicit(q_nom_old, v_nom_old, p_nom_old, ba_nom_old, bg_nom_old, δx)

            # Propagate the *injected* state one step
            q_i_pred, v_i_pred, p_i_pred, ba_i_pred, bg_i_pred = self._propagate_nominal(
                q_inj, v_inj, p_inj, ba_inj, bg_inj,
                gyro_raw, accel_raw, dt
            )
            # Retract the propagated state to an error state relative to the new nominal state
            chi_pred[i] = self._retract_explicit(self.q, self.v, self.p, self.b_a, self.b_g,
                                                 q_i_pred, v_i_pred, p_i_pred, ba_i_pred, bg_i_pred)

        δx_pred_mean = np.sum(self.Wm[:, None] * chi_pred, axis=0)
        delta_chi = chi_pred - δx_pred_mean # Shape (2n+1, n)
        P_pred = Qd + np.sum(self.Wc[:, None, None] *
                             delta_chi[..., None] @ delta_chi[:, None, :], axis=0)

        # Ensure P remains symmetric
        self.P = 0.5 * (P_pred + P_pred.T)

    def ukf_update_posvel(self, z_meas, R_meas):
        """ Updates the state using visual position and velocity measurements via UKF. """
        # z_meas: 6x1 vector [p_meas, v_meas]
        # R_meas: 6x6 measurement noise covariance matrix

        # 1. Generate sigma points for the current error state (mean=0, covariance=P)
        chi = self._sigma_points(np.zeros(self.n), self.P)
        
        # 2. Transform sigma points through the measurement function h(delta_x)
        #    h(delta_x) extracts position and velocity from the full state (nominal + delta_x)
        Zsig = np.zeros((2 * self.n + 1, 6)) # Transformed sigma points in measurement space (3 pos + 3 vel)
        for i, δx in enumerate(chi):
            # Inject error sigma point into the current nominal state
            q_i, v_i, p_i, ba_i, bg_i = self._inject_explicit(
                self.q, self.v, self.p, self.b_a, self.b_g, δx
            )
            # The predicted measurement is the position and velocity of the injected state
            Zsig[i, :3] = p_i
            Zsig[i, 3:] = v_i

        # 3. Calculate predicted measurement mean from transformed sigma points
        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)

        # 4. Calculate innovation covariance S (Pzz) and state-measurement cross-covariance Pxz
        delta_Z = Zsig - z_pred # Shape (2n+1, 6)
        delta_chi = chi - np.zeros(self.n) # Shape (2n+1, n); error state mean is zero

        Pzz = R_meas + np.sum(self.Wc[:, None, None] *
                              delta_Z[..., None] @ delta_Z[:, None, :], axis=0)

        Pxz = np.sum(self.Wc[:, None, None] *
                     delta_chi[..., None] @ delta_Z[:, None, :], axis=0)
        # 5. Calculate Kalman Gain K
        try:
            # Use pseudo-inverse for potentially ill-conditioned S (Pzz)
            Pzz_inv = np.linalg.pinv(Pzz) 
        except np.linalg.LinAlgError:
            print("Warning: UKF Vision S (Pzz) matrix is singular! Skipping update.")
            return
        
        K = Pxz @ Pzz_inv # Shape (n, 6)

        # 6. Calculate corrected error state estimate
        residual = z_meas - z_pred
        δx_corr = K @ residual # Shape (n,)

        # 7. Update error state covariance P (Joseph form for stability)
        P_updated = self.P - K @ Pzz @ K.T
        # Ensure P remains symmetric and positive semi-definite
        self.P = 0.5 * (P_updated + P_updated.T) 
        # 8. Update nominal state with the corrected error state
        self._update_common(δx_corr)

    def zero_velocity_update(self): # Override for UKF
        """ Updates the state using Zero Velocity Update (ZUPT) via UKF. """
        # Measurement is zero velocity in the body frame.
        z_meas = np.zeros(3)
        R_meas = self.R_zupt # Use pre-defined ZUPT noise

        # 1. Generate sigma points for the current error state
        chi = self._sigma_points(np.zeros(self.n), self.P)
        
        # 2. Transform sigma points through measurement function h(delta_x) = R_body_from_nav(q_nominal (+) delta_theta) @ (v_nominal + delta_v)
        Zsig = np.zeros((2 * self.n + 1, 3)) # Transformed sigma points in measurement space (3 body-frame velocity components)
        for i, δx in enumerate(chi):
            # Inject error sigma point into the current nominal state
            q_i, v_i, p_i, ba_i, bg_i = self._inject_explicit(
                self.q, self.v, self.p, self.b_a, self.b_g, δx
            )
            # Calculate R_body_from_nav for this sigma point's orientation
            q_xyzw_i = convert_quaternion_order(q_i)
            R_nav_from_body_i = R.from_quat(q_xyzw_i).as_matrix()
            R_body_from_nav_i = R_nav_from_body_i.T
            # Measurement function: velocity in body frame
            Zsig[i, :] = R_body_from_nav_i @ v_i
        # 3. Calculate predicted measurement mean from transformed sigma points
        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)

        # 4. Calculate innovation covariance S (Pzz) and state-measurement cross-covariance Pxz
        delta_Z = Zsig - z_pred # Shape (2n+1, 3)
        delta_chi = chi - np.zeros(self.n) # Shape (2n+1, n)

        Pzz = R_meas + np.sum(self.Wc[:, None, None] *
                              delta_Z[..., None] @ delta_Z[:, None, :], axis=0)
        Pxz = np.sum(self.Wc[:, None, None] *
                     delta_chi[..., None] @ delta_Z[:, None, :], axis=0)

        # 5. Calculate Kalman Gain K
        try:
            Pzz_inv = np.linalg.pinv(Pzz)
        except np.linalg.LinAlgError:
            print("Warning: UKF ZUPT S (Pzz) matrix is singular! Skipping update.")
            return
        
        K = Pxz @ Pzz_inv # Shape (n, 3)

        # 6. Calculate corrected error state estimate
        residual = z_meas - z_pred # Residual: (target zero velocity) - (predicted body velocity)
        δx_corr = K @ residual # Shape (n,)

        # 7. Update error state covariance P
        P_updated = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P_updated + P_updated.T)

        # 8. Update nominal state with the corrected error state
        self._update_common(δx_corr)

    def gravity_update(self, accel_raw): # Override for UKF, takes accel_raw
        """ Updates the accelerometer bias using gravity measurement via UKF during static periods. """
        # Measurement is the raw accelerometer reading
        z_meas = accel_raw
        # Measurement noise is accelerometer white noise
        R_meas = (self.sigma_a**2) * np.eye(3)

        # 1. Generate sigma points for the current error state
        chi = self._sigma_points(np.zeros(self.n), self.P)

        # 2. Transform sigma points through the measurement function

        Zsig = np.zeros((2 * self.n + 1, 3)) # Transformed sigma points in measurement space (3 accelerometer components)
        for i, δx in enumerate(chi):
            q_i, v_i, p_i, ba_i, bg_i = self._inject_explicit(
                self.q, self.v, self.p, self.b_a, self.b_g, δx
            )
            # Calculate R_body_from_nav for this sigma point's orientation
            q_xyzw_i = convert_quaternion_order(q_i)
            R_nav_from_body_i = R.from_quat(q_xyzw_i).as_matrix()
            R_body_from_nav_i = R_nav_from_body_i.T
            # Expected gravity vector in body frame for this sigma point
            g_body_i = R_body_from_nav_i @ (-self.g)
            # Measurement function: expected accel reading = g_body + accel_bias
            Zsig[i, :] = g_body_i + ba_i

        # 3. Calculate predicted measurement mean (expected accelerometer reading)
        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)

        # 4. Calculate innovation covariance S (Pzz) and state-measurement cross-covariance Pxz
        delta_Z = Zsig - z_pred # Shape (2n+1, 3)
        delta_chi = chi - np.zeros(self.n) # Shape (2n+1, n)

        Pzz = R_meas + np.sum(self.Wc[:, None, None] *
                              delta_Z[..., None] @ delta_Z[:, None, :], axis=0)
        Pxz = np.sum(self.Wc[:, None, None] *
                     delta_chi[..., None] @ delta_Z[:, None, :], axis=0)

        # 5. Calculate Kalman Gain K
        try:
            Pzz_inv = np.linalg.pinv(Pzz)
        except np.linalg.LinAlgError:
            print("Warning: UKF Gravity S (Pzz) matrix is singular! Skipping update.")
            return

        K = Pxz @ Pzz_inv # Shape (n, 3)

        # 6. Calculate corrected error state estimate
        residual = z_meas - z_pred # Residual: (actual accelerometer reading) - (expected accelerometer reading)
        δx_corr = K @ residual # Shape (n,)

        # 7. Update error state covariance P
        P_updated = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P_updated + P_updated.T)

        # 8. Update nominal state with the corrected error state
        self._update_common(δx_corr)

# === H Y B R I D  F I L T E R  C L A S S ==========================
class ErrorStateKalmanFilterVIO_Hybrid(ErrorStateKalmanFilterVIO):
    """
    Hybrid error-state filter using UKF for delta_theta and ESKF for the remaining 12 dimensions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        # Initialize ESKF components
        # --- UKF parameters: only for delta_theta (3 dimensions) ---
        self.n_h = 3                              # UKF dimension for orientation error
        self.alpha_h, self.beta_h = 0.5, 2.0      # alpha_h (scaling parameter), beta_h (distribution parameter)
        self.kappa_h = 3 - self.n_h               # kappa_h (secondary scaling, Merwe's recommendation: 3-n_h)
        self.lambda_h = self.alpha_h**2 * (self.n_h + self.kappa_h) - self.n_h
        w0_m = self.lambda_h / (self.n_h + self.lambda_h)
        w0_c = w0_m + (1 - self.alpha_h**2 + self.beta_h)
        wi    = 1.0 / (2 * (self.n_h + self.lambda_h))
        self.Wm_h = np.hstack([w0_m, np.full(2*self.n_h, wi)])
        self.Wc_h = np.hstack([w0_c, np.full(2*self.n_h, wi)])

        # Initialize Cholesky factor (S_tt) of P_tt for SR-UKF for orientation error
        # P[0:3,0:3] is initially P_init_ori, which is 1e-6 * eye(3).
        self.S_tt = self._ensure_pd_and_chol(self.P[0:3,0:3].copy())

    def _ensure_pd_and_chol(self, matrix, default_jitter=1e-9, min_eigenvalue_abs=1e-7):
        """ Ensures the matrix is positive definite and returns its Cholesky factor. """
        mat = 0.5 * (matrix + matrix.T) # Ensure symmetry
        try:
            # Try direct Cholesky with a small jitter
            return np.linalg.cholesky(mat + np.eye(mat.shape[0]) * default_jitter)
        except np.linalg.LinAlgError:
            # If it fails, use eigenvalue method
            eigenvalues, eigenvectors = np.linalg.eigh(mat)
            # Clamp small/negative eigenvalues to a small positive value
            eigenvalues[eigenvalues < min_eigenvalue_abs] = min_eigenvalue_abs
            mat_pd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Add jitter again for safety before final Cholesky
            return np.linalg.cholesky(mat_pd + np.eye(mat.shape[0]) * default_jitter)

    # ---------- Sigma points for delta_theta only (from Cholesky factor S_tt) ------------
    def _sigma_points_theta_from_S(self, mean_3, S_tt_chol): # S_tt_chol is L such that P_tt = L L^T
        # Sigma points: mean +/- sqrt(n_h + lambda_h) * L_column_i
        term_matrix = np.sqrt(self.n_h + self.lambda_h) * S_tt_chol
        chi = np.zeros((2*self.n_h+1, 3))
        chi[0] = mean_3
        for i in range(self.n_h):
            chi[i+1]          = mean_3 + term_matrix[:, i]
            chi[self.n_h+1+i] = mean_3 - term_matrix[:, i]
        return chi

    # ---------------- P R E D I C T (Updated with SR-UKF for orientation) --------------------
    def predict(self, gyro_raw, accel_raw, dt):
        if dt <= 0: return

        # ➊  -- Update current nominal state and full P matrix using ESKF-style prediction --
        q_old, v_old, p_old = self.q.copy(), self.v.copy(), self.p.copy()
        ba_old, bg_old      = self.b_a.copy(), self.b_g.copy()
        super().predict(gyro_raw, accel_raw, dt) # ESKF predict (self.P is updated, including P_tt, P_tx, P_xx)

        # ➋ -- Update self.S_tt from the P[0:3,0:3] block updated by ESKF predict --
        P_tt_after_eskf = self.P[0:3,0:3].copy()
        self.S_tt = self._ensure_pd_and_chol(P_tt_after_eskf)

        # ➌  -- UKF propagation for the delta_theta covariance block (P_tt) using S_tt --
        sigma_theta = self._sigma_points_theta_from_S(np.zeros(3), self.S_tt)
        sigma_theta_pred = np.zeros_like(sigma_theta)

        for i, dtheta in enumerate(sigma_theta):
            # Inject delta_theta -> propagate nominal quaternion
            q_i = quaternion_multiply(q_old, delta_quaternion(dtheta)) # q_old is nominal q before ESKF predict
            q_i, _, _, _, _ = self._propagate_nominal( # Propagate the injected state
                q_i, v_old, p_old, ba_old, bg_old, gyro_raw, accel_raw, dt
            ) # v_old, p_old etc. are also nominal states before ESKF predict
            
            dtheta_new = self._retract_explicit( 
                self.q, self.v, self.p, self.b_a, self.b_g, 
                q_i, v_old, p_old, ba_old, bg_old 
 
            )[0:3]
            sigma_theta_pred[i] = dtheta_new

        mean_theta = np.sum(self.Wm_h[:,None] * sigma_theta_pred, axis=0)
        dSigma = sigma_theta_pred - mean_theta
        P_tt_ukf_refined = np.sum(self.Wc_h[:,None,None] * dSigma[...,None] @ dSigma[:,None,:], axis=0)

       
        self.P[0:3, 0:3] = 0.5*(P_tt_ukf_refined + P_tt_ukf_refined.T) # Ensure symmetry
        self.S_tt = self._ensure_pd_and_chol(self.P[0:3,0:3].copy())   # Update S_tt from the refined P_tt

    # --- Helper functions for nominal state propagation and retraction ---
    def _propagate_nominal(self, q, v, p, b_a, b_g, gyro_raw, accel_raw, dt):
        """ Propagates a given nominal state using IMU measurements. """
        if dt <= 0:
            return q, v, p, b_a, b_g
        accel_corrected = accel_raw - b_a
        gyro_corrected = gyro_raw - b_g
        dq_dt = 0.5 * np.array([
            -q[1]*gyro_corrected[0] - q[2]*gyro_corrected[1] - q[3]*gyro_corrected[2],
             q[0]*gyro_corrected[0] + q[2]*gyro_corrected[2] - q[3]*gyro_corrected[1],
             q[0]*gyro_corrected[1] - q[1]*gyro_corrected[2] + q[3]*gyro_corrected[0],
             q[0]*gyro_corrected[2] + q[1]*gyro_corrected[1] - q[2]*gyro_corrected[0]
        ])
        q_new = q + dq_dt * dt
        q_new /= np.linalg.norm(q_new)
        q_xyzw = convert_quaternion_order(q_new)
        R_nav_from_body = R.from_quat(q_xyzw).as_matrix()
        a_nav_start = R.from_quat(convert_quaternion_order(q)).as_matrix() @ accel_corrected + self.g
        a_nav_end = R_nav_from_body @ accel_corrected + self.g
        v_new = v + 0.5 * (a_nav_start + a_nav_end) * dt
        p_new = p + v * dt + 0.25 * (a_nav_start + a_nav_end) * dt**2
        b_a_new = b_a
        b_g_new = b_g
        return q_new, v_new, p_new, b_a_new, b_g_new

    def _retract_explicit(self, q_nom, v_nom, p_nom, ba_nom, bg_nom, q_inj, v_inj, p_inj, ba_inj, bg_inj):
        """ Retracts an injected state back to an error state relative to a nominal state. """
        q_nom_norm = q_nom / np.linalg.norm(q_nom)
        q_inj_norm = q_inj / np.linalg.norm(q_inj)
        delta_q = quaternion_multiply(q_inj_norm, quaternion_conjugate(q_nom_norm))
        delta_q /= np.linalg.norm(delta_q)
        if delta_q[0] < 0: delta_q = -delta_q
        cos_half_angle = np.clip(delta_q[0], -1.0, 1.0 - 1e-12)
        half_angle = np.arccos(cos_half_angle)
        angle = 2 * half_angle
        sin_half_angle = np.sqrt(1.0 - cos_half_angle**2)
        if sin_half_angle < 1e-9:
            delta_theta = np.zeros(3)
        else:
            axis = delta_q[1:] / sin_half_angle
            delta_theta = angle * axis
        delta_v = v_inj - v_nom
        delta_p = p_inj - p_nom
        delta_ba = ba_inj - ba_nom
        delta_bg = bg_inj - bg_nom
        delta_x = np.concatenate([delta_theta, delta_v, delta_p, delta_ba, delta_bg])
        return delta_x
    # --- End of helper functions ---

    # --------- Measurement: visual p,v (ESKF update for linear parts) -----------
    def vision_posvel_update(self, p_meas, v_meas, R_vis):
        # Linear parts use ESKF formulation (no Hessian for theta needed here)
        y = np.concatenate([p_meas - self.p, v_meas - self.v])
        H = np.zeros((6,15)); H[0:3,6:9]=np.eye(3); H[3:6,3:6]=np.eye(3)
        S = H @ self.P @ H.T + R_vis
        K = self.P @ H.T @ np.linalg.inv(S)
        delta_x = K @ y
        self._update_common(delta_x)               # Update nominal state and biases
        I15=np.eye(15); self.P = (I15-K@H)@self.P@(I15-K@H).T + K@R_vis@K.T

    # --------- Measurement: ZUPT (ESKF update) -----------------
    def zero_velocity_update(self):
        # Without iterating over 3D theta sigma points via UKF,
        # the linear ESKF H matrix is sufficient as the v_body measurement is linear in error states (approx).
        super().zero_velocity_update()             # Base ESKF update

# =========================================================
# CSV Result and Summary Functions
# =========================================================
def save_results_to_csv(results, filename):
    data = {
        "#timestamp [ns]": results["timestamps"],
        " q_RS_w []": results["estimated_quaternions"][:,0],
        " q_RS_x []": results["estimated_quaternions"][:,1],
        " q_RS_y []": results["estimated_quaternions"][:,2],
        " q_RS_z []": results["estimated_quaternions"][:,3],
        " v_RS_R_x [m s^-1]": results["estimated_velocities"][:,0],
        " v_RS_R_y [m s^-1]": results["estimated_velocities"][:,1],
        " v_RS_R_z [m s^-1]": results["estimated_velocities"][:,2],
        " p_RS_R_x [m]": results["estimated_positions"][:,0],
        " p_RS_R_y [m]": results["estimated_positions"][:,1],
        " p_RS_R_z [m]": results["estimated_positions"][:,2],
        " b_a_x [m s^-2]": results["estimated_accel_biases"][:,0],
        " b_a_y [m s^-2]": results["estimated_accel_biases"][:,1],
        " b_a_z [m s^-2]": results["estimated_accel_biases"][:,2],
        " b_g_x [rad s^-1]": results["estimated_gyro_biases"][:,0],
        " b_g_y [rad s^-1]": results["estimated_gyro_biases"][:,1],
        " b_g_z [rad s^-1]": results["estimated_gyro_biases"][:,2]
    }
    pd.DataFrame(data).to_csv(filename, index=False)

def save_summary_to_csv(results, seq_name, args, filename):
    summary = {
       "Sequence": seq_name,
       "Alpha_v": args.alpha_v, 
       "Epsilon_v": args.epsilon_v, 
       "Zeta_H": args.zeta_H, 
       "Zeta_L": args.zeta_L, 
       "Beta_p": args.beta_p, 
       "Epsilon_p": args.epsilon_p, 
       "Zeta_p": args.zeta_p, 
       "Entropy_Norm_Min": args.entropy_norm_min, 
       "Pose_Chi2_Norm_Min": args.pose_chi2_norm_min, 
       "Culled_Norm_Min": args.culled_norm_min,
       "W_THR": args.w_thr,
       "D_THR": args.d_thr,
       "Activation_Function": f"casef (s={args.s})",
       "Adaptive": args.adaptive,
       "RMSE_Quaternion_Angular_Deg": str(results["rmse_quat_angular_deg"]) if results["rmse_quat_angular_deg"] is not None else "None",
       "RMSE_Euler_Deg": str(results["rmse_euler_deg"]) if results["rmse_euler_deg"] is not None else "None",
       "RMSE_Velocity": str(results["rmse_vel"]) if results["rmse_vel"] is not None else "None",
       "RMSE_Position": str(results["rmse_pos"]) if results["rmse_pos"] is not None else "None",
       "RMSE_Accel_Bias": str(results["rmse_ba"]) if results["rmse_ba"] is not None else "None",
       "RMSE_Gyro_Bias": str(results["rmse_bg"]) if results["rmse_bg"] is not None else "None"
    }
    df_summary = pd.DataFrame([summary])
    if os.path.exists(filename):
        df_summary.to_csv(filename, index=False, mode="a", header=False)
    else:
        df_summary.to_csv(filename, index=False, mode="w", header=True)

# =========================================================
# ESKF-based VIO Data Processing Function
# =========================================================
def process_vio_data(imu_file, visual_file, sigma_v_map=None, sigma_p_map=None):
    # File reading (similar to previous version, basic error handling)
    imu_data = pd.read_csv(imu_file)
    imu_data['timestamp'] = pd.to_datetime(imu_data['#timestamp [ns]'], unit='ns')
    imu_data['dt'] = imu_data['timestamp'].diff().dt.total_seconds()
    imu_data.loc[0, 'dt'] = 0.0

    needed_cols = [
        ' q_RS_w []',' q_RS_x []',' q_RS_y []',' q_RS_z []',
        ' v_RS_R_x [m s^-1]',' v_RS_R_y [m s^-1]',' v_RS_R_z [m s^-1]',
        ' p_RS_R_x [m]',' p_RS_R_y [m]',' p_RS_R_z [m]',
        ' b_w_RS_S_x [rad s^-1]',' b_w_RS_S_y [rad s^-1]',' b_w_RS_S_z [rad s^-1]',
        ' b_a_RS_S_x [m s^-2]',' b_a_RS_S_y [m s^-2]',' b_a_RS_S_z [m s^-2]',
        'w_RS_S_x [rad s^-1]','w_RS_S_y [rad s^-1]','w_RS_S_z [rad s^-1]',
        'a_RS_S_x [m s^-2]','a_RS_S_y [m s^-2]','a_RS_S_z [m s^-2]'
    ]

    
    for c in needed_cols:
        if c not in imu_data.columns:
            imu_data[c] = 0.0
        imu_data[c] = pd.to_numeric(imu_data[c], errors='coerce')

    # File reading (similar to previous version, basic error handling)
    visual_data = pd.read_csv(visual_file)
    visual_data['timestamp'] = pd.to_datetime(visual_data['#timestamp [ns]'], unit='ns')
    for c in [' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']:
        if c not in visual_data.columns:
            visual_data[c] = 0.0
        visual_data[c] = pd.to_numeric(visual_data[c], errors='coerce')
    visual_data = visual_data.sort_values(by='timestamp').reset_index(drop=True)

    initial_quaternion = imu_data[[' q_RS_w []', ' q_RS_x []',' q_RS_y []',' q_RS_z []']].iloc[0].values
    initial_velocity   = imu_data[[' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]']].iloc[0].values
    if ' p_RS_R_x [m]' in imu_data.columns:
        initial_position = imu_data[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].iloc[0].values
    else:
        initial_position = np.zeros(3)


    initial_gyro_bias = np.array([
        imu_data[' b_w_RS_S_x [rad s^-1]'].iloc[0],
        imu_data[' b_w_RS_S_y [rad s^-1]'].iloc[0],
        imu_data[' b_w_RS_S_z [rad s^-1]'].iloc[0]
    ]) if all(c in imu_data.columns for c in [' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]']) else np.zeros(3)

    initial_accel_bias = np.array([
        imu_data[' b_a_RS_S_x [m s^-2]'].iloc[0],
        imu_data[' b_a_RS_S_y [m s^-2]'].iloc[0],
        imu_data[' b_a_RS_S_z [m s^-2]'].iloc[0]
    ]) if all(c in imu_data.columns for c in [' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]']) else np.zeros(3)

    # --- Instantiate the ESKF version ---
    eskf = ErrorStateKalmanFilterVIO(initial_quaternion, initial_velocity, initial_position,
                                     initial_accel_bias=initial_accel_bias,
                                     initial_gyro_bias=initial_gyro_bias)

    visual_index = 0
    max_visual_index = len(visual_data)
    prev_vis_time = None
    prev_vis_pos = None

    estimated_quaternions = []
    estimated_velocities = []
    estimated_positions = []
    estimated_accel_biases = []
    estimated_gyro_biases = []
    timestamps = []

    true_quaternions = []
    true_velocities = []
    true_positions = []
    true_accel_biases = []
    true_gyro_biases = []
    q_prev = initial_quaternion.copy()

    # --- ZUPT settings ---
    WIN_LEN_STATIC  = args.zupt_win          
    ACC_STD_THR     = args.zupt_acc_thr      
    GYRO_STD_THR    = args.zupt_gyro_thr     
    accel_window = deque(maxlen=WIN_LEN_STATIC)
    gyro_window  = deque(maxlen=WIN_LEN_STATIC) 
    total_static_duration = 0.0      
    for i, row in imu_data.iterrows():
        dt = row['dt']
        if dt <= 0: continue # Valid dt check
        
        gyro_raw = np.array([
            row['w_RS_S_x [rad s^-1]'],
            row['w_RS_S_y [rad s^-1]'],
            row['w_RS_S_z [rad s^-1]']
        ], dtype=float)
        accel_raw = np.array([
            row['a_RS_S_x [m s^-2]'],
            row['a_RS_S_y [m s^-2]'],
            row['a_RS_S_z [m s^-2]']
        ], dtype=float)

        # Prediction step (with raw IMU data)
        eskf.predict(gyro_raw, accel_raw, dt)
        current_time = row['timestamp']

        # --- Static detection & ZUPT / Gravity update ---
        accel_window.append(accel_raw)
        gyro_window.append(gyro_raw)

        if len(accel_window) == WIN_LEN_STATIC:
            # Standard deviation of acceleration norm is used for static detection.
            accel_data_stack = np.vstack(accel_window)
            acc_norm = np.linalg.norm(accel_data_stack, axis=1)
            acc_norm_std = acc_norm.std()

            # Standard deviation of gyroscope norm.
            gyro_data_stack = np.vstack(gyro_window)
            g_norm   = np.linalg.norm(gyro_data_stack, axis=1)
            gyro_norm_std = g_norm.std() # Gyro norm std dev

            if (acc_norm_std < ACC_STD_THR) and (gyro_norm_std < GYRO_STD_THR):
                eskf.zero_velocity_update()
                eskf.gravity_update(accel_raw) 
                window_duration = WIN_LEN_STATIC * dt # Approximate window duration
                total_static_duration += window_duration

                accel_window.clear()
                gyro_window.clear()

        # Visual measurement update (if available)
        while (visual_index < max_visual_index and
               (visual_data.loc[visual_index, 'timestamp'] <= current_time)):

            p_meas = np.array([
                visual_data.loc[visual_index, ' p_RS_R_x [m]'],
                visual_data.loc[visual_index, ' p_RS_R_y [m]'],
                visual_data.loc[visual_index, ' p_RS_R_z [m]']
            ], dtype=float)

            t_vis = visual_data.loc[visual_index, 'timestamp']
            t_vis_ns = t_vis.value

            fixed_sigma_v = None
            sigma_p_val = None

            if sigma_v_map and len(sigma_v_map) > 0:
                if t_vis_ns in sigma_v_map:
                    fixed_sigma_v = sigma_v_map[t_vis_ns]
                else:
                    min_diff = float('inf')
                    chosen_key = None
                    for k in sigma_v_map.keys():
                        diff = abs(k - t_vis_ns)
                        if diff < min_diff:
                            min_diff = diff
                            chosen_key = k
                    # Time matching tolerance for visual data (2ms).
                    if chosen_key is not None and min_diff < 2e6: # (2ms tolerance)
                        fixed_sigma_v = sigma_v_map[chosen_key]

            if sigma_p_map and len(sigma_p_map) > 0:
                if t_vis_ns in sigma_p_map:
                    sigma_p_val = sigma_p_map[t_vis_ns]
                else:
                    min_diff = float('inf')
                    chosen_key = None
                    for k in sigma_p_map.keys():
                        diff = abs(k - t_vis_ns)
                        if diff < min_diff:
                            min_diff = diff
                            chosen_key = k
                    if chosen_key is not None and min_diff < 2e6: # (2ms tolerance)
                        sigma_p_val = sigma_p_map[chosen_key]

            
            # --- Logic for calculating velocity from consecutive visual positions ---
            v_meas_to_use = eskf.v # Default (for an ineffective update if dt_vis is too small)
            dt_vis = 0.0
            if prev_vis_time is not None and prev_vis_pos is not None:
                dt_vis = (t_vis - prev_vis_time).total_seconds()
                if dt_vis > 1e-9: # Avoid very small dt_vis
                    v_meas_to_use = (p_meas - prev_vis_pos) / dt_vis

            I3 = np.eye(3)
            current_sigma_p_sq = eskf.R_vis_6d[0,0] # (fixed_sigma_p**2)
            current_sigma_v_sq = eskf.R_vis_6d[3,3] # (fixed_sigma_v**2)
            if sigma_p_val is not None:
                current_sigma_p_sq = sigma_p_val**2

            # If using sigma_v from adaptive_sigma_v independently, not derived from sigma_p:
            if fixed_sigma_v is not None: # fixed_sigma_v comes from sigma_v_map
                current_sigma_v_sq = fixed_sigma_v**2


            R_vision_current = np.block([
                [current_sigma_p_sq * I3, np.zeros((3,3))],
                [np.zeros((3,3)), current_sigma_v_sq * I3]
            ])

            eskf.vision_posvel_update(p_meas, v_meas_to_use, R_vision_current)

            prev_vis_pos = p_meas
            prev_vis_time = t_vis
            visual_index += 1

        q_est = ensure_quaternion_continuity(eskf.q, q_prev)
        estimated_quaternions.append(q_est)
        estimated_velocities.append(eskf.v.copy())
        estimated_positions.append(eskf.p.copy())
        estimated_accel_biases.append(eskf.b_a.copy()) 
        estimated_gyro_biases.append(eskf.b_g.copy())  
        timestamps.append(row['timestamp'].value)

        q_prev = q_est

        if all(col in imu_data.columns for col in [' q_RS_w []',' q_RS_x []',' q_RS_y []',' q_RS_z []']):
            q_gt = row[[' q_RS_w []',' q_RS_x []',' q_RS_y []',' q_RS_z []']].values
        else:
            q_gt = np.array([np.nan]*4)

        if all(col in imu_data.columns for col in [' v_RS_R_x [m s^-1]',' v_RS_R_y [m s^-1]',' v_RS_R_z [m s^-1]']):
            v_gt = row[[' v_RS_R_x [m s^-1]',' v_RS_R_y [m s^-1]',' v_RS_R_z [m s^-1]']].values
        else:
            v_gt = np.array([np.nan]*3)

        if all(col in imu_data.columns for col in [' p_RS_R_x [m]',' p_RS_R_y [m]',' p_RS_R_z [m]']):
            p_gt = row[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values
        else:
            p_gt = np.array([np.nan]*3)

        # Accel Bias
        if all(c in imu_data.columns for c in [' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]']):
            ba_gt = row[[' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]']].values
        else:
            ba_gt = np.array([np.nan]*3)

        # Gyro Bias
        if all(c in imu_data.columns for c in [' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]']):
            bg_gt = row[[' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]']].values
        else:
            bg_gt = np.array([np.nan]*3)

        true_quaternions.append(q_gt)
        true_velocities.append(v_gt)
        true_positions.append(p_gt)
        true_accel_biases.append(ba_gt) 
        true_gyro_biases.append(bg_gt)  

    # --- Convert results to NumPy arrays ---
    estimated_quaternions = np.array(estimated_quaternions, dtype=float)
    estimated_velocities = np.array(estimated_velocities, dtype=float)
    estimated_positions = np.array(estimated_positions, dtype=float)
    estimated_accel_biases = np.array(estimated_accel_biases, dtype=float)
    estimated_gyro_biases = np.array(estimated_gyro_biases, dtype=float)
    timestamps_ns = np.array(timestamps, dtype='int64')

    true_quaternions = np.array(true_quaternions, dtype=float)
    true_velocities = np.array(true_velocities, dtype=float)
    true_positions = np.array(true_positions, dtype=float)
    true_accel_biases = np.array(true_accel_biases, dtype=float)
    true_gyro_biases = np.array(true_gyro_biases, dtype=float)

    # --- RMSE Calculations ---
    estimated_euler_raw = np.array([quaternion_to_euler(q) for q in estimated_quaternions])
    # NaN check should be on true_quaternions before converting to Euler angles.
    valid_true_q_indices_for_euler = ~np.isnan(true_quaternions).any(axis=1)
    true_euler_raw = np.array([quaternion_to_euler(q) for q in true_quaternions[valid_true_q_indices_for_euler] if not np.isnan(q).any()])


    min_len = min(len(estimated_quaternions), len(true_quaternions))
    rmse_quat_angular_deg = None 
    rmse_euler_deg = None        
    rmse_vel = None
    rmse_pos = None
    rmse_ba = None
    rmse_bg = None
    estimated_euler_deg = estimated_euler_raw 
    true_euler_deg = true_euler_raw

    if min_len > 0:
        valid_gt_indices = np.where(
            ~np.isnan(true_quaternions[:min_len]).any(axis=1) &
            ~np.isnan(true_velocities[:min_len]).any(axis=1) &
            ~np.isnan(true_positions[:min_len]).any(axis=1) &
            ~np.isnan(true_accel_biases[:min_len]).any(axis=1) &
            ~np.isnan(true_gyro_biases[:min_len]).any(axis=1)
        )[0]

        if len(valid_gt_indices) > 0:
            min_len_valid = len(valid_gt_indices)
            est_q_valid = estimated_quaternions[valid_gt_indices]
            true_q_valid = true_quaternions[valid_gt_indices]
            est_v_valid = estimated_velocities[valid_gt_indices]
            true_v_valid = true_velocities[valid_gt_indices]
            est_p_valid = estimated_positions[valid_gt_indices]
            true_p_valid = true_positions[valid_gt_indices]
            est_ba_valid = estimated_accel_biases[valid_gt_indices]
            true_ba_valid = true_accel_biases[valid_gt_indices]
            est_bg_valid = estimated_gyro_biases[valid_gt_indices]
            true_bg_valid = true_gyro_biases[valid_gt_indices]

            # Quaternion Angular RMSE (adapted from plots.py, in degrees)
            aligned_quaternions_for_angular_rmse = align_quaternions(est_q_valid, true_q_valid)
            
            delta_q_array = [quaternion_multiply(qe, quaternion_conjugate(qg)) 
                             for qg, qe in zip(true_q_valid, aligned_quaternions_for_angular_rmse)]
            
            angles_rad_list = []
            for delta_q_val in delta_q_array:
                # Taking the absolute value of the w component and clipping is important
                cos_half_angle = np.clip(abs(delta_q_val[0]), -1.0, 1.0) 
                angle_rad = 2 * np.arccos(cos_half_angle)
                angles_rad_list.append(angle_rad)
            
            angles_deg_array = np.array(angles_rad_list) * 180.0 / np.pi
            # RMSE of angular errors (direct RMSE as target error is 0 degrees)
            rmse_quat_angular_deg = np.sqrt(np.mean(angles_deg_array**2))

            est_euler_deg_valid = np.array([quaternion_to_euler(q) for q in aligned_quaternions_for_angular_rmse])
            true_euler_deg_valid = np.array([quaternion_to_euler(q) for q in true_q_valid])

            # Ensure angle continuity (unwrap)
            est_euler_deg_valid_unwrapped = ensure_angle_continuity(est_euler_deg_valid, threshold=180) # threshold=180 (degrees) instead of 2*pi for rad
            true_euler_deg_valid_unwrapped = ensure_angle_continuity(true_euler_deg_valid, threshold=180)

            estimated_euler_deg = est_euler_deg_valid_unwrapped # Store for results
            true_euler_deg = true_euler_deg_valid_unwrapped     # Store for results
            
            # calculate_angle_rmse function correctly computes differences in degrees
            rmse_euler_deg = calculate_angle_rmse(est_euler_deg_valid_unwrapped, true_euler_deg_valid_unwrapped)


            # Velocity RMSE
            diff_v = est_v_valid - true_v_valid
            rmse_vel = np.sqrt((diff_v**2).mean(axis=0))

            # Position RMSE
            diff_p = est_p_valid - true_p_valid
            rmse_pos = np.sqrt((diff_p**2).mean(axis=0))

            # Accel Bias RMSE
            diff_ba = est_ba_valid - true_ba_valid
            rmse_ba = np.sqrt((diff_ba**2).mean(axis=0))

            # Gyro Bias RMSE
            diff_bg = est_bg_valid - true_bg_valid
            rmse_bg = np.sqrt((diff_bg**2).mean(axis=0))

    # --- Collect Results in a Dictionary ---
    results = {
        "timestamps": timestamps_ns,
        "estimated_quaternions": estimated_quaternions,
        "estimated_euler_deg": estimated_euler_deg, 
        "estimated_euler_raw": estimated_euler_raw, 
        "estimated_velocities": estimated_velocities,
        "estimated_positions": estimated_positions,
        "estimated_accel_biases": estimated_accel_biases,
        "estimated_gyro_biases": estimated_gyro_biases,
        "true_quaternions": true_quaternions,
        "true_euler_deg": true_euler_deg,           
        "true_euler_raw": true_euler_raw,           
        "rmse_quat_angular_deg": rmse_quat_angular_deg, 
        "rmse_euler_deg": rmse_euler_deg,               
        "rmse_vel": rmse_vel,
        "rmse_pos": rmse_pos,
        "rmse_ba": rmse_ba,
        "rmse_bg": rmse_bg,
        "total_static_duration": total_static_duration
    }
    return results

# =========================================================
# Function to process a single sequence (e.g., MH01)
# =========================================================
def run_sequence(seq_name, config, summary_file):
    imu_file_path = f"imu_interp_gt/{seq_name}_imu_with_interpolated_groundtruth.csv"
    sequence_number_str = seq_name[2:]  # Example: MH01 -> "01"
    # Convert to int to remove leading zero (e.g., "01" -> 1), then format into path
    visual_file_path = f"VO/vo_pred_super_best/mh{int(sequence_number_str)}_ns.csv"

    if args.adaptive:
        sigma_v_map = compute_adaptive_sigma_v(config, visual_file_path, seq_name)
        sigma_p_map = compute_adaptive_sigma_p(config, visual_file_path, seq_name)
    else:
        sigma_v_map = {}
        sigma_p_map = {}

    print(f"[{seq_name}] Starting...")
    start_time = time.time()
    results = process_vio_data(imu_file_path, visual_file_path, sigma_v_map, sigma_p_map)
    # Check if results is None (in case of processing error)
    if results is None:
        print(f"[{seq_name}] Could not be processed due to an error.")
        return 

    print(f"[{seq_name}] === RMSE Results ===")
    print("Quaternion Angular RMSE (degrees):", results["rmse_quat_angular_deg"]) 
    print("Euler Angle RMSE (degrees):", results["rmse_euler_deg"]) 
    print("Velocity RMSE (m/s):", results["rmse_vel"])
    if results["rmse_pos"] is not None:
        print("Position RMSE (m):************************************************", results["rmse_pos"],"***************")
    if results["rmse_ba"] is not None:
        print("Accel Bias RMSE (m/s^2):", results["rmse_ba"])
    if results["rmse_bg"] is not None:
        print("Gyro Bias RMSE (rad/s):", results["rmse_bg"])
    if "total_static_duration" in results:
        print(f"Total Static Duration (Gravity Update Active): {results['total_static_duration']:.4f} sn")


    detailed_csv = f"outputs/adaptive_{seq_name.lower()}.csv"
    save_results_to_csv(results, filename=detailed_csv)

    if SAVE_RESULTS_CSV:
        save_summary_to_csv(results, seq_name, args, filename=summary_file)

    end_time = time.time()
    print(f"[{seq_name}] Total Execution Time: {end_time - start_time:.4f} s")

# =========================================================
# MAIN PROGRAM: Run sequences in parallel
# =========================================================
if __name__ == "__main__":
    sequences = ["MH01", "MH02", "MH03", "MH04", "MH05"]
    summary_file = SAVE_RESULTS_CSV_NAME

    # --- Initial CSV Check ---
    results_exist = False
    if os.path.exists(summary_file) and SAVE_RESULTS_CSV:
        try:
            df_results = pd.read_csv(summary_file)
            required_cols_check = [
                "Sequence", "Alpha_v", "Epsilon_v", "Zeta_H", "Zeta_L",
                "Beta_p", "Epsilon_p", "Zeta_p", 
                "Entropy_Norm_Min", "Pose_Chi2_Norm_Min", "Culled_Norm_Min",
                "W_THR", "D_THR", "Activation_Function", "Adaptive"
            ]
            if all(col in df_results.columns for col in required_cols_check):
                filtered_df = df_results[
                    (df_results["Alpha_v"] == args.alpha_v) &                    
                    (df_results["Epsilon_v"] == args.epsilon_v) &
                    (df_results["Zeta_H"] == args.zeta_H) &
                    (df_results["Zeta_L"] == args.zeta_L) &
                    (df_results["Beta_p"] == args.beta_p) &
                    (df_results["Epsilon_p"] == args.epsilon_p) &
                    (df_results["Zeta_p"] == args.zeta_p) &
                    (df_results["Entropy_Norm_Min"] == args.entropy_norm_min) & 
                    (df_results["Pose_Chi2_Norm_Min"] == args.pose_chi2_norm_min) & 
                    (df_results["Culled_Norm_Min"] == args.culled_norm_min) & 
                    (df_results["W_THR"] == args.w_thr) &
                    (df_results["D_THR"] == args.d_thr) &
                    (df_results["Activation_Function"] == f"casef (s={args.s})") &
                    (df_results["Adaptive"] == args.adaptive)
                ]
                
                existing_sequences = set(filtered_df["Sequence"].unique())
                # Check if all sequences exist for this parameter set
                if set(sequences).issubset(existing_sequences):
                    results_exist = True
            else:
                 print(f"Warning: {summary_file} is missing expected columns (check normalization parameters?). Skipping check.")

        except pd.errors.EmptyDataError:
            print(f"Warning: {summary_file} is empty. Skipping check.")
        except Exception as e:
            print(f"Warning: Error reading {summary_file}: {e}. Skipping check.")
    if results_exist:
        print(f"Results for this parameter combination (alpha_v={args.alpha_v}, epsilon_v={args.epsilon_v}, zeta_H={args.zeta_H}, zeta_L={args.zeta_L}, beta_p={args.beta_p}, epsilon_p={args.epsilon_p}, zeta_p={args.zeta_p}, entropy_norm_min={args.entropy_norm_min}, pose_chi2_norm_min={args.pose_chi2_norm_min}, culled_norm_min={args.culled_norm_min}, w_thr={args.w_thr}, d_thr={args.d_thr}, activation=casef (s={args.s}), adaptive={args.adaptive}) already exist in '{summary_file}'.")
        print("The software will not be run.")
        sys.exit(0) # Exit program
    # --- End of CSV Check ---

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for seq in sequences:
            fut = executor.submit(run_sequence, seq, config, summary_file)
            futures.append(fut)

        concurrent.futures.wait(futures)

    print("=== Processing of all sequences finished. ===")