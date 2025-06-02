SAVE_RESULTS_CSV = False
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

# Sigma_p weights for just experiment 
parser.add_argument("--beta_p", type=float, default=1, help="Weight for inv_entropy (for sigma_p)")
parser.add_argument("--epsilon_p", type=float, default=1, help="Weight for pose_chi2 (for sigma_p)")
parser.add_argument("--zeta_p", type=float, default=1, help="Weight for culled_keyframes (for sigma_p)")

# Sigma_p Normalization Minimum Values.
parser.add_argument("--entropy_norm_min", type=float, default=0.5, help="Minimum value for Entropy normalization (for sigma_p)")
parser.add_argument("--pose_chi2_norm_min", type=float, default=0.6, help="Minimum value for Pose Chi2 normalization (for sigma_p)")
parser.add_argument("--culled_norm_min", type=float, default=0, help="Minimum value for Culled Keyframes normalization (for sigma_p)")

# Sigma_v weights "_H represent just Rising, _L represent just Falling" for experiment, You can do same for _H and _L
parser.add_argument("--alpha_v", type=float, default=18, help="Weight for Intensity difference (for sigma_v)")
parser.add_argument("--epsilon_v", type=float, default=3.3, help="Weight for Pose_chi2 difference (for sigma_v)")
parser.add_argument("--zeta_H", type=float, default=1, help="Weight for increasing culled_keyframes (for sigma_v)")
parser.add_argument("--zeta_L", type=float, default=0, help="Weight for decreasing culled_keyframes (for sigma_v)")

parser.add_argument("--w_thr", type=float, default=0.20, help="w_thr parameter for image confidence")
parser.add_argument("--d_thr", type=float, default=1, help="d_thr parameter for image confidence")
parser.add_argument("--s", type=float, default=1, help="s parameter for CASEF activation function")
parser.add_argument("--adaptive", action="store_true", default=True, help="Enable adaptive covariance")


parser.add_argument("--zupt_acc_thr", type=float, default=0.1, help="Acceleration std threshold for ZUPT [m/s²]")
parser.add_argument("--zupt_gyro_thr", type=float, default=0.1, help="Gyroscope std threshold for ZUPT [rad/s]")
parser.add_argument("--zupt_win", type=int, default=60, help="Window size for ZUPT (number of samples)")

args = parser.parse_args()

# =========================================================
# Activation Functions
# =========================================================
def casef(x, s):
    """
    Clipped Adaptive Saturation Exponential Function (CASEF)
    For negative inputs returns 0, scales exponentially in the 0–1 range,
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
# Helper Functions (Quaternion Operations etc.)
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

def R_to_quat_wxyz(rotation_matrix):
    # SciPy as_quat() returns [x, y, z, w]
    q_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def quat_wxyz_to_R(q_wxyz):
    # SciPy from_quat() expects [x, y, z, w]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return R.from_quat(q_xyzw).as_matrix()

so3_exp = lambda phi: R.from_rotvec(phi).as_matrix()
so3_log = lambda rotation_matrix: R.from_matrix(rotation_matrix).as_rotvec()


def calculate_angle_rmse(predictions, targets):
    diff = np.array([angle_difference(p, t) for p, t in zip(predictions, targets)])
    return np.sqrt((diff ** 2).mean(axis=0))

# =========================================================
# Invariant EKF (IEKF) Class for VIO
# =========================================================
class InvariantEKF_VIO:
    """
    Base Invariant Extended Kalman Filter (IEKF) for Visual-Inertial Odometry.
    Handles nominal state propagation on the SE(3) manifold and error state propagation.
    """
    def __init__(self, initial_quaternion, initial_velocity, initial_position,
                 initial_accel_bias=np.zeros(3), initial_gyro_bias=np.zeros(3),
                 gravity_vector=np.array([0, 0, -9.81]),
                 # Gauss-Markov & Noise Parameters (EuRoC MAV Values)                 
                 tau_a=300.0, tau_g=1000.0,         # Correlation times (s)
                 sigma_wa=2e-5, sigma_wg=2e-5, 
                 sigma_g=2e-4, sigma_a=2e-1):

        # Nominal State
        self.R = quat_wxyz_to_R(initial_quaternion / np.linalg.norm(initial_quaternion))
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
        """ Computes discrete-time state transition matrix (Phi) and process noise covariance (Qd) using Van Loan's method. """
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # Continuous-time error state dynamics matrix A (15x15)
        A = np.zeros((15, 15))
        A[0:3, 0:3] = -skew(gyro_corrected)    # d(delta_theta)/dt
        A[0:3, 12:15] = -I3                    # d(delta_theta)/dt dependency on delta_b_g
        A[3:6, 0:3] = -R_nav_from_body @ skew(accel_corrected) # d(delta_v)/dt dependency on delta_theta
        A[3:6, 9:12] = -R_nav_from_body        # d(delta_v)/dt dependency on delta_b_a
        A[6:9, 3:6] = I3                       # d(delta_p)/dt dependency on delta_v
        A[9:12, 9:12] = -1/self.tau_a * I3     # d(delta_b_a)/dt (Gauss-Markov)
        A[12:15, 12:15] = -1/self.tau_g * I3   # d(delta_b_g)/dt (Gauss-Markov)

        # Matrix creation for Van Loan (30x30)
        M = np.zeros((30, 30))
        M[0:15, 0:15] = A * dt
        M[0:15, 15:30] = self.Q_c * dt
        M[15:30, 15:30] = -A.T * dt # Component of the Van Loan formulation structure

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
        
        # Store old state for p,v propagation
        v_old = self.v.copy()
        p_old = self.p.copy()
        R_old = self.R.copy() # Not strictly needed for this formulation if self.R is updated sequentially

        # Correct biases
        accel_corrected = accel_raw - self.b_a
        gyro_corrected = gyro_raw - self.b_g

        # --- Nominal State Update ---
        # Rotation update
        self.R = self.R @ so3_exp(gyro_corrected * dt)

        # Acceleration in navigation frame using *new* R
        a_nav = self.R @ accel_corrected + self.g
        
        # Velocity and Position update (using old v for p update consistency with ESKF form)
        self.v = v_old + a_nav * dt
        self.p = p_old + v_old * dt + 0.5 * a_nav * dt**2

        # Bias state remains constant (only error state is updated)
        # Nominal biases are considered constant between updates; their uncertainty is handled in P.

        # --- Error State Covariance Update (Van Loan) ---
        Phi, Qd = self._compute_van_loan_matrices(self.R, accel_corrected, gyro_corrected, dt)

        # Update covariance: P = Phi * P * Phi^T + Qd
        P_new = Phi @ self.P @ Phi.T + Qd
        # Ensure P remains symmetric
        self.P = 0.5 * (P_new + P_new.T)

        # Nominal state (self.R, self.v, self.p) already updated above.

    def _update_common(self, delta_x):
        """ Common nominal state update logic after an error state correction. """
        delta_theta = delta_x[0:3]
        self.R = self.R @ so3_exp(delta_theta)

        self.v += delta_x[3:6]
        self.p += delta_x[6:9]
        self.b_a += delta_x[9:12]
        self.b_g += delta_x[12:15]


    def zero_velocity_update(self):
        R_body_from_nav = self.R.T

        # Measurement: velocity in navigation frame expressed in body frame
        v_body_predicted = R_body_from_nav @ self.v
        y = -v_body_predicted # Measurement residual

        H = np.zeros((3, 15))
        # Jacobian of y = 0 - v_body, so Jacobian is -[v_body]x.
        H[:, 0:3] = skew(v_body_predicted) 
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

        # Update nominal state with error correction
        self._update_common(delta_x)
        # Hard reset velocity to zero after correction
        self.v = np.zeros(3)
        
        I15 = np.eye(15)
        P_new = (I15 - K @ H) @ self.P @ (I15 - K @ H).T + K @ self.R_zupt @ K.T # Joseph form
        self.P = 0.5 * (P_new + P_new.T) # Ensure symmetry

    def vision_posvel_update(self, p_meas, v_meas, R_vision):
        """ Updates state with visual position and velocity measurements. R_vision is the measurement noise covariance. """
        y = np.concatenate([p_meas - self.p, v_meas - self.v]) # Measurement residual (6x1)
        H = np.zeros((6, 15))
        H[0:3, 6:9] = np.eye(3)  # H_p: Effect of position error on measurement
        H[3:6, 3:6] = np.eye(3)  # H_v: Effect of velocity error on measurement

        # Kalman update steps
        S = H @ self.P @ H.T + R_vision # Uses updated R_vision
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Vision S matrix is not invertible!")
            return # Update cannot be performed
        K = self.P @ H.T @ S_inv # Kalman gain (15x6)

        delta_x = K @ y # Error state correction (15x1)

        # Update state and covariance (Joseph Form)
        self._update_common(delta_x)
        I15 = np.eye(15)
        P_new = (I15 - K @ H) @ self.P @ (I15 - K @ H).T + K @ R_vision @ K.T # Joseph form
        self.P = 0.5 * (P_new + P_new.T) # Ensure symmetry

    def gravity_update(self, accel_raw):
        """ Updates accelerometer bias using gravity during static periods. """
        # self.R is the current estimate of R_nav_from_body
        R_body_from_nav = self.R.T

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
        
        self._update_common(dx)

        I15 = np.eye(15)
        P_new = (I15 - K @ H) @ self.P @ (I15 - K @ H).T + K @ R_acc @ K.T # Joseph form
        self.P = 0.5 * (P_new + P_new.T) # Ensure symmetry

# === H Y B R I D  F I L T E R  C L A S S ==========================
class InvariantHybridKF_VIO(InvariantEKF_VIO):
    """
    Hybrid Invariant Kalman Filter for VIO.
    Uses Cubature Kalman Filter (CKF) to refine the orientation error covariance (P_tt)
    and the base Invariant EKF (IEKF) for all other state components and cross-covariances.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        # Initialize InvariantEKF_VIO components
        # --- CKF parameters: only for δθ (3 dimensions) ---
        self.n_ckf = 3  # Dimension of the state for CKF (orientation error)
        self.m_ckf = 2 * self.n_ckf  # Number of cubature points

        # Weights for cubature points
        self.W_ckf = np.full(self.m_ckf, 1.0 / self.m_ckf)

        # Base cubature points (xi_j = sqrt(n_ckf) * [e_j; -e_j])
        self.base_cubature_points = np.zeros((self.m_ckf, self.n_ckf))
        sqrt_n_ckf = np.sqrt(self.n_ckf)
        for j in range(self.n_ckf):
            self.base_cubature_points[j, j] = sqrt_n_ckf
            self.base_cubature_points[j + self.n_ckf, j] = -sqrt_n_ckf # type: ignore

        # Initialize Cholesky factor (S_tt) of P_tt (orientation error covariance block)
        # P[0:3,0:3] is initially P_init_ori, which is 1e-6 * eye(3).
        self.S_tt = self._ensure_pd_and_chol(self.P[0:3,0:3].copy())

    def _cubature_points_theta_from_S(self, S_tt_chol): # S_tt_chol is L such that P_tt = L L^T
        """ Generates cubature points for the 3D orientation error (delta_theta) from its Cholesky factor. """
        # Cubature points: X_j = S_tt_chol @ xi_j
        # where xi_j are sqrt(n_ckf) * [e_1, -e_1, e_2, -e_2, ..., e_n, -e_n]
        # And mean is assumed to be zero for error state
        cub_pts = np.zeros((self.m_ckf, self.n_ckf))
        for i in range(self.m_ckf):
            cub_pts[i, :] = S_tt_chol @ self.base_cubature_points[i, :]
        return cub_pts

    def _ensure_pd_and_chol(self, matrix, default_jitter=1e-9, min_eigenvalue_abs=1e-7):
        """ Ensures the matrix is positive definite and returns its Cholesky factor. """
        mat = 0.5 * (matrix + matrix.T) # Ensure symmetry
        try:
            # Attempt direct Cholesky with a small jitter
            return np.linalg.cholesky(mat + np.eye(mat.shape[0]) * default_jitter)
        except np.linalg.LinAlgError:
            # If it fails, use the eigenvalue method
            eigenvalues, eigenvectors = np.linalg.eigh(mat)
            # Clamp small/negative eigenvalues to a small positive value
            eigenvalues[eigenvalues < min_eigenvalue_abs] = min_eigenvalue_abs
            mat_pd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Add jitter again for safety before the final Cholesky
            return np.linalg.cholesky(mat_pd + np.eye(mat.shape[0]) * default_jitter)

    # ---------------- P R E D I C T (with CKF for orientation error covariance) --------------------
    def predict(self, gyro_raw, accel_raw, dt):
        if dt <= 0: return

        # ➊  -- Perform full IEKF prediction (updates nominal state and full P matrix) --
        R_old, v_old, p_old = self.R.copy(), self.v.copy(), self.p.copy()
        ba_old, bg_old = self.b_a.copy(), self.b_g.copy()
        super().predict(gyro_raw, accel_raw, dt) # Base IEKF predict updates self.R, self.v, self.p and self.P

        # ➋ -- Get the Cholesky factor of the orientation error covariance P[0:3,0:3] updated by IEKF --
        P_tt_after_eskf = self.P[0:3,0:3].copy()
        self.S_tt = self._ensure_pd_and_chol(P_tt_after_eskf)

        # ➌  -- Refine P[0:3,0:3] using CKF propagation for orientation error (delta_theta) --
        # Generate cubature points for delta_theta (mean is zero) using S_tt from IEKF update
        cubature_delta_theta_points = self._cubature_points_theta_from_S(self.S_tt)
        propagated_delta_theta_points = np.zeros((self.m_ckf, self.n_ckf))

        for i, dtheta in enumerate(cubature_delta_theta_points):
            # Inject orientation error cubature point into old nominal rotation R_old
            R_i = R_old @ so3_exp(dtheta)
            # Propagate the full state starting from perturbed R_i and old v,p,ba,bg
            # This uses the same nominal propagation model as the IEKF.
            R_i_prop, v_i_prop, p_i_prop, ba_i_prop, bg_i_prop = self._propagate_nominal(
                R_i, v_old, p_old, ba_old, bg_old, gyro_raw, accel_raw, dt
            )
            # Retract the propagated perturbed state against the new IEKF nominal state (self.R, self.v, etc.)
            # to get the propagated error cubature point in the Lie algebra.
            delta_x_retracted = self._retract_explicit(
                self.R, self.v, self.p, self.b_a, self.b_g, # New nominal state
                R_i_prop, v_i_prop, p_i_prop, ba_i_prop, bg_i_prop # Propagated perturbed state
            )
            dtheta_new = delta_x_retracted[0:3]
            propagated_delta_theta_points[i] = dtheta_new

        # Calculate predicted mean of propagated delta_theta points
        # This mean should ideally be close to zero.
        mean_propagated_delta_theta = np.sum(self.W_ckf[:, None] * propagated_delta_theta_points, axis=0)

        # Calculate refined P_tt using CKF
        P_tt_ckf_refined = np.zeros((self.n_ckf, self.n_ckf))
        for i in range(self.m_ckf):
            diff = propagated_delta_theta_points[i] - mean_propagated_delta_theta
            P_tt_ckf_refined += self.W_ckf[i] * np.outer(diff, diff)
       
        self.P[0:3, 0:3] = 0.5*(P_tt_ckf_refined + P_tt_ckf_refined.T) # Make symmetric
        self.S_tt = self._ensure_pd_and_chol(self.P[0:3,0:3].copy())   # Update S_tt from the CKF-refined P_tt

    # --- Helper functions for CKF part (nominal state propagation and retraction) ---
    def _propagate_nominal(self, R_curr, v_curr, p_curr, b_a_curr, b_g_curr, gyro_raw, accel_raw, dt):
        """ Helper to propagate a given nominal state (R,v,p,ba,bg) using IMU measurements. Used by CKF. """
        if dt <= 0:
            return R_curr, v_curr, p_curr, b_a_curr, b_g_curr

        accel_corrected = accel_raw - b_a_curr
        gyro_corrected = gyro_raw - b_g_curr

        R_new = R_curr @ so3_exp(gyro_corrected * dt)

        a_nav_start = R_curr @ accel_corrected + self.g
        a_nav_end = R_new @ accel_corrected + self.g
        v_new = v_curr + 0.5 * (a_nav_start + a_nav_end) * dt
        p_new = p_curr + v_curr * dt + 0.25 * (a_nav_start + a_nav_end) * dt**2
        
        b_a_new = b_a_curr # Biases are random walks, nominal part constant here
        b_g_new = b_g_curr
        return R_new, v_new, p_new, b_a_new, b_g_new

    def _retract_explicit(self, R_nom, v_nom, p_nom, ba_nom, bg_nom, R_inj, v_inj, p_inj, ba_inj, bg_inj):
        """ Helper to retract an 'injected/perturbed' state to an error state relative to a nominal state. Used by CKF. """
        # Rotation error
        delta_R_matrix = R_nom.T @ R_inj
        delta_theta = so3_log(delta_R_matrix)

        # Other errors
        delta_v = v_inj - v_nom
        delta_p = p_inj - p_nom
        delta_ba = ba_inj - ba_nom
        delta_bg = bg_inj - bg_nom
        delta_x = np.concatenate([delta_theta, delta_v, delta_p, delta_ba, delta_bg])
        return delta_x

    # --- End of  helper functions ---

    # --------- Measurement updates call base IEKF methods -----------
    def vision_posvel_update(self, p_meas, v_meas, R_vis):
        # The hybrid CKF refinement is only in the predict step for orientation covariance.
        # Measurement updates use the standard IEKF formulation from the base class.
        y = np.concatenate([p_meas - self.p, v_meas - self.v])
        H = np.zeros((6,15)); H[0:3,6:9]=np.eye(3); H[3:6,3:6]=np.eye(3)
        S = H @ self.P @ H.T + R_vis
        K = self.P @ H.T @ np.linalg.inv(S)
        delta_x = K @ y
        self._update_common(delta_x)               # Update nominal state + biases
        I15=np.eye(15); self.P = (I15-K@H)@self.P@(I15-K@H).T + K@R_vis@K.T

    def zero_velocity_update(self):
        super().zero_velocity_update() # Uses base IEKF ZUPT update

# =========================================================
# CSV Results and Summary Functions
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
# VIO Data Processing Function using Invariant Hybrid KF
# =========================================================
def process_vio_data(imu_file, visual_file, sigma_v_map=None, sigma_p_map=None):
    # Read IMU data
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
    # Read visual data
    visual_data = pd.read_csv(visual_file)
    visual_data['timestamp'] = pd.to_datetime(visual_data['#timestamp [ns]'], unit='ns')
    for c in [' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']:
        if c not in visual_data.columns:
            visual_data[c] = 0.0
        visual_data[c] = pd.to_numeric(visual_data[c], errors='coerce')
    visual_data = visual_data.sort_values(by='timestamp').reset_index(drop=True)
    # Initialize filter state with first IMU reading or defaults
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
    
    # --- Instantiate the Hybrid version ---
    iekf = InvariantHybridKF_VIO(initial_quaternion, initial_velocity, initial_position,
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

    true_quaternions = [] # Ground truth
    true_velocities = []
    true_positions = []
    true_accel_biases = []
    true_gyro_biases = []
    R_prev_for_continuity_conversion = iekf.R.copy() # For converting back to quat for logging

    # --- ZUPT settings ---
    WIN_LEN_STATIC  = args.zupt_win          
    ACC_STD_THR     = args.zupt_acc_thr      
    GYRO_STD_THR    = args.zupt_gyro_thr     
    accel_window = deque(maxlen=WIN_LEN_STATIC)
    gyro_window  = deque(maxlen=WIN_LEN_STATIC) 
    total_static_duration = 0.0      

    for i, row in imu_data.iterrows():
        dt = row['dt']
        if dt <= 0: continue # Skip if delta time is not valid
        
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
        
        # Prediction step (with raw data) - Calls the predict method of the instantiated filter (Hybrid in this case)
        iekf.predict(gyro_raw, accel_raw, dt)
        current_time = row['timestamp']

        # --- Static detection & ZUPT / Gravity update ---
        accel_window.append(accel_raw)
        gyro_window.append(gyro_raw)

        if len(accel_window) == WIN_LEN_STATIC:
            # Calculate standard deviation of acceleration norm
            accel_data_stack = np.vstack(accel_window)
            acc_norm = np.linalg.norm(accel_data_stack, axis=1)
            acc_norm_std = acc_norm.std()

            # Calculate standard deviation of gyroscope norm
            gyro_data_stack = np.vstack(gyro_window)
            g_norm   = np.linalg.norm(gyro_data_stack, axis=1)
            gyro_norm_std = g_norm.std()

            if (acc_norm_std < ACC_STD_THR) and (gyro_norm_std < GYRO_STD_THR):
                iekf.zero_velocity_update()
                iekf.gravity_update(accel_raw) 
                window_duration = WIN_LEN_STATIC * dt # Approximate window duration
                total_static_duration += window_duration
                # Clear windows after ZUPT/Gravity update to detect next static period
                accel_window.clear()
                gyro_window.clear()
        
        # Visual measurement update (if available and timestamp matches)
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

            # --- Logic for calculating velocity from consecutive positions for visual velocity measurement ---
            v_meas_to_use = iekf.v # Default to current filter velocity
            dt_vis = 0.0
            if prev_vis_time is not None and prev_vis_pos is not None:
                dt_vis = (t_vis - prev_vis_time).total_seconds()
                if dt_vis > 1e-9: # Avoid very small dt
                    v_meas_to_use = (p_meas - prev_vis_pos) / dt_vis

            I3 = np.eye(3)
            current_sigma_p_sq = iekf.R_vis_6d[0,0] # Default sigma_p squared
            current_sigma_v_sq = iekf.R_vis_6d[3,3] # Default sigma_v squared

            if sigma_p_val is not None:
                current_sigma_p_sq = sigma_p_val**2
            if fixed_sigma_v is not None: # fixed_sigma_v is obtained from sigma_v_map
                current_sigma_v_sq = fixed_sigma_v**2


            R_vision_current = np.block([
                [current_sigma_p_sq * I3, np.zeros((3,3))],
                [np.zeros((3,3)), current_sigma_v_sq * I3]
            ])

            iekf.vision_posvel_update(p_meas, v_meas_to_use, R_vision_current)

            prev_vis_pos = p_meas
            prev_vis_time = t_vis
            visual_index += 1

        # Convert current R to quaternion for logging and continuity
        q_current_est_wxyz = R_to_quat_wxyz(iekf.R)
        q_prev_for_continuity_wxyz = R_to_quat_wxyz(R_prev_for_continuity_conversion)
        q_est = ensure_quaternion_continuity(q_current_est_wxyz, q_prev_for_continuity_wxyz)
        
        estimated_quaternions.append(q_est)
        estimated_velocities.append(iekf.v.copy())
        estimated_positions.append(iekf.p.copy())
        estimated_accel_biases.append(iekf.b_a.copy()) 
        estimated_gyro_biases.append(iekf.b_g.copy())  
        timestamps.append(row['timestamp'].value)
        R_prev_for_continuity_conversion = iekf.R.copy() # Update for next iteration

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

            # Quaternion Angular RMSE (adapted from plots.py and in degrees)
            aligned_quaternions_for_angular_rmse = align_quaternions(est_q_valid, true_q_valid)
            
            delta_q_array = [quaternion_multiply(qe, quaternion_conjugate(qg)) 
                             for qg, qe in zip(true_q_valid, aligned_quaternions_for_angular_rmse)]
            
            angles_rad_list = []
            for delta_q_val in delta_q_array:
                # Absolute value of w component is clipped to ensure valid acos input
                cos_half_angle = np.clip(abs(delta_q_val[0]), -1.0, 1.0) 
                angle_rad = 2 * np.arccos(cos_half_angle)
                angles_rad_list.append(angle_rad)
            
            angles_deg_array = np.array(angles_rad_list) * 180.0 / np.pi
            # RMSE of angular errors (direct root mean square, as target error is 0 degrees)
            rmse_quat_angular_deg = np.sqrt(np.mean(angles_deg_array**2))

            est_euler_deg_valid = np.array([quaternion_to_euler(q) for q in aligned_quaternions_for_angular_rmse])
            true_euler_deg_valid = np.array([quaternion_to_euler(q) for q in true_q_valid])

            # Ensure angle continuity (unwrap)
            est_euler_deg_valid_unwrapped = ensure_angle_continuity(est_euler_deg_valid, threshold=180) # (degrees)
            true_euler_deg_valid_unwrapped = ensure_angle_continuity(true_euler_deg_valid, threshold=180)
            estimated_euler_deg = est_euler_deg_valid_unwrapped # Store for results
            true_euler_deg = true_euler_deg_valid_unwrapped     # Store for results
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

    # --- Collect Results in Dictionary ---
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
    visual_file_path = f"VO/vo_pred_super_512/mh{int(sequence_number_str)}_ns.csv"    

    if args.adaptive:
        sigma_v_map = compute_adaptive_sigma_v(config, visual_file_path, seq_name)
        sigma_p_map = compute_adaptive_sigma_p(config, visual_file_path, seq_name)
    else:
        sigma_v_map = {}
        sigma_p_map = {}

    print(f"[{seq_name}] Starting...")
    start_time = time.time()
    results = process_vio_data(imu_file_path, visual_file_path, sigma_v_map, sigma_p_map)
    if results is None:
        print(f"[{seq_name}] Could not be processed due to an error.")
        return 

    print(f"[{seq_name}] === RMSE Results ===")
    print("Quaternion Angular RMSE (degrees):", results["rmse_quat_angular_deg"]) 
    print("Euler Angle RMSE (degrees):", results["rmse_euler_deg"]) 
    print("Velocity RMSE (m/s):", results["rmse_vel"])
    if results["rmse_pos"] is not None:
        print("Position RMSE (m):", results["rmse_pos"])
    if results["rmse_ba"] is not None:
        print("Accel Bias RMSE (m/s^2):", results["rmse_ba"])
    if results["rmse_bg"] is not None:
        print("Gyro Bias RMSE (rad/s):", results["rmse_bg"])
    if "total_static_duration" in results:
        print(f"Total Static Duration (Gravity Update Active): {results['total_static_duration']:.4f} s")


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
                # Check if all sequences exist
                if set(sequences).issubset(existing_sequences):
                    results_exist = True
            else:
                 print(f"Warning: Expected columns are missing in {summary_file} (Were normalization parameters checked?). Skipping check.")

        except pd.errors.EmptyDataError:
            print(f"Warning: {summary_file} is empty. Skipping check.")
        except Exception as e:
            print(f"Warning: Error reading {summary_file}: {e}. Skipping check.")

    if results_exist:
        print(f"Results for this parameter combination (alpha_v={args.alpha_v}, epsilon_v={args.epsilon_v}, zeta_H={args.zeta_H}, zeta_L={args.zeta_L}, beta_p={args.beta_p}, epsilon_p={args.epsilon_p}, zeta_p={args.zeta_p}, entropy_norm_min={args.entropy_norm_min}, pose_chi2_norm_min={args.pose_chi2_norm_min}, culled_norm_min={args.culled_norm_min}, w_thr={args.w_thr}, d_thr={args.d_thr}, activation=casef (s={args.s}), adaptive={args.adaptive}) already exist in '{summary_file}'.")
        print("The software will not run.")
        sys.exit(0) # Exit program
    # --- End of CSV Check ---

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for seq in sequences:
            fut = executor.submit(run_sequence, seq, config, summary_file)
            futures.append(fut)

        concurrent.futures.wait(futures)

    print("=== Processing of all sequences finished. ===")