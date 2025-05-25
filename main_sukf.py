SAVE_RESULTS_CSV = True
SAVE_RESULTS_CSV_NAME = "results.csv" # Filename for storing summary results.

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

# Adaptive covariance parameters for sigma_p
parser.add_argument("--beta_p", type=float, default=1, help="Weight for the inverse entropy metric in adaptive sigma_p calculation.")
parser.add_argument("--epsilon_p", type=float, default=1, help="Weight for the pose chi-squared metric in adaptive sigma_p calculation.")
parser.add_argument("--zeta_p", type=float, default=1, help="Weight for the culled keyframes metric in adaptive sigma_p calculation.")

# Normalization minimum values for sigma_p components. Increasing these values shifts the normalization curve downwards.
parser.add_argument("--entropy_norm_min", type=float, default=0, help="Minimum normalization threshold for the entropy metric in sigma_p.")
parser.add_argument("--pose_chi2_norm_min", type=float, default=1, help="Minimum normalization threshold for the pose chi-squared metric in sigma_p.")
parser.add_argument("--culled_norm_min", type=float, default=0, help="Minimum normalization threshold for the culled keyframes metric in sigma_p.")

# Adaptive covariance parameters for sigma_v
parser.add_argument("--alpha_v", type=float, default=5, help="Weight for the intensity difference metric in adaptive sigma_v calculation.")
parser.add_argument("--epsilon_v", type=float, default=2, help="Weight for the pose chi-squared difference metric in adaptive sigma_v calculation.")
parser.add_argument("--zeta_H", type=float, default=1, help="Weight for an increase in culled keyframes in adaptive sigma_v calculation.")
parser.add_argument("--zeta_L", type=float, default=0, help="Weight for a decrease in culled keyframes in adaptive sigma_v calculation.")

parser.add_argument("--w_thr", type=float, default=0.25, help="Weight threshold for image confidence, used in adaptive covariance scaling.")
parser.add_argument("--d_thr", type=float, default=0.99, help="Depth threshold for image confidence, used in adaptive covariance scaling.")
parser.add_argument("--s", type=float, default=3.0, help="Parameter 's' for the CASEF activation function, controlling its steepness.")
parser.add_argument("--adaptive", action="store_true", default=False, help="Enable adaptive covariance estimation for visual measurements.")

# Zero-velocity update (ZUPT) parameters
parser.add_argument("--zupt_acc_thr", type=float, default=0.1, help="Standard deviation threshold of accelerometer readings for ZUPT detection [m/s²].")
parser.add_argument("--zupt_gyro_thr", type=float, default=0.1, help="Standard deviation threshold of gyroscope readings for ZUPT detection [rad/s].")
parser.add_argument("--zupt_win", type=int, default=60, help="Sliding window size (number of samples) for ZUPT static detection.")

args = parser.parse_args()

# =========================================================
# Aktivasyon Fonksiyonları
# =========================================================
def casef(x, s):
    """
    Clipped Adaptive Saturation Exponential Function (CASEF).
    This function provides a tunable non-linear mapping, crucial for scaling adaptive covariance metrics.
    It returns 0 for negative inputs, scales exponentially within the [0, 1] range,
    and saturates at 1 for inputs greater than or equal to 1.
    The parameter 's' controls the steepness of the exponential curve.
    """
    x_clip = np.clip(x, 0.0, 1.0)

    if np.isclose(s, 0.0):
        return x_clip # Linear behavior for s near 0
    MAX_SAFE_EXP_ARG = 100 
    MIN_SAFE_EXP_ARG = -100

    # Handle cases where 's' is extremely large or small to prevent numerical issues.
    if s > MAX_SAFE_EXP_ARG:
        return 1.0 if np.isclose(x_clip, 1.0) else 0.0

    if s < MIN_SAFE_EXP_ARG:
        return 0.0 if np.isclose(x_clip, 0.0) else 1.0

    exp_s = np.exp(s)
    numerator = np.exp(s * x_clip) - 1.0
    denominator = exp_s - 1.0

    # Fallback to linear behavior if the denominator is near zero (s was extremely close to 0).
    if np.isclose(denominator, 0.0):
        return x_clip # Fallback to linear behavior

    return numerator / denominator

activation_func = lambda x: casef(x, args.s)

# Configuration dictionary aggregating parameters for adaptive covariance and other modules.
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
# Yardımcı Fonksiyonlar (Quaternion İşlemleri vb.)
# =========================================================
def skew(vector):
    """
    Computes the skew-symmetric matrix of a 3D vector.
    This operation is fundamental in representing cross-product operations as matrix multiplications, e.g., in ESKF Jacobians.
    """
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def convert_quaternion_order(q_wxyz):
    """
    Converts a quaternion from [w, x, y, z] order to [x, y, z, w] order.
    This conversion is primarily for compatibility with libraries like scipy.spatial.transform.Rotation.
    """
    q_wxyz = np.array(q_wxyz)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

def quaternion_multiply(q1, q2):
    """
    Performs quaternion multiplication (Hamilton product) of two quaternions.
    Inputs q1 and q2 are assumed to be in [w, x, y, z] (scalar-first) order.
    """
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """
    Computes the conjugate of a quaternion.
    Input q is assumed to be in [w, x, y, z] (scalar-first) order.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def delta_quaternion(delta_theta):
    """
    Constructs a small rotation quaternion from a 3D small angle error vector (delta_theta).
    This is a first-order approximation used for injecting angular errors into the nominal quaternion state.
    """
    return np.concatenate([[1.0], 0.5 * delta_theta])


def quaternion_to_euler(q):
    """
    Converts a quaternion to Euler angles (XYZ convention, degrees).
    Input q is assumed to be in [w, x, y, z] (scalar-first) order.
    """
    q_xyzw = [q[1], q[2], q[3], q[0]]
    r = R.from_quat(q_xyzw)
    return r.as_euler('xyz', degrees=True)

def ensure_quaternion_continuity(q, q_prev):
    """Ensures quaternion continuity by addressing the double-cover property."""
    if np.dot(q, q_prev) < 0:
        return -q
    return q

def align_quaternions(estimated_quaternions, true_quaternions):
    min_length = min(len(estimated_quaternions), len(true_quaternions))
    aligned_quaternions = np.copy(estimated_quaternions[:min_length])
    """
    Aligns a sequence of estimated quaternions with a sequence of true quaternions
    to maintain continuity and minimize angular errors for consistent error evaluation.
    """
    for i in range(min_length):
        if np.dot(estimated_quaternions[i], true_quaternions[i]) < 0:
            aligned_quaternions[i] = -aligned_quaternions[i]
    return aligned_quaternions

def ensure_angle_continuity(angles, threshold=180):
    """
    Unwraps angles to ensure continuity, preventing large jumps due to angle wrapping.
    Crucial for accurate plotting and RMSE calculation of angular trajectories.
    """
    return np.unwrap(angles, axis=0, period=2 * threshold)

def angle_difference(angle1, angle2):
    """
    Calculates the shortest angular difference between two angles in degrees,
    correctly handling the 360-degree wrap-around.
    """
    diff = angle1 - angle2
    return (diff + 180) % 360 - 180

def calculate_angle_rmse(predictions, targets):
    """Calculates Root Mean Squared Error (RMSE) for angular data (degrees), robust to wrapping."""
    diff = np.array([angle_difference(p, t) for p, t in zip(predictions, targets)])
    return np.sqrt((diff ** 2).mean(axis=0))

# =========================================================
# ESKF Sınıfı (Bias Tahmini ile) - Original Base Class
# =========================================================
class ErrorStateKalmanFilterVIO:
    """
    Implements a standard Error-State Kalman Filter (ESKF) for Visual-Inertial Odometry (VIO).
    The filter estimates orientation (quaternion), velocity, position, and IMU biases.
    This class serves as a base for more advanced filter implementations like UKF.
    """
    def __init__(self, initial_quaternion, initial_velocity, initial_position,
                 initial_accel_bias=np.zeros(3), initial_gyro_bias=np.zeros(3),
                 gravity_vector=np.array([0, 0, -9.81]),
                 tau_a=300.0, tau_g=1000.0, sigma_wa=2e-5, sigma_wg=2e-5, 
                 sigma_g=2e-4, sigma_a=2e-1):

        # Nominal Durum
        self.q = initial_quaternion / np.linalg.norm(initial_quaternion) # Ensure initial norm
        self.v = initial_velocity
        self.p = initial_position
        self.b_a = initial_accel_bias
        self.b_g = initial_gyro_bias
        self.g = gravity_vector

        # Hata Durumu Kovaryansı (15x15)
        # Error state vector: [delta_theta, delta_v, delta_p, delta_b_a, delta_b_g]
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

        # IMU Noise Parameters and Bias Correlation Times
        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        self.sigma_wa = sigma_wa 
        self.sigma_wg = sigma_wg 
        self.tau_a = tau_a
        self.tau_g = tau_g

        # Continuous-Time Process Noise Covariance (Q_c): 15x15 matrix for error state dynamics
        Qc = np.zeros((15,15))
        Qc[0:3,  0:3]   = (self.sigma_g**2) * np.eye(3)  # Gyroscope white noise contribution
        Qc[3:6,  3:6]   = (self.sigma_a**2) * np.eye(3)  # Accelerometer white noise contribution
        Qc[9:12, 9:12]  = (self.sigma_wa**2) * np.eye(3) # Accelerometer bias random walk contribution
        Qc[12:15,12:15] = (self.sigma_wg**2) * np.eye(3) # Gyroscope bias random walk contribution
        self.Q_c = Qc

        # Measurement Noise Covariances (R)
        self.R_zupt = np.diag([1e-3, 1e-3, 1e-3]) # Covariance for Zero-Velocity Update (ZUPT)
        fixed_sigma_p = 1e0 
        fixed_sigma_v = 1e-2
        self.R_vis_6d = np.diag([fixed_sigma_p**2, fixed_sigma_p**2, fixed_sigma_p**2,
                                 fixed_sigma_v**2, fixed_sigma_v**2, fixed_sigma_v**2]) # Covariance for visual position/velocity measurements

    def _compute_van_loan_matrices(self, R_nav_from_body, accel_corrected, gyro_corrected, dt):
        """
        Computes the discrete-time state transition matrix (Phi) and discrete-time
        process noise covariance matrix (Qd) for the error state dynamics using the Van Loan method.
        This method ensures consistency between state propagation and covariance propagation.
        """
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # Construct the continuous-time error state dynamics matrix A (15x15)
        # A represents the Jacobian of the continuous-time error state dynamics.
        A = np.zeros((15, 15))
        A[0:3, 0:3] = -skew(gyro_corrected)    # d(delta_theta)/dt
        A[0:3, 12:15] = -I3                    # d(delta_theta)/dt bağımlılığı delta_b_g'ye
        A[3:6, 0:3] = -R_nav_from_body @ skew(accel_corrected) # d(delta_v)/dt bağımlılığı delta_theta'ya
        A[3:6, 9:12] = -R_nav_from_body        # d(delta_v)/dt bağımlılığı delta_b_a'ya
        A[6:9, 3:6] = I3                       # d(delta_p)/dt bağımlılığı delta_v'ye
        A[9:12, 9:12] = -1/self.tau_a * I3     # d(delta_b_a)/dt (Gauss-Markov)
        A[12:15, 12:15] = -1/self.tau_g * I3   # d(delta_b_g)/dt (Gauss-Markov)


        # Form the augmented matrix M for the Van Loan method (30x30)
        # M = [[ A*dt, Qc*dt ],
        #      [  0,  -A.T*dt ]]
        M = np.zeros((30, 30))
        M[0:15, 0:15] = A * dt
        M[0:15, 15:30] = self.Q_c * dt
        M[15:30, 15:30] = -A.T * dt

        # Compute the matrix exponential of M
        M_exp = expm(M)

        # Extract the discrete state transition matrix (Phi) and discrete process noise covariance (Qd)
        # from the partitioned matrix exponential of M.
        Phi = M_exp[0:15, 0:15] # expm(A*dt)
        Qd = M_exp[0:15, 15:30] @ Phi.T 
        # Ensure Qd is symmetric for numerical stability.
        Qd = 0.5 * (Qd + Qd.T)

        return Phi, Qd

    def predict(self, gyro_raw, accel_raw, dt):
        """
        Performs the prediction step of the ESKF, propagating nominal state and error covariance.
        """
        if dt <= 0:
            return
        q = self.q; v = self.v; p = self.p; b_a = self.b_a; b_g = self.b_g

        # Biasları düzelt
        accel_corrected = accel_raw - b_a
        gyro_corrected = gyro_raw - b_g

        # --- Nominal State Propagation ---
        # Quaternion update using Euler integration
        dq_dt = 0.5 * np.array([
            -q[1]*gyro_corrected[0] - q[2]*gyro_corrected[1] - q[3]*gyro_corrected[2],
             q[0]*gyro_corrected[0] + q[2]*gyro_corrected[2] - q[3]*gyro_corrected[1],
             q[0]*gyro_corrected[1] - q[1]*gyro_corrected[2] + q[3]*gyro_corrected[0],
             q[0]*gyro_corrected[2] + q[1]*gyro_corrected[1] - q[2]*gyro_corrected[0]
        ])
        q_new = q + dq_dt * dt
        q_new /= np.linalg.norm(q_new) # Normalizasyon
        # Compute rotation matrix from the newly predicted quaternion
        q_xyzw = convert_quaternion_order(q_new)
        R_nav_from_body = R.from_quat(q_xyzw).as_matrix()

        # Velocity and position update using IMU measurements and gravity
        a_nav = R_nav_from_body @ accel_corrected + self.g
        v_new = v + a_nav * dt
        p_new = p + v * dt + 0.5 * a_nav * dt**2 

        # Biases are modeled as random walks and remain constant during nominal state propagation
        b_a_new = b_a
        b_g_new = b_g

        # --- Error State Covariance Propagation (using Van Loan method) ---
        # Compute discrete state transition matrix (Phi) and process noise covariance (Qd)
        Phi, Qd = self._compute_van_loan_matrices(R_nav_from_body, accel_corrected, gyro_corrected, dt)
        # Propagate the error state covariance: P_new = Phi * P * Phi^T + Qd
        P_new = Phi @ self.P @ Phi.T + Qd
        # Ensure P remains symmetric
        self.P = 0.5 * (P_new + P_new.T)

        # Nominal durumu güncelle
        self.q = q_new
        self.v = v_new
        self.p = p_new
        self.b_a = b_a_new
        self.b_g = b_g_new

    def _update_common(self, delta_x):
        """
        Applies the error state correction (delta_x) to the nominal state variables.
        This retraction operation updates the quaternion via multiplication and other states via addition.
        """
        
        dq = delta_quaternion(delta_x[0:3])
        self.q = quaternion_multiply(self.q, dq)
        self.q /= np.linalg.norm(self.q) 

        self.v += delta_x[3:6]
        self.p += delta_x[6:9]
        self.b_a += delta_x[9:12]
        self.b_g += delta_x[12:15]

    def zero_velocity_update(self):
        """
        Performs a Zero-Velocity Update (ZUPT) to correct the filter state
        when the IMU is detected to be stationary.
        """
        q_xyzw = convert_quaternion_order(self.q)
        R_nav_from_body = R.from_quat(q_xyzw).as_matrix()
        R_body_from_nav = R_nav_from_body.T

        # Ölçüm: Navigasyon çerçevesindeki hızın body çerçevesindeki ifadesi
        v_body_predicted = R_body_from_nav @ self.v
        y = -v_body_predicted # Ölçüm artığı (residual)

        # Construct the Jacobian matrix H for the ZUPT measurement
        # H relates the error state to the measurement residual.
        H = np.zeros((3, 15))
        H[:, 0:3] = -skew(R_body_from_nav @ self.v) 
        H[:, 3:6] = R_body_from_nav            

        # Kalman güncelleme adımları
        S = H @ self.P @ H.T + self.R_zupt
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: ZUPT innovation covariance matrix S is singular. Skipping ZUPT update.")
            return
        K = self.P @ H.T @ S_inv # Kalman kazancı (15x3)

        delta_x = K @ y # Hata durumu düzeltmesi (15x1)

        # Reset velocity to zero as per ZUPT assumption.
        self.v = np.zeros(3)
        # Correct other states based on the error state correction.
        # Note: The H matrix structure implies that delta_x[0:3] (orientation error)
        # is not directly corrected by this specific measurement update, but other parts of delta_x are.
        # The quaternion itself is not directly reset/updated here beyond the _update_common if it were called.
        # However, the standard ESKF ZUPT often directly resets velocity and corrects biases/position.
        # Here, we apply corrections based on delta_x for consistency with a general update.
        # For a strict ZUPT, one might only correct biases and position if velocity is reset.
        # The current delta_x will have components for orientation error if K maps y to it.
        # Let's apply the parts of delta_x that are relevant given H.
        # self.q is not updated via delta_x[0:3] here, as H[0:3] is non-zero.
        # self.p, self.b_a, self.b_g are updated.
        self.p += delta_x[6:9]
        self.b_a += delta_x[9:12]
        self.b_g += delta_x[12:15]
        
        I15 = np.eye(15)
        P_new = (I15 - K @ H) @ self.P @ (I15 - K @ H).T + K @ self.R_zupt @ K.T
        self.P = 0.5 * (P_new + P_new.T) # Ensure symmetry

    def vision_posvel_update(self, p_meas, v_meas, R_vision):
        """Updates the filter state using visual position and velocity measurements."""
        y = np.concatenate([p_meas - self.p, v_meas - self.v]) # Ölçüm artığı (6x1)
        H = np.zeros((6, 15))
        H[0:3, 6:9] = np.eye(3)  # H_p: Pozisyon hatasının ölçüme etkisi
        H[3:6, 3:6] = np.eye(3)  # H_v: Hız hatasının ölçüme etkisi

        # Kalman güncelleme adımları
        S = H @ self.P @ H.T + R_vision # Güncellenmiş R_vision kullanılır

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Vision innovation covariance matrix S is singular. Skipping vision update.")
            return
        K = self.P @ H.T @ S_inv # Kalman kazancı (15x6)

        delta_x = K @ y # Hata durumu düzeltmesi (15x1)

        # Apply the error state correction and update covariance
        self._update_common(delta_x)
        I15 = np.eye(15)
        P_new = (I15 - K @ H) @ self.P @ (I15 - K @ H).T + K @ R_vision @ K.T
        self.P = 0.5 * (P_new + P_new.T) # Ensure symmetry

    def gravity_update(self, accel_raw): # Takes accel_raw only now
        """
        Updates the accelerometer bias using the gravity vector as a measurement
        when the system is detected to be stationary.
        """
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
        H[:, 0:3] = skew(R_body_from_nav @ self.g)
        # Jacobian w.r.t delta_b_a
        H[:, 9:12] = -np.eye(3)
        
        # Measurement noise covariance for accelerometer
        R_acc = (self.sigma_a**2) * np.eye(3)

        # Kalman güncellemesi
        S  = H @ self.P @ H.T + R_acc
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Gravity Update innovation covariance matrix S is singular. Skipping gravity update.")
            return
        K  = self.P @ H.T @ S_inv
        dx = K @ y
        
        # Apply the error state correction to the nominal state variables.
        # Note: The H matrix structure implies that delta_x[3:6] (velocity error),
        # delta_x[6:9] (position error), and delta_x[12:15] (gyro bias error)
        # are not directly corrected by this measurement.
        # The quaternion is corrected via dx[0:3] through _update_common if it were called.
        # Here, we apply corrections based on dx for consistency.
        # self.q is not updated via dx[0:3] here.
        self.v += dx[3:6]
        self.p += dx[6:9]
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]

        I15 = np.eye(15)
        P_new = (I15 - K @ H) @ self.P @ (I15 - K @ H).T + K @ R_acc @ K.T
        self.P = 0.5 * (P_new + P_new.T) # Ensure symmetry

# =========================================================
# UKF-ESKF Sınıfı (Additive UKF + SUT)
# =========================================================
class ErrorStateKalmanFilterVIO_UKF(ErrorStateKalmanFilterVIO):
    """
    Implements an Error-State Kalman Filter for VIO using an Unscented Kalman Filter (UKF)
    for the error-state propagation and update steps. This approach aims to better handle
    non-linearities compared to the standard ESKF's linearization.
    """
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
        self.n = 15                # Error-state dimension: [delta_theta, delta_v, delta_p, delta_b_a, delta_b_g]
        self.α = 0.5               # SUT scaling parameter alpha (controls spread of sigma points)
        self.β = 2.0               # SUT scaling parameter beta (incorporates prior knowledge of distribution, 2.0 is optimal for Gaussian)
        self.κ = 3 - self.n        # SUT scaling parameter kappa (secondary scaling, 3-n often recommended)
        self.λ = self.α**2 * (self.n + self.κ) - self.n # Composite scaling parameter lambda
        
        # Sigma point weights (Merwe's Scaled Unscented Transform)
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.λ))) # Weights for mean
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.λ))) # Weights for covariance
        self.Wm[0] = self.λ / (self.n + self.λ)                         # Weight for mean (center point)
        self.Wc[0] = self.Wm[0] + (1 - self.α**2 + self.β)             # Weight for covariance (center point)

    def _sigma_points(self, x_mean, P):
        """
        Generates sigma points for the Unscented Transform based on the current error-state
        mean (typically zero) and covariance P.
        """
        num_states = P.shape[0] # Should be self.n
        if num_states != self.n:
             print(f"Warning: P matrix size {P.shape} does not match state size {self.n} in _sigma_points")
             # This indicates a potential inconsistency in state dimension management.


        # Ensure P is positive semi-definite before Cholesky
        P = 0.5 * (P + P.T) # Ensure symmetry
        # Add small diagonal jitter for numerical stability
        P_stable = P + np.eye(num_states) * 1e-9 

        try:
            # Use Cholesky decomposition.
            S = np.linalg.cholesky((num_states + self.λ) * P_stable)
        except np.linalg.LinAlgError:
            print(f"Error: Cholesky decomposition failed in _sigma_points even after jitter.")
            # Fallback to a degenerate set of sigma points (all at the mean)
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
        """
        Injects an error state (delta_x) into a given nominal state (q_nom, v_nom, etc.)
        to produce a perturbed absolute state. This is used for sigma point propagation.
        """
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
        """
        Retracts an absolute state (q_inj, v_inj, etc.) back to an error state (delta_x)
        relative to a given nominal state (q_nom, v_nom, etc.). This is the inverse of injection
        and is used after propagating sigma points.
        """
        # Ensure nominal quaternion is normalized
        q_nom_norm = q_nom / np.linalg.norm(q_nom)
        q_inj_norm = q_inj / np.linalg.norm(q_inj)

        delta_q = quaternion_multiply(quaternion_conjugate(q_nom_norm), q_inj_norm)
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
        """
        Propagates a given nominal state (q, v, p, b_a, b_g) forward in time by dt
        using IMU measurements (gyro_raw, accel_raw).
        """
        if dt <= 0:
            # Return current state if dt is invalid
            return q, v, p, b_a, b_g 

        # Correct biases
        accel_corrected = accel_raw - b_a
        gyro_corrected = gyro_raw - b_g

        # Quaternion update using first-order Euler integration
        dq_dt = 0.5 * np.array([
            -q[1]*gyro_corrected[0] - q[2]*gyro_corrected[1] - q[3]*gyro_corrected[2],
             q[0]*gyro_corrected[0] + q[2]*gyro_corrected[2] - q[3]*gyro_corrected[1],
             q[0]*gyro_corrected[1] - q[1]*gyro_corrected[2] + q[3]*gyro_corrected[0],
             q[0]*gyro_corrected[2] + q[1]*gyro_corrected[1] - q[2]*gyro_corrected[0]
        ])
        q_new = q + dq_dt * dt
        q_new /= np.linalg.norm(q_new) # Normalize

        # Rotation matrix from the *new* quaternion
        q_xyzw = convert_quaternion_order(q_new)
        R_nav_from_body = R.from_quat(q_xyzw).as_matrix()

        # Velocity and Position update (using trapezoidal integration for better accuracy)
        a_nav_start = R.from_quat(convert_quaternion_order(q)).as_matrix() @ accel_corrected + self.g
        a_nav_end = R_nav_from_body @ accel_corrected + self.g
        v_new = v + 0.5 * (a_nav_start + a_nav_end) * dt # Trapezoidal rule for velocity
        p_new = p + v * dt + 0.25 * (a_nav_start + a_nav_end) * dt**2 # More accurate position update

        # Biases remain constant during propagation in this model
        b_a_new = b_a
        b_g_new = b_g

        return q_new, v_new, p_new, b_a_new, b_g_new

    def predict(self, gyro_raw, accel_raw, dt):
        """
        Predicts the next state using the Unscented Kalman Filter for the error state.
        This involves propagating the nominal state and then propagating the error state
        sigma points through the system dynamics.
        """
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
        self.b_a, self.b_g = ba_new, bg_new # Biases are assumed constant during this propagation step

        # --- (b) UKF Error State Covariance Propagation ---
        # Calculate necessary components for Qd based on OLD nominal state
        accel_corrected_old = accel_raw - ba_nom_old
        gyro_corrected_old = gyro_raw - bg_nom_old
        q_xyzw_old = convert_quaternion_order(q_nom_old)
        R_nav_from_body_old = R.from_quat(q_xyzw_old).as_matrix()

        # Get discrete process noise Qd using Van Loan on OLD state.
        # This Qd represents the discretized process noise for the error state dynamics.
        _, Qd = self._compute_van_loan_matrices(R_nav_from_body_old, accel_corrected_old, gyro_corrected_old, dt)
        
        # 1. Generate sigma points for the error state (around zero mean)
        # Use current covariance P before prediction
        chi = self._sigma_points(np.zeros(self.n), self.P)
        chi_pred = np.zeros_like(chi)

        # 2. Propagate sigma points through the non-linear dynamics
        for i, δx in enumerate(chi):
            # Inject error sigma point into the *old* nominal state
            q_inj, v_inj, p_inj, ba_inj, bg_inj = self._inject_explicit(q_nom_old, v_nom_old, p_nom_old, ba_nom_old, bg_nom_old, δx)

            # Propagate the *injected* state one step
            q_i_pred, v_i_pred, p_i_pred, ba_i_pred, bg_i_pred = self._propagate_nominal(
                q_inj, v_inj, p_inj, ba_inj, bg_inj,
                gyro_raw, accel_raw, dt
            )

            # Retract the propagated injected state back to an error state relative to the *newly propagated* nominal state
            chi_pred[i] = self._retract_explicit(self.q, self.v, self.p, self.b_a, self.b_g,
                                                 q_i_pred, v_i_pred, p_i_pred, ba_i_pred, bg_i_pred)
        # 3. Calculate the predicted error state mean and covariance
        # The predicted error state mean should be close to zero.

        δx_pred_mean = np.sum(self.Wm[:, None] * chi_pred, axis=0)
        delta_chi = chi_pred - δx_pred_mean # Shape (2n+1, n)
        P_pred = Qd + np.sum(self.Wc[:, None, None] *
                             delta_chi[..., None] @ delta_chi[:, None, :], axis=0)

        # Ensure P remains symmetric
        self.P = 0.5 * (P_pred + P_pred.T)

    def ukf_update_posvel(self, z_meas, R_meas):
        """
        Updates the filter state using visual position and velocity measurements (z_meas)
        via the Unscented Kalman Filter update mechanism. R_meas is the measurement noise
        covariance.
        """
        # z_meas: 6x1 vector [p_meas, v_meas]
        # R_meas: 6x6 measurement noise covariance matrix

        # 1. Generate sigma points for the current error state (mean=0, cov=P)
        chi = self._sigma_points(np.zeros(self.n), self.P)
        
        # 2. Propagate sigma points through the measurement function h(x)
        #    h(x) extracts position and velocity from the full state obtained by injecting dx
        #    The measurement function h(x) effectively extracts position and velocity
        #    from the full state obtained by injecting the error sigma point dx.
        Zsig = np.zeros((2 * self.n + 1, 6)) # Transformed sigma points in measurement space
        for i, δx in enumerate(chi):
            # Inject error into current nominal state (self.q, self.v, ...)
            q_i, v_i, p_i, ba_i, bg_i = self._inject_explicit(
                self.q, self.v, self.p, self.b_a, self.b_g, δx
            )
            # The measurement is simply the position and velocity of the injected state
            Zsig[i, :3] = p_i
            Zsig[i, 3:] = v_i

        # 3. Calculate predicted measurement mean
        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)

        # 4. Calculate innovation covariance Pzz and cross-covariance Pxz
        delta_Z = Zsig - z_pred # Shape (2n+1, 6)
        # Error sigma points are already relative to zero mean for the error state
        delta_chi = chi # Shape (2n+1, n)

        Pzz = R_meas + np.sum(self.Wc[:, None, None] *
                              delta_Z[..., None] @ delta_Z[:, None, :], axis=0)

        Pxz = np.sum(self.Wc[:, None, None] *
                     delta_chi[..., None] @ delta_Z[:, None, :], axis=0)

        # 5. Calculate Kalman Gain K
        try:
            # Use pseudo-inverse for robustness against potentially ill-conditioned Pzz
            Pzz_inv = np.linalg.pinv(Pzz) 
        except np.linalg.LinAlgError:
            print("Warning: UKF Vision Pzz matrix is singular! Skipping update.")
            return
        
        K = Pxz @ Pzz_inv # Shape (n, 6)

        # 6. Calculate corrected error state
        residual = z_meas - z_pred
        δx_corr = K @ residual # Shape (n,)

        # 7. Update state covariance P
        P_updated = self.P - K @ Pzz @ K.T
        # Ensure P remains symmetric and positive semi-definite
        self.P = 0.5 * (P_updated + P_updated.T) 

        # 8. Update nominal state using the common update function from base class
        self._update_common(δx_corr)

    def zero_velocity_update(self): # Override for UKF
        """
        Performs a Zero Velocity Update (ZUPT) using the UKF mechanism.
        The measurement is zero velocity in the body frame, and the measurement function
        involves the orientation (non-linear).
        """
        # Measurement is zero velocity in the body frame.
        z_meas = np.zeros(3)
        R_meas = self.R_zupt # Use pre-defined ZUPT noise

        # 1. Generate sigma points
        chi = self._sigma_points(np.zeros(self.n), self.P)
        
        # 2. Propagate sigma points through measurement function h(x) = R_body_from_nav(q) @ v
        Zsig = np.zeros((2 * self.n + 1, 3)) # Transformed sigma points in measurement space (body velocity)
        for i, δx in enumerate(chi):
            # Inject error into current nominal state
            q_i, v_i, p_i, ba_i, bg_i = self._inject_explicit(
                self.q, self.v, self.p, self.b_a, self.b_g, δx
            )
            # Calculate R_body_from_nav for this sigma point's orientation
            q_xyzw_i = convert_quaternion_order(q_i)
            R_nav_from_body_i = R.from_quat(q_xyzw_i).as_matrix()
            R_body_from_nav_i = R_nav_from_body_i.T
            # Measurement function: velocity in body frame
            Zsig[i, :] = R_body_from_nav_i @ v_i

        # 3. Calculate predicted measurement mean
        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)

        # 4. Calculate innovation covariance Pzz and cross-covariance Pxz
        delta_Z = Zsig - z_pred # Shape (2n+1, 3)
        delta_chi = chi # Shape (2n+1, n), error sigma points relative to zero mean

        Pzz = R_meas + np.sum(self.Wc[:, None, None] *
                              delta_Z[..., None] @ delta_Z[:, None, :], axis=0)

        Pxz = np.sum(self.Wc[:, None, None] *
                     delta_chi[..., None] @ delta_Z[:, None, :], axis=0)

        # 5. Calculate Kalman Gain K
        try:
            Pzz_inv = np.linalg.pinv(Pzz)
        except np.linalg.LinAlgError:
            print("Warning: UKF ZUPT Pzz matrix is singular! Skipping update.")
            return
        
        K = Pxz @ Pzz_inv # Shape (n, 3)

        # 6. Calculate corrected error state
        residual = z_meas - z_pred # Residual: 0 - predicted body velocity
        δx_corr = K @ residual # Shape (n,)

        # 7. Update state covariance P
        P_updated = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P_updated + P_updated.T)

        # 8. Update nominal state
        self._update_common(δx_corr)

    def gravity_update(self, accel_raw): # Override for UKF, takes accel_raw
        """
        Updates accelerometer bias using gravity measurement via UKF during static periods.
        The measurement is the raw accelerometer reading. The measurement function involves
        transforming the gravity vector into the body frame using the orientation,
        making it non-linear.
        """
        # Measurement is the raw accelerometer reading
        z_meas = accel_raw
        # Measurement noise is accelerometer white noise
        R_meas = (self.sigma_a**2) * np.eye(3)

        # 1. Generate sigma points
        chi = self._sigma_points(np.zeros(self.n), self.P)

        # 2. Propagate sigma points through measurement function
        # h(x) = R_body_from_nav(q) @ (-g_nav) + b_a
        Zsig = np.zeros((2 * self.n + 1, 3)) # Transformed sigma points (expected accelerometer reading)
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
            # Measurement function: expected accel reading = g_body + bias
            Zsig[i, :] = g_body_i + ba_i

        # 3. Calculate predicted measurement mean (expected accel reading)
        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)

        # 4. Calculate innovation covariance Pzz and cross-covariance Pxz
        delta_Z = Zsig - z_pred # Shape (2n+1, 3)
        delta_chi = chi # Shape (2n+1, n), error sigma points relative to zero mean

        Pzz = R_meas + np.sum(self.Wc[:, None, None] *
                              delta_Z[..., None] @ delta_Z[:, None, :], axis=0)

        Pxz = np.sum(self.Wc[:, None, None] *
                     delta_chi[..., None] @ delta_Z[:, None, :], axis=0)

        # 5. Calculate Kalman Gain K
        try:
            Pzz_inv = np.linalg.pinv(Pzz)
        except np.linalg.LinAlgError:
            print("Warning: UKF Gravity Pzz matrix is singular! Skipping update.")
            return

        K = Pxz @ Pzz_inv # Shape (n, 3)

        # 6. Calculate corrected error state
        residual = z_meas - z_pred # Residual: actual accel - expected accel
        δx_corr = K @ residual # Shape (n,)

        # 7. Update state covariance P
        P_updated = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P_updated + P_updated.T)

        # 8. Update nominal state
        self._update_common(δx_corr)

# === H I B R I T  F I L T R E  S I N I F I ==========================
class ErrorStateKalmanFilterVIO_Hybrid(ErrorStateKalmanFilterVIO):
    # IMPORTANT NOTE: Despite the class name "ErrorStateKalmanFilterVIO_Hybrid",
    # this implementation in "main_sukf.py" functions as a FULL Unscented Kalman Filter (UKF)
    # for the entire 15-dimensional error state. It overrides the prediction and update
    # methods of the base ESKF class with UKF logic.
    """
    Implements a full Unscented Kalman Filter (UKF) for the 15-dimensional error state
    [delta_theta, delta_v, delta_p, delta_b_a, delta_b_g] in a VIO context.
    This class inherits from a base ESKF but replaces its core filtering mechanisms
    with the UKF approach (sigma point propagation, unscented transform for updates)
    to better handle system non-linearities.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Initialize base ESKF components (state, noise params)

        # --- UKF Parameters for the full 15D error state ---
        self.n = 15                # Full error-state dimension
        self.α = 0.5               # UKF scaling parameter alpha
        self.β = 2.0               # UKF scaling parameter beta (optimal for Gaussian)
        self.κ = 3 - self.n        # UKF scaling parameter kappa
        self.λ = self.α**2 * (self.n + self.κ) - self.n # Composite scaling parameter lambda
        
        # Sigma point weights for mean and covariance (Merwe's Scaled Unscented Transform)
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.λ))) # Weights for mean
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.λ))) # Weights for covariance
        self.Wm[0] = self.λ / (self.n + self.λ)                         # Weight for mean (center point)
        self.Wc[0] = self.Wm[0] + (1 - self.α**2 + self.β)             # Weight for covariance (center point)

    # --- Helper functions needed from UKF (copied or adapted) ---
    def _sigma_points(self, x_mean, P):
        """
        Generates sigma points for the Unscented Transform based on the error-state
        mean (typically zero) and covariance P.
        """
        num_states = P.shape[0] 
        if num_states != self.n:
             print(f"Warning: P matrix size {P.shape} does not match state size {self.n} in _sigma_points")

        P = 0.5 * (P + P.T) 
        P_stable = P + np.eye(num_states) * 1e-9 

        try:
            S = np.linalg.cholesky((num_states + self.λ) * P_stable)
        except np.linalg.LinAlgError:
            print(f"Error: Cholesky decomposition failed in _sigma_points even after jitter.")
            chi = np.zeros((2 * num_states + 1, num_states))
            chi[0] = x_mean
            return chi 

        chi = np.zeros((2 * num_states + 1, num_states))
        chi[0] = x_mean                   
        for i in range(num_states):
            chi[i + 1]        = x_mean + S[:, i] 
            chi[num_states + 1 + i] = x_mean - S[:, i] 
        return chi

    def _inject_explicit(self, q_nom, v_nom, p_nom, ba_nom, bg_nom, delta_x):
        """
        Injects an error state (delta_x) into a given nominal state (q_nom, v_nom, etc.)
        to produce a perturbed absolute state for sigma point propagation.
        """
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

    # ---------------- P R E D I C T (Full UKF Logic) --------------------
    def predict(self, gyro_raw, accel_raw, dt):
        """
        Predicts the next state using the Unscented Kalman Filter for the full error state.
        This involves propagating the nominal state and then propagating the error state
        sigma points through the system dynamics using the unscented transform.
        """
        if dt <= 0:
            return

        # --- (a) Nominal State Propagation ---
        q_nom_old, v_nom_old, p_nom_old = self.q.copy(), self.v.copy(), self.p.copy()
        ba_nom_old, bg_nom_old = self.b_a.copy(), self.b_g.copy()

        q_new, v_new, p_new, ba_new, bg_new = self._propagate_nominal(
            q_nom_old, v_nom_old, p_nom_old, ba_nom_old, bg_nom_old,
            gyro_raw, accel_raw, dt
        )
        self.q, self.v, self.p = q_new, v_new, p_new
        self.b_a, self.b_g = ba_new, bg_new

        # --- (b) UKF Error State Covariance Propagation ---
        # Calculate components for Qd based on the *old* nominal state, consistent with ESKF theory.
        accel_corrected_old = accel_raw - ba_nom_old
        gyro_corrected_old = gyro_raw - bg_nom_old
        q_xyzw_old = convert_quaternion_order(q_nom_old)
        R_nav_from_body_old = R.from_quat(q_xyzw_old).as_matrix()

        # Discrete process noise for the error state dynamics.
        _, Qd = self._compute_van_loan_matrices(R_nav_from_body_old, accel_corrected_old, gyro_corrected_old, dt)

        # 1. Generate sigma points for the current error state (centered around zero mean).
        chi = self._sigma_points(np.zeros(self.n), self.P)
        chi_pred = np.zeros_like(chi)

        for i, δx in enumerate(chi):
            q_inj, v_inj, p_inj, ba_inj, bg_inj = self._inject_explicit(q_nom_old, v_nom_old, p_nom_old, ba_nom_old, bg_nom_old, δx)

            q_i_pred, v_i_pred, p_i_pred, ba_i_pred, bg_i_pred = self._propagate_nominal(
                q_inj, v_inj, p_inj, ba_inj, bg_inj,
                gyro_raw, accel_raw, dt
            )

            # Retract the propagated injected state back to an error state relative to the *newly propagated* nominal state.
            chi_pred[i] = self._retract_explicit(self.q, self.v, self.p, self.b_a, self.b_g,
                                                 q_i_pred, v_i_pred, p_i_pred, ba_i_pred, bg_i_pred)
        # 3. Calculate the predicted error state mean and covariance.
        # The predicted error state mean should ideally be close to zero.
        δx_pred_mean = np.sum(self.Wm[:, None] * chi_pred, axis=0)
        delta_chi = chi_pred - δx_pred_mean 
        P_pred = Qd + np.sum(self.Wc[:, None, None] *
                             delta_chi[..., None] @ delta_chi[:, None, :], axis=0)
        # Ensure P remains symmetric.
        self.P = 0.5 * (P_pred + P_pred.T)

    # _propagate_nominal and _retract_explicit are essential helper functions for the UKF predict step.
    # These versions are consistent with a full UKF implementation.
    # or confirm they are the same as UKF's versions.
    # The existing _propagate_nominal and _retract_explicit in Hybrid seem to match UKF's.

    # --- Retained Helper functions (already copied from UKF in original Hybrid) ---
    def _propagate_nominal(self, q, v, p, b_a, b_g, gyro_raw, accel_raw, dt): # Retained
        if dt <= 0:
            return q, v, p, b_a, b_g
        """
        Propagates a given nominal state (q, v, p, b_a, b_g) forward by dt
        using IMU measurements. This is a helper for sigma point propagation.
        """
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

    def _retract_explicit(self, q_nom, v_nom, p_nom, ba_nom, bg_nom, q_inj, v_inj, p_inj, ba_inj, bg_inj): # Retained
        """
        Retracts an absolute state (q_inj, v_inj, etc.) back to an error state (delta_x)
        relative to a given nominal state (q_nom, v_nom, etc.). Inverse of injection.
        """
        q_nom_norm = q_nom / np.linalg.norm(q_nom)
        q_inj_norm = q_inj / np.linalg.norm(q_inj)
        delta_q = quaternion_multiply(quaternion_conjugate(q_nom_norm), q_inj_norm)
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
    # --- End of copied helper functions ---
    
    # --------- Ölçüm: görsel p,v  (Full UKF Logic) -----------
    def vision_posvel_update(self, p_meas, v_meas, R_vision):
        """
        Updates the filter state using visual position and velocity measurements via the UKF.
        The measurement model extracts position and velocity from the sigma points.
        R_vision is the measurement noise covariance.
        """
        z_meas = np.concatenate([p_meas, v_meas]) # Measurement: [p_meas, v_meas]
        R_meas = R_vision # Measurement noise covariance

        # 1. Generate sigma points for the current error state.
        chi = self._sigma_points(np.zeros(self.n), self.P)
        
        # 2. Propagate sigma points through the measurement function h(x).
        Zsig = np.zeros((2 * self.n + 1, 6)) 
        for i, δx in enumerate(chi):
            q_i, v_i, p_i, ba_i, bg_i = self._inject_explicit(
                self.q, self.v, self.p, self.b_a, self.b_g, δx
            )
            Zsig[i, :3] = p_i
            Zsig[i, 3:] = v_i

        # 3. Calculate predicted measurement mean.
        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)

        # 4. Calculate innovation covariance Pzz and cross-covariance Pxz.
        delta_Z = Zsig - z_pred 
        delta_chi = chi - np.zeros(self.n) 
        
        Pzz = R_meas + np.sum(self.Wc[:, None, None] *
                              delta_Z[..., None] @ delta_Z[:, None, :], axis=0)

        Pxz = np.sum(self.Wc[:, None, None] *
                     delta_chi[..., None] @ delta_Z[:, None, :], axis=0)

        try:
            # 5. Calculate Kalman Gain K.
            Pzz_inv = np.linalg.pinv(Pzz) 
        except np.linalg.LinAlgError:
            print("Warning: UKF Vision Pzz matrix is singular! Skipping update.")
            return
        
        K = Pxz @ Pzz_inv 

        # 6. Calculate corrected error state.
        residual = z_meas - z_pred
        δx_corr = K @ residual 

        # 7. Update state covariance P.
        P_updated = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P_updated + P_updated.T) 

        # 8. Update nominal state.
        self._update_common(δx_corr)

    # --------- Ölçüm: ZUPT (Full UKF Logic) -----------------
    def zero_velocity_update(self):
        """
        Performs a Zero Velocity Update (ZUPT) using the UKF mechanism.
        The measurement is zero velocity in the body frame; the measurement function
        involves the orientation, making it non-linear.
        """
        z_meas = np.zeros(3)
        R_meas = self.R_zupt 

        # 1. Generate sigma points.
        chi = self._sigma_points(np.zeros(self.n), self.P)
        
        # 2. Propagate sigma points through measurement function h(x) = R_body_from_nav(q) @ v.
        Zsig = np.zeros((2 * self.n + 1, 3)) 
        for i, δx in enumerate(chi):
            q_i, v_i, p_i, ba_i, bg_i = self._inject_explicit(
                self.q, self.v, self.p, self.b_a, self.b_g, δx
            )
            q_xyzw_i = convert_quaternion_order(q_i) # Convert to [x,y,z,w] for SciPy
            R_nav_from_body_i = R.from_quat(q_xyzw_i).as_matrix()
            R_body_from_nav_i = R_nav_from_body_i.T
            Zsig[i, :] = R_body_from_nav_i @ v_i

        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)

        delta_Z = Zsig - z_pred 
        delta_chi = chi # Error sigma points are already relative to zero mean.

        # 4. Calculate innovation covariance Pzz and cross-covariance Pxz.
        Pzz = R_meas + np.sum(self.Wc[:, None, None] *
                              delta_Z[..., None] @ delta_Z[:, None, :], axis=0)

        Pxz = np.sum(self.Wc[:, None, None] *
                     delta_chi[..., None] @ delta_Z[:, None, :], axis=0)

        try:
            # 5. Calculate Kalman Gain K.
            Pzz_inv = np.linalg.pinv(Pzz)
        except np.linalg.LinAlgError:
            print("Warning: UKF ZUPT Pzz matrix is singular! Skipping update.")
            return
        
        K = Pxz @ Pzz_inv 
        residual = z_meas - z_pred 
        # 6. Calculate corrected error state.
        δx_corr = K @ residual 

        # 7. Update state covariance P.
        P_updated = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P_updated + P_updated.T)

        # 8. Update nominal state.
        self._update_common(δx_corr)

    # --------- Ölçüm: Gravity Update (Full UKF Logic) -----------
    def gravity_update(self, accel_raw): 
        """
        Updates accelerometer bias using gravity measurement via UKF during static periods.
        The measurement is the raw accelerometer reading. The measurement function,
        h(x) = R_body_from_nav(q) @ (-g_nav) + b_a, is non-linear due to orientation.
        """
        z_meas = accel_raw
        R_meas = (self.sigma_a**2) * np.eye(3)

        # 1. Generate sigma points.
        chi = self._sigma_points(np.zeros(self.n), self.P)

        # 2. Propagate sigma points through measurement function.
        Zsig = np.zeros((2 * self.n + 1, 3)) 
        for i, δx in enumerate(chi):
            q_i, v_i, p_i, ba_i, bg_i = self._inject_explicit(
                self.q, self.v, self.p, self.b_a, self.b_g, δx
            )
            q_xyzw_i = convert_quaternion_order(q_i) # Convert for SciPy
            R_nav_from_body_i = R.from_quat(q_xyzw_i).as_matrix()
            R_body_from_nav_i = R_nav_from_body_i.T
            g_body_i = R_body_from_nav_i @ (-self.g)
            Zsig[i, :] = g_body_i + ba_i

        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)
        delta_Z = Zsig - z_pred 
        delta_chi = chi # Error sigma points are relative to zero mean.

        # 4. Calculate innovation covariance Pzz and cross-covariance Pxz.
        Pzz = R_meas + np.sum(self.Wc[:, None, None] *
                              delta_Z[..., None] @ delta_Z[:, None, :], axis=0)
        Pxz = np.sum(self.Wc[:, None, None] *
                     delta_chi[..., None] @ delta_Z[:, None, :], axis=0)

        try:
            Pzz_inv = np.linalg.pinv(Pzz)
            # 5. Calculate Kalman Gain K.
        except np.linalg.LinAlgError:
            print("Warning: UKF Gravity Pzz matrix is singular! Skipping update.")
            return

        K = Pxz @ Pzz_inv 
        residual = z_meas - z_pred 
        δx_corr = K @ residual 
        # 6. Calculate corrected error state.

        # 7. Update state covariance P.
        P_updated = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P_updated + P_updated.T)
        # 8. Update nominal state.
        self._update_common(δx_corr)

# =========================================================
# CSV Sonuç ve Özet Fonksiyonları
# =========================================================
def save_results_to_csv(results, filename):
    """Saves detailed trajectory and bias estimation results to a CSV file."""

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
    """Appends a summary of RMSE results and configuration parameters to a CSV file."""
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
# ESKF Tabanlı VIO Verisi İşleme Fonksiyonu
# =========================================================
def process_vio_data(imu_file, visual_file, sigma_v_map=None, sigma_p_map=None):
    """
    Processes VIO data from IMU and visual measurements using the UKF-ESKF.
    Handles data loading, filter initialization, main processing loop, and RMSE calculation.
    """
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

    visual_data = pd.read_csv(visual_file)
    visual_data['timestamp'] = pd.to_datetime(visual_data['#timestamp [ns]'], unit='ns')
    for c in [' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']:
        if c not in visual_data.columns:
            visual_data[c] = 0.0
        visual_data[c] = pd.to_numeric(visual_data[c], errors='coerce')
    visual_data = visual_data.sort_values(by='timestamp').reset_index(drop=True)

    # Initialize filter with first IMU reading or default values
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

    # Instantiate the UKF-based filter (referred to as "Hybrid" in class name, but functions as full UKF).
    # This is the core of the VIO system, performing sensor fusion.
    eskf = ErrorStateKalmanFilterVIO_Hybrid(initial_quaternion, initial_velocity, initial_position,
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

    # Parameters for Zero-Velocity Update (ZUPT) based static detection
    WIN_LEN_STATIC  = args.zupt_win          
    ACC_STD_THR     = args.zupt_acc_thr      
    GYRO_STD_THR    = args.zupt_gyro_thr     
    accel_window = deque(maxlen=WIN_LEN_STATIC)
    gyro_window  = deque(maxlen=WIN_LEN_STATIC) 
    total_static_duration = 0.0 # Tracks total time ZUPT/Gravity updates are active.

    for i, row in imu_data.iterrows():
        dt = row['dt']
        if dt <= 0: continue # Geçerli dt kontrolü
        
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

        # Perform filter prediction step using IMU data. This invokes the UKF prediction.
        eskf.predict(gyro_raw, accel_raw, dt)
        current_time = row['timestamp']

        # Static detection logic for ZUPT and Gravity updates
        accel_window.append(accel_raw)
        gyro_window.append(gyro_raw)

        if len(accel_window) == WIN_LEN_STATIC:
            # Calculate standard deviation of accelerometer and gyroscope norms over the window
            accel_data_stack = np.vstack(accel_window)
            acc_norm = np.linalg.norm(accel_data_stack, axis=1)
            acc_norm_std = acc_norm.std()

            gyro_data_stack = np.vstack(gyro_window)
            g_norm   = np.linalg.norm(gyro_data_stack, axis=1)
            gyro_norm_std = g_norm.std()


            if (acc_norm_std < ACC_STD_THR) and (gyro_norm_std < GYRO_STD_THR):
                eskf.zero_velocity_update()
                eskf.gravity_update(accel_raw) # Use current raw acceleration for gravity update
                window_duration = WIN_LEN_STATIC * dt # Yaklaşık pencere süresi
                total_static_duration += window_duration

                accel_window.clear()
                gyro_window.clear()

        # Görsel ölçüm güncellemesi (varsa)
        # Process visual measurements if available up to the current IMU time
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

            # Retrieve adaptive sigma_v, if enabled and available for the current visual timestamp
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
                    # Match with a tolerance (e.g., 2ms)
                    if chosen_key is not None and min_diff < 2e6: #(2ms)
                        fixed_sigma_v = sigma_v_map[chosen_key]
            # Retrieve adaptive sigma_p, if enabled and available
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
                    if chosen_key is not None and min_diff < 2e6: # Match with a tolerance (e.g., 2ms)
                        sigma_p_val = sigma_p_map[chosen_key]

            
            # Visual velocity is derived from consecutive position measurements.
            v_meas_to_use = eskf.v # Default to current filter velocity if dt_vis is too small
            dt_vis = 0.0
            if prev_vis_time is not None and prev_vis_pos is not None:
                dt_vis = (t_vis - prev_vis_time).total_seconds()
                if dt_vis > 1e-9: # Avoid division by zero or very small dt
                    v_meas_to_use = (p_meas - prev_vis_pos) / dt_vis
            
            # Construct the measurement noise covariance R_vision for this update
            I3 = np.eye(3)
            current_sigma_p_sq = eskf.R_vis_6d[0,0] # (fixed_sigma_p**2)
            current_sigma_v_sq = eskf.R_vis_6d[3,3] # (fixed_sigma_v**2)

            if sigma_p_val is not None:
                current_sigma_p_sq = sigma_p_val**2
            
            # If adaptive sigma_v is computed, use it.
            if fixed_sigma_v is not None: # fixed_sigma_v is from the adaptive sigma_v computation
                current_sigma_v_sq = fixed_sigma_v**2

            R_vision_current = np.block([
                [current_sigma_p_sq * I3, np.zeros((3,3))],
                [np.zeros((3,3)), current_sigma_v_sq * I3]
            ])

            eskf.vision_posvel_update(p_meas, v_meas_to_use, R_vision_current)
            # Store current visual measurement for next velocity calculation
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

        q_prev = q_est # Store for quaternion continuity check in the next iteration
        # Store ground truth data if available for RMSE calculation
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

    # Convert result lists to NumPy arrays for efficient processing
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

    # --- Root Mean Squared Error (RMSE) Computation for Performance Evaluation ---
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
    
    # Calculate RMSE if ground truth is available and valid
    if min_len > 0: # Ensure there is data to compare
        valid_gt_indices = np.where(
            ~np.isnan(true_quaternions[:min_len]).any(axis=1) &
            ~np.isnan(true_velocities[:min_len]).any(axis=1) &
            ~np.isnan(true_positions[:min_len]).any(axis=1) &
            ~np.isnan(true_accel_biases[:min_len]).any(axis=1) &
            ~np.isnan(true_gyro_biases[:min_len]).any(axis=1)
        )[0]

        if len(valid_gt_indices) > 0: # Ensure there are valid ground truth entries
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
                # Ensure the scalar part (w) is positive and clipped for arccos stability
                cos_half_angle = np.clip(abs(delta_q_val[0]), -1.0, 1.0) 
                angle_rad = 2 * np.arccos(cos_half_angle)
                angles_rad_list.append(angle_rad)
            
            angles_deg_array = np.array(angles_rad_list) * 180.0 / np.pi
            # RMSE of these angular errors (target error is 0 degrees)
            rmse_quat_angular_deg = np.sqrt(np.mean(angles_deg_array**2))

            est_euler_deg_valid = np.array([quaternion_to_euler(q) for q in aligned_quaternions_for_angular_rmse])
            true_euler_deg_valid = np.array([quaternion_to_euler(q) for q in true_q_valid])

            # Ensure angle continuity (unwrap)
            est_euler_deg_valid_unwrapped = ensure_angle_continuity(est_euler_deg_valid, threshold=180)
            true_euler_deg_valid_unwrapped = ensure_angle_continuity(true_euler_deg_valid, threshold=180) # Ensure continuity for true Euler angles as well

            estimated_euler_deg = est_euler_deg_valid_unwrapped # Store unwrapped Euler angles
            true_euler_deg = true_euler_deg_valid_unwrapped     # Store unwrapped true Euler angles
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

    # Package results into a dictionary
    results = {
        "timestamps": timestamps_ns,
        "estimated_quaternions": estimated_quaternions,
        "estimated_euler_deg": estimated_euler_deg, # Unwrapped Euler angles
        "estimated_euler_raw": estimated_euler_raw, # Raw (wrapped) Euler angles
        "estimated_velocities": estimated_velocities,
        "estimated_positions": estimated_positions,
        "estimated_accel_biases": estimated_accel_biases,
        "estimated_gyro_biases": estimated_gyro_biases,
        "true_quaternions": true_quaternions,
        "true_euler_deg": true_euler_deg,           # Unwrapped true Euler angles
        "true_euler_raw": true_euler_raw,           # Raw (wrapped) true Euler angles
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
# Tek bir sekansı (örn. MH01) işleyen fonksiyon
# =========================================================
def run_sequence(seq_name, config, summary_file):
    """
    Processes a single VIO dataset sequence (e.g., "MH01").
    It handles file paths, adaptive covariance computation (if enabled), and result saving.
    """
    imu_file_path = f"imu_interp_gt/{seq_name}_imu_with_interpolated_groundtruth.csv"
    sequence_number_str = seq_name[2:]  # Örnek: MH01 -> "01"
    # Convert to int to remove leading zero (e.g., "01" -> 1), then format into path
    visual_file_path = f"VO/vo_pred_super_best/mh{int(sequence_number_str)}_ns.csv"

    if args.adaptive:
        sigma_v_map = compute_adaptive_sigma_v(config, visual_file_path, seq_name)
        sigma_p_map = compute_adaptive_sigma_p(config, visual_file_path, seq_name)
    else:
        sigma_v_map = {}
        sigma_p_map = {}

    print(f"[{seq_name}] Processing started...")
    start_time = time.time()
    results = process_vio_data(imu_file_path, visual_file_path, sigma_v_map, sigma_p_map)
    if results is None:
        print(f"[{seq_name}] Could not be processed due to an error.")
        return 

    print(f"[{seq_name}] === RMSE Sonuçları ===")
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
        print(f"Toplam Statik Süre (Gravity Update Aktif): {results['total_static_duration']:.4f} sn")


    detailed_csv = f"outputs/adaptive_{seq_name.lower()}.csv"
    save_results_to_csv(results, filename=detailed_csv)

    if SAVE_RESULTS_CSV:
        save_summary_to_csv(results, seq_name, args, filename=summary_file)

    end_time = time.time()
    print(f"[{seq_name}] Total Execution Time: {end_time - start_time:.4f} s")

# =========================================================
# ANA PROGRAM: Sekansları paralel çalıştırma
# =========================================================
if __name__ == "__main__":
    sequences = ["MH01", "MH02", "MH03", "MH04", "MH05"]
    summary_file = SAVE_RESULTS_CSV_NAME # Centralized summary file for all runs.

    # --- Başlangıçta CSV Kontrolü ---
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
                # Check if results for all sequences with the current parameters already exist.
                if set(sequences).issubset(existing_sequences):
                    results_exist = True
            else:
                 print(f"Warning: Expected columns for parameter matching are missing in {summary_file}. Skipping pre-run check.")

        except pd.errors.EmptyDataError:
            print(f"Warning: {summary_file} is empty. Skipping check.")
        except Exception as e:
            print(f"Warning: Error reading {summary_file}: {e}. Skipping check.")

    if results_exist:
        print(f"Results for the current parameter combination already exist in '{summary_file}'. Skipping execution.")
        sys.exit(0) # Programdan çık
    # --- CSV Kontrolü Sonu ---

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for seq in sequences:
            # Pass the config dictionary to run_sequence
            fut = executor.submit(run_sequence, seq, config, summary_file)
            futures.append(fut)

        concurrent.futures.wait(futures)

    print("=== Processing of all sequences finished. ===")