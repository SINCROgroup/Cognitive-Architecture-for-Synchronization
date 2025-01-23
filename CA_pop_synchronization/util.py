import cmath
import os
import numpy as np
from typing import Sequence


def wrap_angle(angle_rads: float, domain: str) -> float:
    if domain == '-pi to pi':
        return ((angle_rads + np.pi) % (2 * np.pi)) - np.pi
    elif domain == '0 to 2pi':
        return angle_rads % (2 * np.pi)
    else:
        raise ValueError(f"The value of domain '{domain}' is not acceptable.")


def compute_average_phasor(angles: Sequence[float]) -> float:
    return np.mean(np.exp(1j * np.array(angles)))


def compute_average_angle(angles: Sequence[float]) -> float:
    return cmath.phase(compute_average_phasor(angles))


def compute_order_parameter(angles: Sequence[float]) -> float:
    return np.abs(compute_average_phasor(angles))


def create_folder_if_not_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")


def saturate_soft(x: float, thres: float, p: float) -> float:
    return x / ((1 + np.abs(x / thres) ** p) ** (1 / p))


def saturate_hard(x: float, thres: float) -> float:
    if x <= - thres:  return - thres
    elif x >= thres:  return thres
    else:             return x


def bump_fun(x: float) -> float:
    if abs(x) >= 1:  return 0
    else:            return np.exp(1 - 1 / (1 - np.abs(x) ** 2))

def kuramoto_dynamics(theta_old, num_nodes, natural_frequency, dt, coupling, adjacency_matrix, wrapping_domain):
    delta_theta = theta_old.reshape(num_nodes, 1)-theta_old.reshape(1, num_nodes)  # Matrix of all the phase differences (theta_i - theta_j)
    sin_delta_theta = np.sin(-delta_theta)  # Matrix of sines of all the phase differences. (minus because of theta_j - theta_i)
    theta_dots = natural_frequency + coupling * np.sum(adjacency_matrix * sin_delta_theta, 1)
    theta_new = theta_old + dt * theta_dots
    
    return np.array([wrap_angle(theta, wrapping_domain) for theta in theta_new])


def produce_spike_signal_L0(omega_L0, A_spike, interv_spike, length_spike, time_end, dt): # This function produces a test signal to simulate a L0. Don't use it during experiments with real people
    time = np.arange(0, time_end, dt)
    phase_L0_vec = omega_L0 * time
    for spike_time in np.arange(0, time_end, interv_spike):
        spike_indices = (time >= spike_time) & (time < spike_time + length_spike)
        phase_L0_vec[spike_indices] -= A_spike
    phase_L0_vec = np.mod(phase_L0_vec + np.pi, 2 * np.pi) - np.pi
    return phase_L0_vec

