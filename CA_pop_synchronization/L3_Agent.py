import tensorflow as tf
import cmath
import math
import numpy as np
from typing import Sequence # For type hinting numpy arrays
from util import wrap_angle, compute_average_phasor

# TODO MC: can we implement type hinting for the methods?

# ---------------------------------------------------------------- DQN BASED L3 AGENT
class L3_Agent:
    def __init__(self, n_nodes : int, l3_adjacency : Sequence[int], omega_vals : Sequence[float], coupling_strength : float, model_path : str, dt : float = 0.01, virtual_agent_index : int = 0):
                                                
        self.n_nodes = n_nodes
        self.l3_adjacency = l3_adjacency # l3_adjacency[i] = 1  if l3 is linked to i-th participant and 0 otherwise
        self.omega_vals = omega_vals
        self.dt = dt
        self.coupling_strength = coupling_strength
        self.coupling_vals = l3_adjacency * self.coupling_strength
        # self.model = keras.models.load_model(model_path, compile=False)
        loaded_model = tf.saved_model.load(model_path)
        self.model = loaded_model.signatures['serving_default']
        self.actions = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.action_length = len(self.actions)
        self.virtual_agent_index = virtual_agent_index
        self.wrapping_domain = '-pi to pi'
        self.omega_sat = 15     # Saturation value of omega in rad/s

    def l3_update(self, theta_old : float) -> float:
        delta_theta = theta_old - theta_old[self.virtual_agent_index]
        sin_delta_theta = np.sin(delta_theta)
        theta_new = theta_old[self.virtual_agent_index] + self.dt * (self.omega_vals[self.virtual_agent_index] + sum(self.coupling_vals * sin_delta_theta))
        return theta_new

    def get_observation(self, theta_obs : Sequence[float]) -> tuple[float, float, float]: 
        obs_pos = np.zeros(len(theta_obs))
        theta_a = theta_obs[self.virtual_agent_index]
        for i in range(1, len(theta_obs)):
            theta_i = theta_obs[i]
            obs_pos[i] = wrap_angle(theta_a - theta_i, self.wrapping_domain)
        average_phasor = compute_average_phasor(obs_pos[1:])
        mean_obs_pos = cmath.phase(average_phasor)
        var_obs_pos = 1 - np.abs(average_phasor)

        return mean_obs_pos, var_obs_pos, self.omega_vals[self.virtual_agent_index]

    def get_greedy_action(self, obs_input : Sequence[float]) -> int: # Take the greedy action for deployment
        state_tensor = tf.convert_to_tensor(obs_input)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor)
        return np.argmax(action_probs['dense_2'].numpy()[0])

    def get_epsilon_greedy_action(self, obs_input : Sequence[float], epsilon : float) -> int: # Take the epsilon-greedy action for learning
        if np.random.random() > epsilon:
            return self.get_greedy_action(obs_input, self.action_length)
        else:
            return np.random.randint(0, self.action_length)

    def take_action(self, a : int) -> float:
        d_omega = self.actions[a]
        return d_omega

    def l3_update_phase(self, theta: float) -> float:
        obs_input = self.get_observation(theta)
        action = self.get_greedy_action(obs_input)
        omega_delta = self.take_action(action)
        self.omega_vals[self.virtual_agent_index] += omega_delta
        self.omega_vals[self.virtual_agent_index] = np.clip(self.omega_vals[self.virtual_agent_index], -self.omega_sat, self.omega_sat) # Saturation on the value of omega

        return self.l3_update(theta)