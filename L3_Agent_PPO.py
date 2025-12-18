import os
import numpy as np
import cmath
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, alpha=0.0003, fc1_dims=128, fc2_dims=128, chkpt_dir='Checkpoints/'): # 256 256 Tanh
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo_theta_dots')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, n_actions),
            nn.Tanh(),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        mean = self.actor(state)

        return mean

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Kuramoto:
    def __init__(self, ID, omega, theta):
        self.ID = ID
        self.omega = omega
        self.theta = theta
        self.is_virtual = False

    def init_omega(self,omega_min,omega_max):
        self.omega = np.random.uniform(omega_min,omega_max)


class L3(Kuramoto):
    def __init__(self, ID, omega, theta, n_actions, input_dims, omega_sat, alpha=0.00003, model_path='model'):
        super().__init__(ID, omega, theta)
        self.omega_sat = omega_sat
        self.is_virtual = True

        self.actor = ActorNetwork(input_dims, n_actions, alpha, chkpt_dir=model_path)

    def load_model(self):
        print('...loading models...')
        self.actor.load_checkpoint()

    def choose_action_mean(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        action = self.actor(state)
        action = T.squeeze(action).item()
        action = action*self.omega_sat

        return action

    def get_state(self, estimated_phases, n_participants):  # Returns the state of the MDP for the virtual agent
        delta_theta = estimated_phases.reshape(n_participants, 1) - estimated_phases.reshape(1, n_participants)  # Matrix of all the phase differences
        # Remove 0 from phase differences matrix
        delta_theta = delta_theta[~np.eye(delta_theta.shape[0], dtype=bool)].reshape(n_participants, n_participants-1)
        obs_pos = delta_theta[0, :]
        omega_a = self.omega
        order_parameter_complex = 1/(n_participants-1) * np.sum(np.exp(1j*obs_pos))
        mean_obs_pos = cmath.phase(order_parameter_complex)
        var_obs_pos = 1 - np.abs(order_parameter_complex)

        return np.array((mean_obs_pos, var_obs_pos, omega_a))
    
    def get_state_with_theta_dots(self, estimated_phases, estimated_theta_dots, n_participants):  # It behaves as get_state and includes also mean and variance of theta_dots
        delta_theta = estimated_phases.reshape(n_participants, 1) - estimated_phases.reshape(1,n_participants)  # Matrix of all the phase differences
        # Remove 0 from phase differences matrix
        delta_theta = delta_theta[~np.eye(delta_theta.shape[0], dtype=bool)].reshape(n_participants, n_participants - 1)
        obs_pos = delta_theta[0, :]
        omega_a = self.omega
        order_parameter_complex = 1 / (n_participants-1) * np.sum(np.exp(1j * obs_pos))
        r_bar = np.abs(order_parameter_complex)
        mean_obs_pos = cmath.phase(order_parameter_complex)
        var_obs_pos = 1 - r_bar

        mask = np.ones(estimated_theta_dots.shape[0], dtype=bool)
        mask[0] = False  # Mask theta_dot of the virtual agents
        mean_theta_dot = np.mean(estimated_theta_dots[~mask] - estimated_theta_dots[mask])
        var_theta_dot = np.var(estimated_theta_dots[~mask] - estimated_theta_dots[mask])

        return np.array((mean_obs_pos, var_obs_pos, mean_theta_dot, var_theta_dot, omega_a))
