import socket, select, re, sys, signal, os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from L2_Agent import L2Agent
from L3_Agent import L3_Agent
from Phase_estimator_pca_online import Phase_estimator_pca_online
from util import create_folder_if_not_exists, kuramoto_dynamics
from typing import Sequence # For type hinting numpy array

class L2Wrapper():
    def __init__(self, model_path : str, save_path : str, ID : int =0, amplitude : float =15, omega : float =2, n_participants : int =3,
                 omega_parts=np.array([0, 3.4, 4.6]), c_strength : float =0.25, filtering_coef : float = 1/5, 
                 dt : float = 0.01, thres_min : float = 0, thres_max : float = np.pi/2, speed_thres : float = 4, damping_factor : float = 4,
                 mode_saturation : str = "soft", p_norm : int   = 20, virtual_agent_index : int   = 0):
        
        self.ID = ID  # Python CA instance ID
        self.amplitude = amplitude  # Movement amplitude
        self.omega = omega  # Movement frequency
        self.x = 0
        self.y = 0
        self.z = 0
        self.z_amp_ratio = 0.1
        self.initial_position = 0
        self.initial_phase = 0
        self.n_participants = n_participants
        self.filtering_coef = filtering_coef # Filter coefficient of the amplitudes first-order filter
        

        self.graph_nx = nx.complete_graph(self.n_participants)
        self.adjacency_matrix = nx.to_numpy_array(self.graph_nx)

        self.l3_agent = L3_Agent(n_participants, self.adjacency_matrix[0], omega_parts, c_strength, model_path, dt, virtual_agent_index)

        self.l2_phase = []
        self.l2_agent = L2Agent(dt, thres_min, thres_max, speed_thres, damping_factor, mode_saturation, p_norm, virtual_agent_index)

        self.window_pca = 4  # duration of the time window [seconds] in which the PCA is operated
        self.interval_between_pca = 1  # time interval [seconds] separating consecutive computations of the PCA

        self.estimators_live = []
        for _ in range(self.n_participants):
            self.estimators_live.append(Phase_estimator_pca_online(self.window_pca,
                                                                   self.interval_between_pca))  # One estimator for each participant
            
        self.amplitude_k1 = 0.9     # Amplitude of oscillation of the first kuramoto
        self.amplitude_k2 = 1.4     # Amplitude of oscillation of the second kuramoto      
        self.kuramoto_phases = [np.zeros(self.n_participants)]

        self.time_history = [0]
        self.phases_history = [np.zeros(self.n_participants)]
        self.positions_history = []
        self.save_path = save_path
        create_folder_if_not_exists(save_path)

    def reset_CA(self):
        # STORE DATA
        self.save_data()
        self.plot_phases(np.stack(self.phases_history))

        # RESET THE PHASE ESTIMATORS
        self.estimators_live = []
        for _ in range(self.n_participants):
            self.estimators_live.append(Phase_estimator_pca_online(self.window_pca, self.interval_between_pca))

        self.kuramoto_phases = [np.zeros(self.n_participants)]

        self.phases_history = [np.zeros(self.n_participants)]
        self.time_history = [0]
        self.positions_history = []
    
    # This function extracts the 3D data position coming from UE
    def parse_TCP_string(self, string : str) -> tuple[bool, Sequence[float]]:
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        flag = len(numbers) == 3 * self.l3_agent.n_nodes + 1
        return flag, numbers[0:-1], numbers[-1]

    def set_initial_position(self, position : list[list[float]]):
        self.initial_position = position[:, self.l2_agent.virtual_agent_index]
        self.initial_phase = 0
        self.positions_history.append(position.T)

     # Calculates the next position and formats the message to be sent to UE for animation
    def update_position(self, positions : list[list[float]], delta_t : float, time : float, phase_L0 : float) -> str:
        # positions contains the neighbors 3D end effectors
        self.positions_history.append(positions.T)
        theta = np.arctan2(self.z, self.y)
        ic(theta)
        self.l2_phase.append(theta)

        # Here we are considering that the L0 signal is already a phase. If not, take as input the position of L0 and convert to phase
        # TODO: For experiments with real people, remember to add a phase estimator for the L0 and update_position should take as input the position of the L0

        ic(time)

        phases = []
        positive_amplitudes = []
        negative_amplitudes = []
        for i in range(
                self.n_participants):  # Collect phases of all participants. The ones from other participants are estimated
            if i != self.l2_agent.virtual_agent_index:
                phases.append(self.estimators_live[i].estimate_phase(positions[:, i], time))
                positive_amplitudes.append(self.estimators_live[i].amplitude_pos_p)
                negative_amplitudes.append(self.estimators_live[i].amplitude_pos_n)
            else:
                phases.append(theta - self.initial_phase)

        ic(phases)

        self.l2_agent.dt = self.l3_agent.dt = delta_t

        self.time_history.append(self.time_history[-1] + delta_t)
        self.phases_history.append(np.array(phases))

        theta_old = np.array((self.l2_phase[-1],self.kuramoto_phases[-1][1],self.kuramoto_phases[-1][2])) # Collect thetas at time t-1
        theta_next_kuramoto = kuramoto_dynamics(theta_old,self.l3_agent.n_nodes,self.l3_agent.omega_vals,self.l3_agent.dt, self.l3_agent.coupling_strength, self.adjacency_matrix,
                                           self.l3_agent.wrapping_domain) # Compute theta of Kuramoto agents at time t
        self.kuramoto_phases.append(theta_next_kuramoto)
        theta_next_l3 = self.l3_agent.l3_update_phase(np.array(phases))  # This function changes the omega of the L3 and outputs its new phase at time t

        theta_next_l2 = self.l2_agent.compute_phase_L2(phase_L0, theta_next_l3)  # Compute the new phase of L2 by blending the ones of L0 and L3
        ic(theta_next_l2)

        # First order filtering on the amplitude
        if np.cos(theta_next_l2 > 0):  steady_state_val = np.mean(positive_amplitudes)
        else:                          steady_state_val = np.mean(negative_amplitudes)
        self.amplitude = (1-self.filtering_coef*self.l2_agent.dt) * self.amplitude + self.filtering_coef * self.l2_agent.dt * steady_state_val

        ic(self.amplitude)

        self.y = self.amplitude * np.cos(theta_next_l3)
        self.z = self.amplitude * np.sin(theta_next_l3)

        y_k1 = self.amplitude_k1 * np.cos(self.kuramoto_phases[-1][1])
        z_k1 = self.amplitude_k1 * np.sin(self.kuramoto_phases[-1][1])

        y_k2 = self.amplitude_k2 * np.cos(self.kuramoto_phases[-1][2])
        z_k2 = self.amplitude_k2 * np.sin(self.kuramoto_phases[-1][2])

        message = 'X=' + str(self.initial_position[0]) + ' Y=' + str(self.initial_position[1] + self.y) + ' Z=' + str(
            self.initial_position[2] + self.z_amp_ratio * np.abs(self.z))  # Format data as UE Vector
        message = message + ';X=' + str(self.initial_position[0]) + ' Y=' + str(
            self.initial_position[1] + y_k1) + ' Z=' + str(self.initial_position[2] + self.z_amp_ratio * np.abs(z_k1))
        message = message + ';X=' + str(self.initial_position[0]) + ' Y=' + str(
            self.initial_position[1] + y_k2) + ' Z=' + str(self.initial_position[2] + self.z_amp_ratio * np.abs(z_k2))

        return message
    
    def plot_phases(self, phases: list[list[float]]):
        # Plotting
        colors = ['red', 'blue', 'magenta', 'yellow', 'orange', 'olive', 'cyan']
        plt.figure()
        for i in range(self.n_participants):
            if i == self.l2_agent.virtual_agent_index:
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'L3 {i + 1}')
            else:
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'VH {i + 1}')

        plt.title('Phases of Experiment')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Phases (radiants)')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{self.save_path}\\phases_plot.png')
        plt.close()

        # PLOT ESTIMATION ERROR
        plt.figure()
        for i in range(1, self.n_participants): plt.plot(self.time_history[:-1],
                                                         np.abs(phases[1:, i] - np.array(self.kuramoto_phases)[:-1, i]),
                                                         color=colors[i], label=f'VH {i + 1}')

        plt.title('Phase Estimation error')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Absolute error (radiants)')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{self.save_path}\\phases_estimation_error.png')
        plt.close()

    def save_data(self):
        np.save(f'{self.save_path}/phases_history.npy',   np.stack(self.phases_history))
        np.save(f'{self.save_path}/postions_history.npy', np.array(self.positions_history))
        np.save(f'{self.save_path}/time_history.npy',     np.stack(self.time_history))

    @staticmethod
    def start_connection(address : str, port : int) -> tuple[str, int]:
        # Create a TCP sockets
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Bind the socket to the server address and port
            server_socket.bind((address, port))
        except socket.error as e:
            print("Connection error: %s" % e)

        # Listen for incoming connections
        server_socket.listen(1)  # Limit number of connections to L3 socket
        print(f'Server listening on {address}:{port}')

        # Wait for a client connection
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()
        print(f'Connection from {client_address}')

        return connection, client_address

