import socket, select, re, sys, signal, os
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from L3_Agent_PPO import L3, Kuramoto
from Phase_estimator_pca_online import Phase_estimator_pca_online
from util import create_folder_if_not_exists, l3_update_theta, get_time_string
from typing import Sequence # For type hinting numpy array

class L3_Wrapper():

    def __init__(self, model_path : str, save_path : str, ID : int =0, amplitude : float =15, omega : float =2, n_participants : int =3,
                 omega_parts=np.array([0, 3.4, 3.4]), c_strength : float =0.25, omega_sat : float = 15):
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
        self.omega_sat = omega_sat
        self.c_strength = c_strength

        self.l3_agent = L3(0,omega_parts[0],0,n_actions=1,input_dims=(5,), omega_sat=omega_sat, model_path=model_path)
        self.l3_agent.load_model()

        self.window_pca = 4  # duration of the time window [seconds] in which the PCA is operated
        self.interval_between_pca = 1  # time interval [seconds] separating consecutive computations of the PCA

        self.AGENTS = []
        self.AGENTS.append(self.l3_agent)
        for i in range(1, self.n_participants):
            self.AGENTS.append(Kuramoto(i, 0, 0))

        self.estimators_live = []
        for _ in range(self.n_participants):
            self.estimators_live.append(Phase_estimator_pca_online(self.window_pca,
                                                                   self.interval_between_pca))  # One estimator for each participant

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

        self.phases_history = [np.zeros(self.n_participants)]
        self.time_history = [0]
        self.positions_history = []

     # This function extracts the 3D data position coming from UE
    def parse_TCP_string(self, string : str) -> tuple[bool, Sequence[float]]:
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        flag = len(numbers) == 3 * self.n_participants + 1
        return flag, numbers[0:-1], numbers[-1]

    def set_initial_position(self, position : list[list[float]]):
        self.initial_position = position[:, 0] # 0 because it is always the index of L3
        self.initial_phase = 0
        self.positions_history.append(position.T)
    
    # Calculates the next position and formats the message to be sent to UE for animation
    def update_position(self, positions : list[list[float]], delta_t : float, time : float) -> str:
        # positions contains the neighbors 3D end effectors
        self.positions_history.append(positions.T)
        theta = np.arctan2(self.z, self.y)
        ic(theta)

        ic(time)

        phases = [] # Vector of the real phases
        for i in range(self.n_participants):  # Collect phases of all participants. The ones from other participants are estimated
            if self.AGENTS[i].is_virtual == False: 
                phases.append(self.estimators_live[i].estimate_phase(positions[:, i], time))
            else:
                phases.append(theta - self.initial_phase)

        ic(phases)

        self.time_history.append(self.time_history[-1] + delta_t)
        self.phases_history.append(np.array(phases))

        phases_dots = (self.phases_history[-1] - self.phases_history[-2])/delta_t  # Compute the phase dots
        observation = self.l3_agent.get_state_with_theta_dots(np.array(phases), phases_dots, self.n_participants)
        self.l3_agent.omega = self.l3_agent.choose_action_mean(observation)  # Compute the new omega for the virtual agent
        ic(self.l3_agent.omega)

        l3_theta_next = l3_update_theta(np.array(phases), self.l3_agent.omega, coupling=self.c_strength, dt=delta_t)

        self.y = self.amplitude * np.cos(l3_theta_next)
        self.z = self.amplitude * np.sin(l3_theta_next)

        message = 'X=' + str(self.initial_position[0]) + ' Y=' + str(self.initial_position[1] + self.y) + ' Z=' + str(
            self.initial_position[2] + self.z_amp_ratio * np.abs(self.z))  # Format data as UE Vector

        return message
    
    def plot_phases(self, phases: list[list[float]]):
        # Plotting
        colors = ['red', 'blue', 'magenta', 'yellow', 'orange', 'olive', 'cyan']
        plt.figure()
        for i in range(self.n_participants):
            if self.AGENTS[i].is_virtual == True: 
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'L3 {i + 1}')
            else:
                plt.plot(self.time_history, phases[:, i], color=colors[i], label=f'VH {i + 1}')

        plt.title('Phases of Experiment')
        plt.xlabel('time  (seconds)')
        plt.ylabel('Phases (radiants)')
        plt.legend()
        plt.grid(True)

        time_string = get_time_string()

        plt.savefig(f'{self.save_path}\\'+time_string+'phases_plot.png')
        plt.close()

    def save_data(self):
        time_string = get_time_string()
        np.save(f'{self.save_path}/'+time_string+'phases_history.npy',   np.stack(self.phases_history))
        np.save(f'{self.save_path}/'+time_string+'positions_history.npy', np.array(self.positions_history))
        np.save(f'{self.save_path}/'+time_string+'time_history.npy',     np.stack(self.time_history))

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