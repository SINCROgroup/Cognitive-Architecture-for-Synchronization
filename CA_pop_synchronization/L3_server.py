import socket
import select
import re
import numpy as np
import sys
import signal
from icecream import ic
from L3_engine import L3Agent, Phase_estimator_pca_online
import sys

ic.configureOutput(prefix='DEBUG | ')
# ic.disable()                              # Uncomment to stop debugging messages

class L3_Wrapper():

    def __init__(self, model_path, ID = 0, amplitude = 10, omega = 2, partecipants = 3, omega_parts = np.array([0, 3.4, 4.6]), c_strenght = 1.25):
        self.ID = ID                # Python CA instance ID
        self.amplitude = amplitude  # Movement amplitude
        self.omega = omega          # Movement frequency                
        self.x = 0 
        self.y = 0  
        self.z = 0  
        self.z_amp_ratio = 0.1    
        self.intial_position = 0
        self.intial_phase = 0
        self.partecipants = partecipants

        self.l3_phase = []
        self.l3_agent = L3Agent(partecipants, omega_parts, c_strenght, model_path)

        self.window_pca           = 4     # duration of the time window [seconds] in which the PCA is operated
        self.interval_between_pca = 1     # time interval [seconds] separating consecutive computations of the PCA

        # self.parts_phases = []
        self.estimators_live = []
        for _ in range(self.partecipants):
            self.estimators_live.append(Phase_estimator_pca_online(self.window_pca, self.interval_between_pca))

        self.kuramoto_phases = [np.zeros(self.partecipants)]

    def reset_CA(self):
        # RESET THE PHASE ESTIMATORS
        self.estimators_live = []
        for _ in range(self.partecipants):
            self.estimators_live.append(Phase_estimator_pca_online(self.window_pca, self.interval_between_pca))

        self.kuramoto_phases = [np.zeros(self.partecipants)]

    # This function extracts the 3D data position coming from UE
    def parse_TCP_string(self, string):
        ic(string)
        numbers = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', string)])
        flag = len(numbers) == 3*self.l3_agent.n_nodes + 1
        return flag, numbers[0:-1], numbers[-1]
    
    def set_intial_position(self, position):
        self.intial_position = position
        self.y = -self.amplitude
        self.z = 0
        self.intial_phase = 0

    # Calculates the next position and formats the message to be sent to UE for animation
    def update_position(self, positions, delta_t, time):
        # positions contains the neighbors 3D end effectors
        theta = np.arctan2(self.z, self.y)
        theta = np.mod(theta, 2*np.pi)  # wrap to [0, 2pi)
        ic(theta)
        self.l3_phase.append(theta)

        ic(time)

        phases = [theta - self.intial_phase]
        for i in range(self.l3_agent.n_nodes - 1):
            phases.append(agent.estimators_live[i].estimate_phase(positions[:, i+1], time))           # CA assumes that first position element is the L3 itself

        ic(phases)

        self.l3_agent.dt = delta_t
        theta_next = self.l3_agent.l3_update_phase(np.array(phases))        
        # self.kuramoto_phases.append(self.l3_agent.update_phases(self.kuramoto_phases[-1]))                  # comment this and next line and uncomment upper line if L3 is connected with VR agents 
        # theta_next = self.kuramoto_phases[-1][self.l3_agent.virtual_agent]

        ic(theta_next)

        self.y = self.amplitude * np.cos(theta_next)
        self.z = self.amplitude * np.sin(theta_next)

        message = 'X=' + str(self.intial_position[0]) + ' Y=' + str(self.intial_position[1] + self.y) + ' Z=' + str(self.intial_position[2] + self.z_amp_ratio * np.abs(agent.z))    # Format data as UE Vector
        return message

    @staticmethod
    def start_connection(address, port):
        # Create a TCP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Bind the socket to the server address and port
            server_socket.bind((address, port))
        except socket.error as e:
            print("Connection error: %s" % e)

        # Listen for incoming connections
        server_socket.listen(1)         # Limit number of connections to L3 socket
        print(f'Server listening on {address}:{port}')

        # Wait for a client connection
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()
        print(f'Connection from {client_address}')

        return connection, client_address


if __name__ == "__main__":
    parameters = sys.argv[1:]

    # Validate inputs to ensure they are numbers
    validated_inputs = []
    error = False
    for param in parameters:
        try:
            validated_inputs.append(int(param))  # try converting to float
        except ValueError:
            print(f'Invalid input: {param} is not a number, please insert the number of partecipants connected to the L3 CA.')
            error = True
            sys.exit(1)

    # Implement additional error handling
    if len(validated_inputs) == 0 or validated_inputs[0] <= 0:
        print('Error: Input is not valid, please insert the number of partecipants connected to the L3 CA.')
        sys.exit(1)
    elif not error:
        n_partecipants = validated_inputs[0] + 1            # participant number is inteded as the number that the L3 is connected to

    ic(n_partecipants)

    agent = L3_Wrapper('CA_pop_synchronization\model', partecipants = n_partecipants)

    # Set the server address and port (must match with socket in UE)
    SERVER_ADDRESS = 'localhost'
    SERVER_PORT = 12345

    connection, client_address = agent.start_connection(SERVER_ADDRESS, SERVER_PORT)

    def signal_handler(sig, frame):
        message = 'quit'
        connection.send(message.encode('utf-8'))
        print('Control + C pressed, closing socket...')
        connection.close()
        sys.exit(0)

    # Register the signal handler for SIGINT (Control + C)
    signal.signal(signal.SIGINT, signal_handler)

    time = 0

    # Receive data stream from UE
    while True:    
        try:
            ic(f'waiting for data')
            ready_to_read, ready_to_write, exception = select.select([connection], [], [], 5)
            if ready_to_read: 
                data = connection.recv(1024).decode()
                ic(f'Received data: {data}')

            if data == '' or not(ready_to_read or ready_to_write or exception): 
                print('Connection with client terminated')
                connection.close()
                
                agent.reset_CA()
                print('Cognitive Architecture reset completed')
                connection, client_address = agent.start_connection(SERVER_ADDRESS, SERVER_PORT)
    
                time = 0

            else:
                _, position, delta_t = agent.parse_TCP_string(data)
                position = np.reshape(position, (agent.partecipants, 3)).T

                if time == 0: agent.intial_position = position[:, 0]
                
                time += delta_t
                ic(position, delta_t)
                message = agent.update_position(position, delta_t, time)

                _, ready_to_write, _ = select.select([], [connection], [])
                if ready_to_write: connection.send(message.encode('utf-8'))
                ic(f'Message sent: {message}')

        except KeyboardInterrupt:
            # Handle the Control + C key press gracefully
            signal_handler(None, None)    