import socket, select, re, sys, signal, os   # TODO check if some imports are unnecessary
import numpy as np
from icecream import ic
from L3_Wrapper import L3_Wrapper


ic.configureOutput(prefix='DEBUG | ')
# ic.disable()                              # Uncomment to stop debugging messages

if __name__ == "__main__":

    parameters = sys.argv[1:]  # Takes the parameters (number of participants) from command line

    # Validate inputs to ensure they are numbers
    n_participants = 0
    error = False

    try:
        n_participants = (int(parameters[0]))  # try converting to float
    except ValueError:
        print(
            f'Invalid input: {parameters[0]} is not a number, please insert the number of participants connected to the L3 CA.')
        error = True
        sys.exit(1)

        # Implement additional error handling
    if n_participants <= 0 or len(parameters) < 2:
        print(
            'Error: Input is not valid, please insert the number of participants connected to the L3 CA and a valid path to store simlation data')
        sys.exit(1)
    elif not error:
        n_participants = n_participants + 1  # participant number is inteded as the number that the L3 is connected to
        path_to_data = parameters[1]

    # n_participants = 3
    # path_to_data = 'simulation_data'

    agent = L3_Wrapper('model', save_path=path_to_data, n_participants=n_participants)

    # Set the server address and port (must match with socket in UE)
    SERVER_ADDRESS = 'localhost'
    SERVER_PORT = 12345

    connection, client_address = agent.start_connection(SERVER_ADDRESS, SERVER_PORT)


    def signal_handler(sig, frame):
        message = 'quit'
        connection.send(message.encode('utf-8'))
        print('Control + C pressed, closing socket...')
        agent.reset_CA()
        connection.close()
        sys.exit(0)


    # Register the signal handler for SIGINT (Control + C)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl + C to interrupt the game

    time = 0

    # Receive data stream from UE
    while True:
        try:
            ic(f'waiting for data')
            ready_to_read, ready_to_write, exception = select.select([connection], [], [],
                                                                     5)  # 5 is the timeout time (in seconds)
            if ready_to_read:  # Wait for the connection to start reading
                data = connection.recv(1024).decode()  # Receive data for a maximum of 1024 bytes
                ic(f'Received data: {data}')

            if data == '' or not (ready_to_read or ready_to_write or exception):
                print('Connection with client terminated')
                connection.close()

                agent.reset_CA()
                print('Cognitive Architecture reset completed')
                connection, client_address = agent.start_connection(SERVER_ADDRESS, SERVER_PORT)

                time = 0

            else:
                _, position, delta_t = agent.parse_TCP_string(data)  # Extract data coming from Unreal Engine
                position = np.reshape(position, (agent.n_participants, 3)).T

                if time == 0: agent.set_initial_position(position)

                time += delta_t  # Update time
                ic(position, delta_t)
                message = agent.update_position(position, delta_t, time)  # Update position

                _, ready_to_write, _ = select.select([], [connection], [])
                if ready_to_write: connection.send(
                    message.encode('utf-8'))  # Send the new positions and time to Unreal Engine
                ic(f'Message sent: {message}')

        except KeyboardInterrupt:
            # Handle the Control + C key press gracefully
            signal_handler(None, None)
