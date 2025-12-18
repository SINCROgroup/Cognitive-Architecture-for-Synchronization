# Cognitive Architecture for Synchronization Demo 

To set up your machine:
- Install python packages running `pip install -r python_packages.txt`.

## Functionalities of the Repository  

Folder `CA_pop_synchronization` contains the python capable of:
- Connecting with TCP sockets and stream data (by default expects connection by the same local machine).
- Extract phases from 3D positions of an oscillator-like motion.
- Simulate an interaction with 1 trained DQN artificial L3 avatar and an arbitrary number of humans.

## PoP Synchronization Demo
To run the demo application:
- Move in the folder CA_pop_synchronization and launch the python script `main.py` passing the number of participant the L3 has to interact with and wait for the TCP socket to be online (see prompt messages).

Example usage:
`python main.py 2 simulation_data`

Useful information
- The TCP connection consists in the exchange of strings containig 3D end-effector positions and the time interval (delta_t) between each message. The string messages are structured as: `X={end-effector x value} Y={end-effector y value} Z={end-effector z value}; {delta_t value}`.
- If multiple positions need to be sent concatenate them in a single string and leave the delta_t value as last item in the string.

