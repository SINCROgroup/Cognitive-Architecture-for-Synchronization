# Cognitive Architecture for Synchronization Demo 

To set up your machine:
- Install python packages running `pip install -r python_packages.txt`.
- Download, install and abilitate the Unreal Engine [Socketer](https://github.com/How2Compute/Socketer) plugin.

## Functionalities of the Repository  
Folder `L3_test` contains an Unreal Engine project capable of:
- Connecting with TCP sockets and stream data via TCP (by default expects connection by the same local machine).
- Render Avatar Behaviour using end-effector position using an IK rig.

Folder `CA_pop_synchronization` contains the python capable of:
- Connecting with TCP sockets and stream data (by default expects connection by the same local machine).
- Extract phases from 3D positions of an oscillator-like motion.
- Simulate an interaction with 2 kuramoto oscillators and 1 trained DQN artificial L3 avatar.

Folder `CA_falcon_heavy` contains the python capable of:
- Connecting with TCP sockets and stream data (by default expects connection by the same local machine).
- Extract phases from 3D positions of an oscillator-like motion.
- Compute global and invidual sinchronization indexes.

## Code Structure 
To debug character animation, navigate to `L3_test\Content\Characters\Mannequins\Animations` using Content Drawer and open `L3_synch`.
This blueprint connects the character behaviour with Python commands using TCP socket connection, the basic workflow is as follows:
- First launch the python server hosting the CA running `L3_server.py`.
- By running the UE blueprint, the rendered engine creates an instance of the socket and tries to connect to the server through the event graph.
- The `L3_server.py` waits for position data, generates the character behaviour and sends back data to the rendering engine.
- `L3_client.py` sends data points in the 3D space used as target values for the left hand IK control.
- Such data is stored in a local varible and, it is continously checked by the animgraph to set target location for the left hand IK rig which updates the full body posture.

## PoP Synchronization Demo
To run the demo application:
- Move in the folder CA_pop_synchronization and launch the python script `L3_server.py` passing the number of participant the L3 has to interact with and wait for the TCP socket to be online (see prompt messages).
- Launch the  `L3_test` unreal project to render simulated L3 behaviour

Example usage:
`python main.py 2 simulation_data`

Useful information
- The TCP connection consists in the exchange of strings containig 3D end-effector positions and the time interval (delta_t) between each message. The string messages are structured as: `X={end-effector x value} Y={end-effector y value} Z={end-effector z value}; {delta_t value}`.
- If multiple positions need to be sent concatenate them in a single string and leave the delta_t value as last item in the string.
- The IK rig used by the animation blueprint `L3_synch` is located in the folder `L3_test\Content\Characters\Mannequins\Rigs`.

## Falcon Heavy Demo
To run the demo application with single socket:
- Launch the python script `L3_server.py` and wait for the TCP socket to be online (see prompt messages).
- Launch the unreal prject to establish TCP connection and render visual animation based on synchronization indexes.

To run the demo application with dual socket:
- Launch TCP socket on the machine that hosts sensor data from participants.
- Launch the python script `L3_server_dual_socket.py` to connect with the source TCP socket.s
- Launch the unreal prject to receive and render the synchronization indexes.

Useful information
- The TCP connection consists in the exchange of strings
  - INPUT: 3D end-effector positions and the time interval (delta_t) between each message. The string messages are structured as: `X={end-effector x value} Y={end-effector y value} Z={end-effector z value};... ;{delta_t value}`.
  - OUTPUT: synchronization indexes in same participants order of how the positions are sent. The string messages are structured as: `global synch index, participant 1 synch index; participant 2 synch index; ...`.
- If multiple positions need to be sent, concatenate them in a single string and leave the delta_t value as last item in the string. 
