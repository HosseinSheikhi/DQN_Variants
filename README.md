# DQN_Variants
In this repository different variants of DQN for both Classic Control and Atari Games of open ai gym environments using TF 2.2 are implemented.

# Variants
- Natural DQN
- Double DQN
- Prioritized DQN
- Dueling DQN (TODO)


# How to Run?
Execeutable file can be find in environments folder
	- Classic_Control.py to run classic control from open ai gym environments like Cartpole
	- Atari_Games.py to run atari games from openai-gym environments like Breakout

To run each file you have to set the name of the environment, type of the DQN variant and mode of the programm

	-variant could be "nature_dqn", "double_dqn", "priotrized_dqn"
	-mode could be either "train" or "test" 

If running the test mode you must enter the epsiode number you gonna load it weights. Make sure such weights are already trained, otherwise change episode number to weights which exists in saved_model 