import torch as t
import torch.nn as nn
import torch.nn.function as F
import torch.optim as optim
import numpy as np


# Deep Q Learning:
#	Model free   (No need to know the dynamics of the game)
#	bootstrapped (self-starting process that is supposed to continue without external input)
#	Off policy   (policy used to generate actions)
class DeepQNetwork(nn.Module):
	#				 , Learning rate,
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
		super(DeepQNetwork, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions

		#					 v Input(unpacks list)  v Output
		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		self.fc2 = nn.Linear( self.fc1_dims,   self.fc2_dims)
		self.fc3 = nn.Linear( self.fc2_dims,   self.n_actions)
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	# Foward propagation
	def foward(self, state):

		# Pass state to first fully connected layer
		x = F.relu(self.fc1(state))

		# Pass that output to the second fully connected layer
		x = F.relu(self.fc2(x))

		# Pass that output to the third fully connected layer
		action = self.fc3(x)
		# Return. Do not activate because we want the agents raw estimate, values could be negative
		return action

class Agent():
				# gamma:   Determins waiting for future rewards,
				# epsilon: Explore/Explout dilema, how long does the agent spend exploring vs taking best known action
				# lr: Leaning rate
				# batch_size: Batch of memory we are learning from
				# n_actions: Number of available actions
				# max_mem_size: Max memory size
				# eps_end: epsilon ends at
				# eps_dec: decrements epsilon every time step
	def __init__(self,
				gamma,
				epsilon,
				lr,
				inpput_dims,
				bath_size,
				n_actions,
				max_mem_size=100000,
				eps_end=0.01,
				eps_dec=5e-4
				):

		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_min = eps_end
		self.eps_dec = eps_dec
		self.lr = lr
		self.action_space = [i for i in range(n_actions)]
		self.mem_size = max_mem_size
		self.batch_size = batch_size

		# Memory counter: Position of the first available memory for storing agents memory
		self.mem_cntr = 01

		# Evaluation neworks: 
		self.Q_eval = DeepQNetwork(self.lr, n_actions=nn_actions, input_dims=input_dims,
								   fc1_dims=256, fc2_dims=256)

		# Method of storing agents state in memory
		self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

		# Keeps track of agents new states
		self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

		# 
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

		# Used as a mask for setting values of nes states to zeros
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

		)

	def store_transition(self, state, action, reward, state_, done):
		
		# Position of first unoccupied memory
		# Modulo to wrap around to write over old memory with new agents memory
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.action_memory[index] = action
		self.terminal_memory[index] = done

		# Increment to know that memory was filled
		self.mem_cntr +=1

	# Funtion to chose actions
	def choose_action(self, observation):
		
		# If true take best known action
		if np.random.random() > self.epsilon:

			# Send variables we want to perform computations on to our device
			state = T.tensor([observation]).to(self.Q_eval.device)
			# Pass state through Deep Q network
			actions = self.Q_eval.forward(state)
			# Get max value
			action = T.argmax(actions).item()
		# Else random action
		else:
			action = np.random.random.choice(self.action_space)

		return action

	# 
	def learn(self):
		
		# If memory no full with data to learn from return
		if self.mem_cntr < self.batch_size:
			return

		# Zero parameter gradiant in optimzer
		# Particular to PyTorch: Used to ensure that we aren’t tracking any unnecessary
		# information when we train our neural network
		# https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
		self.Q_eval.optimizer.zero_grad()

		# Position of our maximum memory to select subset of memory up to the last filled memory
		max_mem = min(self.mem_cntr, slef.mem_size)

		# Choose random memory that has yet to be chosen   v replace == false to no use previously choseen memory
		batch = np.random.choice(max_mem, self.batch_size, replace=false)

		# Bach index for book keeping, needed for proper array slicing
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		# Converting numpy array subset of agent memory to PyTorch Tensor
		# This step is needed for all batch types except action batch
		state_batch = T.tensor(self.state_memory([batch]).to(self.Q_eval.device)
		new_state_batch = T.tensor(self.new_state_memory([batch]).to(self.Q_eval.device)
		reward_batch = T.tensor(self.reward_memory([batch]).to(self.Q_eval.device)
		terminal_batch = T.tensor(self.terminal_memory([batch]).to(self.Q_eval.device)

		# 
		action_batch = self.action_memory[batch]

		# Only take values of actions taken (Reason for array slicing)
		# We want to be moving our agent's estimate for the value of the current state towards
		# the maximal value for the next state
		# (Tilting towards selecting maximal actions)
		q_eval = self.Q_eval.forward(state_batch)[Batch_index, action_batch]

		# Same as above but for next state
		q_next = self.Q_ecal.forward(new_state_batch)

		# Terminal state must be zero
		q_next[teminal_batch] = 0.0

		# This is what we are updating our estimates towards
		q_target = reward_batch + self.gamma *       T.max(q_next, dim=1)[0]

		# 
		loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
		loss.backwards()
		self.Q_val.optimizer.step()

		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
						else self.eps_min