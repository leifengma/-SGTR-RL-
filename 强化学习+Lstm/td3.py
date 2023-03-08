import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime as dt
import random

class ReplayBuffer:
	# The memory to store transitions as the agent plays the environment
	def __init__ (self,max_size = 1e6):
		"""
		Args:
			max_size: The total transitions the agent can store without
				deleting older transitions.
		"""
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add (self,transition):
		"""
		Store transitions in the memory buffer.
		The Order is state, next_state, actions, reward, done.
		"""
		if len (self.storage) == self.max_size:
			self.storage[int (self.ptr)] = transition
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append (transition)

	def sample (self,batch_size):
		""" Retrieve samples from the memory buffer
		Args:
			batch_size: the amount of transitions to be randomly
				sampled at one time.
		"""
		ind = np.random.randint (0,len (self.storage),size=batch_size)
		batch_states,batch_next_states,batch_actions,batch_rewards,batch_dones = [],[],[],[],[]
		for i in ind:
			state,next_state,action,reward,done = self.storage[i]
			batch_states.append (np.array (state,copy=False))
			batch_next_states.append (np.array (next_state,copy=False))
			batch_actions.append (np.array (action,copy=False))
			batch_rewards.append (np.array (reward,copy=False))
			batch_dones.append (np.array (done,copy=False))
		return np.array (batch_states),np.array (batch_next_states),np.array (batch_actions),np.array (
			batch_rewards).reshape (-1,1),np.array (batch_dones).reshape (-1,1)


class Actor(keras.Model):
	"""Creates an actor network"""
	def __init__(self, state_dim, action_dim, min_action, max_action):

		"""
		Args:
			state_dim: The dimensions of the state the environment will produce.
				The input for the network.
			action_dim: The dimensions of the actions the environment can take.
				The output for the network.
			max_action: The maximum possible action that the environment can have
				for one particular action. The output is scaled following the
				tanh activation function.
		"""
		super(Actor, self).__init__()
		# self.layer_1 = keras.layers.Dense(state_dim, activation='relu',
		#                                   kernel_initializer=tf.keras.initializers.VarianceScaling(
		#                                   	scale=1./3., distribution = 'uniform'))
		self.layer_1 = layers.Conv1D(32, 3, activation='hard_sigmoid',input_shape=(10,12))
		self.layer_1_0 = layers.GlobalAveragePooling1D()
		self.layer_2 = keras.layers.Dense(16, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		# self.layer_3 = keras.layers.Dense(18, activation='relu',
		#                                  kernel_initializer=tf.keras.initializers.VarianceScaling(
		#                                  	scale=1./3., distribution = 'uniform'))
		self.layer_4 = keras.layers.Dense(action_dim, activation='sigmoid',
		                                 kernel_initializer=tf.random_uniform_initializer(
		                                 	minval=-3e-3, maxval=3e-3))
		self.max_action, self.min_action = max_action, min_action

	def call(self, obs):
		if obs.ndim==2:
			obs = np.array([obs])
		x = self.layer_1(obs)
		x = self.layer_1_0(x)
		x = self.layer_2(x)
		# x = self.layer_3(x)
		x = self.layer_4(x)
		x = x * (self.max_action - self.min_action) + self.min_action
		return x


class Critic(keras.Model):
	"""Creates two critic networks"""
	def __init__(self, state_dim, action_dim):
		"""
		Args:
			state_dim: The dimensions of the state the environment will produce.
				The first input for the network.
			action_dim: The dimensions of the actions the environment can take.
				The second input for the network.
		"""
		super(Critic, self).__init__()
		# preprocess layer
		self.layer_0 = keras.layers.Conv1D (32,3,activation='hard_sigmoid',input_shape=(128,10,12))
		self.layer_0_0 = keras.layers.GlobalAveragePooling1D ()
		self.layer_0_1 = keras.layers.Concatenate(axis=1)
		# The First Critic NN
		self.layer_1 = keras.layers.Dense(state_dim + action_dim, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_2 = keras.layers.Dense(36, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_3 = keras.layers.Dense(18, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_4 = keras.layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(
			minval=-3e-3, maxval=3e-3))
		# The Second Critic NN
		self.layer_5 = keras.layers.Dense(state_dim + action_dim, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_6 = keras.layers.Dense(36, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_7 = keras.layers.Dense(18, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_8 = keras.layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(
			minval=-3e-3, maxval=3e-3))

	def call(self, obs, actions):
		x0_0 = self.layer_0(obs)
		x0_0 = self.layer_0_0(x0_0)
		x0 = self.layer_0_1([x0_0,actions])
		# forward propagate the first NN
		x1 = self.layer_1(x0)
		x1 = self.layer_2(x1)
		x1 = self.layer_3(x1)
		x1 = self.layer_4(x1)
		# forward propagate the second NN
		x2 = self.layer_5(x0)
		x2 = self.layer_6(x2)
		x2 = self.layer_7(x2)
		x2 = self.layer_8(x2)
		return x1, x2

	def Q1(self, state, action):
		x0_0 = self.layer_0 (state)
		x0_0 = self.layer_0_0 (x0_0)
		x0 = self.layer_0_1([x0_0,action])
		x1 = self.layer_1(x0)
		x1 = self.layer_2(x1)
		x1 = self.layer_3(x1)
		x1 = self.layer_4(x1)
		return x1


class TD3(object):
	"""
		Addressing Function Approximation Error in Actor-Critic Methods"
		by Fujimoto et al. arxiv.org/abs/1802.09477
	"""
	def __init__(
		self,
		state_dim,
		action_dim,
		min_action,
		max_action,
		current_time = None,
		summaries: bool = False,
		gamma = 0.99,
		tau = 0.005,
		noise_std = 0.2,
		noise_clip = 0.5,
		expl_noise = 1,
		actor_train_interval = 2,
		actor_lr = 1e-4,
		critic_lr = 1e-4,
		critic_loss_fn = None
	):

		"""
		Args:
			state_dim: The dimensions of the state the environment will produce.
				This is the input for the Actor network and one of the inputs
				for the Critic network.
			action_dim: The dimensions of the actions the environment can take.
				This is the output for the Actor network and one of the inputs
				for the Critic network.
			max_action: The maximum possible action for the environment. Actions
				will be clipped by this value after noise is added.
			current_time: The date and time to use for folder creation.
			summaries: A bool to gather Tensorboard summaries.
			gamma: The discount factor for future rewards.
			tau: The factor that the target networks are soft updated.
			noise_std: The scale factor to add noise to learning.
			noise_clip: The maximum noise that can be added to actions during
				learning,
			expl_noise: The scale factor for noise during action selection.
			actor_train_interval: The interval at which the Actor network
				is trained and the target networks are soft updated.
			actor_lr: The learning rate used for SGA of the Actor network.
			critic_lr: The learning rate used for SGD of the Critic network
			critic_loss_fn: The loss function of the Critic network. If none
				tf.keras.losses.Huber() is used.
		"""


		self.actor = Actor(state_dim, action_dim, min_action, max_action)
		self.actor_target = Actor(state_dim, action_dim, min_action, max_action)
		for t, e in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
			t.assign(e)
		self.actor_optimizer = keras.optimizers.Adam(lr=actor_lr)


		self.critic = Critic(state_dim, action_dim)
		self.critic_target = Critic(state_dim, action_dim)
		for t, e in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
			t.assign(e)
		self.critic_optimizer = keras.optimizers.Adam(lr=critic_lr)
		if critic_loss_fn is not None:
			self.critic_loss_fn = critic_loss_fn
		else:
			self.critic_loss_fn = tf.keras.losses.Huber()


		self.action_dim = action_dim
		self.state_dim = state_dim
		self.min_action = min_action
		self.max_action = max_action
		self.gamma = gamma
		self.tau = tau
		self.noise_std = noise_std
		self.noise_clip = noise_clip
		self.expl_noise = expl_noise
		self.actor_train_interval = actor_train_interval
		self.summaries = summaries
		if current_time is None:
			self.current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
		else:
			self.current_time = current_time
		if self.summaries:
			self.train_writer = tf.summary.create_file_writer('./logs/' + self.current_time)



		self.train_it = 0

	def select_action(self, state, noise: bool = False):
		# Action selection by the actor_network.
		#state = state.reshape(1, -1)
		action = self.actor.call(state)[0].numpy()
		if noise:
			noise = noise = tf.random.normal(action.shape, mean=0, stddev=self.expl_noise)
			action = tf.clip_by_value(action + noise, self.min_action, self.max_action)
		return action



	def train(self, replay_buffer, batch_size=128):
		# training of the Actor and Critic networks.
		self.train_it += 1

		# create a sample of transitions
		batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)

		# calculate a' and add noise
		next_actions = self.actor_target.call(batch_next_states)

		noise = tf.random.normal(next_actions.shape, mean=0, stddev=self.noise_std)
		noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
		noisy_next_actions = tf.clip_by_value(next_actions + noise, self.min_action, self.max_action)

		# calculate the min(Q(s', a')) from the two critic target networks
		target_q1, target_q2 = self.critic_target.call(batch_next_states, noisy_next_actions)
		target_q = tf.minimum(target_q1, target_q2)

		# calculate the target Q(s, a)
		td_targets = tf.stop_gradient(batch_rewards + (1 - batch_dones) * self.gamma * target_q)

		# Use gradient descent on the critic network
		trainable_critic_variables = self.critic.trainable_variables

		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(trainable_critic_variables)
			model_q1, model_q2 = self.critic.call(batch_states, batch_actions)
			critic_loss = (self.critic_loss_fn(td_targets, model_q1) + self.critic_loss_fn(td_targets, model_q2))
		critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
		self.critic_optimizer.apply_gradients(zip(critic_grads, trainable_critic_variables))

		# create tensorboard summaries
		if self.summaries:
			if self.train_it % 100 == 0:
				td_error_1 = td_targets - model_q1
				td_error_2 = td_targets - model_q2
				with self.train_writer.as_default():
					tf.summary.scalar('td_target_mean', tf.reduce_mean(td_targets), step = self.train_it)
					tf.summary.scalar('td_target_max', tf.reduce_max(td_targets), step = self.train_it)
					tf.summary.scalar('td_target_min', tf.reduce_min(td_targets), step = self.train_it)

					tf.summary.scalar('pred_mean_1', tf.reduce_mean(model_q1), step = self.train_it)
					tf.summary.scalar('pred_max_1', tf.reduce_max(model_q1), step = self.train_it)
					tf.summary.scalar('pred_min_1', tf.reduce_min(model_q1), step = self.train_it)

					tf.summary.scalar('pred_mean_2', tf.reduce_mean(model_q2), step = self.train_it)
					tf.summary.scalar('pred_max_2', tf.reduce_max(model_q2), step = self.train_it)
					tf.summary.scalar('pred_min_2', tf.reduce_min(model_q2), step = self.train_it)

					tf.summary.scalar('td_error_mean_1', tf.reduce_mean(td_error_1), step = self.train_it)
					tf.summary.scalar('td_error_mean_abs_1', tf.reduce_mean(tf.abs(td_error_1)), step = self.train_it)
					tf.summary.scalar('td_error_max_1', tf.reduce_max(td_error_1), step = self.train_it)
					tf.summary.scalar('td_error_min_1', tf.reduce_min(td_error_1), step = self.train_it)

					tf.summary.scalar('td_error_mean_2', tf.reduce_mean(td_error_2), step = self.train_it)
					tf.summary.scalar('td_error_mean_abs_2', tf.reduce_mean(tf.abs(td_error_2)), step = self.train_it)
					tf.summary.scalar('td_error_max_2', tf.reduce_max(td_error_2), step = self.train_it)
					tf.summary.scalar('td_error_min_2', tf.reduce_min(td_error_2), step = self.train_it)

					tf.summary.histogram('td_targets_hist', td_targets, step = self.train_it)
					tf.summary.histogram('td_error_hist_1', td_error_1, step = self.train_it)
					tf.summary.histogram('td_error_hist_2', td_error_2, step = self.train_it)
					tf.summary.histogram('pred_hist_1', model_q1, step = self.train_it)
					tf.summary.histogram('pred_hist_2', model_q2, step = self.train_it)



		# Use gradient ascent on the actor network at a set interval
		if self.train_it % self.actor_train_interval == 0:
			trainable_actor_variables = self.actor.trainable_variables

			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(trainable_actor_variables)
				actor_loss = -tf.reduce_mean(self.critic.Q1(batch_states, self.actor.call(batch_states)))
			actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
			self.actor_optimizer.apply_gradients(zip(actor_grads, trainable_actor_variables))

			# update the weights in the critic and actor target models
			for t, e in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
				t.assign(t * (1 - self.tau) + e * self.tau)

			for t, e in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
				t.assign(t * (1 - self.tau) + e * self.tau)

			# create tensorboard summaries
			if self.summaries:
				if self.train_it % 100 == 0:
					with self.train_writer.as_default():
						tf.summary.scalar('actor_loss', actor_loss, step = self.train_it)


	def save(self, steps):
		# Save the weights of all the models.
		self.actor.save_weights('./models/{}/actor_{}'.format(self.current_time, steps))
		self.actor_target.save_weights('./models/{}/actor_target_{}'.format(self.current_time, steps))

		self.critic.save_weights('./models/{}/critic_{}'.format(self.current_time, steps))
		self.critic_target.save_weights('./models/{}/critic_target_{}'.format(self.current_time, steps))


	def load(self, file_name, path_num):
		actor_path=r"M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\models\\"+f"{file_name}\\actor_{path_num}"
		actor_target_path=r"M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\models\\"+f"{file_name}\\actor_target_{path_num}"
		critic_path=r"M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\models\\"+f"{file_name}\\critic_{path_num}"
		critic_target_path=r"M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\models\\"+f"{file_name}\\critic_target_{path_num}"
		# Save the weights of all the models.
		self.actor.load_weights(actor_path)
		self.actor_target.load_weights(actor_target_path)

		self.critic.load_weights(critic_path)
		self.critic_target.load_weights(critic_target_path)



