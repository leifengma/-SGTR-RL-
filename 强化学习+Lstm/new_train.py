import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import random
import os,time

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import threading

from td3 import ReplayBuffer,TD3
from rl_lstm import SGTR_env

def get_str_time ():
	s = '_'
	for i in time.localtime (time.time ())[0:6]:
		s += str (i) + '_'
	return s


current_time = get_str_time()
if not os.path.exists('./logs/' + current_time):
	os.makedirs('./logs/' + current_time)

if not os.path.exists('./rl_model/models/' + current_time):
	os.makedirs('./rl_model/models/' + current_time)


# initialise the environment
env_model = keras.Sequential ([
	layers.LSTM (units=256,input_shape=(10,12),return_sequences=True),
	layers.Dropout (0.4),
	layers.LSTM (units=256,return_sequences=True),
	layers.Dropout (0.3),
	layers.LSTM (units=128,return_sequences=True),
	layers.LSTM (units=32),
	layers.Dense (11)
])
env_model.compile (optimizer='adam',loss='mse')
env_model.load_weights (r'M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\LSTM模型\best_model.hdf5')
with open (
		r'M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\LSTM模型\用于训练lstm的数据.npy',
		'rb') as f:
	train_datasets = np.load (f,allow_pickle=True)
env = SGTR_env(env_model,-56,train_datasets[0],train_datasets[1],train_datasets[2])
# env = wrappers.Monitor(env, save_dir, force = True)
# env.seed(0)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
max_action,min_action = 3.41,-0.81
result_writer = tf.summary.create_file_writer('./logs/' + current_time)


def evaluate_policy(policy, eval_episodes=5):
	global env,action_dim,state_dim,memory
	# during training the policy will be evaluated without noise
	print("Evaluating Policy..........")
	avg_reward = 0.
	file_name=str (get_str_time ())
	try:
		os.makedirs (
			r"M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\eval结果图\\" + file_name)
	except:
		print ("file exist")
	for _ in range(eval_episodes):
		state = env.reset()
		done = False
		counter=0

		speed_rec = []
		temp_rec = []
		action_rec = []
		rewards = []
		reward_sum = []
		episode_reward = 0
		while counter<2000:  # not done and
			counter+=1
			action = policy.select_action(state)  # action = policy.select_action (state,noise=True)
			next_state, reward, done, _ = env.step(action)
			avg_reward += reward

			action = action.data.tolist ()[0]
			action_rec.append (action)
			next_state = next_state.astype (np.float32)
			episode_reward += reward
			rewards.append (reward)
			reward_sum.append (episode_reward)

			ori_temp_before_last = env.cal_origin_val (1,state[-1][1])
			ori_temp_last = env.cal_origin_val (1,next_state[-1][1])
			temp_rec.append (ori_temp_before_last)
			# 赋值state，累计总reward，步数
			state = next_state

			temp_change_speed = (ori_temp_last - ori_temp_before_last) * 3600
			speed_rec.append (temp_change_speed)
		plt.figure (figsize=(20,15))
		plt.subplot (2,3,1)
		plt.plot (speed_rec)
		plt.plot ([-56]*len(speed_rec),'r--')
		plt.title ('speed_rec')
		plt.subplot (2,3,2)
		plt.plot (temp_rec)
		plt.title ('temp_rec')
		plt.subplot (2,3,3)
		plt.plot (action_rec)
		plt.title ('action_rec')
		plt.subplot (2,3,4)
		plt.plot (reward_sum)
		plt.title ('reward_sum')
		plt.subplot (2,3,5)
		plt.plot (rewards)
		plt.title ('rewards')
		plt.savefig(r"M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\eval结果图\\"+file_name+"\\"+str(get_str_time())+".svg")
		plt.close ()

	avg_reward /= eval_episodes
	print ("---------------------------------------")
	print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
	print ("---------------------------------------")
	return avg_reward


# initialise the replay buffer
memory = ReplayBuffer()
# initialise the policy
policy = TD3(state_dim, action_dim, min_action, max_action, current_time=current_time, summaries=True)




def train ():
	global env,action_dim,state_dim,memory,policy
	max_timesteps = 2e6
	start_timesteps = 1e4
	total_timesteps = 0
	eval_freq = 5e3
	save_freq = 1e5
	eval_counter = 0
	episode_num = 0
	episode_reward = 0
	max_episode_steps=2000
	done=True
	action_keep_count=0
	action_keep_val=0
	action=0

	# 接着上次训练
	# policy.load ("_2022_7_29_11_24_42_",20)

	while total_timesteps < max_timesteps:

		if done:

			# print the results at the end of the episode
			if total_timesteps != 0:
				print (
					f'Episode: {episode_num}  | Timesteps: {total_timesteps}/{max_timesteps} | Episode Reward: {round (episode_reward,3)} |--------Process Done Flag-------')
				with result_writer.as_default ():
					tf.summary.scalar ('total_reward',episode_reward,step=episode_num)

			if eval_counter > eval_freq:
				eval_counter %= eval_freq
				evaluate_policy (policy)

			state = env.reset ()

			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# the environment will play the initial episodes randomly
		if action_keep_count ==0:
			if total_timesteps < start_timesteps:
				action = (max_action-min_action) * np.random.random()+min_action
			else:  # select an action from the actor network with noise
				action = policy.select_action (state,noise=True)
			action = np.array(action,dtype='float32').reshape(1,)
			action_keep_val=action
			action_keep_count = 10
		else:
			action = action_keep_val
			action_keep_count -= 1






		# the agent plays the action
		next_state,reward,done,info = env.step (action)
		state = np.array(state).astype(np.float32)
		reward = np.array(reward).astype(np.float32)
		next_state = next_state.astype (np.float32)
		action = np.array (action).astype(np.float32)

		# add to the total episode reward
		episode_reward += reward

		# check if the episode is done
		done_bool = float (done)
		done_bool = np.array(done_bool).astype(np.float32)
		done = True if episode_timesteps + 1 == max_episode_steps else False
		# add to the memory buffer

		memory.add ((state,next_state,action,reward,done_bool))

		# show info
		print (
			f'Episode: {episode_num}  | Action: {action} | Step: {episode_timesteps} | Step Reward: {reward} |',
			end='\r')

		# update the state, episode timestep and total timestep
		state = next_state
		episode_timesteps += 1
		total_timesteps += 1
		eval_counter += 1

		# train after the first episode
		if total_timesteps > start_timesteps:
			policy.train (memory)

		# save the model
		if total_timesteps % save_freq == 0:
			policy.save (int (total_timesteps / save_freq))









def test ():
	global env,action_dim,state_dim,memory,policy
	# 装载模型权重

	for test_count in range(20):
		policy.load ("_2022_7_29_11_24_42_",test_count+1)
		state = env.reset () # 初始化state
		state = state.astype (np.float32)  # 整理state的类型
		speed_rec = []
		temp_rec = []
		action_rec = []
		rewards = []
		reward_sum=[]
		episode_reward=0
		for step in tqdm (range (2000)):
			action = policy.select_action(state,noise=False)  # + tf.clip_by_value (tf.random.normal ([1],0,0.006),-1,1)
			action = action.data.tolist ()[0]
			action_rec.append (action)
			# 与环境进行交互
			next_state,reward,done,_ = env.step (action)
			next_state = next_state.astype (np.float32)
			action = np.array (action).tolist ()
			done = 1 if done == True else 0
			#print(reward)
			episode_reward+=reward
			rewards.append(reward)
			reward_sum.append(episode_reward)

			ori_temp_before_last = env.cal_origin_val (1,state[1])
			ori_temp_last = env.cal_origin_val (1,next_state[1])
			temp_rec.append (ori_temp_before_last)
			# 赋值state，累计总reward，步数
			state = next_state

			temp_change_speed = (ori_temp_last - ori_temp_before_last) * 3600
			speed_rec.append (temp_change_speed)
		plt.figure(figsize=(20,15))
		plt.subplot (2,3,1)
		plt.plot (speed_rec)
		plt.title ('speed_rec')
		plt.subplot (2,3,2)
		plt.plot (temp_rec)
		plt.title ('temp_rec')
		plt.subplot (2,3,3)
		plt.plot (action_rec)
		plt.title ('action_rec')
		plt.subplot (2,3,4)
		plt.plot (reward_sum)
		plt.title ('reward_sum')
		plt.subplot (2,3,5)
		plt.plot (rewards)
		plt.title ('rewards')
		plt.show ()


def plot_reward():
	global env
	temp = []
	xlist = []
	for i in range (int (10000 / 0.01)):
		xlist.append (-5000 + 0.01 * i)

	for i in xlist:
		aaa = env.cal_reward (i)
		temp.append (aaa)
	plt.plot (xlist,temp)
	plt.show ()


if __name__ == '__main__':
	# def multi_thread ():
	# 	threads = []
	# 	for i in range(10):
	# 		threads.append (
	# 			threading.Thread (target=train,args=(True,f"Thread{i}"))
	# 		)
	#
	# 	for thread in threads:
	# 		thread.start ()
	#
	# 	for thread in threads:
	# 		thread.join ()
	#
	# try:
	# 	multi_thread()
	# except:
	# 	rewards_rec = pd.DataFrame (data={'rewards': rewards})
	# 	rewards_rec.to_csv (r"rewards_recoard/" + model_name + 'rewards.csv')
	# 	td3_trainer.save_weights ('rl_model/last_model',model_name)
	# 	replay_data = np.array (replay_buffer.buffer,dtype='object')
	# 	np.save ("numpy_binary",replay_data)


	train()
	# test()