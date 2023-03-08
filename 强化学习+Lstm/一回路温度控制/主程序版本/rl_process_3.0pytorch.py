
import pandas as pd
import random
import os, time

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import keras
import sys

# sys.path.append("..")
# from LSTM模型 import keras_model  # 导入

model = keras.models.load_model(
    r'M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\一回路温度控制\温度模型\模型\best_model_epoch42.hdf5')
#

data_path = r'M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\一回路温度控制\一回路温度train数据.npy'
train_data = np.load(data_path, allow_pickle=True)
train_data = train_data.astype('float64')
men_std = np.load("一回路温度mean,std数据.npy", allow_pickle=True)


def get_str_time():
    s = '_'
    for i in time.localtime(time.time())[0:6]:
        s += str(i) + '_'
    return s


class SGTR_env():
    def __init__(self, model, set_point, train_datasets, mean, std):
        self.action_space = np.array([0] * 1)
        self.observation_space = np.array([0] * 12)
        self.mean = mean
        self.std = std
        self.response = []
        self.set_point = set_point
        self.train_datasets = train_datasets
        self.model = model
        self.state = self.train_datasets[1100]  # [random.randint(0, train_datasets.shape[0])]
        self.step_count = 0  # 步数计数

    def reset(self):
        self.step_count = 0
        # begin_index = range(1800, self.train_datasets.shape[0] - 3000, 100)
        state = self.train_datasets[1100]
        #  state = self.train_datasets[random.randint(0, self.train_datasets.shape[0]-1)]
        self.state = np.array(state)
        return np.array(state).reshape(120, )

    def cal_origin_val(self, pos, now_val):
        """
		计算未归一化的值
		"""
        val = now_val * self.std[pos] + self.mean[pos]
        return val

    def justice_down(self, next_state, step):
        """
		判断是否达到失败条件，deltaT<10或70分钟内未能实现一二回路压力平衡（小于1MP）
		"""
        ori_deltaT = self.cal_origin_val(6, next_state[-1, 6])
        # ori_pressure = self.cal_origin_val(0,next_state[-1, 0])
        # if ori_deltaT < 7.9 and step > 100:  # or (step>4200 and ori_pressure<1):
        #     return True
        #
        # else:
        #     return False
        return False

    def cal_reward(self, error):
        if error > -0.01 and error < 0.01:
            reward_1 = 2
        else:
            reward_1 = 0
        reward_2 = -abs(error)
        reward = reward_1 + reward_2
        return reward

    def step(self, action):
        self.step_count += 1
        self.state[-1, -1] = action
        # model(test_input, training=False)
        next_variable_state = np.array(self.model(np.array([self.state]), training=False))
        next_action = action
        zip_state_action = np.append(next_variable_state, next_action).reshape(1, -1)
        next_state = np.row_stack((self.state, zip_state_action))
        next_state = np.delete(next_state, 0, axis=0)
        # ori_temp_last = self.cal_origin_val(1, next_state[-1, 1])
        # ori_temp_before_last = self.cal_origin_val(1, next_state[-2, 1])
        error = self.set_point - next_state[-1, 1]
        reward = self.cal_reward(error)
        done = self.justice_down(next_state, self.step_count)
        self.state = next_state

        return np.array(next_state).reshape(120, ), reward, done, {}


# import tensorflow_probability as tfp
# import tensorlayer as tl
#
# tfd = tfp.distributions
# Normal = tfd.Normal
#
# tl.logging.set_verbosity(tl.logging.DEBUG)

random.seed(2)
np.random.seed(2)


#####################  hyper parameters  ####################
# choose env
ENV = 'Pendulum-v0'
action_range = 5.  # scale action, [-action_range, action_range]

# RL training
# RL training
max_frames = 1000  # total number of steps for training
test_frames = 300  # total number of steps for testing
max_steps = 500  # maximum number of steps for one episode
batch_size = 32  # udpate batchsize
explore_steps = 100  # 500 for random action sampling in the beginning of training
update_itr = 3  # repeated updates for single step
hidden_dim = 128  # size of hidden layers for networks
q_lr = 3e-4  # q_net learning rate
policy_lr = 3e-4  # policy_net learning rate
policy_target_update_interval = 3  # delayed steps for updating the policy network and target networks
explore_noise_scale = 1.0  # range of action noise for exploration
eval_noise_scale = 0.5  # range of action noise for evaluation of action value
reward_scale = 1.  # value range of reward
replay_buffer_size = 1000  # size of replay buffer


###############################  TD3  ####################################

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e4)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=1):
    eval_env = SGTR_env(model=model, set_point=0.0739, train_datasets=train_data, mean=men_std[0], std=men_std[1])

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        for i in range(1000):
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


file_name = f"TD3_SGTR_32"
action_max, action_min = 3.9575705208415273, -1.114517271633452
state_dim = 120
action_dim = 1
max_action = 3.9575705208415273
kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": 0.99,
    "tau": 0.005,
}
if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./models"):
    os.makedirs("./models")
torch.manual_seed(32)
np.random.seed(32)


def train():
    # initialization of env
    # env = NormalizedActions(gym.make(ENV))
    env = SGTR_env(model=model, set_point=0.0739, train_datasets=train_data, mean=men_std[0], std=men_std[1])
    kwargs["policy_noise"] = 0.2 * max_action
    kwargs["noise_clip"] = 0.5 * max_action
    kwargs["policy_freq"] = 2
    policy = TD3(**kwargs)
    # initialization of buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, TD3, 32)]

    rewards = []  # 记录每个EP的总reward
    max_reward = -np.infty
    for episode in range(1000):
        t0 = time.time()
        state = env.reset()  # 初始化state
        state = state.astype(np.float32)  # 整理state的类型
        episode_reward = 0
        for t in range(500):
            # Select action randomly or according to policy
            if episode < 200:
                aaa = random.uniform(action_min, action_max)
                action = np.array(aaa)
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * 0.1, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if (episode * 500 + t + 1) % 64 == 0:
                for updata in range(3):
                    policy.train(replay_buffer, 32)

        print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
              .format(episode, 1000, episode_reward, time.time() - t0))
        rewards.append(episode_reward)
        if episode_reward > max_reward:
            max_reward = episode_reward
            policy.save(f"./models/{file_name}")

    np.save(r"rewards_recoard/" + get_str_time() + 'rewards.npy', rewards)


def test():
    from tqdm import tqdm
    env = SGTR_env(model=model, set_point=0.0739, train_datasets=train_data, mean=men_std[0], std=men_std[1])

    # initialization of buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # initialization of trainer
    td3_trainer = TD3_Trainer(replay_buffer, hidden_dim=hidden_dim,
                              policy_target_update_interval=policy_target_update_interval, \
                              action_range=action_range, q_lr=q_lr, policy_lr=policy_lr)
    td3_trainer.load_weights()
    for test_count in range(1):

        state = env.reset()  # 初始化state
        state = state.astype(np.float32)  # 整理state的类型
        speed_rec = []
        temp_rec = []
        action_rec = []
        reward_sum = []
        episode_reward = 0

        if test_count < 1:
            print('intialize')
            _ = td3_trainer.policy_net(
                [state], is_train=False
            )  # need an extra call to make inside functions be able to use forward
            _ = td3_trainer.target_policy_net([state], is_train=False)

        for step in tqdm(range(max_steps)):
            action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)
            done = 1 if done == True else 0

            action_rec.append(action)
            # 与环境进行交互

            action = np.array(action).tolist()
            done = 1 if done == True else 0
            # print(reward)
            episode_reward += reward
            reward_sum.append(episode_reward)

            ori_temp_before_last = env.cal_origin_val(1, state[1])
            ori_temp_last = env.cal_origin_val(1, next_state[1])
            temp_rec.append(ori_temp_before_last)
            # 赋值state，累计总reward，步数
            state = next_state

            temp_change_speed = (ori_temp_last - ori_temp_before_last) * 3600
            speed_rec.append(temp_change_speed)
        np.save(r"./强化学习记录/speed_rec.npy", speed_rec)
        plt.figure(figsize=(20, 15))
        plt.subplot(2, 2, 1)
        plt.plot(speed_rec)
        plt.title('speed_rec')
        plt.subplot(2, 2, 2)
        plt.plot(temp_rec)
        plt.title('temp_rec')
        plt.subplot(2, 2, 3)
        plt.plot(action_rec)
        plt.title('action_rec')
        plt.subplot(2, 2, 4)
        plt.plot(reward_sum)
        plt.title('reward_sum')
        plt.show()




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

    # train(True,"Thread0")

    train()
    # kwargs["policy_noise"] = 0.2 * max_action
    # kwargs["noise_clip"] = 0.5 * max_action
    # kwargs["policy_freq"] = 2
    # policy = TD3(**kwargs)
    # policy.load(r"M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\一回路温度控制\models\TD3_SGTR_32")
    # eval_policy(policy, TD3, 32)
