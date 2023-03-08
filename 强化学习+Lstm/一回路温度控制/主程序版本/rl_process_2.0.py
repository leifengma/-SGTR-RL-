
from tensorflow import keras
import pandas as pd
import random
import os, time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.display import clear_output
import math
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model

tfd = tfp.distributions
Normal = tfd.Normal

import sys
sys.path.append("..")
from LSTM模型 import keras_model  # 导入

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
        self.state = self.train_datasets[1100] # [random.randint(0, train_datasets.shape[0])]
        self.step_count = 0  # 步数计数

    def reset(self):
        self.step_count = 0
        # begin_index = range(1800, self.train_datasets.shape[0] - 3000, 100)
        state = self.train_datasets[1100]
        #  state = self.train_datasets[random.randint(0, self.train_datasets.shape[0]-1)]
        self.state = np.array(state)
        return np.array(state).reshape(120,)

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
        if error>-0.01 and error<0.01:
            reward_1=2
        else:
            reward_1=0
        reward_2=-abs(error)
        reward=reward_1+reward_2
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
        error=self.set_point-next_state[-1,1]
        reward = self.cal_reward(error)
        done = self.justice_down(next_state, self.step_count)
        self.state = next_state

        return np.array(next_state).reshape(120,), reward, done, {}


# import tensorflow_probability as tfp
# import tensorlayer as tl
#
# tfd = tfp.distributions
# Normal = tfd.Normal
#
# tl.logging.set_verbosity(tl.logging.DEBUG)

random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)  # reproducible

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
replay_buffer_size = 1000 # size of replay buffer


###############################  TD3  ####################################


class ReplayBuffer:
    '''
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    '''

    def __init__(self, capacity):
        self.capacity = capacity        #buffer的最大值
        self.buffer = []                #buffer列表
        self.position = 0               #当前输入的位置，相当于指针

    def push(self, state, action, reward, next_state, done):
        #如果buffer的长度小于最大值，也就是说，第一环的时候，需要先初始化一个“空间”，这个空间值为None，再给这个空间赋值。
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)



class QNetwork(Model):
    ''' the network for evaluate values of state-action pairs: Q(s,a) '''

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=1):
        super(QNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=64, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=64, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    ''' the network for generating non-determinstic (Gaussian distributed) action from the state input '''

    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=5., init_w=1):
        super(PolicyNetwork, self).__init__()

        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=64, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=32, act=tf.nn.relu, W_init=w_init, in_channels=64, name='policy3')

        self.output_linear = Dense(n_units=num_actions, W_init=w_init, \
        b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=32, name='policy_output')

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        output = tf.nn.sigmoid(self.output_linear(x))  # unit range output [0, 1]

        return output

    def evaluate(self, state, eval_noise_scale):
        '''
        generate action with state for calculating gradients;
        eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        '''
        state = state.astype(np.float32)        #状态的type整理
        action = self.forward(state)            #通过state计算action，注意这里action范围是[-1,1]

        action = self.action_range * action     #映射到游戏的action取值范围

        # add noise
        normal = Normal(0, 1)                   #建立一个正态分布
        eval_noise_clip = 2 * eval_noise_scale  #对噪声进行上下限裁剪。eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale      #弄个一个noisy和action的shape一致，然后乘以scale
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)  #对noisy进行剪切，不要太大也不要太小
        action = action                 #action加上噪音

        return action

    #输入state，输出action
    def get_action(self, state, explore_noise_scale):
        ''' generate action with state for interaction with envronment '''
        action = self.forward([state])          #这里的forward函数，就是输入state，然后通过state输出action。只不过形式不一样而已。最后的激活函数式tanh，所以范围是[-1, 1]
        action = action.numpy()[0]              #获得的action变成矩阵。
        action=np.array(action_max - action_min) * (action) + action_min+ tf.random.uniform([1], -0.006, 0.006)
        # add noise
        #normal = Normal(0, 1)                   #生成normal这样一个正态分布
        #noise = normal.sample(action.shape) * explore_noise_scale       #在正态分布中抽样一个和action一样shape的数据，然后乘以scale
        # action = self.action_range * action      #action乘以动作的范围，加上noise

        return action.numpy()

    def sample_action(self,state):
        ''' generate random actions for exploration '''
        #a = tf.random.uniform([self.num_actions], -1, 1)
        aaa = tf.clip_by_value(tf.random.normal([1], state[-1], 3.5), action_min, action_max)
        return aaa.numpy()


class TD3_Trainer():

    def __init__(
            self, replay_buffer, hidden_dim, action_range, policy_target_update_interval=1, q_lr=3e-4, policy_lr=3e-4
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        # 用两个Qnet来估算，doubleDQN的想法。同时也有两个对应的target_q_net
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        # 把net 赋值给target_network
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        self.update_cnt = 0     #更新次数
        self.policy_target_update_interval = policy_target_update_interval      #策略网络更新频率

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)

    #在网络初始化的时候进行硬更新
    def target_ini(self, net, target_net):
        ''' hard-copy update for initializing target networks '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    #在更新的时候进行软更新
    def target_soft_update(self, net, target_net, soft_tau):
        ''' soft update the target net with Polyak averaging '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
                # 原来参数占比 + 目前参数占比
            )
        return target_net

    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        ''' update all networks in TD3 '''
        self.update_cnt += 1        #计算更新次数
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)     #从buffer sample数据

        reward = reward[:, np.newaxis]  # expand dim， 调整形状，方便输入网络
        done = done[:, np.newaxis]

        # 输入s',从target_policy_net计算a'。注意这里有加noisy的
        new_next_action = self.target_policy_net.evaluate(
            next_state, eval_noise_scale=eval_noise_scale
        )  # clipped normal noise

        # 归一化reward.(有正有负)
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        # 把s'和a'堆叠在一起，一起输入到target_q_net。
        # 有两个qnet，我们取最小值
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))

        #计算target_q的值，用于更新q_net
        #之前有把done从布尔变量改为int，就是为了这里能够直接计算。
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_input = tf.concat([state, action], 1)  # input of q_net

        #更新q_net1
        #这里其实和DQN是一样的
        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        #更新q_net2
        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))

        # Training Policy Function
        # policy不是经常updata的，而qnet更新一定次数，才updata一次
        if self.update_cnt % self.policy_target_update_interval == 0:
            #更新policy_net
            with tf.GradientTape() as p_tape:
                # 计算 action = Policy(s)，注意这里是没有noise的
                new_action = self.policy_net.evaluate(
                    state, eval_noise_scale=0.0
                )  # no noise, deterministic policy gradients

                #叠加state和action
                new_q_input = tf.concat([state, new_action], 1)
                # ''' implementation 1 '''
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                ''' implementation 2 '''
                predicted_new_q_value = self.q_net1(new_q_input)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)    #梯度上升
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            # 软更新target_network三个
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save_weights(self):  # save trained weights
        tl.files.save_npz(self.q_net1.trainable_weights, name='model_q_net1.npz')
        tl.files.save_npz(self.q_net2.trainable_weights, name='model_q_net2.npz')
        tl.files.save_npz(self.target_q_net1.trainable_weights, name='model_target_q_net1.npz')
        tl.files.save_npz(self.target_q_net2.trainable_weights, name='model_target_q_net2.npz')
        tl.files.save_npz(self.policy_net.trainable_weights, name='model_policy_net.npz')
        tl.files.save_npz(self.target_policy_net.trainable_weights, name='model_target_policy_net.npz')

    def load_weights(self):  # load trained weights
        tl.files.load_and_assign_npz(name='model_q_net1.npz', network=self.q_net1)
        tl.files.load_and_assign_npz(name='model_q_net2.npz', network=self.q_net2)
        tl.files.load_and_assign_npz(name='model_target_q_net1.npz', network=self.target_q_net1)
        tl.files.load_and_assign_npz(name='model_target_q_net2.npz', network=self.target_q_net2)
        tl.files.load_and_assign_npz(name='model_policy_net.npz', network=self.policy_net)
        tl.files.load_and_assign_npz(name='model_target_policy_net.npz', network=self.target_policy_net)



action_dim = 1  # 动作空间
state_dim = 120
action_max, action_min = 3.9575705208415273, -1.114517271633452

def train():
    # initialization of env
    # env = NormalizedActions(gym.make(ENV))
    env = SGTR_env(model=model, set_point=0.0739, train_datasets=train_data, mean=men_std[0], std=men_std[1])


    # initialization of buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # initialization of trainer
    td3_trainer = TD3_Trainer(replay_buffer, hidden_dim=hidden_dim,
                              policy_target_update_interval=policy_target_update_interval, \
                              action_range=action_range, q_lr=q_lr, policy_lr=policy_lr)

    # set train mode
    td3_trainer.q_net1.train()
    td3_trainer.q_net2.train()
    td3_trainer.target_q_net1.train()
    td3_trainer.target_q_net2.train()
    td3_trainer.policy_net.train()
    td3_trainer.target_policy_net.train()

    # training loop
    rewards = []  # 记录每个EP的总reward
    max_reward = -np.infty
    for frame_idx in range(max_frames):  # 小于最大步数，就继续训练
        t0 = time.time()
        state = env.reset()  # 初始化state
        state = state.astype(np.float32)  # 整理state的类型
        episode_reward = 0
        if frame_idx < 1:  # 第一次的时候，要进行初始化trainer
            print('intialize')
            _ = td3_trainer.policy_net(
                [state])  # need an extra call here to make inside functions be able to use model.forward
            _ = td3_trainer.target_policy_net([state])

        # 开始训练
        for step in range(max_steps):
            if frame_idx > 0:  # 如果小于500步，就随机，如果大于就用get-action
                action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)  # 带有noisy的action
            else:
                action = td3_trainer.policy_net.sample_action(state=state)

            # 与环境进行交互
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)
            done = 1 if done == True else 0

            # 记录数据在replay_buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # 赋值state，累计总reward，步数
            state = next_state
            episode_reward += reward

            # 如果数据超过一个batch_size的大小，那么就开始更新
            if len(replay_buffer) > batch_size:
                for i in range(update_itr):  # 注意：这里更新可以更新多次！
                    td3_trainer.update(batch_size, eval_noise_scale=0.5, reward_scale=1.)


            if done:
                break

        print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
              .format(frame_idx, 1000, episode_reward, time.time() - t0))
        rewards.append(episode_reward)
        if episode_reward>max_reward:
            max_reward=episode_reward
            td3_trainer.save_weights()
    np.save(r"rewards_recoard/" + get_str_time() + 'rewards.npy',rewards)

def test():
    from tqdm import  tqdm
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
                [state],is_train=False
            )  # need an extra call to make inside functions be able to use forward
            _ = td3_trainer.target_policy_net([state],is_train=False)

        for step in tqdm(range(max_steps)):
            action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)
            done = 1 if done == True else 0



            action_rec.append(action)
            # 与环境进行交互


            action = np.array(action).tolist()
            done = 1 if done == True else 0
            #print(reward)
            episode_reward += reward
            reward_sum.append(episode_reward)

            ori_temp_before_last = env.cal_origin_val(1, state[1])
            ori_temp_last = env.cal_origin_val(1, next_state[1])
            temp_rec.append(ori_temp_before_last)
            # 赋值state，累计总reward，步数
            state = next_state

            temp_change_speed = (ori_temp_last - ori_temp_before_last) * 3600
            speed_rec.append(temp_change_speed)
        np.save(r"./强化学习记录/speed_rec.npy",speed_rec)
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


def plot_reward():
    global env
    temp = []
    xlist = []
    for i in range(int(10000 / 0.01)):
        xlist.append(-5000 + 0.01 * i)

    for i in xlist:
        aaa = env.cal_reward(i)
        temp.append(aaa)
    plt.plot(xlist, temp)
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
    #test()