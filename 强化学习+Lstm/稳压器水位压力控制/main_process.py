import sys

sys.path.append("..")
from tensorflow import keras
import numpy as np
import random

import math

import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

import tensorlayer as tl

model = keras.models.load_model(r'M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\稳压器水位压力控制\稳压器模型\best_model_epoch22.hdf5')

data_path = r'M:\work\project_program\bishe\pyproject\RL_pid_control_critic_actor\强化学习+Lstm\稳压器水位压力控制\train_datasets' \
            r'.npy '
train_data = np.load(data_path, allow_pickle=True)
mean, std = train_data[1].astype('float64'), train_data[2].astype('float64')
train_data = train_data[0].astype('float64')
pass

def get_str_time():
    s = '_'
    for i in time.localtime(time.time())[0:6]:
        s += str(i) + '_'
    return s


class SGTR_env():
    def __init__(self, model, set_point, train_datasets, mean, std):
        self.action_space = np.array([0] * 4)
        self.observation_space = np.array([0] * 8)
        self.mean = mean
        self.std = std
        self.response = []
        self.set_point = set_point
        self.train_datasets = train_datasets
        self.model = model
        self.state = self.train_datasets[1500]
        self.step_count = 0  # 步数计数

    def reset(self):
        self.step_count = 0
        # begin_index = [2500,6500,11000,14500]
        begin_index = 1500 # range(1800, self.train_datasets.shape[0] - 3000, 100)
        state = self.train_datasets[begin_index] # random.sample(begin_index, 1)[0]]  # 10420]
        #  state = self.train_datasets[random.randint(0, self.train_datasets.shape[0]-1)]
        # 加入设定值信息
        self.state = np.array(state)
        return_state=np.array(state)
        return_state=return_state.reshape(1,80)
        return_state=np.append(return_state,np.array(self.set_point).reshape(1,2),axis=1)
        # 加入误差信息
        set_value = [self.set_point[0], self.set_point[1]]
        now_level = self.state[-1, 0]
        now_press = self.state[-1, 1]
        error = [set_value[0] - now_level, set_value[1] - now_press]
        return_state = np.append(return_state, np.array(error).reshape(1, 2), axis=1)

        return return_state

    def cal_origin_val(self, pos, now_val):
        """
        计算未归一化的值
        """
        val = now_val * self.std[pos] + self.mean[pos]
        return val

    def cal_changed_val(self, pos, now_val):
        """
        计算归一化后的值
        """
        val = (now_val - self.mean[pos]) / self.std[pos]
        return val

    def justice_down(self, next_state, step):
        """
        判断是否达到失败条件，deltaT<10或70分钟内未能实现一二回路压力平衡（小于1MP）
        """
        ori_deltaT = self.cal_origin_val(6, next_state[-1, 6])
        # ori_pressure = self.cal_origin_val(0,next_state[-1, 0])
        if ori_deltaT < 10:  # or (step>4200 and ori_pressure<1):
            return True

        else:
            return False

    def cal_reward(self, errors,states):
        level_k,level_k_1=states[-1][0],states[-2][0]
        press_k, press_k_1 = states[-1][1], states[-2][1]
        # 对达到设定值奖励
        b_1=2 if abs(errors[0]) < 0.001 else 0
        b_2=2 if abs(errors[1]) < 0.001 else 0
        reward_1=b_1+b_2

        # 对远离设定值惩罚
        c_1=-abs(errors[0])
        c_2=-abs(errors[1])
        reward_2=2*(c_1+c_2)

        # 对被控量剧烈变化惩罚
        d_1=-0.5*abs(level_k-level_k_1)
        d_2=-0.5*abs(press_k-press_k_1)
        reward_3=0  # d_1+d_2

        # 势函数，正态分布与RewardShaping进行比较
        # 正态分布
        mu, sigma = 0, 4
        f_1 = (1 / (math.sqrt(2 * math.pi) * sigma) * math.e ** (-(errors[0] - mu) ** 2 / (2 * sigma ** 2))) * 40
        f_2 = (1 / (math.sqrt(2 * math.pi) * sigma) * math.e ** (-(errors[1] - mu) ** 2 / (2 * sigma ** 2))) * 40
        # # RewardShaping
        # coef,rms=0.5,0.5
        # f_1 = coef * math.e **(-errors[0] ** 2 / (2 * rms))
        # f_2 = coef * math.e **(-errors[1] ** 2 / (2 * rms))
        reward_4= f_1+f_2

        reward=reward_1+reward_2

        return reward

    def step(self, action):
        self.step_count += 1
        self.state[-1, 4:] = action[:]
        # model(test_input, training=False)
        next_variable_state = np.array(self.model(np.array([self.state]), training=False))
        next_action = action
        zip_state_action = np.append(next_variable_state, next_action).reshape(1, -1)
        next_state = np.row_stack((self.state, zip_state_action))
        next_state = np.delete(next_state, 0, axis=0)
        self.state = next_state

        # level_set_value = self.cal_changed_val(0, self.set_point[0])
        # press_set_value = self.cal_changed_val(1, self.set_point[1])
        set_value = [self.set_point[0], self.set_point[1]]

        now_level = next_state[-1, 0]
        now_press = next_state[-1, 1]
        error = [set_value[0] - now_level, set_value[1] - now_press]

        reward = self.cal_reward(error,next_state)
        done = False  # self.justice_down(next_state, self.step_count)

        state=self.state
        return_state = np.array(state)
        return_state = return_state.reshape(1, 80)
        return_state = np.append(return_state, np.array(self.set_point).reshape(1, 2), axis=1)
        return_state = np.append(return_state, np.array(error).reshape(1, 2), axis=1)
        return return_state, reward, done, {}


#####################  hyper parameters  ####################
RANDOMSEED = 1  # random seed

EP_MAX = 500  # total number of episodes for training
EP_LEN = 200  # total number of steps for each episode
GAMMA = 0.9  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH = 32  # update batchsize
A_UPDATE_STEPS = 10  # actor update steps
C_UPDATE_STEPS = 10  # critic update steps
S_DIM, A_DIM = 84, 4  # state dimension, action dimension
EPS = 1e-8  # epsilon

# 注意：这里是PPO1和PPO2的相关的参数。
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty  PPO1
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better  PPO2
][1]  # choose the method for optimization


# METHOD=METHOD[1]
###############################  PPO  ####################################

class PPO(object):
    '''
    PPO 类
    '''

    def __init__(self):

        # 构建critic网络：
        # 输入state，输出V值
        tfs = tl.layers.Input([None, S_DIM], tf.float32, 'state')
        l1 = tl.layers.Dense(100, tf.nn.relu)(tfs)
        v = tl.layers.Dense(1)(l1)
        self.critic = tl.models.Model(tfs, v)
        self.critic.train()

        # 构建actor网络：
        # actor有两个actor 和 actor_old， actor_old的主要功能是记录行为策略的版本。
        # 输入是state，输出是描述动作分布的mu和sigma
        self.actor = self._build_anet('pi', trainable=True)
        self.actor_old = self._build_anet('oldpi', trainable=False)
        self.actor_opt = tf.optimizers.Adam(A_LR)
        self.critic_opt = tf.optimizers.Adam(C_LR)

    def a_train(self, tfs, tfa, tfadv):
        '''
        更新策略网络(policy network)
        '''
        # 输入时s，a，td-error。这个和AC是类似的。
        tfs = np.array(tfs, np.float32)         #state
        tfa = np.array(tfa, np.float32)         #action
        tfadv = np.array(tfadv, np.float32)     #td-error


        with tf.GradientTape() as tape:

            # 【敲黑板】这里是重点！！！！
            # 我们需要从两个不同网络，构建两个正态分布pi，oldpi。
            mu, sigma = self.actor(tfs)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, sigma_old = self.actor_old(tfs)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            # 在新旧两个分布下，同样输出a的概率的比值
            # 除以(oldpi.prob(tfa) + EPS)，其实就是做了import-sampling。怎么解释这里好呢
            # 本来我们是可以直接用pi.prob(tfa)去跟新的，但为了能够更新多次，我们需要除以(oldpi.prob(tfa) + EPS)。
            # 在AC或者PG，我们是以1,0作为更新目标，缩小动作概率到1or0的差距
            # 而PPO可以想作是，以oldpi.prob(tfa)出发，不断远离（增大or缩小）的过程。
            ratio = pi.prob(tfa) / (oldpi.prob(tfa) + EPS)
            # 这个的意义和带参数更新是一样的。
            surr = ratio * tfadv

            # 我们还不能让两个分布差异太大。
            # PPO1
            if METHOD['name'] == 'kl_pen':
                tflam = METHOD['lam']
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - tflam * kl))
            # PPO2：
            # 很直接，就是直接进行截断。
            else:  # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(ratio * tfadv,  #surr
                               tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * tfadv)
                )
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)

        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        '''
        更新actor_old参数。
        '''
        for p, oldp in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldp.assign(p)

    def c_train(self, tfdc_r, s):
        '''
        更新Critic网络
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32) #tfdc_r可以理解为PG中就是G，通过回溯计算。只不过这PPO用TD而已。

        with tf.GradientTape() as tape:
            v = self.critic(s)
            advantage = tfdc_r - v                  # 就是我们说的td-error
            closs = tf.reduce_mean(tf.square(advantage))

        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def cal_adv(self, tfs, tfdc_r):
        '''
        计算advantage，也就是td-error
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)           # advantage = r - gamma * V(s_)
        return advantage.numpy()

    def update(self, s, a, r):
        '''
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        '''
        s, a, r = s.astype(np.float32), a.astype(np.float32), r.astype(np.float32)

        self.update_old_pi()
        adv = self.cal_adv(s, r)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

        # update actor
        #### PPO1比较复杂:
        # 动态调整参数 adaptive KL penalty
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution

        #### PPO2比较简单，直接就进行a_train更新:
        # clipping method, find this is better (OpenAI's paper)
        else:
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv)

        # 更新 critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)

    def _build_anet(self, name, trainable):
        '''
        Build policy network
        :param name: name
        :param trainable: trainable flag
        :return: policy network
        '''
        # 连续动作型问题，输出mu和sigma。
        tfs = tl.layers.Input([None, S_DIM], tf.float32, name + '_state')
        l1 = tl.layers.Dense(100, tf.nn.relu, name=name + '_l1')(tfs)

        a = tl.layers.Dense(A_DIM, tf.nn.tanh, name=name + '_a')(l1)
        mu = tl.layers.Lambda(lambda x: x * 2, name=name + '_lambda')(a)

        sigma = tl.layers.Dense(A_DIM, tf.nn.softplus, name=name + '_sigma')(l1)

        model = tl.models.Model(tfs, [mu, sigma], name)

        if trainable:
            model.train()
        else:
            model.eval()
        return model

    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        s = s[np.newaxis, :].astype(np.float32)
        mu, sigma = self.actor(s)                   # 通过actor计算出分布的mu和sigma
        pi = tfp.distributions.Normal(mu, sigma)    # 用mu和sigma构建正态分布
        a = tf.squeeze(pi.sample(1), axis=0)[0]     # 根据概率分布随机出动作
        return np.clip(a, -2, 2)                    # 最后sample动作，并进行裁剪。

    def get_v(self, s):
        '''
        计算value值。
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]  # 要和输入的形状对应。
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5('model/ppo_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/ppo_actor_old.hdf5', self.actor_old)
        tl.files.save_weights_to_hdf5('model/ppo_critic.hdf5', self.critic)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/ppo_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/ppo_actor_old.hdf5', self.actor_old)
        tl.files.load_hdf5_to_weights_in_order('model/ppo_critic.hdf5', self.critic)





def train():
    env = SGTR_env(model, [1., -0.4], train_data, mean, std)

    # reproducible
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    ppo = PPO()

    all_ep_r = []
    max_rew=-np.infty
    # 更新流程：
    for ep in range(EP_MAX):
        env.set_point = [1., -0.4]
        s = env.reset()
        # if ep >= 500:
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        t0 = time.time()
        for t in range(EP_LEN):  # in one episode


            a = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)  # 对奖励进行归一化。有时候会挺有用的。所以我们说说，奖励是个主观的东西。
            s = s_
            ep_r += r

            # # N步更新的方法，每BATCH步了就可以进行一次更新
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                v_s_ = ppo.get_v(s_)  # 计算n步中最后一个state的v_s_

                # 和PG一样，向后回溯计算。
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                # 所以这里的br并不是每个状态的reward，而是通过回溯计算的V值
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)

        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)

        if ep_r > max_rew:
            ppo.save_ckpt()
            max_rew = ep_r

        print(
            'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                ep, EP_MAX, ep_r,
                time.time() - t0
            )
        )


    np.save("all_ep_r_改变设定值.npy", all_ep_r)

def test():
    env = SGTR_env(model, [-0.1, -0.2], train_data, mean, std)

    # reproducible
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    ppo = PPO()

    all_ep_r = []
    ppo.load_ckpt()
    while True:
        s = env.reset()
        for i in range(EP_LEN):
            s, r, done, _ = env.step(ppo.choose_action(s))
            if done:
                break


if __name__ == '__main__':

    train()
    #test()



