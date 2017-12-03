# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from forest import *
import time
import sys
import matplotlib.pyplot as plt


class DeepQNet:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.1,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False):

        self.useConvNet = False;

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        if (self.useConvNet):
            self._build_conv_net()
        else:
            # consist of [target_net, evaluate_net]
            self._build_net()

        # target net (using fixed q-target method)
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        # eval net
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        # update the paras in the target net
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    # TODO
    # Use Conv Net
    def _build_conv_net(self):
        raise (NotImplementedError)
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def choose_action_for_test(self, observation):

        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        print(type(actions_value))
        print(actions_value)
        return pick_based_on_possibility(actions_value[0])


    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })
        if (int(self.learn_step_counter % 500 == 0)):
            print('%d,cost:%d' % (self.learn_step_counter, cost))

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def train(env, net, episode_num=500):
    step = 0
    log_gameTime = []
    for episode in range(episode_num):
        print('No.%d episode!' % episode)
        # initial observation
        pos = random.randint(0, env.cell_num - 1)
        observation = env.get_observation(pos)
        gameTime = 0

        # refresh env
        env.re_initialize()

        while True:
            gameTime += 1

            # RL choose action based on observation
            action = net.choose_action(observation)

            # RL take action and get next observation and reward
            # observation_, reward, done = env.step(action)
            next_pos = env.cells[pos]['actions'][action_mapping[action]]
            reward = env.get_reward(next_pos)
            observation_ = env.get_observation(next_pos)

            net.store_transition(observation, action, reward, observation_)

            if (step > 200):
                net.learn()

            # swap observation
            observation = observation_
            pos = next_pos
            # break while loop when end of this episode
            if gameTime > 40000:
                print('!')
                log_gameTime.append(gameTime)
                gameTime = 0
                break
            step += 1
            env.env_move_forward()

    # end of game
    print('game over')
    plt.plot(np.arange(len(log_gameTime)), log_gameTime)
    plt.ylabel('Game Time')
    plt.xlabel('Episode')
    plt.show()
    # env.destroy()


def test(env, net, episode_num=1):
    log_gameTime = []
    for episode in range(episode_num):
        print('No.%d episode!' % episode)
        # initial observation
        pos = random.randint(0, env.cell_num - 1)
        observation = env.get_observation(pos)
        total_rewards = 500

        env.re_initialize()

        env.print_map(pos)

        while True:

            action = net.choose_action_for_test(observation)

            next_pos = env.cells[pos]['actions'][action_mapping[action]]
            reward = env.get_reward(next_pos)
            total_rewards += reward
            observation_ = env.get_observation(next_pos)

            net.store_transition(observation, action, reward, observation_)

            observation = observation_
            pos = next_pos
            env.env_move_forward()

            env.print_map(pos)
            time.sleep(0.5)

            if total_rewards <= 0:
                print('!')

                break

    # end of game
    # print('game over')
    # plt.plot(np.arange(len(log_gameTime)), log_gameTime)
    # plt.ylabel('Game Time')
    # plt.xlabel('Episode')
    # plt.show()
    # env.destroy()



def pick_based_on_possibility(l):
    min_ele = min(l)
    li = []
    sum = 0
    for x in l:
        li.append(x - min_ele)
    for x in li:
        sum += x
    for i in range(len(li)):
        li[i] = li[i] / sum + (li[i - 1] if i > 0 else 0)
    # print(li)
    rand = random.random()
    for i in range(len(li)):
        if li[i] >= rand:
            return i

    return -1


if __name__ == "__main__":
    env = Forest(row=7, col=7, mushroom=5, trap=2, tree=2, carnivore=1, disaster_p=0)
    RL = DeepQNet(n_actions=5,
                  n_features=env.cell_num,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.1,
                  replace_target_iter=200,
                  memory_size=2000,
                  # output_graph=True
                  )
    train(env, RL, episode_num=1)
    test(env, RL)
    RL.plot_cost()
