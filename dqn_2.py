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
            output_graph=False,
            useConvNet = False):

        self.useConvNet = useConvNet;

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
        self.memory = np.zeros((self.memory_size, (n_features**2) * 2 + 2))

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
        self.s = tf.placeholder(tf.float32, [None, self.n_features, self.n_features, 1], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features, self.n_features, 1], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action


        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)


        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')


        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='target')


        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            #self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            #self._train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    # TODO
    # Use Conv Net
    def _build_conv_net(self):
        #raise(NotImplementedError)
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features, self.n_features, 1], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features, self.n_features, 1], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action


        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)


        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            
            ############################################CONV1##########################################
            ew1_shape = [3, 3, 1, 32]
            eweight1 = tf.get_variable(name='conv_kernel_1', shape=ew1_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=50))
            eb1_shape = [32]
            ebias1 = tf.get_variable(name='conv_bias_1', shape=eb1_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=50))
            epooling1_shape = [1, 2, 2, 1]
            
            ec1 = conv_out = tf.nn.conv2d(self.s, eweight1, strides=[1, 1, 1, 1], padding="SAME")
            ecr1 = tf.nn.relu(ec1 + ebias1)
            epl1 = tf.nn.max_pool(ecr1, strides=epooling1_shape,ksize=epooling1_shape, padding="SAME")
            ############################################CONV2##########################################
            ew2_shape = [3, 3, 32, 32]
            eweight2 = tf.get_variable(name='conv_kernel_2', shape=ew2_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=50))
            eb2_shape = [32]
            ebias2 = tf.get_variable(name='conv_bias_2', shape=eb2_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=50))
            epooling2_shape = [1, 2, 2, 1]
            
            ec2 = conv_out = tf.nn.conv2d(epl1, eweight2, strides=[1, 1, 1, 1], padding="SAME")
            ecr2 = tf.nn.relu(ec2 + ebias2)
            epl2 = tf.nn.max_pool(ecr2, strides=epooling2_shape,ksize=epooling2_shape, padding="SAME")
            ############################################DENSE##########################################
            es1=epl2.get_shape()
            evector_length = es1[1].value * es1[2].value * es1[3].value
            eflatten = tf.reshape(epl2, shape=[-1, evector_length])
            
            e1 = tf.layers.dense(eflatten, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')


        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            ############################################CONV1##########################################
            tw1_shape = [3, 3, 1, 32]
            tweight1 = tf.get_variable(name='conv_kernel_1', shape=tw1_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=50))
            tb1_shape = [32]
            tbias1 = tf.get_variable(name='conv_bias_1', shape=tb1_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=50))
            tpooling1_shape = [1, 2, 2, 1]
            
            tc1 = conv_out = tf.nn.conv2d(self.s_, tweight1, strides=[1, 1, 1, 1], padding="SAME")
            tcr1 = tf.nn.relu(tc1 + tbias1)
            tpl1 = tf.nn.max_pool(tcr1, strides=tpooling1_shape,ksize=tpooling1_shape, padding="SAME")
            ############################################CONV2##########################################
            tw2_shape = [3, 3, 32, 32]
            tweight2 = tf.get_variable(name='conv_kernel_2', shape=tw2_shape,
                                     initializer=tf.glorot_uniform_initializer(seed=50))
            tb2_shape = [32]
            tbias2 = tf.get_variable(name='conv_bias_2', shape=tb2_shape,
                                   initializer=tf.glorot_uniform_initializer(seed=50))
            tpooling2_shape = [1, 2, 2, 1]
            
            tc2 = conv_out = tf.nn.conv2d(tpl1, tweight2, strides=[1, 1, 1, 1], padding="SAME")
            tcr2 = tf.nn.relu(tc2 + tbias2)
            tpl2 = tf.nn.max_pool(tcr2, strides=tpooling2_shape,ksize=tpooling2_shape, padding="SAME")
            ############################################DENSE##########################################
            ts1=tpl2.get_shape()
            tvector_length = ts1[1].value * ts1[2].value * ts1[3].value
            tflatten = tf.reshape(tpl2, shape=[-1, tvector_length])
            
            t1 = tf.layers.dense(tflatten, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='target')


        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            #self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            #self._train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

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
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation.reshape(1,self.n_features,self.n_features,1)})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action


    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            #print('\ntarget_params_replaced\n')


        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        x=batch_memory.shape[0]
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features**2].reshape(x,self.n_features,self.n_features,1),
                self.a: batch_memory[:, self.n_features**2],
                self.r: batch_memory[:, self.n_features**2 + 1],
                self.s_: batch_memory[:, -self.n_features**2:].reshape(x,self.n_features,self.n_features,1),
            })
        if ( int(self.learn_step_counter % 5000 == 0) ):
            print('%d,cost:%d' % (self.learn_step_counter, cost))

        self.cost_his.append(cost)


        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def plot_cost(self):
        cost_his_=np.array(self.cost_his)
        for i in range(len(cost_his_)):
            if (cost_his_[i]>5000):
                cost_his_[i] = 5000
        plt.plot(np.arange(len(cost_his_)), cost_his_)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def train(env, net, episode = 500, vision = 13):
    step = 0
    log_gameTime = []
    for episode in range(episode):
        print('No.%d episode!' % (episode+1))
        # initial observation
        pos = env.init_env(refresh = True)
        observation = env.get_observation().flatten()
        #observation = np.reshape(observation, (vision, vision, 1))
        total_rewards = 1000
        gameTime = 0

        # refresh env

        while True:
            gameTime += 1
            
            # RL choose action based on observation
            action = net.choose_action(observation)

            # RL take action and get next observation and reward
            #observation_, reward, done = env.step(action)
            reward = env.get_reward(action)
            total_rewards += reward
            observation_ = env.get_observation().flatten()
            #observation_ = np.reshape(observation_, (vision, vision, 1))
            #print(observation.shape)
            net.store_transition(observation, action, reward, observation_)

            #if (step > 200) and (step % 5 == 0):
            if (step > 200):
                net.learn()

            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if (total_rewards < 0) or (gameTime > 100000):
                #print('!')
                log_gameTime.append(gameTime)
                gameTime = 0
                total_rewards = 500
                break
            step += 1


    # end of game
    print('game over')
    plt.plot(np.arange(len(log_gameTime)), log_gameTime)
    plt.ylabel('Game Time')
    plt.xlabel('Episode')
    plt.show()
    
    average_time = []
    accumulator = 0
    count = 0
    for i in range(len(log_gameTime)):
        count += 1
        accumulator += log_gameTime[i]
        average_time.append(accumulator/count)
    plt.plot(np.arange(len(average_time)), average_time)
    plt.ylabel('Average Game Time')
    plt.xlabel('Episode')
    plt.show()
        
        
    
    #env.destroy()


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
    train(env, RL)
    #env.mainloop()
    RL.plot_cost()