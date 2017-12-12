import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from forest_2 import *
from dqn import *

config_map = {
            'rewards' : {'mushroom' : 10,
                    'carniv' : -100,
                    'tree' : 5,
                    'nothing' : -0.5,
                    'trap' : -5},
            'value' : {'agent' : 7,
                    'mushroom' : 29,
                    'carniv' : -99,
                    'tree' : 19,
                    'nothing' : 0.5,
                    'trap' : -4},
            'amount' : {'mushroom' : 2000,
                    'carniv' : 90,
                    'tree' : 1000,
                    'trap' : 200},
            'vision': 13,
            'mushroom_revive': 100,
            'carniv_th': 4,
            'carniv_giveup': 0.3
            }

features = config_map['vision']

env = Forest_2(map_size = [100, 100],
               map_para = config_map)
env.test()

RL = DeepQNet(n_actions=5,
            n_features=features,
            learning_rate=0.01,
            reward_decay=0.9 ,
            e_greedy=0.1,
            replace_target_iter=200,
            memory_size=2000,
            useConvNet = False
            # output_graph=True
            )
with tf.Session() as sess:
    #saver = tf.train.Saver()
    #sess.run(tf.global_variables_initializer())
    train(env, RL, episode = 30000, vision = config_map['vision'])
    #saver.save(sess, 'model/{}'.format('c64@3-c64@3-d100-d100-d100_vi13_epi300_Adam'))
    #env.mainloop()
    #RL.plot_cost()