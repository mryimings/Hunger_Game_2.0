# -*- coding:utf-8 -*-
from __future__ import print_function
import forest
import numpy as np
import random
import matplotlib.pyplot as plt
from forest import action_mapping


def Q_Learing(forest_env, num_episode=100000, gamma=0.95, lr=0.1, e=0.1, max_iter=2000000, alpha=0.1):
    q = np.zeros((forest_env.cell_num, 5))
    with open('/Users/yiming/Documents/ReinforceLearning/final_proj/logs/log.txt', 'w') as log_file:
        max_survival_time = 0
        # draw pic
        max_survival_forPlot = np.zeros(num_episode)
        episode_num_forPlot = np.arange(num_episode)
        average_survival_list = np.zeros(num_episode)
        average_survival = 100
        for num in range(num_episode):
            log_file.write('----------------------------------------------------------\n')
            log_file.write('The log of episode '+str(num)+' is as below:\n')
            log_file.write('0,100|')
            forest_env.re_initialize()
            pos = 0
            iter_times = 0
            total_reward = 100
            while iter_times < max_iter and total_reward > 0:
                action = e_greedy_pick(q, pos, e)
                next_pos = forest_env.cells[pos]['actions'][action_mapping[action]]
                reward = forest_env.get_reward(next_pos)
                total_reward += reward
                #print (total_reward)
                q[pos][action] = (1-lr)*q[pos][action] + lr * (reward + gamma * max(q[next_pos]))
                forest_env.env_move_forward()
                iter_times += 1
                pos = next_pos
                log_file.write(str(pos)+','+str(total_reward)+'|')
                #print('iter',iter_times)
            max_survival_time = max(max_survival_time, iter_times)
            average_survival = (1-alpha)*average_survival + alpha*iter_times
            log_file.write('\n')
            # draw pic
            # print (iter_times)
            max_survival_forPlot[num] = iter_times
            average_survival_list[num] = average_survival
            print(num, iter_times)

    # plt.subplot(221)
    # plt.plot(episode_num_forPlot, max_survival_forPlot)
    # plt.title(s='Max survival time')
    #
    # plt.subplot(222)
    # plt.plot(episode_num_forPlot, average_survival_list)
    # plt.title(s='Average survival time')
    #
    # plt.show()

    return q, max_survival_time


def e_greedy_pick(Q, state, e):
    if (random.random()>e):
        return np.argmax(Q[state])
    else:
        return random.randint(0,4)


def render_episode(q, forest_env):
    pass




policy_map = {0: 'U', 1:'R', 2: 'D', 3: 'L', 4: 'S'}


def find_optimal_action(l):
    max_index = 0
    max_element = l[0]
    for i in range(len(l)):
        if l[i] > max_element:
            max_element = l[i]
            max_index = i
    return policy_map[max_index]


def print_model(q, col):
    for i in range(0, len(q)):
        print(str(i)+':'+find_optimal_action(q[i]), end=' ')
        if i % col == col - 1:
            print('')



if __name__ == '__main__':
    pass
    f = forest.Forest(row=5, col=5, mushroom=5, trap=1, tree=1, carnivore=1, disaster_p=0)
    f.print_map(0)

    model, max_survival_time = Q_Learing(f, num_episode=5000, e=0.2, lr=0.1, gamma=0.7, max_iter=100000)
    print_model(model, f.col_num)
    # print max_survival_time
    del f
