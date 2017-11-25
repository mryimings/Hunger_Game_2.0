import forest
import numpy as np
import random
import matplotlib.pyplot as plt
from forest import action_mapping

def Q_Learing(forest_env, num_episode=100000, gamma=0.95, lr=0.1, e=0.1, max_iter=200000, alpha=0.1):
    q = np.zeros((forest_env.cell_num, 5))
    max_survival_time = 0
    # draw pic
    max_survival_forPlot = np.zeros(num_episode) 
    episode_num_forPlot = np.arange(num_episode)
    average_survival_list = np.zeros(num_episode)
    average_survival = 100
    for num in range(num_episode):
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
            #print('iter',iter_times)
        max_survival_time = max(max_survival_time, iter_times)
        average_survival = (1-alpha)*average_survival + alpha*iter_times

        # draw pic
        # print (iter_times)
        max_survival_forPlot[num] = iter_times
        average_survival_list[num] = average_survival

    plt.subplot(221)
    plt.plot(episode_num_forPlot, max_survival_forPlot)
    plt.title(s='Max survival time')

    plt.subplot(222)
    plt.plot(episode_num_forPlot, average_survival_list)
    plt.title(s='Average survival time')

    plt.show()

    print("finished!")

    return q, max_survival_time


def e_greedy_pick(Q, state, e):
    if (random.random()>e):
        return np.argmax(Q[state])
    else:
        return random.randint(0,4)

if __name__ == '__main__':
    f = forest.Forest(row=10, col=10, mushroom=5, trap=1, tree=1, carnivore=1, disaster_p=0)
    forest.Forest.print_forest(f)
    model, max_survival_time = Q_Learing(f, num_episode=2500)
    print max_survival_time
    del f
    
