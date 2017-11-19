import forest
import numpy as np
import random
import matplotlib.pyplot as plt
from forest import action_mapping

def Q_Learing(forest_env, num_episode=100000, gamma=0.95, lr=0.1, e=0.1, max_iter=200000):
    q = np.zeros((forest_env.cell_num, 5))
    max_suvival_time = 0
    # draw pic
    x = np.zeros(num_episode)
    y = np.arange(num_episode)
    for num in range(num_episode):
        forest_env.re_initialize()
        pos = random.randint(0, forest_env.cell_num - 1)
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
        max_suvival_time = max(max_suvival_time, iter_times)

        #draw pic
        print (iter_times)
        x[num] = iter_times

    plt.plot(y , x)
    plt.show()

    return q, max_suvival_time


def e_greedy_pick(Q, state, e):
    if (random.random()>e):
        return np.argmax(Q[state])
    else:
        return random.randint(0,4)

if __name__ == '__main__':
    f= forest.Forest(row=7, col=7, mushroom=3, trap=0, tree=0, carnivore=0, disaster_p=0)
    forest.Forest.print_forest(f)
    model, max_survival_time = Q_Learing(f, num_episode=1000)
    print max_survival_time