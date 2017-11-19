import forest
import numpy
import random
from forest import action_mapping


def Q_Learing(forest_env, num_episode=100, gamma=0.95, lr=0.1, e=0.1, max_iter=200000):
    q = numpy.zeros((forest_env.cell_num, 5))
    max_survival_time = 0
    for num in range(num_episode):
        pos = random.randint(0, forest_env.cell_num - 1)
        iter_times = 0
        total_reward = 100
        while iter_times < max_iter and total_reward > 0:
            action = e_greedy_pick(q, pos, e)
            next_pos = forest_env.cells[pos]['actions'][action_mapping[action]]
            reward = forest_env.get_reward(next_pos)
            total_reward += reward
            q[pos][action] = (1-lr)*q[pos][action] + lr * (reward + gamma * max(q[next_pos]))
            forest_env.env_move_forward()
            pos = next_pos
            #print pos, total_reward
            iter_times += 1
            pos = next_pos
            forest_env.print_map(pos)
        max_survival_time = max(max_survival_time, iter_times)
        if max_survival_time == max_iter:
            break
    return q, max_survival_time


def e_greedy_pick(Q, state, e):
    if (random.random() > e):
        return numpy.argmax(Q[state])
    else:
        return random.randint(0, 4)


if __name__ == '__main__':

    rewards = {"blank":-1,
            "tree_max":20,
            "tree_punishment":-10,
            "trap":-100,
            "mushroom":60,
            "mushroom_refresh":100,
            "carnivores":-1000,
            "disaster":-10
            }

    f = forest.Forest(row=10, col=10, mushroom=4, trap=0, tree=0, carnivore=0, disaster_p=0, rewards=rewards)
    #f.print_map()
    model, max_survival_time = Q_Learing(f, num_episode=10000)
    print(max_survival_time)
