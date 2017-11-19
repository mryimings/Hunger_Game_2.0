import forest
import numpy
import random
from forest import action_mapping

def Q_Learing(forest_env, num_episode=100000, gamma=0.95, lr=0.1, e=0.1, max_iter=200000):
    q = numpy.zeros((forest_env.cell_num, 5))
    max_suvival_time = 0
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
            print pos, total_reward
            iter_times += 1
        max_suvival_time = max(max_suvival_time, iter_times)
    return q, max_suvival_time


def e_greedy_pick(Q, state, e):
    if random.random() < e:
        return numpy.argmax(Q[state])
    else:
        return random.randint(0, 4)

if __name__ == '__main__':
    f = forest.Forest(row=5, col=5, mushroom=3, trap=0, tree=0, carnivore=0, disaster_p=0)
    model, max_survival_time = Q_Learing(f, num_episode=10)
    print max_survival_time
    del f