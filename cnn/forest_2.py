# -*- coding: utf-8 -*-

import random
import numpy as np

action_mapping = {0: 'up', 1: 'right', 2: 'down', 3: 'left', 4: 'stay'}

config_map = {
            'rewards' : {'mushroom' : 30,
                    'carniv' : -100,
                    'tree' : 20,
                    'nothing' : 0,
                    'trap' : -10},
            'value' : {'agent' : 7,
                    'mushroom' : 29,
                    'carniv' : -99,
                    'tree' : 19,
                    'nothing' : 0.5,
                    'trap' : -9},
            'amount' : {'mushroom' : 1500,
                    'carniv' : 30,
                    'tree' : 500,
                    'trap' : 0},
            'vision': 13,
            'mushroom_revive': 100,
            'carniv_th': 2,
            'carniv_giveup': 0.2
            }

class Forest_2():


    def __init__(self,
                map_size = [100, 100],
                map_para = config_map):

        self.width_map = map_size[0]
        self.height_map = map_size[1]

        self.rewards = map_para['rewards']
        self.value = map_para['value']
        self.amount = map_para['amount']

        self.vision = int(map_para['vision'])
        self.half_vision = int(self.vision/2)
        self.mushroom_revive_step = map_para['mushroom_revive']
        self.carniv_th = map_para['carniv_th']
        self.carniv_giveup = map_para['carniv_giveup']


        self.mushroom_pos = np.zeros((self.amount['mushroom'],2)).astype(int)
        self.tree_pos = np.zeros((self.amount['tree'],2)).astype(int)
        self.carniv_pos = np.zeros((self.amount['carniv'],2)).astype(int)
        self.trap_pos = np.zeros((self.amount['trap'],2)).astype(int)

        self.entire_map = self.value['nothing'] * np.ones((self.width_map, self.height_map))
        self.mushroom_map = np.zeros((self.width_map, self.height_map))
        self.mushroom_revive = np.ones((self.width_map, self.height_map))
        self.tree_map = np.zeros((self.width_map, self.height_map))
        self.trap_map = np.zeros((self.width_map, self.height_map))

        self.rand_env = np.arange(self.width_map*self.height_map)
        self.agent_pos = [np.random.randint(self.width_map),np.random.randint(self.height_map)]


        self.first_time = True
        self.init_env(refresh = True)


    def init_env(self, refresh = True):
        self.agent_pos = [np.random.randint(self.width_map),np.random.randint(self.height_map)]

        if (refresh or self.first_time):
            self.mushroom_map = np.zeros((self.width_map, self.height_map))


            self.first_time = False
            np.random.shuffle(self.rand_env)
            flag = 0
            for i in range(self.mushroom_pos.shape[0]):
                self.mushroom_pos[i][0] = int(self.rand_env[flag]/self.width_map)
                self.mushroom_pos[i][1] = int(self.rand_env[flag]%self.width_map)
                flag += 1
            for i in range(self.tree_pos.shape[0]):
                self.tree_pos[i][0] = int(self.rand_env[flag]/self.width_map)
                self.tree_pos[i][1] = int(self.rand_env[flag]%self.width_map)
                flag += 1
            for i in range(self.carniv_pos.shape[0]):
                self.carniv_pos[i][0] = int(self.rand_env[flag]/self.width_map)
                self.carniv_pos[i][1] = int(self.rand_env[flag]%self.width_map)
                flag += 1
            for i in range(self.trap_pos.shape[0]):
                self.trap_pos[i][0] = int(self.rand_env[flag]/self.width_map)
                self.trap_pos[i][1] = int(self.rand_env[flag]%self.width_map)
                flag += 1

        
        self.mushroom_map = np.zeros((self.width_map, self.height_map))
        self.mushroom_revive = []
        for i in range(self.mushroom_pos.shape[0]):
            x = self.mushroom_pos[i][0]
            y = self.mushroom_pos[i][1]
            self.mushroom_map[x][y] = self.value['mushroom'] - self.value['nothing']

        self.tree_map = np.zeros((self.width_map, self.height_map))
        for i in range(self.tree_pos.shape[0]):
            x = self.tree_pos[i][0]
            y = self.tree_pos[i][1]
            self.tree_map[x][y] = self.value['tree'] - self.value['nothing']

        self.trap_map = np.zeros((self.width_map, self.height_map))
        for i in range(self.trap_pos.shape[0]):
            x = self.trap_pos[i][0]
            y = self.trap_pos[i][1]
            self.trap_map[x][y] = self.value['trap'] - self.value['nothing']
            
        self.entire_map = self.value['nothing'] * np.ones((self.width_map, self.height_map))
        self.entire_map = np.add(self.entire_map, self.mushroom_map)
        self.entire_map = np.add(self.entire_map, self.tree_map)
        self.entire_map = np.add(self.entire_map, self.trap_map)
        
        self.map_nocarniv = self.entire_map
        
        self.carniv_map = np.zeros((self.width_map, self.height_map))
        for i in range(self.carniv_pos.shape[0]):
            x = self.carniv_pos[i][0]
            y = self.carniv_pos[i][1]
            self.entire_map[x][y] = self.value['carniv']
        return self.agent_pos


    def get_reward(self, action):
        self.move_forward(action)

        if (self.entire_map[self.agent_pos[0]][self.agent_pos[1]] == self.value['nothing']):
            reward = self.rewards['nothing']

        elif (self.entire_map[self.agent_pos[0]][self.agent_pos[1]] == self.value['mushroom']):
            reward = self.rewards['mushroom']
            self.mushroom_action(x = self.agent_pos[0],
                                y = self.agent_pos[1], 
                                eat = True)

        elif (self.entire_map[self.agent_pos[0]][self.agent_pos[1]] == self.value['carniv']):
            reward = self.rewards['carniv']

        elif (self.entire_map[self.agent_pos[0]][self.agent_pos[1]] == self.value['trap']):
            reward = self.rewards['trap']

        elif (self.entire_map[self.agent_pos[0]][self.agent_pos[1]] == self.value['tree']):
            #reward = 2 + np.random.randint(2*self.rewards['tree'])-self.rewards['tree']
            reward = np.random.randint(self.rewards['tree'])

        else:
            print(self.entire_map[self.agent_pos[0]][self.agent_pos[1]])
            raise ValueError

        return reward


    def carniv_action(self, agent_x, agent_y):
        # raise(NotImplementedError)
        for c in range(self.carniv_pos.shape[0]):
            cx = self.carniv_pos[c][0]
            cy = self.carniv_pos[c][1]
            dx = agent_x - cx
            dy = agent_y - cy
            
            if (((abs(dx) + abs(dy)) < self.carniv_th) and np.random.rand() > self.carniv_giveup):
                # near agent
                if(abs(dx)>abs(dy) and dy != 0 and self.map_nocarniv[cx][cy + int(dy/abs(dy))] == self.value['nothing']):
                    # move in y
                    self.carniv_pos[c][1] += int(dy/abs(dy))
                elif(dx != 0 and self.map_nocarniv[cx + int(dx/abs(dx))][cy] == self.value['nothing']):
                    # move in x
                    self.carniv_pos[c][0] += int(dx/abs(dx))
            else:
                # far away from agent
                act = np.random.randint(5)
                if(act==0):
                    if(cx < self.width_map-2):
                        if(self.map_nocarniv[cx+1][cy] == self.value['nothing']):
                            self.carniv_pos[c] = np.add(self.carniv_pos[c], [1,0])
                elif(act==1):
                    if(cx > 0):
                        if(self.map_nocarniv[cx-1][cy] == self.value['nothing']):
                            self.carniv_pos[c] = np.add(self.carniv_pos[c], [-1,0])
                elif(act==2):
                    if(cy < self.height_map-2):
                        if(self.map_nocarniv[cx][cy+1] == self.value['nothing']):
                            self.carniv_pos[c] = np.add(self.carniv_pos[c], [0,1])
                elif(act==3):
                    if(cy > 0):
                        if(self.map_nocarniv[cx][cy-1] == self.value['nothing']):
                            self.carniv_pos[c] = np.add(self.carniv_pos[c], [0,-1])
                elif(act==4):
                    self.carniv_pos[c] = np.add(self.carniv_pos[c], [0,0])
                else:
                    print(act)
                    raise ValueError
            self.carniv_pos[c] = np.maximum(self.carniv_pos[c],[0 ,0])
            self.carniv_pos[c] = np.minimum(self.carniv_pos[c],[self.height_map-1, self.width_map-1])
                 

    def mushroom_action(self, x = None, y = None, eat = False, fresh = False):
        if(eat):
            self.mushroom_revive.append([x, y, -self.mushroom_revive_step])
            self.map_nocarniv[x][y] = self.value['nothing']
        if(fresh):
            revive = False
            for i in range(len(self.mushroom_revive)):
                
                self.mushroom_revive[i][2] += 1
                if(self.mushroom_revive[i][2] == 0):
                    [x, y, _] = self.mushroom_revive[i]
                    self.map_nocarniv[x][y] = self.value['mushroom']
                    revive = True
            if(revive):
                [x, y, _] = self.mushroom_revive[0]


    def test(self):
        test_map = self.entire_map[5:7]
        test_s = self.get_observation()
        print("SUCCESS!")

        
    
    def move_forward(self, action):
        #pass
        #action_mapping = {0: 'up', 1: 'right', 2: 'down', 3: 'left', 4: 'stay'}
        self.carniv_action(self.agent_pos[0], self.agent_pos[1])
        
        if(action == 0):
            self.agent_pos = np.add(self.agent_pos, [-1, 0])
        elif(action == 1):
            self.agent_pos = np.add(self.agent_pos, [0, 1])
        elif(action == 2):
            self.agent_pos = np.add(self.agent_pos, [1, 0])
        elif(action == 3):
            self.agent_pos = np.add(self.agent_pos, [0, -1])
        elif(action == 4):
            self.agent_pos = np.add(self.agent_pos, [0, 0])
        else:
            print(action)
            raise ValueError

        self.agent_pos = np.maximum(self.agent_pos,[0 ,0])
        self.agent_pos = np.minimum(self.agent_pos,[self.height_map-1, self.width_map-1])
        
        self.mushroom_action(self, fresh = True)

        self.entire_map = self.map_nocarniv
        for i in range(self.carniv_pos.shape[0]):
            x = self.carniv_pos[i][0]
            y = self.carniv_pos[i][1]
            #self.carniv_map[x][y] = self.value['carniv'] - self.value['nothing']
            self.entire_map[x][y] = self.value['carniv']
        
        
    def print_entire_map(self):
        row = ''
        for i in range(self.entire_map.shape[0]):
            for j in range(self.entire_map.shape[1]):
                if (self.entire_map[i][j] == self.value['nothing']):
                    row += '\033[0m■' + ' '
                elif (self.entire_map[i][j] == self.value['mushroom']):
                    row += '\033[33;1m■' + ' '
                elif (self.entire_map[i][j] == self.value['tree']):
                    row += '\033[32;1mT' + ' '
                elif (self.entire_map[i][j] == self.value['carniv']):
                    row += '\033[31;1mC' + ' '
                elif (self.entire_map[i][j] == self.value['trap']):
                    row += '\033[91mW' + ' '
                else:
                    print(self.entire_map[i][j])
                    raise ValueError
                # white
                # row += '\033[0m■' + ' '
                # blue
                # row += '\033[94m■' + ' '
                # red
                # row += '\033[91m■' + ' '
                # yellow
                # row += '\033[93m■' + ' '
            print(row)
            row = ''
        print(row)


    def print_observation(self):
        observation = self.get_observation()
        row = ''
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                if (observation[i][j] == self.value['nothing']):
                    row += '\033[0m■' + ' '
                elif (observation[i][j] == self.value['mushroom']):
                    row += '\033[33;1m■' + ' '
                elif (observation[i][j] == self.value['tree']):
                    row += '\033[32;1mT' + ' '
                elif (observation[i][j] == self.value['carniv']):
                    row += '\033[31;1mC' + ' '
                elif (self.entire_map[i][j] == self.value['trap']):
                    row += '\033[91mW' + ' '
                else:
                    row += '\033[94mX' + ' '
            print(row)
            row = ''
        print(row)



    def get_observation(self):
        [x, y] = self.agent_pos
        padded_map = np.array(self.entire_map)
        padded_map[x][y] += self.value['agent']
        padded_map = np.lib.pad(padded_map, ((self.half_vision, self.half_vision), (self.half_vision, self.half_vision)), 'constant', constant_values = 0)
        x1 = int(x)
        x2 = int(x + 2*self.half_vision + 1)
        y1 = int(y)
        y2 = int(y + 2*self.half_vision + 1)
        return padded_map[x1:x2,y1:y2]



if __name__ == "__main__":
    env = Forest_2()
    env.test()
    #a=int(-2/abs(2))
    #a=np.array([1,2,3])
    #a=np.max(a,1)
    #print(np.random.randint(5))