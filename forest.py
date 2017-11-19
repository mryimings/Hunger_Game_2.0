import random

action_mapping = {0: 'up', 1: 'right', 2: 'down', 3: 'left', 4: 'stay'}


class Forest:
    def __init__(self, row=25, col=20, tree=25, trap=25, mushroom=25, carnivore=25, disaster_p=0.001, rewards=None):

        self.row_num = row
        self.col_num = col
        self.cell_num = row * col
        self.cells = []
        self.tree_num = tree
        self.trap_num = trap
        self.mushroom_num = mushroom
        self.carnivore_num = carnivore
        self.mushroom_states = {}
        self.carnivores = {}  # key is initial position for each carnivore, and value is current position
        self.curr_carnivores = {}  # stores the current position for each carnivore, for better search performance
        self.disaster_area = set()
        self.disaster_last_time = 0
        self.disaster_probability = disaster_p

        self.rewards_blank = rewards.get("blank", -1)
        self.rewards_tree_max = rewards.get("tree_max", 20)
        self.tree_punishment = rewards.get("tree_punishment", -10)
        self.rewards_trap = rewards.get("trap", -100)
        self.rewards_mushroom = rewards.get("mushroom", 50)
        self.mushroom_refresh = rewards.get("mushroom_refresh", 100)
        self.rewards_carnivores = rewards.get("carnivores", -1000)
        self.rewards_disaster = rewards.get("disaster", -10)

        for i in range(self.cell_num):
            curr_cell = {'actions': {}, 'attribute': 'blank'}

            # The cells looks like:
            # 0, 1, 2, 3
            # 4, 5, 6, 7
            # 8, 9, 10, 11

            # up
            curr_cell['actions']['up'] = i if i < self.col_num else i - self.col_num

            # right
            curr_cell['actions']['right'] = i if i % self.col_num == self.col_num - 1 else i + 1

            # down
            curr_cell['actions']['down'] = i if i >= self.cell_num - self.col_num else i + self.col_num

            # right
            curr_cell['actions']['left'] = i if i % self.col_num == 0 else i - 1

            # stay
            curr_cell['actions']['stay'] = i

            self.cells.append(curr_cell)

        self.init_tree()
        self.init_trap()
        self.init_mushroom()
        self.init_carnivore()

    def init_tree(self):
        count_tree = 0
        while count_tree < self.tree_num:
            i = random.randint(0, len(self.cells) - 1)
            if self.cells[i]['attribute'] == 'blank':
                self.cells[i]['attribute'] = 'tree'
                count_tree += 1

    def init_trap(self):
        count_trap = 0
        while count_trap < self.trap_num:
            i = random.randint(0, len(self.cells) - 1)
            if self.cells[i]['attribute'] == 'blank':
                self.cells[i]['attribute'] = 'trap'
                count_trap += 1

    def init_mushroom(self):
        count_mushroom = 0
        while count_mushroom < self.mushroom_num:
            i = random.randint(0, len(self.cells) - 1)
            if self.cells[i]['attribute'] == 'blank':
                self.cells[i]['attribute'] = 'mushroom'
                count_mushroom += 1
                self.mushroom_states[i] = 0

    def init_carnivore(self):
        while len(self.carnivores) < self.carnivore_num:
            i = random.randint(0, len(self.cells))
            if i not in self.carnivores and self.cells[i]['attribute'] == 'blank':
                self.carnivores[i] = i
                self.curr_carnivores[i] = 1

    def get_reward(self, position):
        curr_att = self.cells[position]['attribute']
        ret_reward = 0
        if curr_att == 'blank':
            ret_reward += self.rewards_blank
        elif curr_att == 'tree':
            ret_reward += (random.randint(0, self.rewards_tree_max) +
                           self.tree_punishment)
        elif curr_att == 'trap':
            ret_reward += self.rewards_trap
        elif curr_att == 'mushroom':
            if self.mushroom_states[position] <= 0:
                ret_reward += self.rewards_mushroom
                self.mushroom_states[position] = self.mushroom_refresh
            else:
                ret_reward += -1

        if position in self.curr_carnivores:
            ret_reward += self.rewards_carnivores * self.curr_carnivores[position]

        if position in self.disaster_area:
            ret_reward += self.rewards_disaster

        return ret_reward

    def random_pick_action(self):
        return action_mapping[random.randint(0, 4)]

    # judge if the carnivore is too far from its initial position
    def is_too_far(self, p1, p2, distance=3):
        if abs(p1 / self.col_num - p2 / self.col_num) + abs(p1 % self.col_num - p2 % self.col_num) >= distance:
            return True
        else:
            return False

    def env_move_forward(self):
        for mushroom_pos in self.mushroom_states:
            if self.mushroom_states[mushroom_pos] > 0:
                self.mushroom_states[mushroom_pos] -= 1

        for carn_init_pos in self.carnivores:
            carn_curr_pos = self.carnivores[carn_init_pos]
            self.curr_carnivores[carn_curr_pos] -= 1
            if self.curr_carnivores[carn_curr_pos] == 0:
                self.curr_carnivores.pop(carn_curr_pos)

            # if a carnivore goes too far, it will come back
            if self.is_too_far(p1=carn_init_pos, p2=carn_curr_pos):
                if carn_curr_pos / self.col_num - carn_init_pos / self.col_num < 0:
                    self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions']['down']
                elif carn_curr_pos / self.col_num - carn_init_pos / self.col_num > 0:
                    self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions']['up']
                elif carn_curr_pos % self.col_num - carn_init_pos % self.col_num > 0:
                    self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions']['right']
                elif carn_curr_pos % self.col_num - carn_init_pos % self.col_num < 0:
                    self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions']['left']
                else:
                    print ("how can this be happening!")
                    # self.carnivores[carn_init_pos] = carn_init_pos
            else:
                self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions'][self.random_pick_action()]

            if self.carnivores[carn_init_pos] not in self.curr_carnivores:
                self.curr_carnivores[self.carnivores[carn_init_pos]] = 0
            self.curr_carnivores[self.carnivores[carn_init_pos]] += 1

        if self.disaster_last_time >= 0:
            self.disaster_last_time -= 1
            if self.disaster_last_time == 0:
                self.disaster_area = set()
        else:
            ran = random.random()
            if ran < self.disaster_probability:
                random_pos = random.randint(0, len(self.cells))
                for i in range(len(self.cells)):
                    if not self.is_too_far(i, random_pos):
                        self.disaster_area.add(i)

                self.disaster_last_time = 100

    def print_forest(self):
        for i in range(len(self.cells)):
            if i in self.curr_carnivores:
                print 'carnivore',
            else:
                print self.cells[i]['attribute'],
            if i % self.col_num == self.col_num - 1:
                print('\n')

    def re_initialize(self):
        for mushroom_pos in self.mushroom_states:
            self.mushroom_states[mushroom_pos] = 0

        self.disaster_area = set()
        self.disaster_last_time = 0

        self.curr_carnivores = {}
        for carn in self.carnivores:
            self.carnivores[carn] = carn
            self.curr_carnivores[carn] = 1



