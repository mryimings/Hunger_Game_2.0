import random


class Forest:
    def __init__(self, row=25, col=20, tree=25, trap=25, mushroom=25, carnivore=25, disaster_p=0.001):

        self.row_num = row
        self.col_num = col
        self.cell_num = row*col
        self.cells = []
        self.tree_num = tree
        self.trap_num = trap
        self.mushroom_num = mushroom
        self.carnivore_num = carnivore
        self.mushroom_states = {}
        self.carnivores = {}
        self.curr_carnivores = set()
        self.disaster_area = set()
        self.disaster_last_time = 0
        self.disaster_probability = disaster_p

        for i in range(self.cell_num):

            curr_cell = {'actions':{}, 'attribute': 'blank'}

            # The cells looks like:
            # 0, 1, 2, 3
            # 4, 5, 6, 7
            # 8, 9, 10, 11

            #up
            curr_cell['actions']['up'] = i if i < self.col_num else i - self.col_num

            #right
            curr_cell['actions']['right'] = i if i % self.col_num == self.col_num-1 else i+1

            #down
            curr_cell['actions']['down'] = i if i >= self.cell_num - self.col_num else i + self.col_num

            #right
            curr_cell['actions']['left'] = i if i % self.col_num == 0 else i-1

            #stay
            curr_cell['actions']['stay'] = i

            self.cells.append(curr_cell)

        self.init_tree()
        self.init_trap()
        self.init_mushroom()
        self.init_carnivore()



    def init_tree(self):
        count_tree = 0
        while count_tree < self.tree_num:
            i = random.randint(0, len(self.cells)-1)
            if self.cells[i]['attribute'] == 'blank':
                self.cells[i]['attribute'] = 'tree'
                count_tree += 1

    def init_trap(self):
        count_trap = 0
        while count_trap < self.trap_num:
            i = random.randint(0, len(self.cells)-1)
            if self.cells[i]['attribute'] == 'blank':
                self.cells[i]['attribute'] = 'trap'
                count_trap += 1

    def init_mushroom(self):
        count_mushroom = 0
        while count_mushroom < self.mushroom_num:
            i = random.randint(0, len(self.cells)-1)
            if self.cells[i]['attribute'] == 'blank':
                self.cells[i]['attribute'] = 'mushroom'
                count_mushroom += 1
                self.mushroom_states[i] = 0

    def init_carnivore(self):
        while len(self.carnivores) < self.carnivore_num:
            i = random.randint(0, len(self.cells))
            if i not in self.carnivores:
                self.carnivores[i] = i
                self.curr_carnivores.add(i)

    def get_reward(self, position):
        curr_att = self.cells[position]['attribute']
        ret_reward = 0
        if curr_att == 'blank':
            ret_reward += -1
        elif curr_att == 'tree':
            ret_reward += (random.randint(0,20)-10)
        elif curr_att == 'trap':
            ret_reward += -100
        elif curr_att == 'mushroon':
            if self.mushroom_states[position] >= 0:
                ret_reward += 50
                self.mushroom_states[position] = 100
            else:
                ret_reward += -1

        if position in self.curr_carnivores:
            ret_reward += -100

        if position in self.disaster_area:
            ret_reward += -10

        return ret_reward

    def random_pick_action(self):
        i = random.randint(0, 4)
        if i == 0:
            return 'up'
        elif i == 1:
            return 'right'
        elif i == 2:
            return 'down'
        elif i == 3:
            return 'left'
        elif i == 4:
            return 'stay'
        else:
            return 'what the hell'

    def is_too_far(self, p1, p2, distance=3):
        if abs(p1/self.col_num-p2/self.col_num) + abs(p1%self.col_num-p2%self.col_num) >= distance:
            return True
        else:
            return False

    def env_move_forward(self):
        for mushroom_pos in self.mushroom_states:
            if self.mushroom_states[mushroom_pos] > 0:
                self.mushroom_states[mushroom_pos] -= 1

        for carn_init_pos in self.carnivores:
            carn_curr_pos = self.carnivores[carn_init_pos]
            self.curr_carnivores.remove(carn_curr_pos)
            if self.is_too_far(p1=carn_init_pos, p2=carn_curr_pos):
                if carn_curr_pos/self.col_num - carn_init_pos/self.col_num < 0:
                    self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions']['down']
                elif carn_curr_pos/self.col_num - carn_init_pos/self.col_num > 0:
                    self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions']['up']
                elif carn_curr_pos%self.col_num - carn_init_pos%self.col_num > 0:
                    self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions']['right']
                elif carn_curr_pos%self.col_num - carn_init_pos%self.col_num < 0:
                    self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions']['left']
                else:
                    print "how can this be happening!"
            else:
                next_state = self.random_pick_action()
                self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions'][next_state]
            self.curr_carnivores.add(self.carnivores[carn_init_pos])

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
            print self.cells[i]['attribute']+',',
            if i % self.col_num == self.col_num-1:
                print '\n'



