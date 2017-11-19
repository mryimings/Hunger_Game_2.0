import random

action_mapping = {0: 'up', 1: 'right', 2: 'down', 3: 'left', 4: 'stay'}

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
        self.carnivores = {}  # key is initial position for each carnivore, and value is current position
        self.curr_carnivores = {}  # stores the current position for each carnivore, for better search performance
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
                self.curr_carnivores[i] = 1

    def get_reward(self, position):
        curr_att = self.cells[position]['attribute']
        ret_reward = 0
        if curr_att == 'blank':
            ret_reward += -1
        elif curr_att == 'tree':
            ret_reward += (random.randint(0,20)-10)
        elif curr_att == 'trap':
            ret_reward += -100
        elif curr_att == 'mushroom':
            if self.mushroom_states[position] <= 0:
                ret_reward += 20
                self.mushroom_states[position] = 50
            else:
                ret_reward += -1

        if position in self.curr_carnivores:
            ret_reward += -100 * self.curr_carnivores[position]

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

    # judge if the carnivore is too far from its initial position
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
            self.curr_carnivores[carn_curr_pos] -= 1
            if self.curr_carnivores[carn_curr_pos] == 0:
                self.curr_carnivores.pop(carn_curr_pos)

            # if a carnivore goes too far, it will come back
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
                # self.carnivores[carn_init_pos] = carn_init_pos
            else:
                next_state = self.random_pick_action()
                self.carnivores[carn_init_pos] = self.cells[carn_curr_pos]['actions'][next_state]

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
                print 'carnivore,',
            else:
                print self.cells[i]['attribute']+',',
            if i % self.col_num == self.col_num-1:
                print '\n'

    def re_initialize(self):
        for mushroom_pos in self.mushroom_states:
            self.mushroom_states[mushroom_pos] = 0

        self.disaster_area = set()
        self.disaster_last_time = 0

        self.curr_carnivores = {}
        for carn in self.carnivores:
            self.carnivores[carn] = carn
            self.curr_carnivores[carn] = 1
    
    def print_map(self, agent_position):
        #pass
        # tree mush animal carni disa trap
        """print('\033[0m■')  # white
        print('\033[91m■') # red
        print('\033[94m■') # green
        print('\033[93m■') # yellow
        print('\033[0m■')  # white"""
        row = ''
        for i in range(len(self.cells)):
            if i % self.row_num == 0:
                print(row)
                row = ''
            else:
                if i == agent_position:
                    row += '\033[91mx'
                else:
                    if (self.cells[i]['attribute']) == 'blank':
                        row += '\033[0m■' #white
                    elif (self.cells[i]['attribute']) == 'tree':
                        row += '\033[94m■' #green
                    elif (self.cells[i]['attribute']) == 'trap':
                        row += '\033[91m■' #red
                    elif (self.cells[i]['attribute']) == 'mushroom':
                        row += '\033[93m' #yellow
            print(row)
