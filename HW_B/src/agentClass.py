import numpy as np
import random
import math
import h5py
import csv
import matplotlib.pyplot as plt
import matplotlib
from operator import itemgetter
from itertools import groupby
import tensorflow as tf
import torch
import torch.nn.functional as F
import copy
# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count
        self.location = r'C:\Users\erikn\OneDrive - Chalmers\Advanced Machine Learning\HW_B\src'

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions and Q-table and storage for the rewards
        # This function should not return a value, store Q table etc as attributes of self
                 
        self.Q_table = [] # Rows: states, Columns: actions
        self.states = []
        self.actions = []
        self.state_index = 0
        self.action_index = 0
        self.reward_tot = 0
        self.no_rotations = 4
        self.penalty = -10000
        self.reward_tots = []
        self.episode_list = []
        self.reward_avg = []

        # Declaring all possible actions
        for i in range(int(gameboard.N_col)):
            for j in range(self.no_rotations):
                action = (i, j)
                self.actions.append(action)

        # self.fn_read_state()

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.episode_count' the total number of episodes in the training

    def fn_load_strategy(self,strategy_file_Q,strategy_file_S):
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input parameter strategy_file (used to test how the agent plays)
        
        self.Q_table = []
        self.states = []

        with open(strategy_file_Q, 'r') as file:
            csv_reader = csv.reader(file)
            Q_table_string = list(csv_reader)
            for Q_row_string in Q_table_string:
                Q_row_float = []
                for Q_value in Q_row_string:
                    Q_row_float.append(float(Q_value))
                self.Q_table.append(Q_row_float)

        with open(strategy_file_S, 'r') as file:
            for row in file:
                row_split = row.replace('\n','').split(',')
                tile = int(row_split[0])
                board_rows = row_split[1:]
                board=np.empty((self.gameboard.N_row,self.gameboard.N_col),dtype=np.float32)
                for i in range(self.gameboard.N_row):
                    blocks = board_rows[i].replace("  ", " ").strip(" ").split(" ")
                    for j in range(self.gameboard.N_col):
                        board[i,j] = np.float32(blocks[j])
                state = (tile, board)
                self.states.append(state)
        print(len(self.Q_table), len(self.states))
        print("Loading DONE")

    def fn_read_state(self):
        
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the game board
        # You can for example represent the state as an integer entry in the Q-table
        # This function should not return a value, store the state as an attribute of self

        tile_type = self.gameboard.cur_tile_type
        board_occupation = self.gameboard.board.copy()
        state = (tile_type, board_occupation)

        # If state has already been visited
        flag = True
        index = 0
        for tup in self.states:
            if (tup[0] == state[0]) and np.array_equal(state[1], tup[1]):
                flag = False
                self.state_index = index
                break
            index += 1

        # If state has not been visited: compute what actions are valid and add to Q-table
        if flag:
            self.states.append(state)

            Q_values_init = []
            old_tile_x = self.gameboard.tile_x
            old_tile_orientation = self.gameboard.tile_orientation

            for i in range(len(self.actions)):
                action = self.actions[i]
                tile_x = action[0]
                tile_orientation = action[1]
                valid = self.gameboard.fn_move(tile_x, tile_orientation)
                if valid == 1:
                    Q_values_init.append(-1000)
                else:
                    Q_values_init.append(0)

            self.Q_table.append(Q_values_init)
            self.state_index = len(self.states) - 1

            self.gameboard.tile_x = old_tile_x
            self.gameboard.tile_orientation = old_tile_orientation

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):
        
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the Q-table or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and execute the action by moving the tile to the desired position and orientation
        
        r = np.random.uniform()
        if r <= self.epsilon:
            r = random.randrange(len(self.actions))
            action = self.actions[self.action_index]

            # Choose a random action that's valid, i.e. Q != -1000
            # r = random.randrange(len(self.actions))
            # action = self.actions[r]
            # flag = True
            # while flag:
            #     r = random.randrange(len(self.actions))
            #     if self.Q_table[self.state_index][r] != self.penalty:
            #         self.action_index = r
            #         action = self.actions[self.action_index]
            #         flag = False
        else:
            self.action_index = np.argmax(self.Q_table[self.state_index])
            action = self.actions[self.action_index]
            
            # max_val = max(self.Q_table[self.state_index])
            # no_occ = self.Q_table[self.state_index].count(max_val)

            # # Choose a random action between the top actions if there are multiple max values in the Q-table
            # if no_occ > 1:
            #     flag = True
            #     while flag:
            #         r = random.randrange(len(self.actions))
            #         if max_val == self.Q_table[self.state_index][r]:
            #             self.action_index = r
            #             action = self.actions[self.action_index]
            #             flag = False
            # # Choose action corresponding to the max value in the Q-table
            # else:
            #     self.action_index = self.Q_table[self.state_index].index(max_val)
            #     action = self.actions[self.action_index]

        # Move back tile to original position
        tile_x = action[0]
        tile_orientation = action[1]
        self.gameboard.fn_move(tile_x, tile_orientation)

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not
    
    def fn_reinforce(self,old_state_index,reward):
        
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q table using state and action stored as attributes in self and using function arguments for the old state and the reward
        # This function should not return a value, the Q table is stored as an attribute of self
        # if self.init_state == False:

        Q = self.Q_table[old_state_index][self.action_index]
        self.Q_table[old_state_index][self.action_index] = Q + self.alpha*(reward + max(self.Q_table[self.state_index]) - Q)

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1

            index1 = len(self.reward_tots)-100
            index2 = len(self.reward_tots)-1
            if index1 < 0:
                index1 = 0
            avg = np.sum(self.reward_tots[index1:index2])/(len(self.reward_tots[index1:index2]))
            self.reward_avg.append(avg)

            self.episode_list.append(self.episode)
            self.reward_tots.append(self.reward_tot)
            self.reward_tot = 0

            if self.episode%10==0:
                # print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
                print('episode: '+str(self.episode)+'/'+str(self.episode_count), '\t avg. reward: '+str(avg))
            if self.episode%100==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
                    
                    strategy_file_Q = self.location+'strategies/strategy_file_Q_1c.csv'
                    strategy_file_S = self.location+'strategies/strategy_file_S_1c.csv'

                    # with open(strategy_file_Q, 'w', newline="") as file:
                    #      write = csv.writer(file)
                    #      write.writerows(self.Q_table)
                    
                    # with open(strategy_file_S, 'w') as file:
                    #     for row in self.states:
                    #         tile = row[0]
                    #         board = row[1]
                    #         arr_string = ""
                    #         for row_b in board:
                    #             arr_string += str(row_b) + ","
                    #         arr_string = str(tile) + "," + arr_string.strip(',').replace('.', '').replace('[', '').replace(']', '')
                    #         file.write("%s\n" % arr_string)
                    
                figsize = 6
                title = '/R(E)_1c_2'

                fig, ax = plt.subplots(figsize=(figsize,figsize))
                ax.plot(self.episode_list, self.reward_tots, label='Rewards')
                ax.plot(self.episode_list, self.reward_avg, label='Moving avg. rewards')
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.set_box_aspect(1) 
                ax.set_xlim([0,self.episode])
                ax.legend(loc="lower right", prop={'size': 11})
                # plt.savefig(self.location+'plots'+title+'.png')
                plt.close()
            
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()
            old_state_index = self.state_index

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tot += reward
            
            # Read the new state
            self.fn_read_state()
            
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state_index,reward)




class DQN(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden_nodes, num_outputs):
        super(DQN, self).__init__()
        self.linear1 = torch.nn.Linear(num_inputs, num_hidden_nodes, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(num_hidden_nodes, num_hidden_nodes, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(num_hidden_nodes, num_outputs, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size 
        self.sync_target_episode_count=sync_target_episode_count 
        self.episode=0
        self.episode_count=episode_count
        self.path = 'HW_B/dq_model'
        matplotlib.use('Agg')
        self.location = r'C:\Users\erikn\OneDrive - Chalmers\Advanced Machine Learning\HW_B'
    
    def fn_init(self,gameboard):
        self.gameboard=gameboard
        self.exp_buffer = []
        self.state = 0
        self.reward_tot = 0
        self.no_rotations = 4
        self.reward_tots = []
        self.episode_list = []
        self.reward_avg = []
        self.actions = []
        self.action_idx = 0

        # Declaring all possible actions
        for i in range(int(gameboard.N_col)):
            for j in range(self.no_rotations):
                action = (i, j)
                self.actions.append(action)

        # Initializing network
        self.num_inputs = self.gameboard.N_col * self.gameboard.N_row + len(self.gameboard.tiles)
        num_hidden_nodes = 64
        num_outputs = len(self.actions)
        self.DQN_train = DQN(self.num_inputs, num_hidden_nodes, num_outputs)
        self.DQN_test = copy.deepcopy(self.DQN_train)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.DQN_train.parameters(), lr=self.alpha)
        
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

    def fn_load_strategy(self,strategy_file):
        # self.DQN_train = DQNModel()
        self.DQN_train.load_state_dict(torch.load(self.location+r'/DQN_train.pth'))
        self.DQN_test = copy.deepcopy(self.DQN_train)

        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        tile_type = self.gameboard.cur_tile_type
        board_occupation = self.gameboard.board.copy().flatten()
        tile_arr = np.ones(len(self.gameboard.tiles)) * -1
        tile_arr[tile_type] = tile_arr[tile_type] * -1
        self.state = np.concatenate((board_occupation, tile_arr))

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))
    
    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Choose random action
        epsilon = np.maximum(self.epsilon, 1-(self.episode/self.epsilon_scale))
        r = 10#np.random.rand()
        if r <= epsilon:
            r = np.random.randint(0, len(self.actions)-1)
            self.action_idx = r

        # Choose predicted action
        else:
            self.DQN_train.eval()
            input_tensor = torch.tensor(self.state, dtype=torch.float64)
            pred = self.DQN_train(input_tensor).detach().numpy()  
            self.action_idx = np.argmax(pred)

        action = self.actions[self.action_idx]
        tile_x = action[0]
        tile_orientation = action[1]

        self.gameboard.fn_move(tile_x, tile_orientation)

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self,batch):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        old_states = []
        Q_values = []
        Q_labels = []
        self.DQN_train.eval()
        self.DQN_test.eval()

        for exp in batch:
            
            old_state = exp[0]
            action_idx = exp[1]
            reward = exp[2]
            new_state = exp[3]
            terminal = exp[4]

            old_states.append(torch.tensor(old_state, dtype=torch.float64))
            Q_pred = self.DQN_train(torch.tensor(old_state, dtype=torch.float64))
            Q_values.append(Q_pred[action_idx])
            
            pred = self.DQN_test(torch.tensor(new_state, dtype=torch.float64)).detach().numpy()
            Q_label = reward + (np.max(pred) * terminal)
            Q_labels.append(torch.tensor(Q_label, dtype=torch.float64))

        old_states = torch.stack(old_states)
        Q_values = torch.stack(Q_values)
        Q_labels = torch.stack(Q_labels)

        self.DQN_train.train()
        loss = self.criterion(Q_values, Q_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1

            index1 = len(self.reward_tots)-100
            index2 = len(self.reward_tots)-1
            if index1 < 0:
                index1 = 0
            avg = np.sum(self.reward_tots[index1:index2])/(len(self.reward_tots[index1:index2]))
            self.reward_avg.append(avg)

            self.episode_list.append(self.episode)
            self.reward_tots.append(self.reward_tot)
            self.reward_tot = 0
            
            if self.episode%10==0:
                print('episode: '+str(self.episode)+'/'+str(self.episode_count), '\t avg. reward: '+str(avg))
                # print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%10==0:
                saveEpisodes=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,50000,100000,200000,500000,1000000];
                # if self.episode in saveEpisodes:
                #     # TO BE COMPLETED BY STUDENT
                #     # Here you can save the rewards and the Q-network to data files

                #     torch.save(self.DQN_train.state_dict(), self.location+'strategies/DQN_train_b')
                    
                figsize = 6
                title = '/R(E)_2b'

                fig, ax = plt.subplots(figsize=(figsize,figsize))
                ax.plot(self.episode_list, self.reward_tots, label='Rewards')
                ax.plot(self.episode_list, self.reward_avg, label='Moving avg. rewards')
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.set_box_aspect(1) 
                ax.set_xlim([0,self.episode])
                ax.legend(loc="upper left", prop={'size': 11})
                # plt.savefig(self.location+'plots'+title+'.png')
                plt.close()

            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                    self.DQN_test = copy.deepcopy(self.DQN_train)
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            old_state = self.state.copy()

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()
            if self.gameboard.gameover == 1:
                terminal = 0
            else:
                terminal = 1

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tot += reward

            # Read the new state
            self.fn_read_state()
            next_state = self.state.copy()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer
            exp = [old_state, self.action_idx, reward, next_state, terminal]
            self.exp_buffer.append(exp)

            # print(len(self.exp_buffer))
            
            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                batch = random.sample(self.exp_buffer, self.batch_size)
                self.fn_reinforce(batch)
                if len(self.exp_buffer) >= self.replay_buffer_size + 1:
                    self.exp_buffer.pop(0)




class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()