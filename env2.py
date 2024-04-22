import random
import visual2
from RL_agent2 import RL


ROWS = 7
COLS = 7
Goal = (5, 5)
walls = [(2, 2), (3, 3), (3, 4), (3, 5)]

class Grid:
    def __init__(self, state):
        self.state = state
        self.board = []
        self.ROWS = ROWS
        self.COLS = COLS
        self.action = None
        self.action_state_pairs = {}
        self.rewards = {}
        self.goal = Goal


        for x in range(ROWS):
            row = []
            for y in range(COLS):
                if (x == 0 or y == 0 or x == ROWS - 1 or y == COLS - 1) or (x, y) in walls :
                    row.append(1)  # border or obsticals
                elif (x, y) == Goal:
                    row.append(3)
                else:
                    row.append(0)  # empty cells
            self.board.append(row)

    def reset(self):
        self.state = (1, 1)
        return self.state

    def take_action(self, action):
        actions = {"up": 0,"down": 1,"right": 2,"left": 3} # just for print random_action_code where action already randomly chosen in main
        random_action_code = list(actions.keys())[action]
        print(f"-------------------------------")
        print(f"Next Move: {random_action_code:} ")
            #visual.GridVisual.show_board(self)
        current_state = self.state
        new_state = self.agentmove(action)
        visual2.GridVisual.show_board(self)
        print(f"State,{current_state}, Action, {action}: {random_action_code} , Next State,  {new_state}")
            # print(f"Action-state pairs: {self.action_state_pairs}")
        print("-------------*********************************---------------")
            # update rewards
        reward = self.reward(current_state, new_state, action)
        if current_state not in self.rewards:
            self.rewards[current_state] = {}
        if action not in self.rewards[current_state]:
            self.rewards[current_state][action] = 0
        self.rewards[current_state][action] += reward
        self.state = new_state  # Update the state to the new state

        done_flag = self.state == Goal # to break the loop later in main

        # if current_state not in self.action_state_pairs:
            #    self.action_state_pairs[current_state] = {}
            #if random_action not in self.action_state_pairs[current_state]:
             #   self.action_state_pairs[current_state][random_action] = []

            #self.action_state_pairs[current_state][random_action].append(new_state)
        # Return the new state, reward, current state, and done flag
        return new_state, reward, current_state, done_flag

    def agentmove(self, direction, print_move=True):
        current_state = self.state
        prev_state = self.state

        # (0: up, 1: down, 2: right, 3: left)
        if direction == 0:
            fcol = self.state[0] + 1
            frow = self.state[1]
        elif direction == 1:
            fcol = self.state[0] -1
            frow = self.state[1]
        elif direction == 2:
            fcol = self.state[0]
            frow = self.state[1] +1
        elif direction == 3:
            fcol = self.state[0]
            frow = self.state[1] -1
        else:
            # invalid direction the agent stay in the same position (in case only but not realy needed)
            fcol = self.state[0]
            frow = self.state[1]

            # check if the new position is valid (not a wall and within the grid)
        if 0 <= fcol < self.ROWS and 0 <= frow < self.COLS and self.board[fcol][frow] != 1:
            # Update the state of the agent
            self.state = (fcol, frow)
            if print_move:
                print(f'//////// agent Previous Location: {prev_state}, Expected New Location : {self.state} ////////')
        else:
            if print_move:
                self.state = prev_state
                print('Faced an Obstacle or a Wall')
                print(f'//////// agent Stay at Location : {self.state} ////////')

        return self.state

    def reward(self, current_state, new_state, action):
        if new_state == current_state:  # agent stays in the same state wall or obstacles
            reward = -0.2
        elif new_state == Goal:  # agent reaches the goal
            reward = 1
        else:  # If the agent moves to a new state (empty cell)
            reward = -0.1

        return reward


# (0: up, 1: down, 2: right, 3: left)
#[6][0] [6][1] [6][2] [6][3] [6][4] [6][5] [6][6]
#[5][0] [5][1] [5][2] [5][3] [5][4] [5][5] [5][6]
#[4][0] [4][1] [4][2] [4][3] [4][4] [4][5] [4][6]
#[3][0] [3][1] [3][2] [3][3] [3][4] [3][5] [3][6]
#[2][0] [2][1] [2][2] [2][3] [2][4] [2][5] [2][6]
#[1][0] [1][1] [1][2] [1][3] [1][4] [1][5] [1][6]
#[0][0] [0][1] [0][2] [0][3] [0][4] [0][5] [0][6]
