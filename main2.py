from env2 import Grid, Goal
from visual2 import GridVisual
from RL_agent2 import RL
from policy import extract_optimal_policy
from metric import plot_time_taken_and_steps

import random
import time

if __name__ == "__main__":
    # Init RL agent
    epsilon = 0.8 # high mean more explore new action and low ecploit the action with th hight q-value
    learning_rate = 0.1 # low slower update the q-value and highr opset
    gamma = 0.9 # high long tearm reward & low immidiat reward (short tearm)
    rl_agent = RL(epsilon, learning_rate)

    # create grid with RL agent
    grid = Grid(state=(1, 1) )
    print(f"//////// agent current Location {grid.state} and represented  by the symbol: 2 ////////")
    GridVisual.show_board(grid)

    num_iterations = 1000
    time_taken = []  # list to store time take for each iter #tst
    steps_to_goal = []  # list to store number of steps taken to reach goal for each iter #tst

    for iteration in range(num_iterations):
        start_time = time.time()  # set time for current iter #tst

        # interact with the env(grid) until the goal is reached
        while True:
            random_action = random.randint(0, 3)
            available_actions = [0, 1, 2, 3]
            action = rl_agent.action_policy(random_action, grid.state, available_actions)
            # execute the chosen action in the environment
            new_state, reward, current_state, done_flag = grid.take_action(action)
            # Update Q-values using the RL agent
            rl_agent.update_q_values(current_state, action, reward, new_state, gamma)

            if done_flag:  # exit iter loop if the episode is done ==goal found
                break

        end_time = time.time()  # end time for current iter #tst
        time_taken.append(end_time - start_time)  # record time taken for current iter #tst
        steps_to_goal.append(len(grid.rewards))  # record number of steps taken to reach goal #tst

        # reset the agent's location for the next iteration
        grid.reset()
        print("Goal Reached in this iteration!")
        print(f"Total Iteration: {iteration + 1}")

    # Print the updated q-values
    print("Updated Q-values:")
    for state, actions in rl_agent.q_values.items():
        print(f"State: {state}")
        for action, q_value in actions.items():
            print(f"  Action: {action}, Q-value: {q_value:.4g}")

    # extract optimal policy (action)
    optimal_policy = extract_optimal_policy(rl_agent.q_values)
    print("Optimal Policy:")
    for state, action in optimal_policy.items():
        print(f"State: {state}, Optimal Action: {action}")

    # plt metric eval (drow)
    plot_time_taken_and_steps(time_taken, steps_to_goal)

    print("All iterations completed.")
