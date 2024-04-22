import random

class RL:
    def __init__(self, epsilon, learning_rate):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.q_values = {} # q value table

    def action_policy(self, chosen_action, state, available_actions):
        if random.random() < self.epsilon:
            # explor: choose a random action
            return random.choice(available_actions)
        else:
            # exploit: choose the action with the highest q-value
            if state in self.q_values:
                return max(self.q_values[state], key=self.q_values[state].get) # the key res of the max based on the q-values
            else:
                # if state is not in Q-values, choose a random action
                return random.choice(available_actions)

    def update_q_values(self, state, action, reward, next_state, gamma):
        # init q-value if not present in the q-value table
        if state not in self.q_values:
            self.q_values[state] = {}
        if action not in self.q_values[state]:
            self.q_values[state][action] = 0

            # Get the maximum Q-value for the next state (gamma)
        max_next_q_value = max(self.q_values.get(next_state, {}).values(), default=0)

        # Update q-value using the standard Q-learning update rule with gamma
        self.q_values[state][action] += self.learning_rate * (reward + gamma * max(self.q_values.get(next_state, {}).values(), default=0) - self.q_values[state][action])
        # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))