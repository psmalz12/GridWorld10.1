def extract_optimal_policy(q_values):
    optimal_policy = {}

    for state, actions in q_values.items():
        # find the action with the highest q-value for each state
        optimal_action = max(actions, key=actions.get)
        optimal_policy[state] = optimal_action

    return optimal_policy
