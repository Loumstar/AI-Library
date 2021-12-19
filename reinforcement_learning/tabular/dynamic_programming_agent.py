import numpy as np

class DynamicProgrammingAgent():

    def __init__(self, threshold=0.0001):
        self.threshold = threshold

        self.policy = np.array([])
        self.values = np.array([])

        self.env = None

    def set_env(self, env):
        self.env = env

    def initialise_values_and_policy(self):
        self.policy = np.zeros((self.env.state_size, self.env.action_size))
        self.values = np.zeros(self.env.state_size)

    def get_action_values(self, state):
        action_values = np.zeros(self.env.action_size)

        transition_matrix = self.env.transition_matrix
        reward_matrix = self.env.reward_matrix

        for next_state in range(self.env.state_size):
            immediate_rewards = reward_matrix[state, next_state, :]
            future_rewards = self.env.discount_rate * self.values[next_state]

            action_values += transition_matrix[state, next_state, :] \
                * (immediate_rewards + future_rewards)

        return action_values

    def policy_evaluation(self):
        new_values = np.zeros(self.env.state_size)

        delta = self.threshold
        epochs = 0

        while delta >= self.threshold:
            epochs += 1

            for state in range(self.env.state_size):
                if self.env.is_absorbing(state):
                    continue

                action_values = self.get_action_values(state)
                state_value = np.sum(self.policy[state, :] * action_values)

                new_values[state] = state_value

            delta = max(abs(new_values - self.values))
            self.values = np.copy(new_values)

        return epochs

    def improve_policy(self):
        policy_stable = True

        for state in range(self.env.state_size):
            if self.is_absorbing_state(state):
                continue

            action_values = self.get_action_values(state)
            best_action = np.argmax(action_values)

            if best_action != np.argmax(self.policy[state, :]):
                policy_stable = False

            self.policy[state, :] = 0
            self.policy[state, best_action] = 1

        return policy_stable

    def solve(self, env):
        self.set_env(env)
        self.initialise_values_and_policy()

        is_stable = False
        epochs = 0

        while not is_stable:
            epochs += 1

            # Step 2: Policy Evaluation
            evaluation_epochs = self.policy_evaluation()
            # Step 3: Policy Improvement
            is_stable = self.improve_policy()

            epochs += evaluation_epochs
        
        # Go back to Step 2 if unstable, else return policy and state value function
        return self.policy, self.values