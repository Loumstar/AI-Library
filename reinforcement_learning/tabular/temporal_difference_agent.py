import numpy as np

class TemporalDifferenceAgent(object):

    def __init__(self, numberof_episodes=500, epsilon=0.5, constant_epsilon=False, learning_rate=0.3):
        self.numberof_episodes = numberof_episodes
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.constant_epsilon = constant_epsilon

        self.env = None

        self.policy = np.array([])
        self.state_values = np.array([])
        self.action_values = np.array([])

        self.episodes = list()

    def set_env(self, env):
        self.env = env

    def initialise_values_and_returns(self):
        state_size = self.env.get_state_size()
        action_size = self.env.get_action_size()

        self.action_values = np.random.rand(state_size, action_size)
        self.state_values = np.zeros(state_size)

        self.policy = np.zeros((state_size, action_size))
        self.improve_policy()

        self.episodes = list()

    def update_epsilon(self, episode_number):
        self.epsilon = (self.numberof_episodes - episode_number) / self.numberof_episodes

    def choose_action(self, state):
        actions = np.arange(self.env.get_action_size())
        return np.random.choice(actions, p=self.policy[state])

    def get_discounted_action_value(self, state, action, reward):
        return reward + self.env.get_gamma() * self.action_values[state, action]

    def get_discounted_state_value(self, state, reward):
        return reward + self.env.get_gamma() * self.state_values[state]

    def get_total_reward(self, episode):
        return sum([step[3] for step in episode])

    def update_action_value(self, prev_state, prev_action, state, action, reward):
        discounted_value = self.get_discounted_action_value(state, action, reward)

        self.action_values[prev_state, prev_action] += self.learning_rate * \
            (discounted_value - self.action_values[prev_state, prev_action])

    def update_state_value(self, prev_state, state, reward):
        discounted_value = self.get_discounted_state_value(state, reward)

        self.state_values[prev_state] += self.learning_rate * \
            (discounted_value - self.state_values[prev_state])

    def improve_policy(self):
        low_probability = self.epsilon / self.env.get_action_size()
        high_probability = 1 - self.epsilon + low_probability

        for state in range(self.env.get_state_size()):
            best_action = np.argmax(self.action_values[state, :])
                    
            self.policy[state, :] = low_probability
            self.policy[state, best_action] = high_probability

    def run_episode(self):
        step, state, reward, done = self.env.reset()
        action = self.choose_action(state)

        episode = [tuple([step, state, action, reward])]

        while not done:
            prev_state = state
            prev_action = action

            step, state, reward, done = self.env.step(action)
            episode.append(tuple([step, state, action, reward]))

            action = self.choose_action(state)

            self.update_action_value(prev_state, prev_action, state, action, reward)
            self.update_state_value(prev_state, state, reward)
            
            self.improve_policy()

        self.episodes.append(episode)

        return episode

    def solve(self, env):
        """
        Solve a given Maze environment using Temporal Difference learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """

        self.set_env(env)
        self.initialise_values_and_returns()

        state_value_history = []
        total_reward_history = []

        for i in range(self.numberof_episodes):
            if not self.constant_epsilon:
                self.update_epsilon(i)

            episode = self.run_episode()
                
            state_value_history.append(np.copy(self.state_values))
            total_reward_history.append(self.get_total_reward(episode))

        return self.policy, state_value_history, total_reward_history