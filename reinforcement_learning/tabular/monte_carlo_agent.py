import numpy as np

class MonteCarloAgent():

    def __init__(self, numberof_episodes=500, learning_rate=0.05, epsilon=0.5, 
                constant_epsilon=False, visit_policy="first", update_policy="online",
                batch_size=40):
        
        self.numberof_episodes = numberof_episodes
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.constant_epsilon = constant_epsilon
        
        self.visit_policy = visit_policy
        self.update_policy = update_policy

        self.batch_size = batch_size \
            if self.update_policy == "batch" \
            else 1

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

    def run_episode(self):
        step, state, reward, done = self.env.reset()
        action = self.choose_action(state)

        episode = [tuple([step, state, action, reward])]

        while not done:
            step, state, reward, done = self.env.step(action)
            action = self.choose_action(state)

            episode.append(tuple([step, state, action, reward]))

        self.episodes.append(episode)

        return episode

    def get_total_reward(self, episode):
        return sum([step[3] for step in episode])

    def get_discounted_reward(self, episode, step):
        discounted_reward = 0

        for i, _, _, reward in episode[step + 1:]:
            relative_step = i - (step + 1)
            discounted_reward += reward * (self.env.get_gamma() ** relative_step)

        return discounted_reward

    def update_action_value(self, state, action, reward):
        reward_error = reward - self.action_values[state, action]
        average_increment = self.learning_rate * reward_error

        self.action_values[state, action] += average_increment

    def update_state_value(self, state, reward):
        reward_error = reward - self.state_values[state]
        average_increment = self.learning_rate * reward_error

        self.state_values[state] += average_increment

    def policy_evaluation(self, episode):
        state_action_pairs = set()

        for step, state, action, _ in episode:
            if self.visit_policy == "first" \
            and tuple([state, action]) in state_action_pairs:
                continue

            state_action_pairs.add(tuple([state, action]))
            
            state_reward = self.get_discounted_reward(episode, step)

            self.update_action_value(state, action, state_reward)
            self.update_state_value(state, state_reward)

    def batch_policy_evaluation(self):
        for i in range(self.batch_size):
            index = len(self.episodes) - self.batch_size + i
            self.policy_evaluation(self.episodes[index])

    def improve_policy(self):
        low_probability = self.epsilon / self.env.get_action_size()
        high_probability = 1 - self.epsilon + low_probability

        for state in range(self.env.get_state_size()):
            best_action = np.argmax(self.action_values[state, :])
                    
            self.policy[state, :] = low_probability
            self.policy[state, best_action] = high_probability
    
    def solve(self, env):
        """
        Solve a given Maze environment using Monte Carlo learning
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

            if i % self.batch_size == 0:
                self.batch_policy_evaluation()
                self.improve_policy()
                
            state_value_history.append(np.copy(self.state_values))
            total_reward_history.append(self.get_total_reward(episode))

        return self.policy, state_value_history, total_reward_history