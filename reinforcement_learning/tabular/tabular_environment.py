import numpy as np

import tabulate

class BaseTabularEnvironment():
    def __init__(self, shape=(10, 10), action_size=1):
        self.height, self.width = shape

        self.action_size = action_size
        self.state_size = self.height * self.width

        self.grid = np.arange(self.state_size).reshape(shape)

        self.neighbours = np.zeros((self.state_size, self.state_size))
        self.transition = np.zeros((self.state_size, self.state_size, self.action_size))
        self.rewards = np.zeros((self.state_size, self.state_size, self.action_size))

    def build_maze(self):
        raise NotImplementedError

    def is_absorbing_coord(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def show(self):
        raise NotImplementedError


class GridWorld(BaseTabularEnvironment):

    def __init__(self, shape=(10, 10), action_size=4, obstaces=[], absorbing=[(10, 10)],
                rewards=[500], step_reward=-1, prob_intended=0.75):
        
        super().__init__(shape, action_size)

        self.step_reward = step_reward
        
        self.prob_intended = prob_intended
        self.prob_unintendd = (1 - prob_intended) / (action_size - 1)
        
        self.obstacles = obstaces
        self.absorbing = absorbing
        self.absorbing_rewards = rewards

        self.build_maze()

    def grid_coord_to_state(self, coord):
        return self.grid[coord]
    
    def state_to_grid_coord(self, state):
        if self.invalid_state(state):
            return None

        row = state // self.width
        col = state % self.width

        return (row, col)

    def invalid_coord(self, coord):
        row, col = coord

        return (row < 0 or row >= self.height) \
            or (col < 0 or col >= self.width)

    def invalid_state(self, state):
        return state < 0 or state >= self.state_size

    def is_obstacle_coord(self, coord):
        return coord in self.obstacles

    def is_absorbing_coord(self, coord):
        return coord in self.absorbing

    def is_obstacle_state(self, state):
        coord = self.state_to_grid_coord(state)
        return self.is_obstacle_coord(coord)

    def is_absorbing_state(self, state):
        coord = self.state_to_grid_coord(state)
        return self.is_absorbing_coord(coord)

    def get_neighbours(self, state):
        row, col = self.state_to_grid_coord(state)
        
        neighbour_states = list()
        neighbour_cells = [
            (row - 1, col), # Left
            (row + 1, col), # Right
            (row, col - 1), # Up
            (row, col + 1)  # Down
        ]

        for n in neighbour_cells:
            if self.invalid_coord(n) or self.is_obstacle_coord(n):
                n_state = state
            else:
                n_state = self.grid_coord_to_state(n)
            
            neighbour_states.append(n_state)

        return neighbour_states

    def get_transition_matrix(self, state):
        row, col = self.state_to_grid_coord(state)
        neighbours = self.neighbours[state]

        for action in range(self.action_size):
            n = neighbours[]
            self.transition[state, col, action] += self.prob_intended

    def get_reward_matrix(coord):
        pass

    def build_maze(self):
        for state in range(self.state_size):
            self.neighbours[state, :] = self.get_neighbours(state)
            
            self.update_transition_matrix(state)
            self.update_rewards_matrix(state)