import random
import torch
import gym

from itertools import islice

from .replay_buffer import Transition, ReplayBuffer
from .dqn_model import DeepQNetwork
from .epsilon import DecayingEpsilon

class Model:
    """
    Class used to create, train and test a learner using 
    Deep Q Learning.
    """
    def __init__(self, state_size, action_size, num_episodes=1000, target_update=10, 
                skip_frames=4, memory_size=32768, batch_size=256, gamma=1, epsilon=None, 
                ddqn=False, experience_replay=True, hidden_layers=2, hidden_layer_size=120, 
                device='cpu', env_name="CartPole-v1", print_step=None):

        self.env_name = env_name
        self.env = None

        # Defined by the environment
        self.state_size = state_size
        self.action_size = action_size

        # Model-specific hyperparameters
        self.num_episodes = num_episodes
        self.target_update = target_update
        self.skip_frames = skip_frames

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.epsilon = epsilon if epsilon is not None \
            else DecayingEpsilon(1, 0.01, 150)

        # Apply Double Deep Q Network Learning
        self.ddqn = ddqn
        # Allow the model to randomly sample from memory
        self.experience_replay = experience_replay

        # Sets the batch size to 1 if experience replay is unselected
        if not self.experience_replay and self.batch_size != 1:
            print("Setting batch size to 1 automatically.")
            self.batch_size = 1

        # Neural Network-specific hyperparameters
        self.hidden_layers = hidden_layers
        self.hidden_layer_size = hidden_layer_size
        
        # Other optionals for model setup
        self.device = device
        self.print_step = print_step

        self._initialise_attributes()

    def _initialise_attributes(self):
        """
        Creates all the memory and methods used for training and 
        optimising the target and network policy.

        In it's own method as these all have to be reinitialised 
        when set_params() is called.
        """
        if self.memory_size < self.batch_size:
            raise ValueError("Batch size is greater than memory size.")

        self.memory = ReplayBuffer(self.memory_size)
        # Frame skips increases the number of inputs into the neural network
        self.inputs = self.state_size * self.skip_frames
        # But action size remains the same regardless of frame_skips
        self.outputs = self.action_size
        # Create the models
        self.target_network = self._initialise_model()
        self.policy_network = self._initialise_model()
        # Set the two networks to be identical, and disable training in the target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters())

    def _initialise_env(self):
        """ Simple method to set up the env for training """
        self.env = gym.make(self.env_name)
        self.env.reset()

    def _initialise_model(self):
        return DeepQNetwork(
            self.inputs, self.outputs, 
            self.hidden_layers, 
            self.hidden_layer_size
        ).to(self.device)

    def set_params(self, **params):
        """
        Method to update any of the hyperparameters while keeping 
        the same model instance. Therefore, need to reinitialise 
        the neural networks, memory and optimiser whenever it get called.

        Used for grid searching/plotting different hyperparameters.
        """
        for param, value in params.items():
            if param not in self.__dict__:
                # Check incase parameter was trying to be set but failed.
                raise ValueError(f"{param} not found in instance.")
            
            setattr(self, param, value)
        
        # Reinitialise the models, memory optimisers, etc.
        self._initialise_attributes()

    def _get_batch(self):
        # Return an empty batch if there aren't enough samples
        if len(self.memory) < self.batch_size:
            return Transition([], [], [], [])
    
        if self.experience_replay:
            # If experience replay is selected, randomly sample from memory
            transitions = self.memory.sample(self.batch_size)
        else:
            # Otherwise sample the most recent steps with batch_size
            slice_point = len(self.memory) - self.batch_size
            transitions = list(islice(self.memory.memory, slice_point, len(self.memory)))
        # Transpose the batch into single transition instance of arrays
        return Transition(*zip(*transitions))


    def _next_state_values(self, next_states, mask):
        # Set all next state values to zero (updated so only final states have 0 value)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        # If not all states are final
        if sum(mask) > 0:
            if self.ddqn:
                # Get the actions that maximise reward from the policy network
                policy_state_action_values = self.policy_network(next_states)
                actions = policy_state_action_values.max(1)[1].detach().unsqueeze(1)
                # Get the state action values of these actions when predicted by the target network
                target_state_action_values = self.target_network(next_states).gather(1, actions)
                # Save these to next state values
                next_state_values[mask] = target_state_action_values.squeeze(1)
            else:
                # Get the state action values only from the policy network
                next_state_values[mask] = self.target_network(next_states).max(1)[0].detach()

        return next_state_values

    def _optimise_model(self):
        """
        Optimisation function.
        Slightly altered from the boilerplate code.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Take a sample of steps from memory
        batch = self._get_batch()

        # Get the mask of non-final states for state_batch
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            device=self.device, dtype=torch.bool)

        # Reduce the state batch into those that aren't terminal
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) \
            if sum(non_final_mask) > 0 else torch.empty(0, self.inputs).to(self.device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_network(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            next_state_values = self._next_state_values(non_final_next_states, non_final_mask)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute loss using MSE
        loss = ((state_action_values - expected_state_action_values.unsqueeze(1)) ** 2).sum()

        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()

        # Limit magnitude of gradient for update step
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)

        # Update the model
        self.optimiser.step()

    def _multiply_state(self, state):
        """
        Method that concatenates the same state self.skip_frames times.
        
        Used exclusively for getting deciding the first action 
        given the starting state.
        """
        if state is None:
            return None
        
        states_tuple = tuple([state] * self.skip_frames)
        return torch.cat(states_tuple, 1)

    def _get_state(self):
        # Returns the current state as a float in a 2D torch tensor
        return torch.tensor(self.env.state).float().unsqueeze(0).to(self.device)

    def _select_action_search(self, state):
        """
        Method that selects an action based on the epsilon schedule 
        and policy network. For tranining only.
        """
        if random.random() > self.epsilon():
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1)

        return torch.tensor(
            [[random.randrange(self.action_size)]], 
            device=self.device, 
            dtype=torch.long
        )

    def _run_training_episode(self):
        """
        Method for running an episode while training.
        Main difference between this and self.run_episode() is 
        """
        self.env.reset()
        rewards = 0

        state = self._get_state()
        combined_states = self._multiply_state(state)
        
        done = False

        while not done:
            skipped_frames_rewards = 0
            action = self._select_action_search(combined_states)

            # Create an empty tensor for concatenating k frames together as inputs to the NN
            next_combined_states = torch.zeros((1, self.inputs), device=self.device)
            # For each frame k skipped 
            for k in range(0, self.inputs, self.state_size):
                # Execute the same action repeatedly
                _, reward, done, _ = self.env.step(action.item())
                skipped_frames_rewards += reward

                if done:
                    next_state = None
                    next_combined_states = None
                    break

                # Record the state and add it to the tensor of combined states
                next_state = self._get_state()
                next_combined_states[:, k:k+self.state_size] = next_state

            rewards += skipped_frames_rewards

            # Convert the rewards to a tensor for memory
            skipped_frames_rewards = torch.tensor(
                [skipped_frames_rewards],
                device=self.device)

            # Update the result of the skipped frames to memory
            self.memory.push(
                combined_states, 
                action, 
                next_combined_states, 
                skipped_frames_rewards)
            
            combined_states = next_combined_states
            state = next_state

            # Apply gradient descent
            self._optimise_model()

        # Return the total rewards of the episode
        return rewards

    def train(self):
        """
        Method to train the policy and target networks.
        Again, based off the boilerplate with modifications.
        """
        # Create a new environment
        self._initialise_env()

        reward_history = list()
        epsilon_history = list()

        for i in range(self.num_episodes):
            # Print the current episode running       
            if self.print_step is not None and i % self.print_step == 0:
                print(f"episode {i}/{self.num_episodes}.")

            # Update epsilon and run the episode
            self.epsilon.step(i)
            rewards = self._run_training_episode()

            reward_history.append(rewards)
            epsilon_history.append(self.epsilon())

            # Update the target network every self.target_update epsiodes.
            if i % self.target_update == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # Close the environment
        self.env.close()

        return reward_history, epsilon_history

    def select_action(self, state):
        with torch.no_grad():
            # Select the most sensible action using the target network
            return self.target_network(state).max(1)[1].view(1, 1)

    def run_episode(self, env=None, recorder=None):
        """
        Very similar method to _run_episode_training, but with options 
        to include your own environment and a recorder instance.

        No training is done, and actions are selected wholly based on 
        the target network (and no searching).
        """
        if env is None: 
            self._initialise_env()
        else:
            self.env = env
            self.env.reset()

        # Get state and concatenate with itself to get the first action
        state = self._get_state()
        combined_states = self._multiply_state(state)

        done = False
        timestep = 0
        total_reward = 0

        while not done:
            action = self.select_action(combined_states)

            # Reset the state to an empty tensor for concatenating multiple states
            combined_states = torch.empty((1, self.inputs), device=self.device)
            
            # For each frame k skipped
            for k in range(0, self.inputs, self.state_size):
                # Capture frame if recorder has been specified
                if recorder is not None:
                    recorder.capture_frame()
                
                # Repeat the same action
                _, reward, done, _ = self.env.step(action.item())
                
                # Sum the rewards and increment timestep
                total_reward += reward
                timestep += 1

                if done:
                    next_state = None
                    combined_states = None
                    break
       
                # Get current state and add it to the combined set of states
                next_state = self._get_state()
                combined_states[:, k:k+self.state_size] = next_state
        
        # Close if env was set by the model instance
        if env is None: self.env.close()

        return timestep, total_reward