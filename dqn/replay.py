import numpy as np


class ReplayBuffer:
    """
    Buffer for storing transitions.
    Maintains a memory of related states, actions, rewards, new states, and terminal flags.
    """

    def __init__(self, max_size: int, input_shape, n_actions):
        self.max_size = max_size
        self.input_shape = input_shape
        self.memory_counter = 0

        self.state_memory = np.zeros((self.max_size, input_shape))
        self.action_memory = np.zeros(
            (self.max_size, n_actions), dtype=np.int8)
        self.reward_memory = np.zeros((self.max_size))
        self.new_state_memory = np.zeros((self.max_size, input_shape))
        self.terminal_memory = np.zeros(self.max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, is_done):
        """
        Saves a transition to memory.
        If the current memory is full, the saved state will override
        the oldest transition currently in memory.
        """
        index = self.memory_counter % self.max_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 if is_done else 0

        # Actions are discrete, so we turn them into one-hot vectors
        # so we can easily and efficiently use them to calculate
        # loss and reward later.
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0
        self.action_memory[index] = actions

        self.memory_counter += 1

    def sample_batch(self, batch_size):
        """
        Returns a batch of randomly selected transitions in the memory.
        """
        stored_transitions = min(self.memory_counter, self.max_size)
        batch = np.random.choice(stored_transitions, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminals
